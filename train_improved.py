"""
YOLOv8 小数据集（~1000 张）改进训练脚本

改进方案（易操作，适合课程/毕业设计快速复现）：
1) 损失函数改进：CIoU -> Wise-IoU（WIoU v1）
2) 输入预处理/增强策略改进：小数据集两阶段训练（先强增广再弱增广）

支持：
- 仅训练基线（baseline）
- 仅训练改进版（improved）
- 同时训练并自动输出对比表（both）

示例：
python train_improved.py --data dataset.yaml --mode both --epochs 120 --batch 16 --imgsz 640 --small-data
"""

from __future__ import annotations

import argparse
import csv
import types
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("YOLOv8 baseline vs improved (WIoU + preprocessing policy)")
    p.add_argument("--data", type=str, default="dataset.yaml", help="数据集配置 yaml")
    p.add_argument("--mode", type=str, default="both", choices=["baseline", "improved", "both"], help="训练模式")
    p.add_argument("--epochs", type=int, default=120, help="总训练轮数（1000 张数据建议 80~150）")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--lr0", type=float, default=0.003, help="初始学习率，小数据集建议适当降低")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu")

    # 小数据集两阶段策略
    p.add_argument("--small-data", action="store_true", help="启用两阶段策略（推荐 1000 张左右数据）")
    p.add_argument("--freeze-epochs", type=int, default=30, help="阶段1冻结训练轮数")
    p.add_argument("--mixup-stage1", type=float, default=0.2, help="阶段1 MixUp 强度")
    p.add_argument("--mixup-stage2", type=float, default=0.0, help="阶段2 MixUp 强度")
    p.add_argument("--workers", type=int, default=8)
    return p.parse_args()


def print_env() -> None:
    print("\n" + "=" * 60)
    print("环境信息")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU Memory: {mem:.2f} GB")
    print("=" * 60 + "\n")


def infer_num_classes(data_yaml: str) -> int:
    try:
        with open(data_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if isinstance(cfg.get("nc"), int) and cfg["nc"] > 0:
            return int(cfg["nc"])
    except Exception as e:  # noqa: BLE001
        print(f"[Warn] 读取 nc 失败，使用默认 1 类。原因: {e}")
    return 1


class WIoULoss(torch.nn.Module):
    """
    Wise-IoU (简化实用版)。
    返回值是 loss（越小越好），可直接替代 bbox IoU loss。
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        # boxes: xyxy
        px1, py1, px2, py2 = pred_boxes.unbind(-1)
        tx1, ty1, tx2, ty2 = target_boxes.unbind(-1)

        inter_x1 = torch.maximum(px1, tx1)
        inter_y1 = torch.maximum(py1, ty1)
        inter_x2 = torch.minimum(px2, tx2)
        inter_y2 = torch.minimum(py2, ty2)

        inter_w = (inter_x2 - inter_x1).clamp(min=0)
        inter_h = (inter_y2 - inter_y1).clamp(min=0)
        inter = inter_w * inter_h

        area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
        area_t = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
        union = area_p + area_t - inter + self.eps
        iou = inter / union

        pcx = (px1 + px2) / 2
        pcy = (py1 + py2) / 2
        tcx = (tx1 + tx2) / 2
        tcy = (ty1 + ty2) / 2
        center_dist2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2

        ex1 = torch.minimum(px1, tx1)
        ey1 = torch.minimum(py1, ty1)
        ex2 = torch.maximum(px2, tx2)
        ey2 = torch.maximum(py2, ty2)
        diagonal2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + self.eps

        # WIoU 的核心思想：根据样本几何关系动态调节梯度权重
        distance_penalty = center_dist2 / diagonal2
        dynamic_weight = torch.exp(distance_penalty.detach())  # 放大困难样本
        wiou_loss = (1.0 - iou) * dynamic_weight
        return wiou_loss


def patch_bbox_loss_with_wiou(trainer: Any, wiou: WIoULoss) -> bool:
    """将 Ultralytics Trainer 的 bbox loss 替换为 WIoU。"""
    if not hasattr(trainer, "compute_loss"):
        return False
    compute_loss = trainer.compute_loss
    if not hasattr(compute_loss, "bbox_loss"):
        return False

    bbox_loss = compute_loss.bbox_loss

    def patched_forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        if fg_mask.sum():
            pred_b = pred_bboxes[fg_mask]
            tgt_b = target_bboxes[fg_mask]
            weight = target_scores[fg_mask].sum(-1, keepdim=True)

            iou_loss = wiou(pred_b, tgt_b)
            box_loss = (iou_loss * weight).sum() / target_scores_sum

            dfl_loss = self._df_loss(pred_dist[fg_mask], tgt_b) * weight
            dfl_loss = dfl_loss.sum() / target_scores_sum
        else:
            box_loss = pred_bboxes.sum() * 0.0
            dfl_loss = pred_bboxes.sum() * 0.0

        return box_loss * self.hyp.box, dfl_loss * self.hyp.dfl

    bbox_loss.forward = types.MethodType(patched_forward, bbox_loss)
    return True


def inject_wiou_callback(model: YOLO) -> None:
    wiou = WIoULoss()

    def on_train_start(trainer):
        ok = patch_bbox_loss_with_wiou(trainer, wiou)
        if ok:
            print("[OK] WIoU 已成功注入，替换 CIoU")
        else:
            print("[Warn] WIoU 注入失败，继续使用默认 IoU loss")

    model.add_callback("on_train_start", on_train_start)


def base_train_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "optimizer": "AdamW",
        "lr0": args.lr0,
        "lrf": 0.01,
        "weight_decay": args.weight_decay,
        "patience": 30,
        "cos_lr": True,
        "val": True,
        "plots": True,
        # 输入预处理/增强策略（可直接在课程报告中解释）
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "mosaic": 1.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.3,
        "shear": 0.0,
        "fliplr": 0.5,
    }


def train_baseline(args: argparse.Namespace):
    print("\n" + "-" * 60)
    print("训练 baseline: 原版 YOLOv8n")
    print("-" * 60)
    model = YOLO("yolov8n.pt")
    train_args = base_train_args(args)
    train_args.update({"name": "yolov8_baseline", "mixup": 0.0})
    return model.train(**train_args)


def train_improved(args: argparse.Namespace):
    print("\n" + "*" * 60)
    print("训练 improved: YOLOv8n + WIoU + 小数据预处理策略")
    print("*" * 60)
    model = YOLO("yolov8n.pt")
    inject_wiou_callback(model)

    train_args = base_train_args(args)
    train_args.update({"name": "yolov8_improved"})

    if args.small_data:
        total = args.epochs
        stage1_epochs = min(max(args.freeze_epochs, 1), max(total - 1, 1))
        stage2_epochs = max(total - stage1_epochs, 1)

        print("\n[两阶段策略]")
        print(f"stage1: epochs={stage1_epochs}, freeze=10, mixup={args.mixup_stage1}, mosaic=1.0")
        print(f"stage2: epochs={stage2_epochs}, freeze=0,  mixup={args.mixup_stage2}, mosaic=0.2")

        stage1 = dict(train_args)
        stage1.update({
            "epochs": stage1_epochs,
            "freeze": 10,
            "mixup": args.mixup_stage1,
            "mosaic": 1.0,
            "close_mosaic": 0,
            "name": "yolov8_improved_stage1",
        })
        s1_result = model.train(**stage1)

        last = Path(s1_result.save_dir) / "weights" / "last.pt"
        stage2_model = YOLO(str(last)) if last.exists() else model
        inject_wiou_callback(stage2_model)

        stage2 = dict(train_args)
        stage2.update({
            "epochs": stage2_epochs,
            "freeze": 0,
            "mixup": args.mixup_stage2,
            "mosaic": 0.2,
            "close_mosaic": max(stage2_epochs // 2, 1),
            "resume": False,
            "name": "yolov8_improved",
        })
        return stage2_model.train(**stage2)

    # 单阶段改进训练
    train_args.update({"mixup": 0.1, "mosaic": 0.8, "close_mosaic": 10})
    return model.train(**train_args)


def _read_last_metrics(run_dir: str) -> Optional[Dict[str, float]]:
    csv_path = Path(run_dir) / "results.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[-1]
    return {
        "mAP50": float(row.get("metrics/mAP50(B)", 0.0)),
        "mAP50-95": float(row.get("metrics/mAP50-95(B)", 0.0)),
        "precision": float(row.get("metrics/precision(B)", 0.0)),
        "recall": float(row.get("metrics/recall(B)", 0.0)),
    }


def print_comparison(base_dir: str, imp_dir: str) -> None:
    base = _read_last_metrics(base_dir)
    imp = _read_last_metrics(imp_dir)
    print("\n" + "=" * 70)
    print("改进前后对比（mAP / Recall / Precision）")
    print("=" * 70)
    print(f"{'Metric':<12}{'Baseline':>12}{'Improved':>12}{'Delta':>12}")
    print("-" * 70)
    if not base or not imp:
        print("结果文件缺失，请确认训练完成并存在 results.csv")
        print("=" * 70)
        return

    for k in ["mAP50", "mAP50-95", "precision", "recall"]:
        b = base[k]
        i = imp[k]
        d = i - b
        print(f"{k:<12}{b:>12.4f}{i:>12.4f}{d:>+12.4f}")

    print("=" * 70)
    print(f"Baseline 目录: {base_dir}")
    print(f"Improved 目录: {imp_dir}")


def main() -> None:
    args = parse_args()
    print_env()
    print(f"检测到类别数 nc = {infer_num_classes(args.data)}（来自 {args.data}）")

    if args.mode == "baseline":
        train_baseline(args)
    elif args.mode == "improved":
        train_improved(args)
    else:
        train_baseline(args)
        train_improved(args)
        print_comparison("runs/detect/yolov8_baseline", "runs/detect/yolov8_improved")

    print("\n训练结束。")
    print("可关注目录：")
    print("- runs/detect/yolov8_baseline")
    print("- runs/detect/yolov8_improved")


if __name__ == "__main__":
    main()
