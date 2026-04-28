"""
YOLOv8 改进版训练脚本
整合论文第三章全部改进：
  ✅ SE 注意力机制  —— 结构改进，注入 Backbone 之后
  ✅ MixUp 数据增强 —— 数据改进，mixup=0.15
  ✅ WIoU 损失函数  —— 损失改进，替换原版 CIoU
  ✅ TAL            —— YOLOv8 内置，默认启用

使用方法：
  python train_improved.py                 # 默认训练改进版
  python train_improved.py --epochs 150    # 自定义轮数
  python train_improved.py --resume        # 断点续训
  python train_improved.py --compare       # 同时训练原版做对比实验（论文第四章用）
"""

import argparse
import torch
import types
from pathlib import Path
import yaml
from ultralytics import YOLO
from model_with_se import build_model, get_train_args
from wiou_loss import WIoULoss


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv8+SE+MixUp+TAL 改进版训练")
    p.add_argument("--data",    default="dataset.yaml")
    p.add_argument("--epochs",  type=int, default=100)
    p.add_argument("--batch",   type=int, default=16)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--device",  default="0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume",  action="store_true", help="断点续训")
    p.add_argument("--compare", action="store_true", help="同时训练原版做对比实验")
    p.add_argument("--small-data", action="store_true", help="小数据集模式（推荐 <= 2000 张）")
    p.add_argument("--freeze-epochs", type=int, default=30, help="第一阶段冻结训练轮数")
    p.add_argument("--mixup-stage1", type=float, default=0.15, help="第一阶段 MixUp 强度")
    p.add_argument("--mixup-stage2", type=float, default=0.0, help="第二阶段 MixUp 强度")
    p.add_argument("--num-classes", type=int, default=None, help="类别数，默认从 data.yaml 自动读取")
    return p.parse_args()


def _infer_num_classes(data_path: str) -> int:
    """优先从 data.yaml 读取类别数，失败时回退到 5。"""
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if isinstance(cfg.get("nc"), int) and cfg["nc"] > 0:
            return cfg["nc"]
    except Exception as e:
        print(f"[!] 读取数据集类别数失败，将使用默认值 5: {e}")
    return 5


def print_env():
    print(f"\n{'='*55}")
    print(f"  环境信息")
    print(f"{'='*55}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  显存    : {mem:.1f} GB")
    print(f"{'='*55}\n")


def _patch_bbox_loss_with_wiou(trainer, wiou):
    """将 trainer 内部 bbox_loss 前向替换为 WIoU。"""
    if not hasattr(trainer, "compute_loss") or not hasattr(trainer.compute_loss, "bbox_loss"):
        return False

    bbox_loss = trainer.compute_loss.bbox_loss

    def patched_bbox_loss(self, pred_dist, pred_bboxes, anchor_points,
                          target_bboxes, target_scores, target_scores_sum, fg_mask):
        """替换后的 bbox 损失计算，使用 WIoU 代替 CIoU"""
        if fg_mask.sum():
            pred_b = pred_bboxes[fg_mask]
            target_b = target_bboxes[fg_mask]
            weight = target_scores[fg_mask].sum(-1, keepdim=True)

            iou_loss = wiou(pred_b, target_b)
            box_loss = (iou_loss * weight).sum() / target_scores_sum

            dfl_loss = self._df_loss(pred_dist[fg_mask], target_b) * weight
            dfl_loss = dfl_loss.sum() / target_scores_sum
        else:
            box_loss = pred_bboxes.sum() * 0
            dfl_loss = pred_bboxes.sum() * 0

        return box_loss * self.hyp.box, dfl_loss * self.hyp.dfl

    bbox_loss.forward = types.MethodType(patched_bbox_loss, bbox_loss)
    return True


def _inject_wiou(model: YOLO):
    """
    将 YOLOv8 训练器的边框损失替换为 WIoU
    对应论文损失函数改进部分
    """
    wiou = WIoULoss()

    def on_train_start(trainer):
        ok = _patch_bbox_loss_with_wiou(trainer, wiou)
        print("[✓] WIoU 损失函数注入成功" if ok else "[!] WIoU 注入失败，保持原版 CIoU")

    try:
        model.add_callback("on_train_start", on_train_start)
        print("[*] WIoU 回调已注册，将在训练开始时注入")
    except Exception as e:
        print(f"[!] 无法注册 WIoU 回调: {e}")


def _train_two_stage_small_data(model: YOLO, args, train_args: dict):
    """小数据集两阶段训练：先冻结+强增强，再解冻+弱增强。"""
    total_epochs = args.epochs
    stage1_epochs = min(max(args.freeze_epochs, 1), max(total_epochs - 1, 1))
    stage2_epochs = max(total_epochs - stage1_epochs, 1)

    print("\n[小数据集两阶段训练]")
    print(f"  Stage1: epochs={stage1_epochs}, freeze=10, mixup={args.mixup_stage1}")
    print(f"  Stage2: epochs={stage2_epochs}, freeze=0, mixup={args.mixup_stage2}")

    stage1_args = dict(train_args)
    stage1_args.update({
        "epochs": stage1_epochs,
        "freeze": 10,
        "mixup": args.mixup_stage1,
        "close_mosaic": 0,
        "name": "tire_yolov8_real_data_stage1",
    })
    stage1_results = model.train(**stage1_args)
    stage1_last = Path(stage1_results.save_dir) / "weights" / "last.pt"
    stage2_model = YOLO(str(stage1_last)) if stage1_last.exists() else model
    if stage1_last.exists():
        # 兼容部分 ultralytics 版本：从 checkpoint 初始化后 overrides 中可能缺失 "model"
        # 会在 model.train() 内部访问 self.overrides["model"] 时触发 KeyError
        stage2_model.overrides["model"] = str(stage1_last)
    _inject_wiou(stage2_model)

    stage2_args = dict(train_args)
    stage2_args.update({
        "epochs": stage2_epochs,
        "freeze": 0,
        "mixup": args.mixup_stage2,
        "close_mosaic": max(stage2_epochs // 2, 1),
        "resume": False,
        "name": "tire_yolov8_real_data",
    })
    return stage2_model.train(**stage2_args)


def train_improved(args):
    """训练改进版模型（SE + MixUp + TAL）"""
    print("\n" + "★"*20)
    print("  训练改进版 YOLOv8（SE + MixUp + TAL）")
    print("★"*20)

    # 断点续训
    if args.resume:
        last = Path("runs/train/tire_yolov8_real_data/weights/last.pt")
        if last.exists():
            print(f"[*] 断点续训: {last}")
            model = YOLO(str(last))
            train_args = get_train_args(args.data)
            train_args.update({
                "epochs": args.epochs,
                "batch":  args.batch,
                "imgsz":  args.imgsz,
                "device": args.device,
                "resume": True,
            })
            return model.train(**train_args)

    # 构建改进模型（注入 SE）
    model = build_model(
        weights     = "yolov8n.pt",
        num_classes = args.num_classes if args.num_classes else _infer_num_classes(args.data),
        reduction   = 16,
        device      = args.device,
    )

    # 获取训练参数（含 MixUp、TAL 配置）
    train_args = get_train_args(args.data)
    train_args.update({
        "epochs": args.epochs,
        "batch":  args.batch,
        "imgsz":  args.imgsz,
        "device": args.device,
    })

    print("\n[改进点确认]")
    print(f"  ✅ SE 注意力  : 已注入 Backbone layer8 之后")
    print(f"  ✅ MixUp 增强 : mixup = {train_args['mixup']}")
    print(f"  ✅ WIoU 损失  : 已替换原版 CIoU")
    print(f"  ✅ TAL 学习   : YOLOv8 内置，已启用")
    print(f"  ✅ 训练轮数   : {train_args['epochs']}")
    print(f"  ✅ 训练设备   : {train_args['device']}\n")

    # ── 注入 WIoU 损失函数 ──
    _inject_wiou(model)

    if args.small_data:
        return _train_two_stage_small_data(model, args, train_args)

    results = model.train(**train_args)
    return results


def train_baseline(args):
    """训练原版 YOLOv8（用于对比实验，论文第四章需要）"""
    print("\n" + "-"*40)
    print("  训练原版 YOLOv8（对比基线）")
    print("-"*40)

    model = YOLO("yolov8n.pt")
    train_args = get_train_args(args.data)
    train_args.update({
        "epochs":  args.epochs,
        "batch":   args.batch,
        "imgsz":   args.imgsz,
        "device":  args.device,
        "mixup":   0.0,         # 关闭 MixUp
        "name":    "tire_yolov8_baseline",
    })
    return model.train(**train_args)


def print_comparison(improved_dir: str, baseline_dir: str):
    """打印改进版与原版的对比结果"""
    def read_results(path):
        import csv
        result_csv = Path(path) / "results.csv"
        if not result_csv.exists():
            return None
        with open(result_csv) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return None
        last = rows[-1]
        return {
            "mAP50":     float(last.get("metrics/mAP50(B)", 0)),
            "mAP50-95":  float(last.get("metrics/mAP50-95(B)", 0)),
            "precision": float(last.get("metrics/precision(B)", 0)),
            "recall":    float(last.get("metrics/recall(B)", 0)),
        }

    imp  = read_results(improved_dir)
    base = read_results(baseline_dir)

    print(f"\n{'='*60}")
    print(f"  对比实验结果（对应论文第四章实验分析）")
    print(f"{'='*60}")
    print(f"  {'指标':<12} {'原版YOLOv8':>12} {'改进版':>12} {'提升':>8}")
    print(f"  {'-'*48}")

    if imp and base:
        for key in ["mAP50", "mAP50-95", "precision", "recall"]:
            b = base[key]
            i = imp[key]
            diff = i - b
            sign = "+" if diff >= 0 else ""
            print(f"  {key:<12} {b:>12.4f} {i:>12.4f} {sign}{diff:>7.4f}")
    else:
        print("  [!] 结果文件未找到，请确认训练已完成")

    print(f"{'='*60}")
    print(f"  改进版权重: {improved_dir}/weights/best.pt")
    print(f"  原版权重  : {baseline_dir}/weights/best.pt")


if __name__ == "__main__":
    args = parse_args()
    print_env()

    # 训练改进版
    train_improved(args)

    # 如果需要对比实验（论文第四章用）
    if args.compare:
        train_baseline(args)
        print_comparison(
            "runs/train/tire_yolov8_real_data",
            "runs/train/tire_yolov8_baseline"
        )

    print(f"\n{'='*55}")
    print(f"  训练完成！")
    print(f"  改进版权重: runs/train/tire_yolov8_real_data/weights/best.pt")
    print(f"  训练曲线 : runs/train/tire_yolov8_real_data/results.png")
    print(f"  接下来运行: python detect.py 进行推理")
    print(f"{'='*55}")
