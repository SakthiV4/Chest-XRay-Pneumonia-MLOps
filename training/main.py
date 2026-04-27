"""
main.py -- Training entry point for Chest X-Ray Pneumonia Detection.

Usage:
    python training/main.py --pretrained --endEpoch 10 --batchSize 32
"""
import argparse
import os
import sys
import multiprocessing
from pathlib import Path

import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as trainer

BASE_DIR = Path(__file__).resolve().parent.parent
device   = 'cuda' if torch.cuda.is_available() else 'cpu'

# ── Transforms (PIL-based for ImageFolder) ────────────────────────────────────
transforms_train = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.1, contrast=0.1),
    T.RandomRotation(5),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])

transforms_val = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std =[0.229, 0.224, 0.225]),
])


def parse_args():
    parser = argparse.ArgumentParser(description='Chest X-Ray Pneumonia -- ResNet-50')
    parser.add_argument('--trainPath',      type=str,
                        default=str(BASE_DIR / 'Data/chest_xray/chest_xray/train'))
    parser.add_argument('--valPath',        type=str,
                        default=str(BASE_DIR / 'Data/chest_xray/chest_xray/test'),
                        help='Using test split (official val only has 16 images)')
    parser.add_argument('--batchSize',      type=int,   default=32)
    parser.add_argument('--lr',             type=float, default=0.001)
    parser.add_argument('--endEpoch',       type=int,   default=10)
    parser.add_argument('--startEpoch',     type=int,   default=0)
    parser.add_argument('--pretrained',     action='store_true', default=False)
    parser.add_argument('--experimentName', type=str,   default='ResNet50_Pneumonia')
    parser.add_argument('--trainLimit',     type=int,   default=-1)
    parser.add_argument('--valLimit',       type=int,   default=-1)
    parser.add_argument('--resume',         action='store_true', default=False)
    parser.add_argument('--checkpoint',     type=str,   default=None)
    return parser.parse_args()


# ── MUST be inside __main__ guard for Windows multiprocessing (num_workers > 0)
if __name__ == '__main__':
    multiprocessing.freeze_support()

    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  Chest X-Ray Pneumonia Detection -- ResNet-50")
    gpu_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    print(f"  Device : {device}  ({gpu_name})")
    print(f"  Config : {vars(args)}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading datasets...")
    trn_dl = trainer.get_data(args.trainPath, args.trainLimit,
                               args.batchSize, transforms_train)
    val_dl = trainer.get_data(args.valPath,   args.valLimit,
                               args.batchSize, transforms_val)
    print("DataLoaders ready.\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model, loss_fn, optimizer, scheduler = trainer.get_model(args.pretrained, args.lr)

    if args.resume and args.checkpoint:
        print(f"Resuming from: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        model.to(device)

    # ── Output dirs ───────────────────────────────────────────────────────────
    tb_path          = str(BASE_DIR / 'runs' /
                           f"{args.experimentName}_bs{args.batchSize}_lr{args.lr}")
    saved_models_dir = str(BASE_DIR / 'saved_models' / args.experimentName)

    writer = SummaryWriter(tb_path)
    os.makedirs(saved_models_dir, exist_ok=True)

    print(f"TensorBoard -> {tb_path}")
    print(f"Checkpoints -> {saved_models_dir}\n")

    # ── Train ─────────────────────────────────────────────────────────────────
    best_acc = trainer.train(
        model, trn_dl, val_dl, loss_fn, optimizer, scheduler,
        args.startEpoch, args.endEpoch, writer, args, saved_models_dir
    )

    writer.close()
    print(f"\nDone! Best val acc: {best_acc:.4f}")
    print(f"Best model: {saved_models_dir}/best_model.pth")
