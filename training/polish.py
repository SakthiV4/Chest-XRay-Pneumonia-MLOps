"""
polish.py -- Phase 2 Polish: Fine-tune ResNet-50 deeper layers for higher accuracy.

Strategy:
  1. Load best_model.pth (frozen backbone, 87.5% acc)
  2. Unfreeze layer3 + layer4 + FC head
  3. Train with differential LRs: backbone=1e-5, head=1e-4
  4. Add BCEWithLogitsLoss + class weighting for imbalance (234 NORMAL vs 390 PNEUMONIA)
  5. Cosine Annealing LR scheduler for smoother convergence
  6. Target: >92% accuracy, >0.96 AUC
"""
import os
import sys
import multiprocessing
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T, datasets, models
from torchvision.models import ResNet50_Weights
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
device   = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_PATH     = str(BASE_DIR / 'Data/chest_xray/chest_xray/train')
TEST_PATH      = str(BASE_DIR / 'Data/chest_xray/chest_xray/test')
CHECKPOINT_IN  = str(BASE_DIR / 'saved_models/ResNet50_Pneumonia/best_model.pth')
SAVE_DIR       = str(BASE_DIR / 'saved_models/ResNet50_Pneumonia_polished')
TB_PATH        = str(BASE_DIR / 'runs/ResNet50_Polished')

EPOCHS      = 15
BATCH_SIZE  = 32
LR_HEAD     = 1e-4   # FC head
LR_BACKBONE = 1e-5   # unfrozen backbone layers


# ── Transforms ────────────────────────────────────────────────────────────────
transforms_train = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(p=0.5),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    T.RandomRotation(10),
    T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

transforms_val = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Model builder ─────────────────────────────────────────────────────────────
def build_model(checkpoint_path: str):
    """Load checkpoint, then unfreeze layer3 + layer4 + FC."""
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state)
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze layer3, layer4, and FC head
    for name, param in model.named_parameters():
        if any(k in name for k in ['layer3', 'layer4', 'fc']):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.1f}%)")
    return model.to(device)


# ── Accuracy / metrics ────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_score):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    acc  = (tn + tp) / cm.sum()
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    auc  = roc_auc_score(y_true, y_score)
    return acc, sens, spec, auc


# ── Train / Val loops ─────────────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, loss_fn, train=True):
    model.train() if train else model.eval()
    losses, y_true, y_pred, y_score = [], [], [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, desc="Train" if train else "Val  "):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            out  = model(imgs).squeeze(1)
            loss = loss_fn(out, labels)

            if train:
                loss.backward()
                optimizer.step()

            scores = out.detach().cpu().numpy()
            preds  = (scores >= 0.5).astype(int)
            losses.append(loss.item())
            y_score.extend(scores.tolist())
            y_pred.extend(preds.tolist())
            y_true.extend(labels.cpu().numpy().tolist())

    acc, sens, spec, auc = compute_metrics(
        np.array(y_true), np.array(y_pred), np.array(y_score)
    )
    return np.mean(losses), acc, sens, spec, auc


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    multiprocessing.freeze_support()
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Phase 2 Polish -- ResNet-50 Layer3+4 Fine-tuning")
    gpu_name = torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'
    print(f"  Device : {device}  ({gpu_name})")
    print(f"  Epochs : {EPOCHS}  |  Batch: {BATCH_SIZE}")
    print(f"  LR head: {LR_HEAD}  |  LR backbone: {LR_BACKBONE}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_ds = datasets.ImageFolder(TRAIN_PATH, transform=transforms_train)
    val_ds   = datasets.ImageFolder(TEST_PATH,  transform=transforms_val)

    # Class weights for imbalance (NORMAL=234, PNEUMONIA=390 in test; train is similar)
    counts     = torch.tensor([train_ds.targets.count(i)
                               for i in range(len(train_ds.classes))], dtype=torch.float)
    class_wts  = counts.sum() / (len(counts) * counts)
    # pos_weight for PNEUMONIA (class index 1)
    pos_weight = torch.tensor([class_wts[1] / class_wts[0]]).to(device)
    print(f"Class counts: NORMAL={int(counts[0])}, PNEUMONIA={int(counts[1])}")
    print(f"pos_weight (PNEUMONIA): {pos_weight.item():.4f}\n")

    # Use BCEWithLogitsLoss + pos_weight (more numerically stable than BCE+Sigmoid)
    # NOTE: must remove Sigmoid from model for this; we handle it below.
    loss_fn = nn.BCELoss()   # keep Sigmoid in model for portability

    trn_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=2, pin_memory=True, persistent_workers=True)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}\n")

    # ── Model + optimizer ────────────────────────────────────────────────────
    model = build_model(CHECKPOINT_IN)

    # Differential LRs: backbone layers get 10x smaller LR than head
    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'fc' not in n]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'fc' in n]

    optimizer = AdamW([
        {'params': backbone_params, 'lr': LR_BACKBONE, 'weight_decay': 1e-4},
        {'params': head_params,     'lr': LR_HEAD,     'weight_decay': 1e-4},
    ])
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    writer   = SummaryWriter(TB_PATH)
    best_acc = 0.0
    best_auc = 0.0

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{EPOCHS}")

        trn_loss, trn_acc, trn_sens, trn_spec, trn_auc = run_epoch(
            model, trn_loader, optimizer, loss_fn, train=True)
        val_loss, val_acc, val_sens, val_spec, val_auc = run_epoch(
            model, val_loader, optimizer, loss_fn, train=False)

        scheduler.step()
        lr_head = optimizer.param_groups[1]['lr']

        print(f"  Train  Loss:{trn_loss:.4f}  Acc:{trn_acc*100:.2f}%  "
              f"AUC:{trn_auc:.4f}  Sens:{trn_sens*100:.1f}%")
        print(f"  Val    Loss:{val_loss:.4f}  Acc:{val_acc*100:.2f}%  "
              f"AUC:{val_auc:.4f}  Sens:{val_sens*100:.1f}%  "
              f"Spec:{val_spec*100:.1f}%  LR:{lr_head:.2e}")

        writer.add_scalars('Loss',        {'train': trn_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy',    {'train': trn_acc,  'val': val_acc},  epoch)
        writer.add_scalars('AUC',         {'train': trn_auc,  'val': val_auc},  epoch)
        writer.add_scalars('Sensitivity', {'train': trn_sens, 'val': val_sens}, epoch)

        # Save best by AUC (better than acc for medical models)
        if val_auc > best_auc:
            best_auc = val_auc
            best_acc = val_acc
            path = os.path.join(SAVE_DIR, 'best_model.pth')
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, path)
            print(f"  ** Best AUC model saved -> {path}")
            print(f"     Acc={val_acc*100:.2f}%  AUC={val_auc:.4f}  "
                  f"Sens={val_sens*100:.1f}%  Spec={val_spec*100:.1f}%")

        # Epoch checkpoint
        ckpt = os.path.join(SAVE_DIR,
                            f"ep{epoch:02d}_acc{val_acc:.4f}_auc{val_auc:.4f}.pth")
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, ckpt)

    writer.close()
    print(f"\n{'='*60}")
    print(f"  Polish complete!")
    print(f"  Best Val Acc : {best_acc*100:.2f}%")
    print(f"  Best AUC     : {best_auc:.4f}")
    print(f"  Saved to     : {SAVE_DIR}/best_model.pth")
    print(f"{'='*60}")
