import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models
from torchvision.models import ResNet50_Weights
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Model ─────────────────────────────────────────────────────────────────────
def get_model(pretrained: bool = True, lr: float = 0.001):
    """ResNet-50 with custom binary classification head."""
    print(f"Building ResNet-50 | pretrained={pretrained} | device={device}")
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model   = models.resnet50(weights=weights)

    # Freeze entire backbone
    for param in model.parameters():
        param.requires_grad = False

    # Trainable classification head  (2048 → 256 → 1)
    model.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    for param in model.fc.parameters():
        param.requires_grad = True

    model     = model.to(device)
    loss_fn   = nn.BCELoss()
    optimizer = Adam(model.fc.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1,
        threshold=0.001, min_lr=1e-6, threshold_mode='abs'
    )
    return model, loss_fn, optimizer, scheduler


# ── Data ──────────────────────────────────────────────────────────────────────
def get_data(path: str, limit: int, batch_size: int, transforms):
    """
    Use torchvision.datasets.ImageFolder — much faster than manual glob+cv2.
    Workers=4 on Windows to prefetch while GPU trains.
    """
    dataset = datasets.ImageFolder(root=path, transform=transforms)

    # Optionally limit samples (reproducible subset)
    if 0 < limit < len(dataset):
        from torch.utils.data import Subset
        import random; random.seed(10)
        indices = random.sample(range(len(dataset)), limit)
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,          # Windows-safe: 2 workers with spawn
        pin_memory=True,        # fast CPU->GPU transfer
        persistent_workers=True,
    )
    print(f"Loaded {len(dataset)} samples from {path}  "
          f"[classes: {getattr(dataset, 'classes', 'N/A')}]")
    return loader


# ── Accuracy ──────────────────────────────────────────────────────────────────
def accuracy(y_pred, y_true):
    """Binary accuracy for sigmoid output."""
    preds = (y_pred >= 0.5).float().squeeze()
    if preds.dim() == 0:           # single-sample edge case
        preds = preds.unsqueeze(0)
    return (preds == y_true).float().mean().item()


def get_lr(optimizer):
    for pg in optimizer.param_groups:
        return pg['lr']


# ── Train / Val loops ─────────────────────────────────────────────────────────
def train_epoch(model, loader, loss_fn, optimizer):
    model.train()
    losses, accs = [], []
    for imgs, labels in tqdm(loader, desc="Train"):
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.float().to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(imgs).squeeze(1)
        loss    = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        accs.append(accuracy(outputs.detach(), labels))

    return sum(losses) / len(losses), sum(accs) / len(accs)


def val_epoch(model, loader, loss_fn, scheduler):
    model.eval()
    losses, accs = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Val  "):
            imgs   = imgs.to(device, non_blocking=True)
            labels = labels.float().to(device, non_blocking=True)

            outputs = model(imgs).squeeze(1)
            loss    = loss_fn(outputs, labels)

            losses.append(loss.item())
            accs.append(accuracy(outputs, labels))

    avg_loss = sum(losses) / len(losses)
    scheduler.step(avg_loss)
    return avg_loss, sum(accs) / len(accs)


# ── Main training loop ────────────────────────────────────────────────────────
def train(model, trn_dl, val_dl, loss_fn, optimizer, scheduler,
          start_epoch, end_epoch, writer, args, saved_models_dir):
    best_acc = 0.0

    for epoch in range(start_epoch, end_epoch):
        print(f"\n{'='*60}\nEpoch {epoch+1}/{end_epoch}")

        trn_loss, trn_acc = train_epoch(model, trn_dl, loss_fn, optimizer)
        val_loss, val_acc = val_epoch(model, val_dl, loss_fn, scheduler)
        lr = get_lr(optimizer)

        print(f"  Train Loss: {trn_loss:.4f} | Train Acc: {trn_acc:.4f}")
        print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | LR: {lr:.6f}")

        writer.add_scalars('Loss',     {'train': trn_loss, 'val': val_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': trn_acc,  'val': val_acc},  epoch)

        if val_acc > best_acc:
            best_acc  = val_acc
            best_path = os.path.join(saved_models_dir, 'best_model.pth')
            torch.save({k: v.cpu() for k, v in model.state_dict().items()}, best_path)
            print(f"  ** New best -> {best_path}  (val_acc={val_acc:.4f})")

        ckpt = os.path.join(
            saved_models_dir,
            f"epoch{epoch:02d}_acc{val_acc:.4f}_loss{val_loss:.4f}.pth"
        )
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, ckpt)

    print(f"\nTraining complete. Best Val Acc: {best_acc:.4f}")
    return best_acc
