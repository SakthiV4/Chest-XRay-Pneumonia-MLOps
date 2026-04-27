"""
test.py — Evaluate a trained checkpoint on the test split.

Usage:
    python test.py --model_path ../saved_models/ResNet50_Pneumonia/best_model.pth \
                   --test_data_path ../Data/test \
                   --output_image confusion_matrix.png
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score
)
from tqdm import tqdm

import train as trainer
import transforms

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description='Evaluate Chest X-Ray Pneumonia model')
parser.add_argument('--model_path',      type=str, required=True, help='Path to .pth checkpoint')
parser.add_argument('--test_data_path',  type=str,
                    default='../Data/chest_xray/chest_xray/test',
                    help='Path to test data directory')
parser.add_argument('--output_image',    type=str, default='confusion_matrix.png')
parser.add_argument('--batch_size',      type=int, default=32)
args = parser.parse_args()


def evaluate(model_path: str, data_path: str, output_image: str, batch_size: int):
    # Load model
    model, _, _, _ = trainer.get_model(pretrained=False, lr=0.001)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"Model loaded: {model_path}")

    test_loader = trainer.get_data(data_path, limit=-1, batch_size=batch_size,
                                   transforms=transforms.transforms_val)

    y_pred, y_true, y_scores = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            outputs = model(imgs).squeeze(1)
            scores = outputs.cpu().numpy()
            preds  = (outputs >= 0.5).float().cpu().numpy()

            y_scores.extend(scores.tolist())
            y_pred.extend(preds.tolist())
            y_true.extend(labels.cpu().numpy().tolist())

    classes = ['NORMAL', 'PNEUMONIA']
    cm = confusion_matrix(y_true, y_pred)
    acc  = (cm[0, 0] + cm[1, 1]) / cm.sum()
    auc  = roc_auc_score(y_true, y_scores)

    print(f"\nAccuracy : {acc:.4f}")
    print(f"AUC-ROC  : {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes))
    print("Confusion Matrix:")
    print(cm)

    # Plot
    df_cm = pd.DataFrame(cm, index=classes, columns=classes)
    plt.figure(figsize=(8, 6))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'ResNet-50 Pneumonia Detection\nAcc={acc:.4f}  AUC={auc:.4f}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    print(f"\nConfusion matrix saved → {output_image}")


if __name__ == '__main__':
    evaluate(args.model_path, args.test_data_path, args.output_image, args.batch_size)
