"""
export_onnx.py — Convert best_model.pth to model.onnx for Triton Inference Server.

Usage:
    python export_onnx.py \
        --model_path ../saved_models/ResNet50_Pneumonia/best_model.pth \
        --output_path ../triton/model_repository/pneumonia_model/1/model.onnx

The exported model:
  - Input:  float32 tensor [batch_size, 3, 224, 224]  (dynamic batch axis)
  - Output: float32 tensor [batch_size, 1]             sigmoid confidence score
"""
import argparse
import os

import torch
import torch.onnx

import sys
sys.path.insert(0, os.path.dirname(__file__))  # ensure local imports work
import train as trainer

parser = argparse.ArgumentParser(description='Export ResNet-50 → ONNX')
parser.add_argument('--model_path',  type=str,
                    default='../saved_models/ResNet50_Pneumonia/best_model.pth',
                    help='Path to trained .pth checkpoint')
parser.add_argument('--output_path', type=str,
                    default='../triton/model_repository/pneumonia_model/1/model.onnx',
                    help='Output .onnx file path')
parser.add_argument('--opset',       type=int, default=17,
                    help='ONNX opset version (default 17)')
args = parser.parse_args()

device = 'cpu'  # Export on CPU for maximum compatibility

print(f"\n{'='*60}")
print(f" ResNet-50 → ONNX Export")
print(f" Checkpoint : {args.model_path}")
print(f" Output     : {args.output_path}")
print(f" ONNX opset : {args.opset}")
print(f"{'='*60}\n")

# ── Load model ───────────────────────────────────────────────
model, _, _, _ = trainer.get_model(pretrained=False, lr=0.001)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully.\n")

# ── Dummy input ───────────────────────────────────────────────
# Shape must match what Triton config.pbtxt declares
dummy_input = torch.randn(1, 3, 224, 224, device=device)

# ── Create output directory ───────────────────────────────────
os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

# ── Export ────────────────────────────────────────────────────
print("Exporting to ONNX...")
torch.onnx.export(
    model,
    dummy_input,
    args.output_path,
    opset_version=args.opset,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input':  {0: 'batch_size'},
        'output': {0: 'batch_size'},
    },
    export_params=True,
    do_constant_folding=True,
    verbose=False,
)
print(f"ONNX model saved → {args.output_path}\n")

# ── Verify ────────────────────────────────────────────────────
try:
    import onnx
    onnx_model = onnx.load(args.output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check PASSED ✓")

    # Print graph I/O summary
    for inp in onnx_model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  Input  : {inp.name}  shape={shape}")
    for out in onnx_model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  Output : {out.name}  shape={shape}")
except ImportError:
    print("onnx package not found — skipping verification. Install with: pip install onnx")

# ── Quick runtime check with onnxruntime ──────────────────────
try:
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(args.output_path, providers=['CPUExecutionProvider'])
    x_np = dummy_input.numpy()
    result = sess.run(None, {'input': x_np})
    print(f"\nONNXRuntime inference check PASSED ✓")
    print(f"  Output shape : {result[0].shape}")
    print(f"  Output value : {result[0]}")
except ImportError:
    print("onnxruntime not found — skipping runtime check. Install with: pip install onnxruntime")

print("\nExport complete!")
