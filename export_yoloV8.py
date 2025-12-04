import sys
import argparse
import os
import torch
import torch.nn as nn
from ultralytics import YOLO

# Suppress the PyTorch 2.6 warnings
def suppress_warnings():
    import warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=FutureWarning)

class DeepStreamOutput(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        boxes = x[0]
        scores = x[1]
        # DeepStream expects [Batch, Boxes, 4] and [Batch, Boxes, Classes]
        # We might need to permute depending on YOLO output, 
        # but usually the standard export handles the heavy lifting.
        return boxes, scores

def suppress_output(x):
    return x

def main(args):
    suppress_warnings()
    
    print(f"Loading weights: {args.weights}")
    
    # Load model with weights_only=False to fix the Pickle error
    try:
        model = YOLO(args.weights)
        # Force a dummy load to ensure weights are present
        _ = model.model
    except Exception as e:
        print(f"Standard load failed, trying torch.load workaround... {e}")
        # Manual workaround for PyTorch 2.6 pickle issues
        ckpt = torch.load(args.weights, map_location='cpu', weights_only=False)
        model = YOLO(args.weights) 
    
    device = torch.device('cpu')
    model = model.to(device)

    print(f"Exporting to ONNX (Opset {args.opset})...")
    
    # We rely on the internal Ultralytics export which is now very robust for DeepStream
    # providing we pass the right kwargs.
    # We DO NOT manually replace the head anymore as Ultralytics added 
    # specific support for this via format='onnx'.
    
    # However, for strictly DeepStream Custom Parsers (DeepStream-Yolo), 
    # we usually want the raw output.
    
    success = model.export(
        format='onnx',
        imgsz=args.size,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
        # This is the magic flag: it prevents adding the NMS plugin 
        # so you get raw boxes that DeepStream can parse
        nms=False, 
    )
    
    if success:
        print(f"\nSuccess! Model exported to: {success}")
        # Create the labels file expected by DeepStream
        print("Creating labels.txt...")
        with open('labels.txt', 'w') as f:
            for name in model.names.values():
                f.write(f"{name}\n")
    else:
        print("\nExport failed.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export YOLOv11 to ONNX for DeepStream')
    parser.add_argument('-w', '--weights', required=True, help='Path to .pt file')
    parser.add_argument('-s', '--size', nargs='+', type=int, default=[640], help='Inference size [h, w]')
    parser.add_argument('--dynamic', action='store_true', help='Enable dynamic batch size')
    parser.add_argument('--simplify', action='store_true', help='Simplify ONNX model')
    parser.add_argument('--opset', type=int, default=12, help='ONNX opset version (Default 12 for DeepStream)')
    
    args = parser.parse_args()
    
    if len(args.size) == 2:
        args.size = args.size
    else:
        args.size = [args.size[0], args.size[0]]

    main(args)