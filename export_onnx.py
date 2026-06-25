import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torchvision import models

# Prevent OpenMP warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Reconfigure stdout to support unicode/emojis in Windows console
sys.stdout.reconfigure(encoding='utf-8')

def main():
    checkpoint_path = 'best_RespiScan_FocalLoss.pth'
    onnx_path = 'best_RespiScan_FocalLoss.onnx'
    
    print("🚀 Starting ONNX Export Process...")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: PyTorch model weights not found at {checkpoint_path}")
        sys.exit(1)
        
    # 1. Instantiate the MobileNetV2 architecture
    print("🧱 Creating MobileNetV2 architecture...")
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )
    
    # 2. Load the trained PyTorch weights
    print(f"💾 Loading PyTorch weights from {checkpoint_path}...")
    device = torch.device("cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # 3. Create dummy input vector matching expected shape (1, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 4. Export the model
    print(f"📤 Exporting model to {onnx_path}...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print("✅ Export completed successfully!")
    
    # 5. Run Equivalence Validation (PyTorch vs ONNX Runtime)
    try:
        import onnxruntime as ort
        print("\n🧪 Running Equivalence Validation...")
        
        # PyTorch Inference
        with torch.no_grad():
            torch_out = model(dummy_input)
            torch_prob = torch.softmax(torch_out, dim=1).numpy()
            
        # ONNX Runtime Inference
        ort_session = ort.InferenceSession(onnx_path)
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = ort_session.run(None, ort_inputs)
        
        # Convert logits to probabilities
        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / e_x.sum(axis=1, keepdims=True)
            
        onnx_prob = softmax(ort_outs[0])
        
        print("PyTorch Probs: ", torch_prob)
        print("ONNX Probs:    ", onnx_prob)
        
        # Check tolerance (1e-5)
        diff = np.abs(torch_prob - onnx_prob)
        max_diff = np.max(diff)
        print(f"Maximum absolute difference in probabilities: {max_diff:.2e}")
        
        if max_diff < 1e-5:
            print("🟢 SUCCESS: PyTorch and ONNX outputs are equivalent within tolerance (< 1e-5)!")
        else:
            print("⚠️ WARNING: Outputs differ by more than tolerance!")
            
    except ImportError:
        print("\n⚠️ ONNX Runtime not installed yet. Skipping validation step.")

if __name__ == '__main__':
    main()
