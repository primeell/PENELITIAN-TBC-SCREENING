import os
import sys
import numpy as np
import pywt
import matplotlib
import matplotlib.cm as cm

# Reconfigure stdout to support unicode/emojis in Windows console
sys.stdout.reconfigure(encoding='utf-8')

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
except ImportError:
    print("❌ Error: onnxruntime not found. Please install it using: pip install onnxruntime")
    sys.exit(1)

# Try to import audio libraries
try:
    import librosa
except ImportError:
    librosa = None

# Prevent OpenMP runtime issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------------------------------------------
# CONSTANTS & CALIBRATION
# -------------------------------------------------------------
MODEL_PATH = 'best_RespiScan_FocalLoss.onnx'
DECISION_THRESHOLD = 0.3595
SAMPLE_RATE = 16000
DURATION_SAMPLES = 16000  # 1.0 second at 16kHz

# ImageNet normalization parameters (used during MobileNetV2 training)
NORM_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORM_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_audio(file_path):
    """
    Ingests and standardizes audio signal to exactly 1.0 second at 16kHz mono.
    """
    if librosa is None:
        raise ImportError("librosa is required for prototype audio ingestion. Install via 'pip install librosa'.")
        
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    if len(y) > DURATION_SAMPLES:
        y = y[:DURATION_SAMPLES]
    else:
        padding = DURATION_SAMPLES - len(y)
        y = np.pad(y, (0, padding), mode='constant')
    return y

def preprocess_cwt_to_tensor(y):
    """
    Converts 1D audio array to CWT scalogram image tensor matching ImageNet format.
    Does not write to disk, keeping execution entirely in-memory for speed on Raspberry Pi.
    """
    # 1. Compute Geometric CWT
    widths = np.geomspace(2, 186, num=64)
    cwtmatr, _ = pywt.cwt(y, widths, 'cmor1.5-1.0')
    cwt_abs = np.abs(cwtmatr)
    
    # 2. Replicate Matplotlib's 'jet' colormap mapping and normalization
    # Min-max normalization of the absolute coefficients
    c_min = cwt_abs.min()
    c_max = cwt_abs.max()
    if c_max - c_min > 1e-8:
        cwt_norm = (cwt_abs - c_min) / (c_max - c_min)
    else:
        cwt_norm = np.zeros_like(cwt_abs)
        
    # Apply Jet color mapping in-memory (returns shape HxWx4)
    colormap = matplotlib.colormaps['jet']
    cwt_rgb = colormap(cwt_norm)[:, :, :3]  # Strip Alpha channel -> shape (64, 16000, 3)
    
    # 3. Resize to 224x224 (expects RGB)
    # Using simple bilinear resizing via numpy to avoid PIL dependency on minimal environments
    # Resizing H (64) and W (16000) to 224x224
    from PIL import Image
    # Convert numpy RGB (0.0 to 1.0) to uint8 (0 to 255) for PIL
    img_uint8 = (cwt_rgb * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8).resize((224, 224), Image.BILINEAR)
    
    # Convert back to float32 and scale to [0, 1]
    img_array = np.array(img, dtype=np.float32) / 255.0  # Shape: (224, 224, 3)
    
    # 4. Apply ImageNet Normalization: (image - mean) / std
    img_normalized = (img_array - NORM_MEAN) / NORM_STD
    
    # 5. Transpose to PyTorch shape Channel-First: (3, 224, 224) and add batch dim -> (1, 3, 224, 224)
    img_tensor = np.transpose(img_normalized, (2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    return img_tensor

def main():
    if len(sys.argv) < 2:
        print("💡 Usage: python pi_run.py <path_to_cough_audio.wav>")
        print("   Example: python pi_run.py sample.wav")
        sys.exit(1)
        
    audio_path = sys.argv[1]
    
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file '{audio_path}' not found.")
        sys.exit(1)
        
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: ONNX optimized model file '{MODEL_PATH}' not found.")
        print("   Please run 'export_onnx.py' first to generate the ONNX file.")
        sys.exit(1)
        
    print("--------------------------------------------------")
    print("🩺 RespiScan Raspberry Pi / Edge Screening Client")
    print("--------------------------------------------------")
    
    # 1. Load Audio
    print(f"🔊 Loading audio file: {audio_path}...")
    try:
        y = load_audio(audio_path)
    except Exception as e:
        print(f"❌ Audio loading failed: {e}")
        sys.exit(1)
        
    # 2. Extract and Normalize CWT
    print("🖼️ Extracting geometric CWT scalogram (in-memory)...")
    try:
        input_tensor = preprocess_cwt_to_tensor(y)
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        sys.exit(1)
        
    # 3. Load ONNX model and run inference
    print(f"🤖 Loading ONNX engine: {MODEL_PATH}...")
    try:
        session = ort.InferenceSession(MODEL_PATH)
        input_name = session.get_inputs()[0].name
        
        # Run Session
        outputs = session.run(None, {input_name: input_tensor})
        logits = outputs[0]
        
        # Softmax computation
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        prob_tb = probs[0, 1]  # Class 1 is TB
        
    except Exception as e:
        print(f"❌ Inference engine failed: {e}")
        sys.exit(1)
        
    # 4. Present screening outcome
    print("\n================ SCREENING OUTCOME ================")
    print(f"TB Suspect Probability: {prob_tb * 100:.2f}%")
    print(f"Optimized Decision Threshold: {DECISION_THRESHOLD:.4f}")
    
    if prob_tb >= DECISION_THRESHOLD:
        print("🚨 Outcome: HIGH RISK (Suspected TB)")
        print("👉 Action: Recommend clinical confirmatory GeneXpert testing.")
    else:
        print("🟢 Outcome: LOW RISK (Non-TB)")
        print("👉 Action: Standard health monitoring (advice medical review if symptoms persist).")
    print("===================================================\n")

if __name__ == '__main__':
    main()
