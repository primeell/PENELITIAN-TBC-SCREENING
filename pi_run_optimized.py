import os
import sys
import wave
import numpy as np
import pywt
from PIL import Image

# Reconfigure stdout to support unicode/emojis in Windows console
sys.stdout.reconfigure(encoding='utf-8')

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
except ImportError:
    print("❌ Error: onnxruntime not found. Please install it using: pip install onnxruntime")
    sys.exit(1)

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

def load_wav_lightweight(file_path, target_samples=16000):
    """
    Lightweight audio loading using Python's native 'wave' library and numpy.
    Eliminates dependency on librosa, soundfile, and libsndfile.
    Includes simple linear resampling if the sampling rate is not 16kHz.
    """
    with wave.open(file_path, 'rb') as w:
        channels = w.getnchannels()
        rate = w.getframerate()
        width = w.getsampwidth()
        n_frames = w.getnframes()
        raw_frames = w.readframes(n_frames)
        
        # 1. Convert raw frames to float numpy array normalized to [-1.0, 1.0]
        if width == 2:
            y = np.frombuffer(raw_frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif width == 1:
            y = (np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        else:
            raise ValueError(f"Unsupported sample width: {width} bytes. Only 8-bit PCM and 16-bit PCM are supported.")
            
        # 2. Downmix stereo to mono
        if channels > 1:
            y = y.reshape(-1, channels).mean(axis=1)
            
        # 3. Simple linear interpolation resampling if sampling rate != 16000 Hz
        if rate != SAMPLE_RATE:
            duration_sec = len(y) / rate
            target_len = int(duration_sec * SAMPLE_RATE)
            xp = np.linspace(0, duration_sec, len(y))
            x_new = np.linspace(0, duration_sec, target_len)
            y = np.interp(x_new, xp, y)
            
        # 4. Pad or crop to exactly target_samples (16000)
        if len(y) > target_samples:
            y = y[:target_samples]
        else:
            padding = target_samples - len(y)
            y = np.pad(y, (0, padding), mode='constant')
            
        return y

def jet_colormap_numpy(v):
    """
    Algebraic numpy-only implementation of Jet colormap.
    Eliminates dependency on matplotlib for colormap coloring.
    """
    # Clip v to [0, 1] just in case
    v = np.clip(v, 0.0, 1.0)
    
    # Calculate R, G, B channels algebraically
    r = np.clip(np.minimum(4.0 * v - 1.5, -4.0 * v + 4.5), 0.0, 1.0)
    g = np.clip(np.minimum(4.0 * v - 0.5, -4.0 * v + 3.5), 0.0, 1.0)
    b = np.clip(np.minimum(4.0 * v + 0.5, -4.0 * v + 2.5), 0.0, 1.0)
    
    # Stack along the last axis to get HxWx3
    return np.stack([r, g, b], axis=-1)

def preprocess_cwt_optimized(y):
    """
    Converts 1D audio array to CWT scalogram image tensor matching ImageNet format.
    Runs entirely in-memory using only numpy, pywt, and Pillow. Zero Matplotlib dependencies.
    """
    # 1. Compute Geometric CWT
    widths = np.geomspace(2, 186, num=64)
    cwtmatr, _ = pywt.cwt(y, widths, 'cmor1.5-1.0')
    cwt_abs = np.abs(cwtmatr)
    
    # 2. Min-max normalization
    c_min = cwt_abs.min()
    c_max = cwt_abs.max()
    if c_max - c_min > 1e-8:
        cwt_norm = (cwt_abs - c_min) / (c_max - c_min)
    else:
        cwt_norm = np.zeros_like(cwt_abs)
        
    # 3. Apply Jet Colormap (numpy-only)
    cwt_rgb = jet_colormap_numpy(cwt_norm)  # shape (64, 16000, 3)
    
    # 4. Resize to 224x224 using PIL (extremely fast and lightweight)
    img_uint8 = (cwt_rgb * 255.0).astype(np.uint8)
    img = Image.fromarray(img_uint8).resize((224, 224), Image.Resampling.BILINEAR)
    
    # Convert back to float32 and scale to [0.0, 1.0]
    img_array = np.array(img, dtype=np.float32) / 255.0  # Shape: (224, 224, 3)
    
    # 5. Apply ImageNet Normalization: (image - mean) / std
    img_normalized = (img_array - NORM_MEAN) / NORM_STD
    
    # 6. Transpose to PyTorch shape Channel-First: (1, 3, 224, 224)
    img_tensor = np.transpose(img_normalized, (2, 0, 1))
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    return img_tensor

def main():
    if len(sys.argv) < 2:
        print("💡 Usage: python pi_run_optimized.py <path_to_cough_audio.wav>")
        sys.exit(1)
        
    audio_path = sys.argv[1]
    
    if not os.path.exists(audio_path):
        print(f"❌ Error: Audio file '{audio_path}' not found.")
        sys.exit(1)
        
    # Attempt to use the standalone embedded ONNX model if available, fallback to default ONNX
    onnx_model = 'best_RespiScan_FocalLoss_embedded.onnx'
    if not os.path.exists(onnx_model):
        onnx_model = MODEL_PATH
        
    if not os.path.exists(onnx_model):
        print(f"❌ Error: ONNX model '{onnx_model}' not found.")
        print("   Please ensure 'best_RespiScan_FocalLoss_embedded.onnx' or 'best_RespiScan_FocalLoss.onnx' is in the workspace.")
        sys.exit(1)
        
    print("--------------------------------------------------")
    print("🩺 RespiScan Optimized Edge Screening Client (Pi/SBC)")
    print("--------------------------------------------------")
    print(f"📦 Using model file: {onnx_model}")
    
    # 1. Load Audio
    print(f"🔊 Loading audio file: {audio_path}...")
    try:
        y = load_wav_lightweight(audio_path)
    except Exception as e:
        print(f"❌ Audio loading failed: {e}")
        sys.exit(1)
        
    # 2. Extract and Normalize CWT
    print("🖼️ Extracting CWT scalogram (optimized in-memory)...")
    try:
        input_tensor = preprocess_cwt_optimized(y)
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        sys.exit(1)
        
    # 3. Load ONNX model and run inference
    print(f"🤖 Loading ONNX engine...")
    try:
        session = ort.InferenceSession(onnx_model)
        input_name = session.get_inputs()[0].name
        
        # Run inference
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
        print("👉 Action: Standard health monitoring.")
    print("===================================================\n")

if __name__ == '__main__':
    main()
