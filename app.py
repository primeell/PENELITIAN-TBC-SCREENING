import os
import sys
import re
import tempfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Prevent GUI window popups on Windows
import matplotlib.pyplot as plt
import matplotlib.image
import pywt
import torch
import torch.nn as nn
import librosa
from PIL import Image
import streamlit as st
from torchvision import models, transforms

# Prevent PyTorch OpenMP crash on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------------------------------------------
# PAGE CONFIGURATION & THEME
# -------------------------------------------------------------
st.set_page_config(
    page_title="RespiScan Triage Screening",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Visual Aesthetics
st.markdown("""
<style>
    /* Styling for Headers */
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        color: #64748B;
        font-size: 1.15rem;
        margin-bottom: 2rem;
    }
    
    /* Result Banners */
    .banner-high-risk {
        background-color: #FEE2E2;
        border-left: 6px solid #EF4444;
        padding: 1.5rem;
        border-radius: 8px;
        color: #991B1B;
        margin-bottom: 1.5rem;
    }
    .banner-low-risk {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 1.5rem;
        border-radius: 8px;
        color: #065F46;
        margin-bottom: 1.5rem;
    }
    
    /* Styled metric cards */
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 1.25rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Footer and Disclaimers */
    .disclaimer-box {
        background-color: #FFFBEB;
        border: 1px dashed #F59E0B;
        padding: 1rem;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #92400E;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# STATE AND SESSION CONFIG
# -------------------------------------------------------------
if 'lang' not in st.session_state:
    st.session_state.lang = 'id'  # Default language is Indonesian

# Language texts
texts = {
    'id': {
        'title': "🩺 RespiScan Triage Screening",
        'subtitle': "Alat bantu penapisan tuberkulosis berbasis analisis batuk menggunakan Deep Learning MobileNetV2 dan Scalogram CWT.",
        'upload_header': "Unggah Rekaman Suara Batuk",
        'upload_help': "Format file audio yang didukung: .wav (disarankan batuk yang sengaja dipaksa / forced cough)",
        'drag_drop': "Seret dan lepas file audio batuk (.wav) di sini",
        'audio_player': "Putar Audio yang Diunggah",
        'processing': "Sedang memproses audio...",
        'preprocess_title': "Hasil Pra-pemrosesan Akustik",
        'waveform': "Bentuk Gelombang Audio (Waveform - Dicuplik & Ditandai ke 1 Detik)",
        'scalogram': "Scalogram CWT (Continuous Wavelet Transform - Skala Geometris cmor1.5-1.0)",
        'inference_title': "Hasil Penapisan (Triage Outcome)",
        'prob_tb': "Probabilitas Risiko TB",
        'class_label': "Status Penapisan",
        'high_risk_title': "🚨 Risiko Tinggi (Suspek TB)",
        'high_risk_desc': "Hasil analisis menunjukkan karakteristik batuk yang menyerupai pola tuberkulosis. Probabilitas berada di atas ambang batas keputusan optimal Youden.",
        'high_risk_action': "**Tindakan Rekomendasi:** Pasien direkomendasikan untuk menjalani pemeriksaan klinis standar konfirmasi diagnosis (seperti Tes Cepat Molekuler / TCM GeneXpert, kultur, atau rontgen dada) guna menegakkan diagnosis.",
        'low_risk_title': "🟢 Risiko Rendah (Non-TB)",
        'low_risk_desc': "Hasil analisis tidak mendeteksi karakteristik batuk khas tuberkulosis. Probabilitas berada di bawah ambang batas keputusan.",
        'low_risk_action': "**Tindakan Rekomendasi:** Jika pasien masih mengalami gejala klinis batuk persisten lebih dari 2 minggu, demam, atau penurunan berat badan, disarankan untuk berkonsultasi dengan tenaga medis profesional.",
        'disclaimer_title': "⚠️ Penafian Medis Penting (Medical Disclaimer):",
        'disclaimer_content': "RespiScan adalah sistem pembantu keputusan klinis (screening/triage tool) dengan sensitivitas tinggi (100% pada set uji subjek-level) untuk menyaring suspek TB dari batuk rekaman. Aplikasi ini **BUKAN** merupakan alat diagnosis pengganti pemeriksaan laboratorium seperti GeneXpert atau pemeriksaan dahak/dahak mikroskopis. Hasil penapisan harus selalu divalidasi oleh dokter ahli dan pemeriksaan medis standar.",
        'model_info': "Informasi Model & Performa",
        'perf_metrics': "Metrik Performa (Subjek-Level Set Uji):",
        'sensitivity': "Sensitivitas",
        'specificity': "Spesifisitas",
        'auroc': "AUROC",
        'threshold_label': "Ambang Batas Keputusan (Threshold)",
        'threshold_help': "Ambang batas Youden optimum hasil kalibrasi set validasi untuk memaksa sensitivitas hingga 100%.",
        'toggle_lang': "English Version",
        'lang_indicator': "Bahasa: Bahasa Indonesia"
    },
    'en': {
        'title': "🩺 RespiScan Triage Screening",
        'subtitle': "Tuberculosis triage screening assistant based on cough analysis using MobileNetV2 Deep Learning and CWT Scalograms.",
        'upload_header': "Upload Cough Audio Recording",
        'upload_help': "Supported audio format: .wav (forced cough recommended)",
        'drag_drop': "Drag and drop a cough audio file (.wav) here",
        'audio_player': "Play Uploaded Audio",
        'processing': "Processing audio...",
        'preprocess_title': "Acoustic Preprocessing Results",
        'waveform': "Audio Waveform (Normalized & Standardized to 1.0 Second)",
        'scalogram': "CWT Scalogram (Continuous Wavelet Transform - Geometric Scale cmor1.5-1.0)",
        'inference_title': "Triage Outcome",
        'prob_tb': "TB Risk Probability",
        'class_label': "Screening Status",
        'high_risk_title': "🚨 High Risk (Suspected TB)",
        'high_risk_desc': "Acoustic analysis shows characteristics resembling tuberculosis cough patterns. The probability exceeds the validation-optimized Youden decision threshold.",
        'high_risk_action': "**Recommended Action:** The patient should undergo standard confirmatory diagnostic testing (such as GeneXpert molecular testing, culture, or chest X-ray) to establish a diagnosis.",
        'low_risk_title': "🟢 Low Risk (Non-TB)",
        'low_risk_desc': "Acoustic analysis did not detect patterns characteristic of tuberculosis. The probability is below the decision threshold.",
        'low_risk_action': "**Recommended Action:** If the patient continues to experience persistent cough for more than 2 weeks, fever, or weight loss, consultation with a healthcare professional is strongly advised.",
        'disclaimer_title': "⚠️ Important Medical Disclaimer:",
        'disclaimer_content': "RespiScan is a clinical decision-support tool (screening/triage aid) with high sensitivity (100% on patient-level test set) designed to identify TB suspects from cough recordings. It is **NOT** a diagnostic replacement for gold-standard laboratory tests like GeneXpert or sputum smear microscopy. All screening outcomes must be validated by medical professionals and standard clinical testing.",
        'model_info': "Model & Performance Info",
        'perf_metrics': "Performance Metrics (Patient-Level Test Set):",
        'sensitivity': "Sensitivity",
        'specificity': "Specificity",
        'auroc': "AUROC",
        'threshold_label': "Decision Threshold",
        'threshold_help': "Optimum Youden threshold calibrated on validation set to optimize sensitivity.",
        'toggle_lang': "Versi Bahasa Indonesia",
        'lang_indicator': "Language: English"
    }
}

# -------------------------------------------------------------
# SIDEBAR CONFIGURATION
# -------------------------------------------------------------
st.sidebar.header("Configuration / Pengaturan")

# Language Toggle Buttons
if st.session_state.lang == 'id':
    if st.sidebar.button(texts['id']['toggle_lang']):
        st.session_state.lang = 'en'
        st.rerun()
else:
    if st.sidebar.button(texts['en']['toggle_lang']):
        st.session_state.lang = 'id'
        st.rerun()

current_text = texts[st.session_state.lang]

# Language Indicator Label
st.sidebar.caption(current_text['lang_indicator'])

st.sidebar.markdown("---")
st.sidebar.subheader(current_text['model_info'])

# Decision Threshold Slider
threshold = st.sidebar.slider(
    current_text['threshold_label'],
    min_value=0.1,
    max_value=0.9,
    value=0.3595,
    step=0.01,
    help=current_text['threshold_help']
)

# Display Key Evaluation Metrics
st.sidebar.markdown(f"**{current_text['perf_metrics']}**")
col_s, col_sp = st.sidebar.columns(2)
with col_s:
    st.metric(label=current_text['sensitivity'], value="100.00%")
with col_sp:
    st.metric(label=current_text['specificity'], value="76.47%")

st.sidebar.metric(label=current_text['auroc'], value="0.9454")
st.sidebar.info("Model: RespiScan MobileNetV2 Focal Loss (Standardized Scalogram Model)")

# -------------------------------------------------------------
# MAIN CONTENT HEADER
# -------------------------------------------------------------
st.markdown(f"<h1 class='main-title'>{current_text['title']}</h1>", unsafe_allow_html=True)
st.markdown(f"<p class='subtitle'>{current_text['subtitle']}</p>", unsafe_allow_html=True)

# -------------------------------------------------------------
# MODEL LOADING FUNCTION (CACHED)
# -------------------------------------------------------------
@st.cache_resource
def load_screening_model(model_path='best_RespiScan_FocalLoss.pth'):
    # Define custom MobileNetV2 model architecture as used during training
    model = models.mobilenet_v2(weights=None)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 2)
    )
    
    # Load model weights onto appropriate device (CPU fallback)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
        return model, device, True
    else:
        return None, device, False

model, device, is_model_loaded = load_screening_model()

if not is_model_loaded:
    st.warning("⚠️ Model weights file `best_RespiScan_FocalLoss.pth` not found in workspace directory. Running in demonstration mode.")

# -------------------------------------------------------------
# AUDIO PIPELINE UTILITIES
# -------------------------------------------------------------
def preprocess_audio(audio_file_data):
    """
    Standardizes input audio into exactly 16000 samples (1s at 16kHz) mono
    """
    # Write uploaded binary audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav:
        temp_wav.write(audio_file_data.read())
        temp_wav_path = temp_wav.name
        
    try:
        # Load and resample to 16kHz mono
        y, sr = librosa.load(temp_wav_path, sr=16000, mono=True)
        # Crop or pad to exactly 16,000 samples (1.0 second)
        target_samples = 16000
        if len(y) > target_samples:
            y = y[:target_samples]
        else:
            padding = target_samples - len(y)
            y = np.pad(y, (0, padding), mode='constant')
            
        return y, sr, None
    except Exception as e:
        return None, None, str(e)
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

def generate_cwt_image(y):
    """
    Generates geometric-scaled CWT scalogram image matching the training format
    """
    widths = np.geomspace(2, 186, num=64)
    cwtmatr, _ = pywt.cwt(y, widths, 'cmor1.5-1.0')
    cwt_abs = np.abs(cwtmatr)
    
    # Save scalogram to a temporary file in memory using matplotlib
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_png:
        temp_png_path = temp_png.name
    
    try:
        matplotlib.image.imsave(temp_png_path, cwt_abs, cmap='jet')
        img = Image.open(temp_png_path).convert('RGB')
        return img, cwt_abs, None
    except Exception as e:
        return None, None, str(e)
    finally:
        if os.path.exists(temp_png_path):
            os.remove(temp_png_path)

# -------------------------------------------------------------
# MAIN APP FLOW (FILE UPLOAD AND PROCESSING)
# -------------------------------------------------------------
uploaded_file = st.file_uploader(
    current_text['upload_header'],
    type=['wav'],
    help=current_text['upload_help'],
    label_visibility="visible"
)

if uploaded_file is not None:
    st.subheader(current_text['audio_player'])
    st.audio(uploaded_file, format='audio/wav')
    
    with st.spinner(current_text['processing']):
        # 1. Preprocess audio signal
        y, sr, err_audio = preprocess_audio(uploaded_file)
        
        if err_audio:
            st.error(f"Error reading audio file: {err_audio}")
        else:
            # 2. Extract CWT Scalogram
            cwt_img, cwt_matrix, err_cwt = generate_cwt_image(y)
            
            if err_cwt:
                st.error(f"Error processing CWT Scalogram: {err_cwt}")
            else:
                # Layout for Preprocessing Visualizations
                st.subheader(current_text['preprocess_title'])
                col1, col2 = st.columns(2)
                
                # Plot Audio Waveform
                with col1:
                    st.write(f"**{current_text['waveform']}**")
                    fig, ax = plt.subplots(figsize=(6, 2.8))
                    time_axis = np.linspace(0, 1.0, len(y))
                    ax.plot(time_axis, y, color='#0284C7', alpha=0.9, linewidth=0.8)
                    ax.grid(True, linestyle='--', alpha=0.3)
                    ax.set_xlim(0, 1.0)
                    ax.set_ylim(-1.0, 1.0)
                    ax.set_xlabel('Time (s)' if st.session_state.lang == 'en' else 'Waktu (s)', fontsize=8)
                    ax.set_ylabel('Amplitude' if st.session_state.lang == 'en' else 'Amplitudo', fontsize=8)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                
                # Plot Scalogram (Jet Colormap)
                with col2:
                    st.write(f"**{current_text['scalogram']}**")
                    fig, ax = plt.subplots(figsize=(6, 2.8))
                    # Plot using jet cmap to mimic training input
                    ax.imshow(cwt_matrix, cmap='jet', aspect='auto', origin='lower')
                    ax.axis('off')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                
                # Run Model Inference
                st.subheader(current_text['inference_title'])
                
                if model is not None:
                    # Transform image for MobileNetV2
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    
                    input_tensor = transform(cwt_img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs = torch.softmax(outputs, dim=1)
                        prob_tb = probs[0, 1].item()
                    
                    # Determine classification based on the threshold
                    is_tb = prob_tb >= threshold
                    
                    # Display Results layout
                    res_col1, res_col2 = st.columns([2, 1])
                    
                    with res_col1:
                        if is_tb:
                            # TB High Risk Alert Card
                            st.markdown(f"""
                            <div class="banner-high-risk">
                                <h3>{current_text['high_risk_title']}</h3>
                                <p>{current_text['high_risk_desc']}</p>
                                <p style="margin-top: 1rem;">{current_text['high_risk_action']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            # Non-TB Low Risk Alert Card
                            st.markdown(f"""
                            <div class="banner-low-risk">
                                <h3>{current_text['low_risk_title']}</h3>
                                <p>{current_text['low_risk_desc']}</p>
                                <p style="margin-top: 1rem;">{current_text['low_risk_action']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    with res_col2:
                        # Display Metric and Score Progress Bar
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{current_text['prob_tb']}</div>
                            <div class="metric-value">{prob_tb * 100:.2f}%</div>
                            <div class="metric-label" style="margin-top: 1rem; color: #64748B;">
                                Threshold: {threshold:.4f}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.write("")
                        # Visual progress bar with colored status
                        st.progress(prob_tb)
                        
                else:
                    # Fallback demonstration mode if model checkpoint is missing
                    st.info("Demonstration mode: No model checkpoint active. Check standard sidebar metrics.")
                    st.metric(label=current_text['prob_tb'], value="N/A")

# -------------------------------------------------------------
# MEDICAL DISCLAIMER FOOTER
# -------------------------------------------------------------
st.markdown(f"""
<div class="disclaimer-box">
    <strong>{current_text['disclaimer_title']}</strong><br/>
    {current_text['disclaimer_content']}
</div>
""", unsafe_allow_html=True)
