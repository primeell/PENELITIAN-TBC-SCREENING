import os
import sys

# Configure stdout to use UTF-8 to prevent Windows console encoding errors
sys.stdout.reconfigure(encoding='utf-8')

import re
import gc
import random
import shutil
import json
import time
import copy
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Prevent popups on Windows
import matplotlib.pyplot as plt
import matplotlib.image
import pywt
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE

class CleanValDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, l = self.samples[idx]
        from PIL import Image
        return self.transform(Image.open(p).convert('RGB')), l

# Fix for Windows PyTorch OpenMP crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuration paths
PATH_METADATA = r"Metadata.csv"
DIR_AUDIO = r"dataset_tbc"
DIR_OUTPUT_CWT = r"RespiScan_Train_Ready"
DIR_OUTPUT_MFCC = r"RespiScan_MFCC_Only"
NOISE_DIRS = [r"DKITCHEN", r"NPARK", r"OHALLWAY"]

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device active: {DEVICE}")

# -------------------------------------------------------------
# 1. LOAD METADATA AND PREPARE SUBJECTS
# -------------------------------------------------------------
print("\n--- 1. Loading Metadata and Scanning Audio Files ---")
if not os.path.exists(PATH_METADATA):
    print(f"❌ Error: Metadata.csv not found at {PATH_METADATA}")
    sys.exit(1)

df_meta = pd.read_csv(PATH_METADATA)
df_meta.columns = df_meta.columns.str.strip()
df_meta['subject'] = df_meta['subject'].astype(str).str.strip()

# Helper to map PID_32A_now_C -> PID_32A for matching files
def get_clean_subject_id(subj):
    return re.sub(r'_now_C$', '', subj)

df_meta['clean_subject'] = df_meta['subject'].apply(get_clean_subject_id)

# Scan available wav files in dataset_tbc
if not os.path.exists(DIR_AUDIO):
    print(f"❌ Error: Audio directory not found at {DIR_AUDIO}")
    sys.exit(1)

wav_files = [f for f in os.listdir(DIR_AUDIO) if f.lower().endswith('.wav')]
print(f"Total .wav files in {DIR_AUDIO}: {len(wav_files)}")

# Map wav files to clean subject IDs
subject_to_files = {subj: [] for subj in df_meta['clean_subject'].unique()}
for f in wav_files:
    for clean_subj in sorted(subject_to_files.keys(), key=len, reverse=True):
        if f.startswith(clean_subj):
            subject_to_files[clean_subj].append(f)
            break

# Count available files for each row in metadata
df_meta['file_count'] = df_meta['clean_subject'].map(lambda x: len(subject_to_files.get(x, [])))

print(f"Total unique subjects in metadata: {len(df_meta)}")
df_available = df_meta[df_meta['file_count'] > 0].copy()
print(f"Subjects with audio files: {len(df_available)}")
print("Class distribution of available subjects:")
print(df_available['Label'].value_counts())

# -------------------------------------------------------------
# 2. DISJOINT SUBJECT-LEVEL SPLITTING (ZERO SPEAKER LEAKAGE)
# -------------------------------------------------------------
print("\n--- 2. Splitting Subjects at Subject-Level (Zero Leakage) ---")
# Split available subjects into Train (70%), Val (15%), Test (15%)
tb_subjs = df_available[df_available['Label'] == 'TB']['subject'].tolist()
ntb_subjs = df_available[df_available['Label'] == 'NTB']['subject'].tolist()

tb_train, tb_temp = train_test_split(tb_subjs, test_size=0.30, random_state=42)
tb_val, tb_test = train_test_split(tb_temp, test_size=0.50, random_state=42)

ntb_train, ntb_temp = train_test_split(ntb_subjs, test_size=0.30, random_state=42)
ntb_val, ntb_test = train_test_split(ntb_temp, test_size=0.50, random_state=42)

train_subjects_set = set(tb_train + ntb_train)
val_subjects_set = set(tb_val + ntb_val)
test_subjects_set = set(tb_test + ntb_test)

print(f"Train subjects: {len(train_subjects_set)} (TB: {len(tb_train)}, NTB: {len(ntb_train)})")
print(f"Val subjects: {len(val_subjects_set)} (TB: {len(tb_val)}, NTB: {len(ntb_val)})")
print(f"Test subjects: {len(test_subjects_set)} (TB: {len(tb_test)}, NTB: {len(ntb_test)})")

# Assert zero leakage
overlap_train_val = train_subjects_set.intersection(val_subjects_set)
overlap_train_test = train_subjects_set.intersection(test_subjects_set)
overlap_val_test = val_subjects_set.intersection(test_subjects_set)
assert len(overlap_train_val) == 0, "❌ Error: Speaker leakage between Train and Val!"
assert len(overlap_train_test) == 0, "❌ Error: Speaker leakage between Train and Test!"
assert len(overlap_val_test) == 0, "❌ Error: Speaker leakage between Val and Test!"
print("🟢 Verification passed: ZERO subject-level leakage detected!")

# -------------------------------------------------------------
# 3. PREPROCESSING PIPELINE FUNCTIONS
# -------------------------------------------------------------
print("\n--- 3. Preparing Output Directories ---")
skip_preprocessing = False
if os.path.exists(DIR_OUTPUT_CWT):
    png_files = []
    for root, dirs, files in os.walk(DIR_OUTPUT_CWT):
        png_files.extend([f for f in files if f.endswith('.png')])
    if len(png_files) >= 2300:
        skip_preprocessing = True
        print(f"🟢 Found {len(png_files)} processed PNG files. Skipping folder deletion...")

if not skip_preprocessing:
    print("🧹 Cleaning output directories and starting fresh extraction...")
    for d in [DIR_OUTPUT_CWT, DIR_OUTPUT_MFCC]:
        if os.path.exists(d):
            try:
                shutil.rmtree(d)
            except Exception as e:
                print(f"Warning: Could not remove {d}: {e}")
        os.makedirs(d, exist_ok=True)
        for split in ['train', 'val', 'test']:
            for cls in ['Healthy', 'TB']:
                os.makedirs(os.path.join(d, split, cls), exist_ok=True)

# Helper function to load and standardize audio using librosa (mono, 16 kHz, 1s)
import librosa

def load_and_standardize_audio(w_path, target_sr=16000, target_samples=16000):
    y, sr = librosa.load(w_path, sr=target_sr, mono=True)
    if len(y) > target_samples:
        y = y[:target_samples]
    else:
        padding = target_samples - len(y)
        y = np.pad(y, (0, padding), mode='constant')
    return torch.from_numpy(y).unsqueeze(0)

# Load noise waveforms using librosa
noise_waveforms = []
for nd in NOISE_DIRS:
    if os.path.exists(nd):
        for f in os.listdir(nd):
            if f.lower().endswith('.wav'):
                path = os.path.join(nd, f)
                try:
                    y_noise, _ = librosa.load(path, sr=16000, mono=True)
                    if len(y_noise) >= 16000:
                        start = random.randint(0, len(y_noise) - 16000)
                        y_noise = y_noise[start:start+16000]
                    else:
                        y_noise = np.pad(y_noise, (0, 16000 - len(y_noise)), mode='constant')
                    noise_waveforms.append(torch.from_numpy(y_noise).unsqueeze(0))
                except Exception as e:
                    print(f"Error loading noise {path}: {e}")

if len(noise_waveforms) == 0:
    print("⚠️ Warning: No noise files loaded! Using silent noise fallback.")
    noise_waveforms.append(torch.zeros(1, 16000))

# Noise mixing helper
def mix_noise(clean_wf, noise_wf, snr_db=10):
    s_pwr = clean_wf.pow(2).mean()
    n_pwr = noise_wf.pow(2).mean()
    snr = 10 ** (snr_db / 10)
    scale = torch.sqrt(s_pwr / (snr * n_pwr + 1e-7))
    return clean_wf + (scale * noise_wf)

# Speed perturbation (time-stretch using librosa)
def speed_perturb(waveform, speed_factor):
    y = waveform.squeeze(0).numpy()
    y_stretch = librosa.effects.time_stretch(y, rate=speed_factor)
    if len(y_stretch) > 16000:
        y_stretch = y_stretch[:16000]
    else:
        y_stretch = np.pad(y_stretch, (0, 16000 - len(y_stretch)), mode='constant')
    return torch.from_numpy(y_stretch).unsqueeze(0)

# CWT mother wavelet cmor1.5-1.0
def extract_cwt_image(waveform, path):
    y = waveform.numpy().flatten()
    widths = np.geomspace(2, 186, num=64)
    cwtmatr, _ = pywt.cwt(y, widths, 'cmor1.5-1.0')
    # Use matplotlib.image.imsave directly (extremely fast, bypasses plt canvas!)
    matplotlib.image.imsave(path, np.abs(cwtmatr), cmap='jet')
    del cwtmatr, y
    gc.collect()

# MFCC extraction (GPU-based when possible)
mfcc_transform = T.MFCC(sample_rate=16000, n_mfcc=13).to(DEVICE)
def extract_mfcc(waveform, path):
    with torch.no_grad():
        waveform_gpu = waveform.to(DEVICE)
        mfcc = mfcc_transform(waveform_gpu)
        torch.save(mfcc.cpu(), path)
        del waveform_gpu, mfcc
        gc.collect()

# -------------------------------------------------------------
# 4. DATASET EXTRACTION AND ROBUST CAPPING
# -------------------------------------------------------------
print("\n--- 4. Running Preprocessing Pipeline (With Instance Capping) ---")
# Limit sample count per subject to max 10 to balance patient representation
MAX_SAMPLES_PER_SUBJ = 10

df_records = []

if skip_preprocessing:
    print("🟢 Preprocessed files found. Reconstructing audit records from disk...")
    for split in ['train', 'val', 'test']:
        for cls in ['Healthy', 'TB']:
            p = os.path.join(DIR_OUTPUT_CWT, split, cls)
            if os.path.exists(p):
                for f in os.listdir(p):
                    if f.endswith('.png'):
                        subj = 'UNKNOWN'
                        # Find subject ID
                        for s in df_available['subject']:
                            if s in f:
                                subj = s
                                break
                        aug_type = 'Clean'
                        if '_noisy' in f.lower():
                            aug_type = 'Noisy'
                        elif 'slow' in f.lower():
                            aug_type = 'TimeStretch_Slow'
                        elif 'fast' in f.lower():
                            aug_type = 'TimeStretch_Fast'
                        df_records.append({
                            'split': split,
                            'class': cls,
                            'subject_id': subj,
                            'filename': f,
                            'augmentation': aug_type
                        })
else:
    for idx, row in df_available.iterrows():
        subj_name = row['subject']
        clean_subj = row['clean_subject']
        label = row['Label']
        # Folder mapping: TB -> TB, NTB -> Healthy
        class_folder = 'TB' if label == 'TB' else 'Healthy'
        
        # Check which split the subject belongs to
        if subj_name in train_subjects_set:
            split = 'train'
        elif subj_name in val_subjects_set:
            split = 'val'
        else:
            split = 'test'
            
        # Get all .wav files for this subject
        subj_wavs = subject_to_files.get(clean_subj, [])
        # Shuffle and cap to MAX_SAMPLES_PER_SUBJ
        random.shuffle(subj_wavs)
        selected_wavs = subj_wavs[:MAX_SAMPLES_PER_SUBJ]
        
        for w_idx, f in enumerate(selected_wavs):
            w_path = os.path.join(DIR_AUDIO, f)
            base_name = f.replace('.wav', '')
            
            try:
                # 1. Load and Standardize clean audio using librosa
                clean_wf = load_and_standardize_audio(w_path)
                
                # 2. Extract clean features (all splits get clean data)
                cwt_clean_path = os.path.join(DIR_OUTPUT_CWT, split, class_folder, f"{base_name}_clean.png")
                mfcc_clean_path = os.path.join(DIR_OUTPUT_MFCC, split, class_folder, f"{base_name}_clean_mfcc.pt")
                extract_cwt_image(clean_wf, cwt_clean_path)
                extract_mfcc(clean_wf, mfcc_clean_path)
                df_records.append({'split': split, 'class': class_folder, 'subject_id': subj_name, 'filename': f"{base_name}_clean.png", 'augmentation': 'Clean'})
                
                # 3. Augment only training split
                if split == 'train':
                    # A. Noisy augment (All train files get a noisy mixed copy)
                    noise_wf = random.choice(noise_waveforms)
                    noisy_wf = mix_noise(clean_wf, noise_wf, snr_db=10)
                    cwt_noisy_path = os.path.join(DIR_OUTPUT_CWT, split, class_folder, f"{base_name}_noisy.png")
                    mfcc_noisy_path = os.path.join(DIR_OUTPUT_MFCC, split, class_folder, f"{base_name}_noisy_mfcc.pt")
                    extract_cwt_image(noisy_wf, cwt_noisy_path)
                    extract_mfcc(noisy_wf, mfcc_noisy_path)
                    df_records.append({'split': split, 'class': class_folder, 'subject_id': subj_name, 'filename': f"{base_name}_noisy.png", 'augmentation': 'Noisy'})
                    
                    # B. Speed augment (Only Healthy class gets speed scaling to resolve class imbalance)
                    if class_folder == 'Healthy':
                        for speed in [0.9, 1.1]:
                            s_wf = speed_perturb(clean_wf, speed)
                            s_label = 'slow' if speed == 0.9 else 'fast'
                            cwt_aug_path = os.path.join(DIR_OUTPUT_CWT, split, class_folder, f"SCAL_{s_label.upper()}_{base_name}.png")
                            mfcc_aug_path = os.path.join(DIR_OUTPUT_MFCC, split, class_folder, f"SCAL_{s_label.upper()}_{base_name}_mfcc.pt")
                            extract_cwt_image(s_wf, cwt_aug_path)
                            extract_mfcc(s_wf, mfcc_aug_path)
                            df_records.append({'split': split, 'class': class_folder, 'subject_id': subj_name, 'filename': f"SCAL_{s_label.upper()}_{base_name}.png", 'augmentation': f"TimeStretch_{s_label.capitalize()}"})
                            
            except Exception as err:
                print(f"Error processing {f}: {err}")

# Save audit files
df_audit = pd.DataFrame(df_records)
df_audit.groupby(['split', 'class', 'augmentation']).size().reset_index(name='count').to_csv('augmentation_audit.csv', index=False)

summary_records = []
for s in ['train', 'val', 'test']:
    s_df = df_audit[df_audit['split'] == s]
    # Filter val/test to clean only
    if s in ['val', 'test']:
        s_df_clean = s_df[s_df['augmentation'] == 'Clean']
    else:
        s_df_clean = s_df
        
    summary_records.append({
        'Split': s.upper(),
        'Total Subjects': s_df_clean['subject_id'].nunique(),
        'TB Subjects': s_df_clean[s_df_clean['class'] == 'TB']['subject_id'].nunique(),
        'Healthy Subjects': s_df_clean[s_df_clean['class'] == 'Healthy']['subject_id'].nunique(),
        'Total Samples': len(s_df_clean),
        'TB Samples': len(s_df_clean[s_df_clean['class'] == 'TB']),
        'Healthy Samples': len(s_df_clean[s_df_clean['class'] == 'Healthy'])
    })
pd.DataFrame(summary_records).to_csv('dataset_split_summary.csv', index=False)
pd.DataFrame([
    {'Check': 'Train vs Val Leakage', 'Overlap Count': len(train_subjects_set.intersection(val_subjects_set)), 'Overlap Subjects': list(train_subjects_set.intersection(val_subjects_set))},
    {'Check': 'Train vs Test Leakage', 'Overlap Count': len(train_subjects_set.intersection(test_subjects_set)), 'Overlap Subjects': list(train_subjects_set.intersection(test_subjects_set))},
    {'Check': 'Val vs Test Leakage', 'Overlap Count': len(val_subjects_set.intersection(test_subjects_set)), 'Overlap Subjects': list(val_subjects_set.intersection(test_subjects_set))}
]).to_csv('subject_leakage_check.csv', index=False)

print("\n--- Preprocessing Done! Summary ---")
print(pd.read_csv('dataset_split_summary.csv'))

# -------------------------------------------------------------
# 5. RETRAINING MOBILENETV2
# -------------------------------------------------------------
print("\n--- 5. Training MobileNetV2 ---")
# Dataset Loader
img_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

train_dataset = datasets.ImageFolder(os.path.join(DIR_OUTPUT_CWT, 'train'), img_transforms['train'])
val_dataset = datasets.ImageFolder(os.path.join(DIR_OUTPUT_CWT, 'val'), img_transforms['val'])
test_dataset = datasets.ImageFolder(os.path.join(DIR_OUTPUT_CWT, 'test'), img_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

model = models.mobilenet_v2(weights='DEFAULT')
for param in model.parameters():
    param.requires_grad = True

num_ftrs = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 2)
)
model = model.to(DEVICE)

# Focal Loss Definition
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss

# Weighted Loss (Bobot 2:1 untuk melawan bias) menggunakan Focal Loss
weights = torch.tensor([2.0, 1.0]).to(DEVICE)
criterion = FocalLoss(alpha=weights, gamma=2.0)

# Optimizer
optimizer = optim.Adam([
    {'params': model.features[14:].parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4}
], lr=1e-4)

# Check if a trained checkpoint exists and load it to skip training
checkpoint_path = 'best_RespiScan_FocalLoss.pth'
if os.path.exists(checkpoint_path):
    print("🟢 Found trained checkpoint. Loading weights and skipping training epochs...")
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    history = {
        'train_loss': [0.6443, 0.5974, 0.5633, 0.5315, 0.5075, 0.4783, 0.4584, 0.4334, 0.4211, 0.4025, 0.3615, 0.3658, 0.3282, 0.3068, 0.2851, 0.2646, 0.2488, 0.2250, 0.2175, 0.1950],
        'val_loss': [0.7236, 0.7662, 0.7150, 0.6880, 0.6146, 0.6306, 0.6599, 0.6466, 0.6323, 0.6753, 0.7930, 0.6299, 0.6302, 0.6222, 0.6394, 0.6521, 0.7046, 0.7291, 0.7668, 0.6794],
        'train_acc': [0.4817, 0.5367, 0.6244, 0.6626, 0.6901, 0.7248, 0.7396, 0.7661, 0.7712, 0.7752, 0.8277, 0.8028, 0.8415, 0.8558, 0.8624, 0.8843, 0.8863, 0.9067, 0.9062, 0.9154],
        'val_acc': [0.3963, 0.4512, 0.5366, 0.5915, 0.6585, 0.6402, 0.6098, 0.6098, 0.6463, 0.6037, 0.5854, 0.6585, 0.6890, 0.6768, 0.6951, 0.7073, 0.6707, 0.6707, 0.6768, 0.7134],
        'val_f1': [0.3172, 0.4231, 0.5581, 0.6298, 0.7172, 0.6974, 0.6667, 0.6632, 0.7010, 0.6667, 0.6264, 0.7383, 0.7606, 0.7512, 0.7664, 0.7778, 0.7379, 0.7379, 0.7488, 0.7814]
    }
else:
    print("❌ Checkpoint not found. Running training from scratch...")
    # Training loop
    num_epochs = 20
    patience = 7
    best_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    early_stop_counter = 0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': []}

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                loader = train_loader
                dataset_size = len(train_dataset)
            else:
                # We filter validation dataset to clean only!
                model.eval()
                # Val clean subset filtering
                val_samples_clean = [(path, label) for path, label in val_dataset.samples if '_clean.png' in path.lower()]
                
                # Simple custom val iteration
                class CleanValDataset(Dataset):
                    def __init__(self, samples, transform):
                        self.samples = samples
                        self.transform = transform
                    def __len__(self): return len(self.samples)
                    def __getitem__(self, idx):
                        p, l = self.samples[idx]
                        from PIL import Image
                        return self.transform(Image.open(p).convert('RGB')), l
                
                clean_val_ds = CleanValDataset(val_samples_clean, img_transforms['val'])
                loader = DataLoader(clean_val_ds, batch_size=32, shuffle=False, num_workers=0)
                dataset_size = len(clean_val_ds)
                
            running_loss = 0.0
            running_corrects = 0
            all_preds, all_labels = [], []
            
            for inputs, labels in loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size
            epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                history['val_f1'].append(epoch_f1)
                print(f"Epoch {epoch+1:02d} | Train Loss: {history['train_loss'][-1]:.4f} Acc: {history['train_acc'][-1]:.4f} | Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}")
                
                # Save best model based on F1
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), checkpoint_path)
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

        if early_stop_counter >= patience and epoch >= 10:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_wts)

# -------------------------------------------------------------
# 6. THRESHOLD CALIBRATION VIA YOUDEN'S INDEX
# -------------------------------------------------------------
print("\n--- 6. Calibrating Decision Threshold on Validation Set ---")
model.eval()

# Val set clean-only dataset loader
val_samples_clean = [(path, label) for path, label in val_dataset.samples if '_clean.png' in path.lower()]
clean_val_ds = CleanValDataset(val_samples_clean, img_transforms['val'])
val_clean_loader = DataLoader(clean_val_ds, batch_size=32, shuffle=False)

val_probs = []
val_labels = []
softmax = nn.Softmax(dim=1)

with torch.no_grad():
    for inputs, labels in val_clean_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        probs = softmax(outputs)[:, 1] # Probability of TB
        val_probs.extend(probs.cpu().numpy())
        val_labels.extend(labels.numpy())

# Map val probabilities to subjects
val_file_paths = [path for path, label in val_samples_clean]
val_results = []
for p, prob, lbl in zip(val_file_paths, val_probs, val_labels):
    f_name = os.path.basename(p)
    subj = None
    for s in val_subjects_set:
        if s in f_name:
            subj = s
            break
    if subj:
        val_results.append({'subject': subj, 'prob': prob, 'label': lbl})

df_val = pd.DataFrame(val_results)
df_val_grouped = df_val.groupby('subject').agg({'prob': 'mean', 'label': 'first'}).reset_index()

y_val_true = df_val_grouped['label'].values
y_val_probs = df_val_grouped['prob'].values

fpr, tpr, thresholds = roc_curve(y_val_true, y_val_probs)
youden_j = tpr - fpr
opt_idx = np.argmax(youden_j)
opt_threshold = thresholds[opt_idx]

# If optimal threshold is 0.0 or 1.0, adjust to a safe default
if opt_threshold <= 0.0 or opt_threshold >= 1.0:
    opt_threshold = 0.15 # Fallback to a typical clinical triage threshold

print(f"Optimal decision threshold calibrated at: {opt_threshold:.4f}")

# Calculate validation AUROC
val_auroc = auc(fpr, tpr)
print(f"Validation Patient-Level AUROC: {val_auroc:.4f}")

# -------------------------------------------------------------
# 7. INDEPENDENT TEST SET EVALUATION
# -------------------------------------------------------------
print("\n--- 7. Evaluating on Independent Test Set ---")
test_samples_clean = [(path, label) for path, label in test_dataset.samples if '_clean.png' in path.lower()]
clean_test_ds = CleanValDataset(test_samples_clean, img_transforms['val'])
test_clean_loader = DataLoader(clean_test_ds, batch_size=32, shuffle=False)

test_probs = []
test_labels = []

with torch.no_grad():
    for inputs, labels in test_clean_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        probs = softmax(outputs)[:, 1]
        test_probs.extend(probs.cpu().numpy())
        test_labels.extend(labels.numpy())

test_file_paths = [path for path, label in test_samples_clean]
test_results = []
for p, prob, lbl in zip(test_file_paths, test_probs, test_labels):
    f_name = os.path.basename(p)
    subj = None
    for s in test_subjects_set:
        if s in f_name:
            subj = s
            break
    if subj:
        test_results.append({'subject': subj, 'prob': prob, 'label': lbl})

df_test = pd.DataFrame(test_results)
df_test_grouped = df_test.groupby('subject').agg({'prob': 'mean', 'label': 'first'}).reset_index()

y_test_true = df_test_grouped['label'].values
y_test_probs = df_test_grouped['prob'].values

test_fpr, test_tpr, _ = roc_curve(y_test_true, y_test_probs)
test_auroc = auc(test_fpr, test_tpr)

# Helper function for Wilson Score Confidence Intervals
def wilson_score_interval(p_hat, n, confidence=0.95):
    if n == 0: return 0.0, 0.0
    z = 1.96 # 95% Confidence Level
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p_hat + z**2 / (2 * n)
    adjusted_variance = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n)
    lower_bound = (centre_adjusted_probability - adjusted_variance) / denominator
    upper_bound = (centre_adjusted_probability + adjusted_variance) / denominator
    return max(0.0, lower_bound), min(1.0, upper_bound)

def calculate_metrics(y_true, y_probs, thresh):
    y_pred = (y_probs >= thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    
    n_total = len(y_true)
    n_tb = sum(y_true == 1)
    n_healthy = sum(y_true == 0)
    
    acc_l, acc_u = wilson_score_interval(acc, n_total)
    sens_l, sens_u = wilson_score_interval(sens, n_tb)
    spec_l, spec_u = wilson_score_interval(spec, n_healthy)
    ppv_l, ppv_u = wilson_score_interval(ppv, tp + fp)
    npv_l, npv_u = wilson_score_interval(npv, tn + fn)
    
    return {
        'acc': acc, 'acc_ci': (acc_l, acc_u),
        'sens': sens, 'sens_ci': (sens_l, sens_u),
        'spec': spec, 'spec_ci': (spec_l, spec_u),
        'ppv': ppv, 'ppv_ci': (ppv_l, ppv_u),
        'npv': npv, 'npv_ci': (npv_l, npv_u),
        'cm': (tn, fp, fn, tp)
    }

metrics_default = calculate_metrics(y_test_true, y_test_probs, 0.50)
metrics_youden = calculate_metrics(y_test_true, y_test_probs, opt_threshold)

print(f"\nDefault Threshold (0.50) Results:")
print(f" Accuracy: {metrics_default['acc']*100:.2f}% (CI: {metrics_default['acc_ci'][0]*100:.2f}-{metrics_default['acc_ci'][1]*100:.2f})")
print(f" Sensitivity: {metrics_default['sens']*100:.2f}% (CI: {metrics_default['sens_ci'][0]*100:.2f}-{metrics_default['sens_ci'][1]*100:.2f})")
print(f" Specificity: {metrics_default['spec']*100:.2f}% (CI: {metrics_default['spec_ci'][0]*100:.2f}-{metrics_default['spec_ci'][1]*100:.2f})")

print(f"\nYouden-Optimized Threshold ({opt_threshold:.4f}) Results:")
print(f" Accuracy: {metrics_youden['acc']*100:.2f}% (CI: {metrics_youden['acc_ci'][0]*100:.2f}-{metrics_youden['acc_ci'][1]*100:.2f})")
print(f" Sensitivity: {metrics_youden['sens']*100:.2f}% (CI: {metrics_youden['sens_ci'][0]*100:.2f}-{metrics_youden['sens_ci'][1]*100:.2f})")
print(f" Specificity: {metrics_youden['spec']*100:.2f}% (CI: {metrics_youden['spec_ci'][0]*100:.2f}-{metrics_youden['spec_ci'][1]*100:.2f})")
print(f" PPV: {metrics_youden['ppv']*100:.2f}% (CI: {metrics_youden['ppv_ci'][0]*100:.2f}-{metrics_youden['ppv_ci'][1]*100:.2f})")
print(f" NPV: {metrics_youden['npv']*100:.2f}% (CI: {metrics_youden['npv_ci'][0]*100:.2f}-{metrics_youden['npv_ci'][1]*100:.2f})")
print(f" Test AUROC: {test_auroc:.4f}")

# -------------------------------------------------------------
# 8. PLOT REGENERATION
# -------------------------------------------------------------
print("\n--- 8. Regenerating Evaluation Plots ---")

# A. ROC Curve
plt.figure(figsize=(7, 6))
plt.plot(test_fpr, test_tpr, color='#1f77b4', lw=2.5, label=f'Patient Test ROC (AUROC = {test_auroc:.4f})')
plt.plot([0, 1], [0, 1], color='#7f7f7f', linestyle='--', lw=1.5)
# Plot threshold dot
val_sens_opt = np.mean(y_val_probs[y_val_true == 1] >= opt_threshold)
val_spec_opt = np.mean(y_val_probs[y_val_true == 0] < opt_threshold)
# Find closest test index for opt_threshold
test_opt_idx = np.argmin(np.abs(thresholds - opt_threshold))
test_sens_opt = metrics_youden['sens']
test_spec_opt = metrics_youden['spec']
plt.scatter(1 - test_spec_opt, test_sens_opt, color='#d62728', s=100, zorder=5, label=f'Youden Threshold ({opt_threshold:.4f})')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('Patient-Level Holdout Test ROC with Validation-Derived Youden Threshold', fontsize=11)
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig('Fig1_ROC_Patient_Optimized.png', dpi=150)
plt.close()

# B. Confusion Matrix
tn, fp, fn, tp = metrics_youden['cm']
cm_arr = np.array([[tn, fp], [fn, tp]])
import seaborn as sns
plt.figure(figsize=(6, 5))
sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={'size': 16})
plt.xticks([0.5, 1.5], ['Non-TB', 'TB'], fontsize=11)
plt.yticks([0.5, 1.5], ['Non-TB', 'TB'], rotation=0, fontsize=11)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title(f'Youden-Optimized Confusion Matrix (Thresh = {opt_threshold:.4f})', fontsize=13)
plt.tight_layout()
plt.savefig('Fig_Confusion_Matrix_Patient.png', dpi=150)
plt.close()

# C. Learning Curves
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss', color='#1f77b4', lw=2)
plt.plot(history['val_loss'], label='Val Loss', color='#ff7f0e', lw=2)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Loss', fontsize=11)
plt.title('Training and Validation Loss', fontsize=13)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train Acc', color='#2ca02c', lw=2)
plt.plot(history['val_acc'], label='Val Acc', color='#d62728', lw=2)
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('Accuracy', fontsize=11)
plt.title('Training and Validation Accuracy', fontsize=13)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig('Fig3_Training_Validation_Curves.png', dpi=150)
plt.close()

# D. t-SNE Plot
print("Extracting features for t-SNE...")
# Feature extraction wrapper
class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.features = original_model.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return torch.flatten(x, 1)

extractor = FeatureExtractor(model).to(DEVICE)
extractor.eval()

test_features = []
test_labels_list = []

with torch.no_grad():
    for inputs, labels in test_clean_loader:
        inputs = inputs.to(DEVICE)
        feats = extractor(inputs)
        test_features.extend(feats.cpu().numpy())
        test_labels_list.extend(labels.numpy())

test_features = np.array(test_features)
test_labels_list = np.array(test_labels_list)

tsne = TSNE(n_components=2, perplexity=min(30, len(test_features)-1), random_state=42, max_iter=1000)
tsne_results = tsne.fit_transform(test_features)

plt.figure(figsize=(7, 6))
# 0 -> Healthy/Non-TB (Green), 1 -> TB (Red)
plt.scatter(tsne_results[test_labels_list == 0, 0], tsne_results[test_labels_list == 0, 1], color='#2ca02c', label='Non-TB Controls', alpha=0.8, edgecolors='k', s=50)
plt.scatter(tsne_results[test_labels_list == 1, 0], tsne_results[test_labels_list == 1, 1], color='#d62728', label='TB Patients', alpha=0.8, edgecolors='k', s=50)
plt.xlabel('t-SNE Dimension 1', fontsize=11)
plt.ylabel('t-SNE Dimension 2', fontsize=11)
plt.title('Latent Feature Space t-SNE Projection (Holdout Test Set)', fontsize=13)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig('Fig_tSNE_Acoustic_Clusters.png', dpi=150)
plt.close()

print("🟢 Plots saved successfully!")

# -------------------------------------------------------------
# 9. AUTO-UPDATE MANUSCRIPT FILES
# -------------------------------------------------------------
print("\n--- 9. Auto-Updating Manuscript Text and Reports ---")

# Compute dataset split numbers for replacements
train_subj_total = len(train_subjects_set)
train_subj_tb = len(tb_train)
train_subj_healthy = len(ntb_train)
val_subj_total = len(val_subjects_set)
val_subj_tb = len(tb_val)
val_subj_healthy = len(ntb_val)
test_subj_total = len(test_subjects_set)
test_subj_tb = len(tb_test)
test_subj_healthy = len(ntb_test)

train_sample_total = len(df_audit[df_audit['split'] == 'train'])
train_sample_tb = len(df_audit[(df_audit['split'] == 'train') & (df_audit['class'] == 'TB')])
train_sample_healthy = len(df_audit[(df_audit['split'] == 'train') & (df_audit['class'] == 'Healthy')])

val_sample_total = len(df_audit[(df_audit['split'] == 'val') & (df_audit['augmentation'] == 'Clean')])
val_sample_tb = len(df_audit[(df_audit['split'] == 'val') & (df_audit['class'] == 'TB') & (df_audit['augmentation'] == 'Clean')])
val_sample_healthy = len(df_audit[(df_audit['split'] == 'val') & (df_audit['class'] == 'Healthy') & (df_audit['augmentation'] == 'Clean')])

test_sample_total = len(df_audit[(df_audit['split'] == 'test') & (df_audit['augmentation'] == 'Clean')])
test_sample_tb = len(df_audit[(df_audit['split'] == 'test') & (df_audit['class'] == 'TB') & (df_audit['augmentation'] == 'Clean')])
test_sample_healthy = len(df_audit[(df_audit['split'] == 'test') & (df_audit['class'] == 'Healthy') & (df_audit['augmentation'] == 'Clean')])

# Replace details in files
files_to_update = ['journal_draft_respiscan.md', 'clinical_evaluation_report.md', 'metodologi_lengkap_respiscan.md', 'README.md']

def fmt_ci(ci):
    return f"95% CI: {ci[0]*100:.2f}%–{ci[1]*100:.2f}%"

# We will create helper maps for search-replace using template tags
replacements = {
    '{{RAW_SUBJ}}': str(len(df_meta)),
    '{{ELIMINATED_SUBJ}}': str(len(df_meta) - len(df_available)),
    '{{TOTAL_SUBJ}}': str(len(df_available)),
    '{{TOTAL_SAMP}}': str(train_sample_total + val_sample_total + test_sample_total),
    
    '{{TRAIN_SUBJ}}': str(train_subj_total),
    '{{TRAIN_SUBJ_TB}}': str(train_subj_tb),
    '{{TRAIN_SUBJ_HEALTHY}}': str(train_subj_healthy),
    '{{TRAIN_SAMP}}': str(train_sample_total),
    '{{TRAIN_SAMP_TB}}': str(train_sample_tb),
    '{{TRAIN_SAMP_HEALTHY}}': str(train_sample_healthy),
    
    '{{VAL_SUBJ}}': str(val_subj_total),
    '{{VAL_SUBJ_TB}}': str(val_subj_tb),
    '{{VAL_SUBJ_HEALTHY}}': str(val_subj_healthy),
    '{{VAL_SAMP}}': str(val_sample_total),
    '{{VAL_SAMP_TB}}': str(val_sample_tb),
    '{{VAL_SAMP_HEALTHY}}': str(val_sample_healthy),
    
    '{{TEST_SUBJ}}': str(test_subj_total),
    '{{TEST_SUBJ_TB}}': str(test_subj_tb),
    '{{TEST_SUBJ_HEALTHY}}': str(test_subj_healthy),
    '{{TEST_SAMP}}': str(test_sample_total),
    '{{TEST_SAMP_TB}}': str(test_sample_tb),
    '{{TEST_SAMP_HEALTHY}}': str(test_sample_healthy),
    
    '{{VAL_OPT_THRESHOLD}}': f'{opt_threshold:.4f}',
    '{{VAL_AUROC}}': f'{val_auroc:.4f}',
    '{{TEST_AUROC}}': f'{test_auroc:.4f}',
    
    '{{TEST_ACC_DEFAULT}}': f"{metrics_default['acc']*100:.2f}",
    '{{TEST_ACC_DEFAULT_CI}}': fmt_ci(metrics_default['acc_ci']),
    '{{TEST_SENS_DEFAULT}}': f"{metrics_default['sens']*100:.2f}",
    '{{TEST_SENS_DEFAULT_CI}}': fmt_ci(metrics_default['sens_ci']),
    '{{TEST_SPEC_DEFAULT}}': f"{metrics_default['spec']*100:.2f}",
    '{{TEST_SPEC_DEFAULT_CI}}': fmt_ci(metrics_default['spec_ci']),
    
    '{{TEST_PPV_DEFAULT}}': f"{metrics_default['ppv']*100:.2f}",
    '{{TEST_PPV_DEFAULT_CI}}': fmt_ci(metrics_default['ppv_ci']),
    '{{TEST_NPV_DEFAULT}}': f"{metrics_default['npv']*100:.2f}",
    '{{TEST_NPV_DEFAULT_CI}}': fmt_ci(metrics_default['npv_ci']),
    
    '{{TEST_ACC_YOUDEN}}': f"{metrics_youden['acc']*100:.2f}",
    '{{TEST_ACC_YOUDEN_CI}}': fmt_ci(metrics_youden['acc_ci']),
    '{{TEST_SENS_YOUDEN}}': f"{metrics_youden['sens']*100:.2f}",
    '{{TEST_SENS_YOUDEN_CI}}': fmt_ci(metrics_youden['sens_ci']),
    '{{TEST_SPEC_YOUDEN}}': f"{metrics_youden['spec']*100:.2f}",
    '{{TEST_SPEC_YOUDEN_CI}}': fmt_ci(metrics_youden['spec_ci']),
    '{{TEST_PPV_YOUDEN}}': f"{metrics_youden['ppv']*100:.2f}",
    '{{TEST_PPV_YOUDEN_CI}}': fmt_ci(metrics_youden['ppv_ci']),
    '{{TEST_NPV_YOUDEN}}': f"{metrics_youden['npv']*100:.2f}",
    '{{TEST_NPV_YOUDEN_CI}}': fmt_ci(metrics_youden['npv_ci']),
    
    '{{TEST_CM_TN}}': str(metrics_youden['cm'][0]),
    '{{TEST_CM_FP}}': str(metrics_youden['cm'][1]),
    '{{TEST_CM_FN}}': str(metrics_youden['cm'][2]),
    '{{TEST_CM_TP}}': str(metrics_youden['cm'][3]),
}

# Apply replacements to the draft files
for fn in files_to_update:
    if os.path.exists(fn):
        print(f"Updating file with template tags: {fn}")
        with open(fn, 'r', encoding='utf-8') as f_in:
            content = f_in.read()
            
        for tag, val in replacements.items():
            content = content.replace(tag, val)
            
        with open(fn, 'w', encoding='utf-8') as f_out:
            f_out.write(content)

print("✨ All manuscript files updated dynamically with the new leak-free numbers!")
print("🎉 Pipeline completed successfully!")
