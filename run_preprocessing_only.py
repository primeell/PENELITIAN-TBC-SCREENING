import os
import sys
import re
import gc
import random
import shutil
import json
import time
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import pywt
import matplotlib
matplotlib.use('Agg')
import matplotlib.image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure stdout to use UTF-8 to prevent Windows console encoding errors
sys.stdout.reconfigure(encoding='utf-8')

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

# For preprocessing, CPU is robust and extremely fast for single-waveform MFCC
DEVICE = torch.device("cpu")

# Helper function to load and standardize audio
def load_and_standardize_audio(w_path, target_sr=16000, target_samples=16000):
    y, sr = librosa.load(w_path, sr=target_sr, mono=True)
    if len(y) > target_samples:
        y = y[:target_samples]
    else:
        padding = target_samples - len(y)
        y = np.pad(y, (0, padding), mode='constant')
    return torch.from_numpy(y).unsqueeze(0)

# Noise mixing helper
def mix_noise(clean_wf, noise_wf, snr_db=10):
    s_pwr = clean_wf.pow(2).mean()
    n_pwr = noise_wf.pow(2).mean()
    snr = 10 ** (snr_db / 10)
    scale = torch.sqrt(s_pwr / (snr * n_pwr + 1e-7))
    return clean_wf + (scale * noise_wf)

# Speed perturbation
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
    matplotlib.image.imsave(path, np.abs(cwtmatr), cmap='jet')

# MFCC extraction
# Global transform defined in child process when loaded
mfcc_transform = T.MFCC(sample_rate=16000, n_mfcc=13).to(DEVICE)
def extract_mfcc(waveform, path):
    with torch.no_grad():
        waveform_gpu = waveform.to(DEVICE)
        mfcc = mfcc_transform(waveform_gpu)
        torch.save(mfcc.cpu(), path)

def process_single_subject(args):
    """
    Worker function to process all waveforms of a single subject.
    args: (subj_name, clean_subj, label, split, selected_wavs, noise_waveforms)
    """
    subj_name, clean_subj, label, split, selected_wavs, noise_waveforms = args
    class_folder = 'TB' if label == 'TB' else 'Healthy'
    records = []
    
    # Re-seed inside process for randomness in noise choice
    # Use clean_subj string hash + random mix as seed
    seed = abs(hash(clean_subj)) % (2**32 - 1)
    random.seed(seed)
    
    for f in selected_wavs:
        w_path = os.path.join(DIR_AUDIO, f)
        base_name = f.replace('.wav', '')
        
        try:
            # 1. Load and Standardize
            clean_wf = load_and_standardize_audio(w_path)
            
            # 2. Extract clean features
            cwt_clean_path = os.path.join(DIR_OUTPUT_CWT, split, class_folder, f"{base_name}_clean.png")
            mfcc_clean_path = os.path.join(DIR_OUTPUT_MFCC, split, class_folder, f"{base_name}_clean_mfcc.pt")
            extract_cwt_image(clean_wf, cwt_clean_path)
            extract_mfcc(clean_wf, mfcc_clean_path)
            records.append({'split': split, 'class': class_folder, 'subject_id': subj_name, 'filename': f"{base_name}_clean.png", 'augmentation': 'Clean'})
            
            # 3. Augment only training split
            if split == 'train':
                # A. Noisy augment
                noise_wf = random.choice(noise_waveforms)
                noisy_wf = mix_noise(clean_wf, noise_wf, snr_db=10)
                cwt_noisy_path = os.path.join(DIR_OUTPUT_CWT, split, class_folder, f"{base_name}_noisy.png")
                mfcc_noisy_path = os.path.join(DIR_OUTPUT_MFCC, split, class_folder, f"{base_name}_noisy_mfcc.pt")
                extract_cwt_image(noisy_wf, cwt_noisy_path)
                extract_mfcc(noisy_wf, mfcc_noisy_path)
                records.append({'split': split, 'class': class_folder, 'subject_id': subj_name, 'filename': f"{base_name}_noisy.png", 'augmentation': 'Noisy'})
                
                # B. Speed augment (Only Healthy)
                if class_folder == 'Healthy':
                    for speed in [0.9, 1.1]:
                        s_wf = speed_perturb(clean_wf, speed)
                        s_label = 'slow' if speed == 0.9 else 'fast'
                        cwt_aug_path = os.path.join(DIR_OUTPUT_CWT, split, class_folder, f"SCAL_{s_label.upper()}_{base_name}.png")
                        mfcc_aug_path = os.path.join(DIR_OUTPUT_MFCC, split, class_folder, f"SCAL_{s_label.upper()}_{base_name}_mfcc.pt")
                        extract_cwt_image(s_wf, cwt_aug_path)
                        extract_mfcc(s_wf, mfcc_aug_path)
                        records.append({'split': split, 'class': class_folder, 'subject_id': subj_name, 'filename': f"SCAL_{s_label.upper()}_{base_name}.png", 'augmentation': f"TimeStretch_{s_label.capitalize()}"})
                        
        except Exception as err:
            print(f"\n❌ Error processing {f}: {err}")
            
    return records


def main():
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
    # 3. PREPROCESSING PIPELINE SETUP
    # -------------------------------------------------------------
    print("\n--- 3. Cleaning Output Directories and Starting Fresh Extraction ---")
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
                
    # Load noise waveforms
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
        
    # Prepare task arguments for multiprocessing
    print("\n--- 4. Running Preprocessing Pipeline in Parallel (Multiprocessing) ---")
    MAX_SAMPLES_PER_SUBJ = 10
    tasks = []
    
    # Re-seed before random selection of wavs for deterministic sample capping
    random.seed(42)
    
    for idx, row in df_available.iterrows():
        subj_name = row['subject']
        clean_subj = row['clean_subject']
        label = row['Label']
        
        if subj_name in train_subjects_set:
            split = 'train'
        elif subj_name in val_subjects_set:
            split = 'val'
        else:
            split = 'test'
            
        subj_wavs = subject_to_files.get(clean_subj, [])
        # copy list to avoid sorting/shuffling differences
        subj_wavs = list(subj_wavs)
        # Sort first to ensure absolute determinism across processes/runs
        subj_wavs.sort()
        random.shuffle(subj_wavs)
        selected_wavs = subj_wavs[:MAX_SAMPLES_PER_SUBJ]
        
        tasks.append((subj_name, clean_subj, label, split, selected_wavs, noise_waveforms))
        
    # Determine the number of worker processes (limit to 4 to prevent MemoryError/virtual memory exhaustion)
    max_workers = min(4, max(1, os.cpu_count() - 2))
    print(f"Starting ProcessPoolExecutor with {max_workers} workers...")
    
    df_records = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_subject, task): task for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing subjects"):
            task = futures[future]
            try:
                records = future.result()
                df_records.extend(records)
            except Exception as e:
                print(f"\n❌ Worker execution failed for subject {task[0]}: {e}")
                
    # Save audit files
    if len(df_records) == 0:
        print("❌ Error: No features extracted successfully!")
        sys.exit(1)
        
    df_audit = pd.DataFrame(df_records)
    df_audit.groupby(['split', 'class', 'augmentation']).size().reset_index(name='count').to_csv('augmentation_audit.csv', index=False)
    
    summary_records = []
    for s in ['train', 'val', 'test']:
        s_df = df_audit[df_audit['split'] == s]
        if s in ['val', 'test']:
            s_df_clean = s_df[s_df['augmentation'] == 'Clean']
        else:
            s_df_clean = s_df
            
        summary_records.append({
            'Split': s.upper(),
            'Total Subjects': s_df_clean['subject_id'].nunique() if len(s_df_clean) > 0 else 0,
            'TB Subjects': s_df_clean[s_df_clean['class'] == 'TB']['subject_id'].nunique() if len(s_df_clean) > 0 else 0,
            'Healthy Subjects': s_df_clean[s_df_clean['class'] == 'Healthy']['subject_id'].nunique() if len(s_df_clean) > 0 else 0,
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
    
    print("\n--- Preprocessing and Feature Extraction Done! ---")
    print(pd.read_csv('dataset_split_summary.csv'))

if __name__ == '__main__':
    main()
