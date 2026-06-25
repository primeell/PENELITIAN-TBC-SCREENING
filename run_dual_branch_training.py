import os
import sys
import re
import gc
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_curve, auc
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configure stdout to use UTF-8 to prevent Windows console encoding errors
sys.stdout.reconfigure(encoding='utf-8')

# Fix for Windows PyTorch OpenMP crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuration paths
DATA_DIR = 'RespiScan_Train_Ready' 
MFCC_DIR = 'RespiScan_MFCC_Only'
PATH_META = 'Metadata.csv'

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Device active: {DEVICE}")

# -------------------------------------------------------------
# 1. CUSTOM DATASET LOADER
# -------------------------------------------------------------
class DualBranchDataset(Dataset):
    """Dataset kustom untuk memuat citra Scalogram 2D dan tensor MFCC 1D secara bersamaan."""
    def __init__(self, base_dir, mfcc_dir, split, image_transforms=None):
        self.image_dir = os.path.join(base_dir, split)
        self.mfcc_dir = os.path.join(mfcc_dir, split)
        self.image_transforms = image_transforms
        
        self.samples = []
        # Petakan class folder ke label numerik (Healthy -> 0, TB -> 1)
        self.class_map = {'Healthy': 0, 'TB': 1}
        
        for class_name, label in self.class_map.items():
            class_img_dir = os.path.join(self.image_dir, class_name)
            if not os.path.exists(class_img_dir):
                continue
            for f in os.listdir(class_img_dir):
                if f.endswith('.png'):
                    img_path = os.path.join(class_img_dir, f)
                    # Match CWT image file to MFCC tensor file by replacing .png with _mfcc.pt
                    mfcc_f = f.replace('.png', '_mfcc.pt')
                    mfcc_path = os.path.join(self.mfcc_dir, class_name, mfcc_f)
                    
                    if os.path.exists(mfcc_path):
                        self.samples.append((img_path, mfcc_path, label))
                        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        img_path, mfcc_path, label = self.samples[idx]
        from PIL import Image
        
        # 1. Load Scalogram
        img = Image.open(img_path).convert('RGB')
        if self.image_transforms:
            img = self.image_transforms(img)
            
        # 2. Load MFCC Tensor (.pt)
        mfcc = torch.load(mfcc_path, weights_only=True).squeeze()
        if len(mfcc.shape) > 1:
            mfcc_feat = torch.mean(mfcc, dim=1) # Rata-rata sumbu waktu (13-dim)
        else:
            mfcc_feat = mfcc
            
        return img, mfcc_feat, label

# -------------------------------------------------------------
# 2. DUAL-BRANCH FUSION MODEL ARCHITECTURE
# -------------------------------------------------------------
class DualBranchFusionNet(nn.Module):
    """Arsitektur Model Hybrid Fusion: MobileNetV2 + MLP."""
    def __init__(self):
        super(DualBranchFusionNet, self).__init__()
        
        # Cabang A: Citra Scalogram 2D (MobileNetV2 Feature Extractor)
        self.scalogram_branch = models.mobilenet_v2(weights='DEFAULT')
        num_ftrs = self.scalogram_branch.classifier[1].in_features
        self.scalogram_branch.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Cabang B: Vektor Akustik 1D MFCC (Dense MLP)
        self.mfcc_branch = nn.Sequential(
            nn.Linear(13, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Cabang Utama Fusion (Penggabung Fitur 256 + 32 = 288)
        self.fusion_classifier = nn.Sequential(
            nn.Linear(288, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, img, mfcc):
        feat_img = self.scalogram_branch(img)
        feat_mfcc = self.mfcc_branch(mfcc)
        
        # Concatenate fitur
        feat_fusion = torch.cat((feat_img, feat_mfcc), dim=1)
        out = self.fusion_classifier(feat_fusion)
        return out

# -------------------------------------------------------------
# 3. TRAINING ROUTINE
# -------------------------------------------------------------
def main():
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

    # Inisialisasi Dataset Loader
    print("--- Loading Datasets ---")
    train_dataset = DualBranchDataset(DATA_DIR, MFCC_DIR, 'train', img_transforms['train'])
    val_dataset = DualBranchDataset(DATA_DIR, MFCC_DIR, 'val', img_transforms['val'])
    test_dataset = DualBranchDataset(DATA_DIR, MFCC_DIR, 'test', img_transforms['val'])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = DualBranchFusionNet().to(DEVICE)
    
    # Fine-tuning: Freezing early layers in MobileNetV2
    for param in model.scalogram_branch.features[:14].parameters():
        param.requires_grad = False

    # Optimize different components with different learning rates
    optimizer = optim.Adam([
        {'params': model.scalogram_branch.features[14:].parameters(), 'lr': 1e-5},
        {'params': model.scalogram_branch.classifier.parameters(), 'lr': 1e-4},
        {'params': model.mfcc_branch.parameters(), 'lr': 1e-4},
        {'params': model.fusion_classifier.parameters(), 'lr': 1e-4}
    ])
    
    # Class weights to handle class imbalance (TB: 2.0, Healthy: 1.0)
    weights = torch.tensor([1.0, 2.0]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Loop Training
    num_epochs = 20
    best_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())

    print("\n--- Starting Dual-Branch Model Training ---")
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train': 
                model.train()
                loader = train_loader
                dataset_len = len(train_dataset)
            else: 
                model.eval()
                loader = val_loader
                dataset_len = len(val_dataset)
                
            running_loss = 0.0
            running_corrects = 0
            all_preds, all_labels = [], []
            
            for imgs, mfccs, labels in loader:
                imgs, mfccs, labels = imgs.to(DEVICE), mfccs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(imgs, mfccs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * imgs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            epoch_loss = running_loss / dataset_len
            epoch_acc = running_corrects.double() / dataset_len
            epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)
            
            if phase == 'val':
                scheduler.step(epoch_f1)
                if epoch_f1 > best_f1:
                    best_f1 = epoch_f1
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), 'best_RespiScan_DualBranch_Fusion.pth')
                    print(f"Epoch {epoch+1:02d} | Train Loss: {prev_train_loss:.4f} Acc: {prev_train_acc:.4f} | Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f} ⭐ Model Saved!")
                else:
                    print(f"Epoch {epoch+1:02d} | Train Loss: {prev_train_loss:.4f} Acc: {prev_train_acc:.4f} | Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F1: {epoch_f1:.4f}")
            else:
                prev_train_loss = epoch_loss
                prev_train_acc = epoch_acc
        
    print(f"\n🏆 Best Validation F1-Score: {best_f1:.4f}")
    model.load_state_dict(best_model_wts)

    # -------------------------------------------------------------
    # 4. VAL SET DECISION THRESHOLD TUNING (YOUDEN'S INDEX)
    # -------------------------------------------------------------
    print("\n--- Calibrating Decision Threshold on Validation Set ---")
    model.eval()
    val_probs = []
    val_labels = []
    softmax = nn.Softmax(dim=1)
    
    with torch.no_grad():
        for imgs, mfccs, labels in val_loader:
            imgs, mfccs = imgs.to(DEVICE), mfccs.to(DEVICE)
            outputs = model(imgs, mfccs)
            probs = softmax(outputs)[:, 1] # Probability of TB
            val_probs.extend(probs.cpu().numpy())
            val_labels.extend(labels.numpy())
            
    df_meta = pd.read_csv(PATH_META)
    df_meta.columns = df_meta.columns.str.strip()
    df_meta['subject'] = df_meta['subject'].astype(str).str.strip()
    
    def get_clean_subject_id(subj):
        return re.sub(r'_now_C$', '', subj)
    df_meta['clean_subject'] = df_meta['subject'].apply(get_clean_subject_id)
    
    # Map val probabilities to subjects
    val_file_paths = [path for path, _, _ in val_dataset.samples]
    val_results = []
    for p, prob, lbl in zip(val_file_paths, val_probs, val_labels):
        f_name = os.path.basename(p)
        subj = None
        for s in df_meta['clean_subject'].unique():
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
    
    if opt_threshold <= 0.0 or opt_threshold >= 1.0:
        opt_threshold = 0.15 # Fallback
        
    val_auroc = auc(fpr, tpr)
    print(f"Optimal decision threshold calibrated at: {opt_threshold:.4f}")
    print(f"Validation Patient-Level AUROC: {val_auroc:.4f}")

    # -------------------------------------------------------------
    # 5. INDEPENDENT TEST SET EVALUATION
    # -------------------------------------------------------------
    print("\n--- Evaluating on Independent Test Set ---")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    test_probs = []
    test_labels = []
    
    with torch.no_grad():
        for imgs, mfccs, labels in test_loader:
            imgs, mfccs = imgs.to(DEVICE), mfccs.to(DEVICE)
            outputs = model(imgs, mfccs)
            probs = softmax(outputs)[:, 1]
            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(labels.numpy())
            
    test_file_paths = [path for path, _, _ in test_dataset.samples]
    test_results = []
    for p, prob, lbl in zip(test_file_paths, test_probs, test_labels):
        f_name = os.path.basename(p)
        subj = None
        for s in df_meta['clean_subject'].unique():
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
    
    # Helper to print metrics
    def print_metrics(y_true, y_probs, thresh, label_str):
        y_pred = (y_probs >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        
        acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        print(f"{label_str} Results:")
        print(f" Accuracy: {acc*100:.2f}%")
        print(f" Sensitivity: {sens*100:.2f}%")
        print(f" Specificity: {spec*100:.2f}%")
        print(f" PPV: {ppv*100:.2f}%")
        print(f" NPV: {npv*100:.2f}%")
        
    print_metrics(y_test_true, y_test_probs, 0.50, "Default Threshold (0.50)")
    print()
    print_metrics(y_test_true, y_test_probs, opt_threshold, f"Youden-Optimized Threshold ({opt_threshold:.4f})")
    print(f" Test AUROC: {test_auroc:.4f}")
    print("\n🎉 Dual-Branch Feature Fusion Pipeline completed successfully!")

if __name__ == '__main__':
    main()
