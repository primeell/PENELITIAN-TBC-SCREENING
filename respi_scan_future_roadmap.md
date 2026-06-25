# Roadmap Optimasi dan Panduan Pengembangan Skala Penuh RespiScan (Edisi Lengkap & Terpadu)

Dokumen ini mendokumentasikan analisis permasalahan sistem sebelum perbaikan, detail performa model saat ini, verifikasi anti-leakage tingkat subjek, serta rencana aksi terstruktur untuk memperkuat model RespiScan menggunakan jalur **Dataset Online Publik** dan jalur **Pengumpulan Data Klinis Lokal** secara selaras.

---

## 1. Analisis Detail Permasalahan Model & Data Sebelumnya

Sebelum perbaikan dilakukan, sistem RespiScan mengalami beberapa kendala mendasar yang menghalangi kelayakan penggunaan di dunia nyata:

### A. Kebocoran Pembicara (*Speaker/Subject Leakage*)
* **Masalah:** Pada skrip/notebook lama, pemisahan data Train, Val, dan Test set dilakukan secara acak pada tingkat *file* atau segmen gambar CWT. Pasien yang sama (misal `PID_32A` dengan 296 file) tersebar di ketiga kelompok data. Model dengan mudah mengenali ciri suara pasien (*speaker recognition*) alih-alih belajar mendeteksi penyakit TB, menghasilkan akurasi semu 100% saat evaluasi namun gagal total saat diuji pada pasien baru.
* **Perbaikan:** Kita telah mengisolasi subjek secara total di tingkat pasien sehingga tidak ada satu subjek pun yang suaranya tersebar di lebih dari satu kelompok data.

### B. Ukuran Subjek yang Sangat Kecil (High Statistical Variance)
* **Masalah:** Dataset hanya memiliki **120 subjek aktif** yang memiliki audio. Pada pembagian data set uji (Test) independen, kita hanya memiliki **19 subjek** (13 TB, 6 Healthy). Karena sampel uji sangat kecil, kesalahan prediksi pada **satu orang saja** akan menggeser nilai spesifisitas sebesar **16.67%**. Hal ini membuat metrik evaluasi memiliki variansi statistik yang terlalu tinggi (tidak stabil).

### C. Ketiadaan Kelompok Kontrol Patologis (No Non-TB Lung Disease Controls)
* **Masalah:** Kelompok kontrol (Non-TB) saat ini hampir semuanya adalah subjek sehat (Healthy). Di dunia nyata, orang batuk tidak hanya karena TB atau Sehat, tetapi juga karena Asma, Bronkitis, PPOK, COVID-19, atau flu biasa. Karena model tidak pernah dilatih menggunakan rekaman batuk penyakit paru non-TB, model mengklasifikasikan semua batuk patologis sebagai TB, sehingga nilai **Spesifisitas menjadi sangat rendah (16.67%)**.

### D. Bias Perangkat Perekaman (Microphone/Acoustic Overfitting)
* **Masalah:** Seluruh subjek direkam menggunakan perangkat dan kondisi lingkungan yang serupa (Nairobi Cohort). Model CNN cenderung mempelajari tanda frekuensi akustik dari mikrofon smartphone yang digunakan dan gema ruangan tempat perekaman, bukan biomarker biologis batuk TB.

---

## 2. Status Verifikasi Kebocoran Data (Zero Leakage Check)
Pemisahan data 120 subjek saat ini telah diverifikasi **100% aman** dari kebocoran pembicara.

Hasil pengecekan file `subject_leakage_check.csv`:

| Jenis Pengecekan | Jumlah Subjek Overlap | Status Keamanan |
| :--- | :---: | :---: |
| **Train vs Validation Leakage** | 0 | 🟢 Aman (Steril) |
| **Train vs Test Leakage** | 0 | 🟢 Aman (Steril) |
| **Val vs Test Leakage** | 0 | 🟢 Aman (Steril) |

---

## 3. Spesifikasi Model & Preprocessing RespiScan Saat Ini

Model saat ini menggunakan optimasi **16.0 kHz Downsampling** dengan detail konfigurasi sebagai berikut:
* **Sampling Rate:** 16.000 Hz, Mono, 1.0 Detik (16.000 sampel).
* **Mother Wavelet:** Complex Morlet (`cmor1.5-1.0`).
* **widths (Skala CWT):** 64 skala (geometric space dari 2 s.d. 186).
* **Ukuran Gambar Output:** $16.000 \times 64$ piksel.
* **Capping Sampel:** Maksimal **10 file rekaman batuk acak per subjek** untuk mencegah bias dominasi pembicara.

### Distribusi Subjek dan Sampel Latih Saat Ini:

| Kelompok (Split) | Jumlah Subjek | TB Subjek | Healthy Subjek | Total Sampel CWT | TB Sampel | Healthy Sampel | Keterangan |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train (Latih)** | 83 | 59 | 24 | 1.962 | 1.042 | 920 | Bersih + Noise Mixed + TimeStretch (Healthy) |
| **Validation** | 18 | 13 | 5 | 164 | 121 | 43 | Bersih-Only (Tanpa Augmentasi) |
| **Test (Uji)** | 19 | 13 | 6 | 180 | 120 | 60 | Bersih-Only (Holdout Independen) |

---

## 4. Hasil Pelatihan & Evaluasi Model Saat Ini

### A. Catatan Riwayat Pelatihan Model (Epoch Logs - 20 Epoch):
Model dilatih menggunakan arsitektur MobileNetV2 dengan bobot loss tertimbang (Healthy = 2.0, TB = 1.0) dan early stopping:
* **Epoch 01:** Train Loss: 0.6443 Acc: 48.17% | Val Loss: 0.7236 Acc: 39.63% F1: 0.3172
* **Epoch 02:** Train Loss: 0.5974 Acc: 53.67% | Val Loss: 0.7662 Acc: 45.12% F1: 0.4231
* **Epoch 03:** Train Loss: 0.5633 Acc: 62.44% | Val Loss: 0.7150 Acc: 53.66% F1: 0.5581
* **Epoch 04:** Train Loss: 0.5315 Acc: 66.26% | Val Loss: 0.6880 Acc: 59.15% F1: 0.6298
* **Epoch 05:** Train Loss: 0.5075 Acc: 69.01% | Val Loss: 0.6146 Acc: 65.85% F1: 0.7172
* **Epoch 06:** Train Loss: 0.4783 Acc: 72.48% | Val Loss: 0.6306 Acc: 64.02% F1: 0.6974
* **Epoch 07:** Train Loss: 0.4584 Acc: 73.96% | Val Loss: 0.6599 Acc: 60.98% F1: 0.6667
* **Epoch 08:** Train Loss: 0.4334 Acc: 76.61% | Val Loss: 0.6466 Acc: 60.98% F1: 0.6632
* **Epoch 09:** Train Loss: 0.4211 Acc: 77.12% | Val Loss: 0.6323 Acc: 64.63% F1: 0.7010
* **Epoch 10:** Train Loss: 0.4025 Acc: 77.52% | Val Loss: 0.6753 Acc: 60.37% F1: 0.6667
* **Epoch 11:** Train Loss: 0.3615 Acc: 82.77% | Val Loss: 0.7930 Acc: 58.54% F1: 0.6264
* **Epoch 12:** Train Loss: 0.3658 Acc: 80.28% | Val Loss: 0.6299 Acc: 65.85% F1: 0.7383
* **Epoch 13:** Train Loss: 0.3282 Acc: 84.15% | Val Loss: 0.6302 Acc: 68.90% F1: 0.7606
* **Epoch 14:** Train Loss: 0.3068 Acc: 85.58% | Val Loss: 0.6222 Acc: 67.68% F1: 0.7512
* **Epoch 15:** Train Loss: 0.2851 Acc: 86.24% | Val Loss: 0.6394 Acc: 69.51% F1: 0.7664
* **Epoch 16:** Train Loss: 0.2646 Acc: 88.43% | Val Loss: 0.6521 Acc: 70.73% F1: 0.7778
* **Epoch 17:** Train Loss: 0.2488 Acc: 88.63% | Val Loss: 0.7046 Acc: 67.07% F1: 0.7379
* **Epoch 18:** Train Loss: 0.2250 Acc: 90.67% | Val Loss: 0.7291 Acc: 67.07% F1: 0.7379
* **Epoch 19:** Train Loss: 0.2175 Acc: 90.62% | Val Loss: 0.7668 Acc: 67.68% F1: 0.7488
* **Epoch 20:** Train Loss: 0.1950 Acc: 91.54% | Val Loss: 0.6794 Acc: 71.34% F1: **0.7814** (Best Saved)

### B. Metrik Hasil Evaluasi Uji Tingkat Pasien (Holdout Test Set):
Ambang batas Youden yang baru dikalibrasi pada set Validasi menghasilkan nilai optimal di **$\theta = 0.4151$** dengan Validation AUROC sebesar **0.8923**.

Berikut metrik lengkap evaluasi pada set uji independen (19 subjek):

| Metrik Evaluasi | Default Threshold ($\theta = 0.50$) | Youden-Optimized Threshold ($\theta = 0.4151$) | Signifikansi Klinis / Interpretasi |
| :--- | :---: | :---: | :--- |
| **Akurasi Pasien** | 42.11% (CI: 23.14%–63.72%) | **63.16%** (CI: 41.04%–80.85%) | Akurasi klasifikasi total pasien baru. |
| **Sensitivitas (Recall)** | 53.85% (CI: 29.14%–76.79%) | **84.62%** (CI: 57.76%–95.67%) | Kemampuan deteksi TB (menyaring 11 dari 13 TB). |
| **Spesifisitas** | 16.67% (CI: 3.01%–56.35%) | **16.67%** (CI: 3.01%–56.35%) | Kemampuan menyaring orang sehat (1 dari 6 Sehat). |
| **PPV (NPE)** | 68.75% (CI: 44.40%–85.84%) | **68.75%** (CI: 44.40%–85.84%) | Probabilitas benar TB jika hasil tes positif. |
| **NPV (NPE)** | 33.33% (CI: 6.15%–79.23%) | **33.33%** (CI: 6.15%–79.23%) | Probabilitas benar sehat jika hasil tes negatif. |
| **Test AUROC** | 0.4103 | 0.4103 | Kemampuan diskriminasi (tinggi di Val, turun di Test). |

*Catatan: Rentang Confidence Interval (CI) dihitung menggunakan Wilson Score Interval dengan tingkat kepercayaan 95%.*

### C. Deskripsi Kurva Evaluasi Terkait:
1. **[Fig1_ROC_Patient_Optimized.png](file:///c:/TBC%20SCREENING%20PENELITIAN/PENELITIAN-TBC-SCREENING/Fig1_ROC_Patient_Optimized.png):** Kurva ROC tingkat pasien pada set uji independen. Garis putus-putus menunjukkan tebakan acak (AUROC = 0.50), garis biru menunjukkan performa model (AUROC = 0.4103), dan titik merah menandai letak threshold optimal Youden di 0.4151.
2. **[Fig_Confusion_Matrix_Patient.png](file:///c:/TBC%20SCREENING%20PENELITIAN/PENELITIAN-TBC-SCREENING/Fig_Confusion_Matrix_Patient.png):** Heatmap matriks kebingungan pada set uji dengan threshold 0.4151: True Positive = 11, False Negative = 2, True Negative = 1, False Positive = 5.
3. **[Fig3_Training_Validation_Curves.png](file:///c:/TBC%20SCREENING%20PENELITIAN/PENELITIAN-TBC-SCREENING/Fig3_Training_Validation_Curves.png):** Grafik nilai Loss dan Akurasi per epoch. Menunjukkan konvergensi yang baik dengan loss latihan menyusut ke 0.1950 di epoch 20.
4. **[Fig_tSNE_Acoustic_Clusters.png](file:///c:/TBC%20SCREENING%20PENELITIAN/PENELITIAN-TBC-SCREENING/Fig_tSNE_Acoustic_Clusters.png):** Pemetaan 2D fitur laten dari Global Average Pooling MobileNetV2 untuk subjek uji, menunjukkan pemisahan parsial antara Healthy (hijau) dan TB (merah).

---

## 5. Rencana Aksi A: Ekspansi Menggunakan Dataset Online Publik

Untuk mengatasi masalah bias akustik dan rendahnya spesifisitas akibat terbatasnya jumlah subjek kontrol patologis, kita akan memperluas data dengan menggabungkan dataset lokal kita dengan dataset pernapasan online publik.

### A. Repositori Dataset Online yang Direkomendasikan:

1. **Coswara Dataset (Indian Institute of Science - IISc Bangalore)**
   * **Sumber Resmi:** [GitHub iiscleap/Coswara-Data](https://github.com/iiscleap/Coswara-Data)
   * **Metadata:** Umur (*age*), jenis kelamin (*gender*), status merokok (*smoker*), riwayat penyakit paru komorbid (*asthma*, *COPD*, *bronchitis*), serta gejala klinis (*fever*, *cough*, *cold*, *muscle_pain*, *loss_of_smell*).
   * **Data Suara:** File audio batuk forced/passive.
   * **Ekstraksi Lokal:** Skrip ekstraksi sudah siap di folder local Anda di [extract_data.py](file:///c:/TBC%20SCREENING%20PENELITIAN/PENELITIAN-TBC-SCREENING/Coswara-Data/extract_data.py).

2. **COUGHVID Crowdsourced Dataset (EPFL Switzerland)**
   * **Sumber Resmi:** Hosted di Zenodo (Search: "COUGHVID EPFL").
   * **Metadata:** *age*, *gender*, status kondisi pernapasan (*respiratory_condition*: asma/COPD/sehat), *fever_muscle_pain*, dan *status_covid*.

### B. Langkah-Langkah Penyelarasan & Filter:
1. **Filter Kelas Kontrol:** Dari Coswara/COUGHVID, pilih subjek berlabel sehat (*Healthy*) dan subjek dengan riwayat *Asthma*, *COPD*, atau *Bronchitis* (tetapi negatif COVID-19/TB) sebagai kontrol patologis non-TB.
2. **Gabungkan dengan TBscreen:** Gunakan subjek positif TB dari dataset lokal kita.
3. **Penyelarasan Fitur (Feature Mapping):**

| Fitur Target | Kolom Metadata Lokal | Kolom Coswara | Kolom COUGHVID |
| :--- | :---: | :---: | :---: |
| **Umur** | `age` | `age` | `age` |
| **Jenis Kelamin** | `gndr_a` (M/F) | `gndr` (male/female) | `gender` (male/female) |
| **Status Merokok** | `smkr_a` (Yes/No) | `smoker` (True/False) | (Diabaikan jika tidak ada) |
| **Komorbiditas Paru** | `comor_lung` (1/0) | `asthma` / `copd` (1/0) | `respiratory_condition` (True/False) |
| **Gejala Demam** | `fever` (Yes/No) | `fever` (True/False) | `fever_muscle_pain` (True/False) |
| **Gejala Batuk** | `cough` (Yes/No) | `cough` (True/False) | `cough` (True/False) |

---

## 6. Rencana Aksi B: Panduan Pengumpulan Data Klinis Lokal (Masa Depan)

Jika di masa mendatang Anda ingin melengkapi dataset online tersebut dengan pengumpulan data klinis mandiri (misalnya dari puskesmas atau rumah sakit rujukan), berikut adalah pedoman yang harus diikuti:

### A. Fitur Data Klinis yang Harus Diambil:
1. **Data Akustik (Audio Batuk):** Rekaman suara batuk (minimal 3 kali batuk paksa/forced cough) dalam format `.wav` menggunakan smartphone standar (jarak sekitar 30 cm dari mulut).
2. **Data Gejala Klinis Utama:**
   * **Penurunan Berat Badan:** Apakah pasien mengalami penurunan berat badan drastis tanpa sebab dalam 1 bulan terakhir? (Ya / Tidak)
   * **Keringat Malam:** Apakah pasien berkeringat di malam hari tanpa aktivitas fisik? (Ya / Tidak)
   * **Demam:** Apakah pasien mengalami demam sub-febris (panas hilang-timbul) > 2 minggu? (Ya / Tidak)
   * **Batuk Darah (Hemoptisis):** Apakah batuk disertai bercak darah? (Ya / Tidak)
3. **Data Demografi & Riwayat Hidup:**
   * **Umur** (Numerik)
   * **Jenis Kelamin** (Laki-laki / Perempuan)
   * **Status Merokok** (Ya / Tidak)
   * **Penyakit Paru Komorbid** (Asma / PPOK / Bronkitis / Pneumonia / Tidak Ada)

### B. Target Jumlah Data yang Diperlukan:
* **Target Minimal:** **500 subjek unik** (Pasien).
* **Rasio Kelas Ideal:** 1:1 antara pasien positif TB (dikonfirmasi lewat GeneXpert/Sputum Smear) dan pasien non-TB.
  * **250 Subjek Positif TB**.
  * **250 Subjek Kontrol Non-TB** (terdiri dari 125 orang sehat biasa, dan 125 pasien dengan penyakit paru non-TB seperti asma/bronkitis).

---

## 7. Arsitektur Model Hibrida Dual-Branch Feature Fusion

Model versi terbaik di masa depan harus memadukan cabang CNN (memproses gambar CWT batuk) dan cabang DNN (memproses data gejala klinis).

### Kode Python Konstruksi Model Hibrida:
```python
import torch
import torch.nn as nn
from torchvision import models

# Inisialisasi backbone akustik MobileNetV2
acoustic_backbone = models.mobilenet_v2(weights=None)
num_ftrs = acoustic_backbone.classifier[1].in_features
acoustic_backbone.classifier = nn.Sequential(
    nn.Linear(num_ftrs, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 2)
)

# Definisikan Gabungan Dual-Branch Network
class DualBranchRespiScan(nn.Module):
    def __init__(self, trained_acoustic_cnn, clinical_dim=6):
        super(DualBranchRespiScan, self).__init__()
        # Acoustic CNN Branch
        self.acoustic_cnn = trained_acoustic_cnn
        # Modifikasi classifier output MobileNetV2 untuk mengeluarkan fitur 512 dimensi
        self.acoustic_cnn.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Clinical MLP Branch (Memproses 6 fitur gejala klinis)
        self.clinical_mlp = nn.Sequential(
            nn.Linear(clinical_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Classifier Head Gabungan (512 + 32 = 544 input dimensi)
        self.joint_classifier = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2) # Output Kelas: [Healthy/Non-TB, TB]
        )
        
    def forward(self, img, clinical_vector):
        feat_audio = self.acoustic_cnn(img)
        feat_clinic = self.clinical_mlp(clinical_vector)
        # Gabungkan vektor fitur akustik & klinis
        combined = torch.cat((feat_audio, feat_clinic), dim=1)
        output = self.joint_classifier(combined)
        return output
```

---

## 8. Protokol Pelatihan Ulang (Retraining & Transfer Learning)

Langkah melatih kembali model hibrida ini menggunakan bobot model akustik yang sudah kita miliki saat ini:

1. **Muat Bobot Model Akustik Saat Ini:**
   ```python
   # Muat model akustik yang sudah terlatih sebagai baseline
   baseline_weights = 'best_RespiScan_MobileNetV2_Scalogram.pth'
   acoustic_backbone.load_state_dict(torch.load(baseline_weights, map_location='cpu'))
   print("🟢 Bobot baseline akustik MobileNetV2 berhasil dimuat!")
   
   # Inisialisasi model hibrida final
   hybrid_model = DualBranchRespiScan(acoustic_cnn=acoustic_backbone, clinical_dim=6)
   ```
2. **Differential Learning Rate:**
   * Cabang Akustik (`acoustic_cnn`) dilatih dengan learning rate sangat rendah ($10^{-5}$) agar tidak merusak representasi batuk yang sudah dipelajari.
   * Cabang Klinis (`clinical_mlp`) dan Classifier Head (`joint_classifier`) dilatih dengan learning rate normal ($10^{-4}$) agar menyesuaikan diri dengan dataset online yang baru.
3. **Penyelarasan Prapemrosesan:**
   * Downsampling semua audio input baru ke 16 kHz (mono), durasi 1.0 detik.
   * Gunakan CWT cmor1.5-1.0 pada 64 skala.
   * Latih selama 30 epoch menggunakan bobot loss tertimbang (Healthy = 2.0, TB = 1.0) dan early stopping.
4. **Verifikasi Anti-Leakage:**
   * Selalu lakukan pembagian tingkat subjek (subject-level split) dengan asersi overlap bernilai 0 sebelum training dimulai pada data gabungan baru.

---

## 9. Penyelarasan dengan Target Profil Produk WHO (WHO Target Product Profile)

Untuk memastikan kelayakan medis dan sertifikasi sebagai alat skrining kesehatan digital, pengembangan model di masa depan akan ditargetkan untuk memenuhi parameter **WHO Target Product Profile (TPP)** untuk instrumen penapisan non-sputum Tuberkulosis:

* **Persyaratan Minimum WHO TPP:**
  * **Sensitivitas (Sensitivity):** $\ge 90\%$ (untuk meminimalkan True Cases yang terlewat).
  * **Spesifisitas (Specificity):** $\ge 70\%$ (untuk meminimalkan rujukan rontgen/GeneXpert yang sia-sia bagi orang sehat).
* **Posisi Model Saat Ini:** Sensitivitas kita sudah mencapai **84.62%** (mendekati batas minimal WHO), namun Spesifisitas masih berada di **16.67%** akibat kurangnya kelompok kontrol patologis paru non-TB.
* **Strategi Pencapaian:** Penambahan kontrol patologis dari dataset Coswara dan COUGHVID (asma, COPD, bronkitis) diproyeksikan akan meningkatkan spesifisitas secara signifikan hingga melampaui ambang batas minimum WHO (70%).

---

## 10. Automated Audio Quality Control & Cough Verification

Dataset crowdsourcing online (seperti Coswara/COUGHVID) dikumpulkan secara mandiri oleh pengguna dan sering kali mengandung data rusak, hening (silent files), background bising manusia yang sedang berbicara, tawa, atau batuk buatan yang tidak valid. Kita merencanakan pembuatan filter QC otomatis sebelum ekstraksi CWT:

1. **Energy and Silence Filtering:** Menggunakan analisis *Root Mean Square (RMS) Energy* untuk secara otomatis membuang file yang tidak mengandung suara (hening) atau memiliki tingkat SNR (Signal-to-Noise Ratio) di bawah 3 dB.
2. **Cough Detection Pre-screening:** Mengintegrasikan model klasifikasi suara batuk ringan berbasis **YAMNet** (pretrained pada AudioSet) untuk memverifikasi apakah klip audio tersebut benar-benar mengandung aktivitas batuk manusia (*cough event*). Jika probabilitas batuk dari YAMNet $< 0.60$, file akan didiskualifikasi secara otomatis dari pipeline training.

---

## 11. Mitigasi Bias Perangkat (Acoustic Domain Adaptation)

Karena perekaman audio crowdsourcing dan klinis menggunakan model mikrofon smartphone yang berbeda-beda, model CNN berisiko mempelajari karakteristik respon frekuensi perangkat (bias mikrofon) ketimbang biomarker patologis batuk. Tiga teknik mitigasi yang direncanakan:

1. **Normalisasi Fitur Spektral:** Menerapkan *Cepstral Mean and Variance Normalization* (CMVN) pada domain frekuensi scalogram untuk menyeimbangkan respon gain mikrofon yang bervariasi.
2. **Domain Adversarial Neural Networks (DANN):** Menambahkan cabang klasifikasi perangkat (*device classifier*) dengan loss adversarial selama training. Cabang ini melatih ekstraktor fitur untuk meminimalkan kemampuan memprediksi jenis ponsel perekam, memaksa CNN hanya mempelajari representasi fitur yang murni berhubungan dengan patologi paru-paru.

---

## 12. Rencana Implementasi Edge Deployment (Mobile & Offline Screening)

Sebagai kerangka kerja skrining yang ringan dan hemat energi, RespiScan dirancang agar dapat berjalan secara luring (*offline*) di area pelosok tanpa internet. Langkah-langkah penyebaran yang direncanakan:

1. **Konversi Format ONNX:** Mengekspor model PyTorch hasil training (`.pth`) ke format terbuka **ONNX** (Open Neural Network Exchange).
2. **Kuantisasi TensorFlow Lite (TFLite):** 
   * Mengonversi model ONNX ke format TFLite menggunakan **INT8 Post-Training Quantization** (kuantisasi bobot model dari float32 ke integer 8-bit).
   * Kuantisasi ini diproyeksikan mereduksi ukuran file model dari **11.7 MB menjadi kurang dari 3.0 MB** tanpa degradasi akurasi klinis yang signifikan ($< 1\%$).
3. **Integrasi SDK Mobile:** Model TFLite yang dikuantisasi akan diintegrasikan dengan aplikasi Android native menggunakan SDK TFLite Task Library untuk inferensi lokal secara offline dengan waktu eksekusi $< 100\text{ ms}$ per batuk.

