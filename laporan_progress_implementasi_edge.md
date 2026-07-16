# Laporan Progres Implementasi Tepi (Edge Deployment) - RespiScan

Dokumen ini mendokumentasikan pencapaian terkini, status pengujian, dan arsitektur teknis dari pemindahan model **RespiScan** dari lingkungan pelatihan (*server/Jupyter*) ke perangkat keras komputasi tepi (*edge hardware*), yaitu **SBC (Raspberry Pi/Radxa Rock)** dan **Perangkat Android Native**.

---

## 🗺️ Peta Arsitektur Implementasi Tepi

```
  [Suara Batuk (Mic)] 
         │
         ▼ (Standardisasi Audio)
  [PCM 16kHz, Mono, 1.0s] 
         │
         ▼ (Ekstraksi CWT In-Memory)
  [224x224 Jet Scalogram Image] 
         │
         ▼ (Normalisasi ImageNet)
  [Tensor: 1, 3, 224, 224]
         │
         ▼ (ONNX Runtime Engine)
  [Inference Model Luring] 
         │
         ▼ (Batas Keputusan Youden: 0.3595)
  [Hasil Triage: Low/High Risk]
```

---

## 📈 Status Pencapaian per Komponen

### 1. ⚙️ Ekspor Model ke ONNX
* **Status:** **100% SELESAI**
* **Deskripsi:** Model PyTorch `best_RespiScan_FocalLoss.pth` berhasil dikonversi ke format ONNX (`opset_version=12`) dengan dimensi input dinamis (*dynamic batching*).
* **Hasil Berkas:**
  * [best_RespiScan_FocalLoss.onnx](file:///c:/TBC%20SCREENING%20PENELITIAN/PENELITIAN-TBC-SCREENING/best_RespiScan_FocalLoss.onnx) (273 KB - memisahkan bobot).
  * [best_RespiScan_FocalLoss_embedded.onnx](file:///c:/TBC%20SCREENING%20PENELITIAN/PENELITIAN-TBC-SCREENING/best_RespiScan_FocalLoss_embedded.onnx) (11.7 MB - bobot tertanam penuh). Berkas *embedded* ini disalin ke direktori aset Android dengan nama `best_RespiScan_FocalLoss.onnx` untuk memudahkan pembacaan satu file.

---

### 2. 🍓 Klien Raspberry Pi & SBC (`pi_run.py`)
* **Status:** **100% SELESAI & TERVERIFIKASI**
* **Deskripsi:** Skrip klien [pi_run.py](file:///c:/TBC%20SCREENING%20PENELITIAN/PENELITIAN-TBC-SCREENING/pi_run.py) telah diuji coba dan terbukti mampu melakukan pemrosesan *in-memory* tanpa menulis gambar perantara ke penyimpanan lokal, yang krusial untuk menjaga umur pakai kartu SD pada Raspberry Pi.
* **Alur Eksekusi:**
  1. Memuat audio `.wav` via `librosa`, melakukan *downsampling* ke 16 kHz, dan melakukan *capping* durasi tepat 1.0 detik.
  2. Mengekstrak CWT `cmor1.5-1.0` secara matematis dengan `pywt.cwt` menggunakan 64 skala geometris.
  3. Menerapkan interpolasi *bilinear* numpy untuk mengubah ukuran ke $224 \times 224$ piksel.
  4. Menerapkan peta warna *Jet colormap* dan normalisasi ImageNet secara programatis.
  5. Melakukan inferensi luring menggunakan `onnxruntime` CPU session.
* **Hasil Pengujian Riil:**
  * **Input Audio Uji:** `dataset_tbc\PID_108A_0_pixel.wav`
  * **TB Suspect Probability:** **41.56%** (di atas threshold Youden 0.3595).
  * **Klasifikasi Akhir:** **🚨 HIGH RISK (Suspected TB)**.

---

### 3. 📱 Aplikasi Android Native (`RespiScanMobile`)
* **Status:** **90% SELESAI (Siap Kompilasi/Build)**
* **Deskripsi:** Kerangka kerja aplikasi Android dibangun secara *native* menggunakan **Kotlin** dan **Jetpack Compose** untuk antarmuka modern yang responsif.
* **Detail Implementasi Kode:**
  * **[AudioCaptureManager.kt](file:///c:/TBC%20SCREENING%20PENELITIAN/PENELITIAN-TBC-SCREENING/RespiScanMobile/app/src/main/java/com/example/respiscanmobile/AudioCaptureManager.kt):** Mengatur perekaman mic menggunakan `AudioRecord` pada tingkat sampel 16 kHz mono 16-bit PCM, lalu dinormalisasi secara instan ke FloatArray range $[-1.0f, 1.0f]$.
  * **[CwtProcessor.kt](file:///c:/TBC%20SCREENING%20PENELITIAN/PENELITIAN-TBC-SCREENING/RespiScanMobile/app/src/main/java/com/example/respiscanmobile/CwtProcessor.kt):** 
    * Memuat filter CWT yang telah diprekomputasi dari berkas biner [wavelet_filters.bin](file:///c:/TBC%20SCREENING%20PENELITIAN/PENELITIAN-TBC-SCREENING/RespiScanMobile/app/src/main/assets/wavelet_filters.bin) (340 KB) di folder aset Android. Teknik ini memotong komputasi fungsi gelombang Morlet kompleks secara dinamis di ponsel yang lambat.
    * Melakukan konvolusi CWT in-place, normalisasi min-max, interpolasi bilinear vertikal dari 64 skala ke 224 baris, pemetaan jet colormap, normalisasi ImageNet, dan penataan saluran *Channel-First* (NCHW) menjadi satu dimensi linier datar siap input.
  * **[OnnxClassifier.kt](file:///c:/TBC%20SCREENING%20PENELITIAN/PENELITIAN-TBC-SCREENING/RespiScanMobile/app/src/main/java/com/example/respiscanmobile/OnnxClassifier.kt):** Memuat model `best_RespiScan_FocalLoss.onnx` dari aset menggunakan pustaka `ai.onnxruntime:onnxruntime-android:1.16.0` untuk mendapatkan hasil probabilitas softmax.
  * **[MainScreen.kt](file:///c:/TBC%20SCREENING%20PENELITIAN/PENELITIAN-TBC-SCREENING/RespiScanMobile/app/src/main/java/com/example/respiscanmobile/ui/main/MainScreen.kt):** Menyediakan UI visual interaktif untuk memulai perekaman suara batuk pasien, memutar audio, melihat visualisasi hasil CWT/waveform, serta menampilkan status rujukan medis (High Risk/Low Risk).

---

## 📋 Ringkasan File Terkait Edge Deployment

| Nama File | Path Relatif | Peran / Deskripsi | Status |
| :--- | :--- | :--- | :--- |
| **`export_onnx.py`** | [export_onnx.py](file:///c:/TBC%20SCREENING%20PENELITIAN/PENELITIAN-TBC-SCREENING/export_onnx.py) | Pengekspor model PyTorch ke format ONNX + verifikasi nilai toleransi. | 🟢 Selesai |
| **`pi_run.py`** | [pi_run.py](file:///c:/TBC%20SCREENING%20PENELITIAN/PENELITIAN-TBC-SCREENING/pi_run.py) | Klien inferensi luring simulasi untuk Raspberry Pi. | 🟢 Selesai |
| **`generate_wavelet_filters.py`** | [generate_wavelet_filters.py](file:///c:/TBC%20SCREENING%20PENELITIAN/PENELITIAN-TBC-SCREENING/generate_wavelet_filters.py) | Skrip pembuat data filter koefisien CWT dalam bentuk berkas biner `.bin`. | 🟢 Selesai |
| **`wavelet_filters.bin`** | [wavelet_filters.bin](file:///c:/TBC%20SCREENING%20PENELITIAN/PENELITIAN-TBC-SCREENING/RespiScanMobile/app/src/main/assets/wavelet_filters.bin) | Koefisien CWT cmor1.5-1.0 siap pakai untuk optimasi matematika Android. | 🟢 Selesai |
| **`OnnxClassifier.kt`** | [OnnxClassifier.kt](file:///c:/TBC%20SCREENING%20PENELITIAN/PENELITIAN-TBC-SCREENING/RespiScanMobile/app/src/main/java/com/example/respiscanmobile/OnnxClassifier.kt) | Logika load dan inferensi model ONNX di aplikasi Android. | 🟢 Selesai |
| **`CwtProcessor.kt`** | [CwtProcessor.kt](file:///c:/TBC%20SCREENING%20PENELITIAN/PENELITIAN-TBC-SCREENING/RespiScanMobile/app/src/main/java/com/example/respiscanmobile/CwtProcessor.kt) | Logika komputasi CWT dan normalisasi tensor di Android. | 🟢 Selesai |
| **`AudioCaptureManager.kt`** | [AudioCaptureManager.kt](file:///c:/TBC%20SCREENING%20PENELITIAN/PENELITIAN-TBC-SCREENING/RespiScanMobile/app/src/main/java/com/example/respiscanmobile/AudioCaptureManager.kt) | Logika perekaman mic 16kHz mono PCM di Android. | 🟢 Selesai |

---

## 🚀 Rencana Langkah Selanjutnya

1. **Build APK Android:** Melakukan kompilasi proyek `RespiScanMobile` untuk menghasilkan APK debug, lalu memasangnya pada ponsel fisik Android untuk menguji proses perekaman mic dan inferensi secara nyata.
2. **Pengukuran Latensi (Profiling):** Mengukur waktu pemrosesan (dalam milidetik) untuk CwtProcessor dan OnnxClassifier di Raspberry Pi dan ponsel Android guna memastikan respons inferensi selesai di bawah **200 ms**.
3. **Optimasi Kuantisasi TFLite (Opsional):** Jika memori ponsel Android sangat terbatas, lakukan kuantisasi INT8 pasca-pelatihan untuk mereduksi ukuran model ONNX dari 11.7 MB menjadi sekitar 3.0 MB.
