# Panduan Metodologi & Strategi RespiScan
*Dokumen ini dirancang sebagai panduan pembelajaran komprehensif bagi Anda untuk memahami, mempelajari, dan mempertanggungjawabkan setiap keputusan ilmiah, preprocessing, model, dan evaluasi yang digunakan dalam penelitian **RespiScan**.*

---

## 🗺️ Peta Alur Sistem RespiScan

Secara garis besar, data batuk diolah melalui alur sistematis berikut:

```
[Raw Audio (645 Subjek)] 
       │ 
       ▼ (Quality Control) 
[593 Subjek Terpilih] (52 subjek bermasalah dieliminasi)
       │ 
       ▼ (Audio Standardization) 
[Standardized 1s Mono, 44.1 kHz] 
       │ 
       ▼ (Subject-Level Splitting) 
┌──────┴────────────────────────┬────────────────────────┐
▼ (Train Split: 449 Subjek)     ▼ (Val Split: 85 Subjek)  ▼ (Test Split: 59 Subjek)
│                               │                         │
├─► Noise Augmentation (10dB)   └─► Evaluasi Clean-Only   └─► Evaluasi Clean-Only
├─► CWT Scalogram (224x224)         (Ambang Youden: 0.1380)   (Holdout Test Metrics)
▼
[MobileNetV2 Training] (Weighted Loss & Differential LR)
```

---

## 1. Dataset: Evolusi, Penyaringan, dan Justifikasi Ukuran

### A. Mengapa Ukuran Cohort Berubah dari Draf Awal?
Dalam draf awal (pilot study), tertulis dataset terdiri dari **136 subjek** menghasilkan **4.258 sampel**. Sedangkan pada versi final, dataset menggunakan **593 subjek** menghasilkan **2.645 sampel**. Perubahan ini didasari reformasi metodologi yang krusial:

*   **Pilot Dataset (Versi Lama):**
    *   *Keterbatasan:* Hanya menggunakan 136 subjek dari database. Agar data terlihat banyak untuk CNN, peneliti melakukan **Overlapping Segment Windowing** (memotong satu rekaman batuk yang panjang menjadi potongan 1 detik secara bertumpang tindih, menghasilkan rata-rata ~31 segmen per subjek).
    *   *Risiko Metodologis:* Model CNN sangat mudah mengenali karakteristik suara unik pembicara (*speaker recognition*). Karena segmen-segmen tersebut bertumpang tindih, model mengalami *overfitting* berat—menghafal sinyal suara subjek yang sama alih-alih mempelajari patologi TBC.
*   **Final Dataset (Versi Baru / Saat Ini):**
    *   *Solusi:* Kami memperluas cohort hingga **593 subjek** (meningkatkan keanekaragaman suara pembicara sebanyak **4.3 kali lipat**).
    *   *Standardisasi Non-Overlapping:* Kami melarang penumpukan segmen bertumpang tindih. Setiap rekaman batuk hanya menghasilkan **1 file scalogram bersih (non-overlapping)**. Ini menghasilkan total **2.645 sampel**. Metodologi ini jauh lebih sahih karena memaksa model mempelajari pola patofisiologi batuk secara umum.

### B. Alur Penyaringan Dataset (Evolution & Curation)
Reviewer jurnal sangat kritis terhadap asal-usul data. Tabel evolusi dataset berikut merangkum proses penyaringan data dari database mentah:

| Tahapan Kurasi | Jumlah Subjek (Pasien) | Jumlah File (Sampel) | Alasan & Kriteria Penyaringan |
| :--- | :---: | :---: | :--- |
| **1. Database Awal (Raw Cohort)** | 645 | — | Database batuk klinis awal dari TBscreen. |
| **2. Tahap Quality Control** | 52 (Dieliminasi) | — | Subjek dikeluarkan karena file audio rusak, rekaman terputus, atau hanya berisi keheningan (silent channels). |
| **3. Dataset Terstandarisasi** | **593** | **2.645** | Subjek yang memiliki rekaman batuk valid. Semua audio dikonversi ke 44.1 kHz, mono, dan dipotong tepat 1.0 detik. |
| **4. Train Partition** | 449 | 2.225 | Digunakan untuk melatih model. Data dicampur dengan bising lingkungan (noisy-mixed) pada SNR 10 dB. |
| **5. Val Partition** | 85 | 254 | Bersih (clean-only). Digunakan untuk menentukan ambang batas keputusan optimal (*Youden's Index*). |
| **6. Test Partition** | 59 | 166 | Bersih (clean-only). Menjadi data uji independen yang benar-benar baru (*unseen holdout*) untuk evaluasi performa akhir. |

---

## 2. Standardisasi Sinyal Audio (Acoustic Preprocessing)

Mengapa kita harus mengubah format audio asli menjadi format khusus (Mono, 44.1 kHz, 1 detik)?

1.  **Mengubah ke Mono (1-Channel):**
    Audio rekaman sering kali memiliki format stereo (2 saluran: kiri dan kanan). Namun, informasi batuk di saluran kiri dan kanan pada dasarnya identik. Mengonversi ke mono dengan merata-ratakan saluran suara (`torch.mean`) mereduksi beban komputasi jaringan saraf tiruan hingga 50% tanpa kehilangan informasi akustik.
2.  **Resampling ke 44.1 kHz (Teorema Nyquist-Shannon):**
    Telinga manusia dapat mendengar frekuensi audio dari 20 Hz hingga 20.000 Hz (20 kHz). Berdasarkan **Teorema Nyquist-Shannon**, untuk mendokumentasikan sinyal berfrekuensi $f$ secara digital tanpa distorsi (*aliasing*), frekuensi perekaman (*sampling rate*) minimal harus **2 kali lipat frekuensi tertinggi** ($2 \times 20\text{ kHz} = 40\text{ kHz}$). Standar industri **44.100 Hz (44.1 kHz)** dipilih untuk memberikan margin keamanan agar detail suara ledakan udara batuk berfrekuensi tinggi tetap terekam utuh.
3.  **Durasi Tepat 1.0 Detik (Cropping / Padding):**
    CNN membutuhkan ukuran input gambar yang seragam. Durasi batuk yang bervariasi (misal ada yang 0.8 detik dan ada yang 3 detik) akan menghasilkan gambar scalogram dengan lebar berbeda.
    *   Jika audio lebih dari 1.0 detik, dipotong (`crop`) untuk mengambil fase ekspulsi batuk utama.
    *   Jika audio kurang dari 1.0 detik, ditambahkan keheningan (`zero-padding`) di bagian akhir.
    *   *Justifikasi:* Durasi 1.0 detik sangat ideal karena fase batuk manusia (ekspulsi udara dan resonansi pita suara) umumnya berlangsung dan selesai dalam rentang waktu tersebut.

---

## 3. Protokol Anti-Leakage: Subject-Level Splitting

Salah satu alasan utama penolakan paper machine learning di jurnal medis terkemuka adalah **Speaker Leakage (Data Leakage)**.

*   **Masalah (Random Split):** Jika Anda memiliki 5 rekaman batuk dari Pasien A, lalu 3 rekaman ditaruh di set Train dan 2 rekaman ditaruh di set Test, model CNN akan dengan mudah mengenali Pasien A di set Test. AI bukan mendeteksi penyakit TB, melainkan menghafal karakteristik pita suara Pasien A (seperti sistem absensi wajah/suara). Hasil akurasi akan terlihat sangat tinggi (misal 99%), namun model akan gagal total saat dibawa ke rumah sakit nyata.
*   **Solusi (Subject-Level Split):** Kami menjamin secara mutlak bahwa seluruh rekaman batuk dari satu subjek/pasien hanya boleh berada di salah satu partisi saja (Train saja, Val saja, atau Test saja). Pasien di set Test adalah **100% orang baru** yang suaranya belum pernah didengar oleh model selama latihan.

---

## 4. Ekstraksi Fitur Akustik: Continuous Wavelet Transform (CWT)

Mengapa kita menggunakan CWT Scalogram (`cmor1.5-1.0`) dan bukan Short-Time Fourier Transform (STFT) / Mel-Spectrogram biasa?

### A. Keterbatasan Fourier (STFT/Mel-Spectrogram) & Prinsip Ketidakpastian Heisenberg
Fourier Transform menggunakan jendela analisis (*analysis window*) dengan ukuran tetap. Hal ini memicu dilema resolusi:
*   Jika jendela **lebar**: Resolusi frekuensi bagus (bisa membedakan nada dekat), tapi resolusi waktu buruk (tidak tahu kapan tepatnya suara itu muncul).
*   Jika jendela **sempit**: Resolusi waktu bagus (tahu kapan suara muncul), tapi resolusi frekuensi buruk.
*   *Masalah pada Batuk:* Batuk memiliki fenomena akustik yang kompleks. Batuk memiliki suara ledakan udara yang terjadi sangat singkat (membutuhkan resolusi waktu tinggi) sekaligus resonansi pita suara berfrekuensi rendah yang konstan (membutuhkan resolusi frekuensi tinggi). STFT tidak bisa memenuhi kedua kebutuhan ini sekaligus.

### B. Solusi Wavelet (CWT)
Wavelet Transform memecahkan masalah ini dengan menerapkan prinsip penskalaan (*dilation* dan *translation*). Menggunakan wavelet induk **complex Morlet (`cmor1.5-1.0`)**, jendela analisis berubah secara dinamis:
*   Pada frekuensi tinggi, jendela mengkerut (memberikan resolusi waktu yang sangat tajam untuk mendeteksi suara klik atau gesekan udara batuk).
*   Pada frekuensi rendah, jendela melebar (memberikan resolusi frekuensi yang tajam untuk menganalisis resonansi paru-paru).

Nilai koefisien CWT yang dihasilkan berupa matriks kompleks 2D. Nilai magnitudo absolutnya diplot sebagai gambar spasial (Scalogram) berukuran 224x224 piksel untuk dimasukkan ke CNN MobileNetV2.

---

## 5. Augmentasi Derau: Environmental Noise Robustness

Sistem screening batuk dirancang untuk ponsel pintar yang akan digunakan di puskesmas atau klinik terpencil yang bising, bukan di studio kedap suara.

*   **Strategi (Class-Balanced Noise Mixing):**
    Kami menyuntikkan suara bising nyata (dapur `DKITCHEN`, taman kota `NPARK`, lorong klinik bising `OHALLWAY`) ke dalam set **Train** dengan Signal-to-Noise Ratio (SNR) sebesar 10 dB.
*   **Mengapa Hanya di Set Train?**
    Pencampuran noise *hanya* dilakukan pada data latih agar model belajar mengabaikan bising latar belakang dan fokus pada struktur gelombang batuk. Sebaliknya, set **Validation dan Test dibiarkan bersih (clean-only)** untuk memastikan evaluasi performa model menggambarkan kemampuan diagnosis yang objektif, tanpa ada manipulasi data buatan (*synthetic performance inflation*).

---

## 6. Weighted Loss Function & Differential Learning Rates

### A. Mengapa Kelas Healthy Diberi Bobot Lebih Besar (Healthy = 2.0, TB = 1.0)?
Secara naluriah, untuk menyaring penyakit menular, kita ingin memberi bobot penalti lebih besar pada kelas sakit (TB weight > Healthy weight). Namun dalam RespiScan, kami melakukan hal sebaliknya:

*   **Alasan (Regularisasi Batas Keputusan):**
    Selama proses pelatihan awal, model cenderung mengalami bias karena ketidakseimbangan dataset, memprediksi terlalu banyak kasus TB (*excessive false-positive*). Dengan memberikan bobot penalti **2.0 pada kelas Healthy**, kami memaksa model selama proses *backpropagation* untuk membangun batas keputusan (*decision boundary*) awal yang stabil dan tidak sembarangan melabeli subjek sehat sebagai TB.
*   **Pemulihan Sensitivitas:**
    Kehilangan sensitivitas akibat penalti Healthy diatasi kemudian pada tahap **kalibrasi threshold Youden** di data validasi. Dengan demikian, kombinasi Weighted Loss (latihan batas kelas yang stabil) dan Youden Calibration (fokus sensitivitas klinik) menghasilkan performa diagnosis yang jauh lebih terkontrol.

### B. Differential Learning Rates
Model menggunakan backbone MobileNetV2 yang sudah dilatih pada jutaan gambar ImageNet.
*   **Feature Extractor (Backbone):** Diberi *learning rate* sangat kecil ($10^{-5}$). Ini menjaga fitur visual dasar (garis, kontras, warna scalogram) agar tidak rusak selama training.
*   **Classifier Head (Top Layer):** Diberi *learning rate* lebih besar ($10^{-4}$). Ini melatih neuron baru untuk fokus mengklasifikasikan perbedaan scalogram kelas Sehat vs TB secara cepat.

---

## 7. Patient-Level Probability Averaging (Soft Voting)

Satu pasien biasanya diminta batuk sebanyak 3 hingga 5 kali (menghasilkan beberapa file audio). Bagaimana AI menentukan diagnosis final pasien?

*   **Salah (Hard Voting):** Melakukan klasifikasi pada setiap file audio batuk, membulatkannya menjadi Sehat (0) atau TB (1), lalu mengambil suara terbanyak (*majority voting*). Metode ini sangat lemah karena membuang nilai probabilitas kontinu (misal, batuk yang bernilai 0.49 dibulatkan menjadi 0, padahal ia sangat dekat dengan batas TB).
*   **Benar (Probability Averaging - Kode Kita):**
    Kami menghitung nilai probabilitas rata-rata dari seluruh rekaman batuk pasien tersebut:
    $$P_j = \frac{1}{K_j} \sum_{k=1}^{K_j} \hat{p}_{jk}$$
    Lalu, keputusan pasien positif TB diambil jika nilai rata-rata probabilitas tersebut melewati batas threshold optimal Youden ($\theta$):
    $$\hat{Y}_{P_j} = \mathbb{I}(P_j \ge \theta)$$
    Metode ini jauh lebih sahih secara klinis karena mempertahankan informasi ketidakpastian (*uncertainty*) model dari setiap rekaman batuk sebelum membuat keputusan akhir.

---

## 8. Kalibrasi Keputusan Klinis via Youden's Index

Mengapa kita tidak memakai threshold default 0.50?

*   **Masalah Threshold 0.50:** Default klasifikasi komputer adalah 0.50. Jika probabilitas $\ge 0.50$, pasien divonis TB. Namun dalam dunia medis, **melewatkan 1 orang pasien TB (False Negative) jauh lebih berbahaya** daripada salah merujuk orang sehat (False Positive). Pasien TB yang terlewat akan kembali ke masyarakat dan menularkan penyakitnya.
*   **Strategi (Youden's Index $J$):**
    Kami menganalisis performa model pada data **Validation**. Youden's Index memaksimalkan jumlah sensitivitas dan spesifisitas pada kurva ROC:
    $$J(\theta) = \text{Sensitivitas}(\theta) + \text{Spesifisitas}(\theta) - 1$$
    Dari pencarian ini, ditemukan bahwa threshold optimal untuk menyaring pasien TB adalah **0.1380**.
*   **Dampak Penggeseran Threshold (Triage Setting):**
    Saat threshold diturunkan ke 0.1380 dan diuji pada set holdout Test:
    *   **Sensitivitas naik menjadi 100.00%** (artinya seluruh 40 pasien TB terdeteksi sempurna, False Negative = 0).
    *   **Spesifisitas turun menjadi 26.32%** (artinya banyak subjek sehat yang dirujuk ke lab untuk tes GeneXpert).
    *   *Kesimpulan Klinis:* Ini adalah karakteristik alat screening penapisan awal (*triage*) yang sangat baik—menangkap semua suspect sakit agar tidak ada penularan bebas, sementara penegakan diagnosis final tetap diserahkan pada tes PCR/GeneXpert.

---

## 9. Explainable AI (t-SNE Latent Space Mapping)

*   **Apa itu t-SNE?**
    MobileNetV2 mengekstrak representasi fitur berupa vektor berdimensi 1280. Manusia tidak bisa membayangkan ruang 1280 dimensi. **t-SNE** memproyeksikan data dimensi tinggi ini ke ruang visual 2D dengan mempertahankan kedekatan jarak antar data.
*   **Bahasa Akademis yang Rigor:**
    Alih-alih mengklaim klaster terpisah sempurna ("clear spatial separation"), dalam manuskrip kami melunakkannya menjadi **"observable clustering tendency"** (kecenderungan pengelompokan yang dapat diamati) dan **"partial visual separation"**. Hal ini disukai oleh reviewer karena bersikap jujur secara ilmiah bahwa model masih memiliki area tumpang tindih (*overlap*) pada batas kelas, namun secara umum sukses mempelajari biomarker batuk TBC secara konsisten.

---

## 10. Justifikasi Pemilihan Metrik Evaluasi Medis (Diagnostic Metrics)

Sistem screening TBC memiliki kebutuhan klinis yang berbeda dari klasifikasi gambar umum. Berikut adalah alasan mengapa metrik-metrik tertentu dipilih dan bagaimana cara membacanya dalam konteks penapisan klinis:

1.  **Sensitivitas (Sensitivity / Recall):**
    *   *Definisi:* Rasio pasien TBC aktif yang dideteksi secara benar oleh model dari keseluruhan pasien yang benar-benar sakit.
    *   *Alasan Pemilihan:* Dalam skenario penapisan awal (*screening/triage*), **Sensitivitas adalah metrik yang paling krusial**. Tujuan utama sistem adalah meminimalkan Negatif Palsu (pasien sakit yang dinyatakan sehat). Penderita TBC aktif yang terlewat akan kembali ke masyarakat dan menularkan penyakitnya. Oleh karena itu, kita sengaja menggeser threshold ke **0.1380** untuk mencapai **Sensitivitas 100.00%**, memastikan tidak ada kasus TBC aktif yang lolos.
2.  **Spesifisitas (Specificity):**
    *   *Definisi:* Rasio subjek sehat yang diidentifikasi secara benar sebagai sehat oleh model.
    *   *Alasan Pemilihan:* Spesifisitas yang tinggi membantu mengurangi beban rujukan laboratorium lanjutan (seperti tes PCR/GeneXpert). Pada threshold Youden kita, spesifisitas turun menjadi **26.32%**. Meskipun rendah (menyebabkan banyak rujukan palsu atau Positif Palsu), ini merupakan trade-off medis yang dapat diterima demi menjamin sensitivitas 100%. Pasien positif palsu akan disaring kembali menggunakan standar emas diagnostik di rumah sakit rujukan.
3.  **Nilai Prediktif Positif (Positive Predictive Value - PPV):**
    *   *Definisi:* Probabilitas bahwa seorang subjek benar-benar sakit TBC ketika tes AI menyatakan hasil "Positif".
    *   *Alasan Pemilihan:* PPV membantu klinisi menilai tingkat kepercayaan dari rujukan yang dikirimkan oleh AI. Nilai PPV kita sebesar **74.07%** menunjukkan bahwa sebagian besar rujukan AI adalah kasus TBC yang terkonfirmasi.
4.  **Nilai Prediktif Negatif (Negative Predictive Value - NPV):**
    *   *Definisi:* Probabilitas bahwa seorang subjek benar-benar sehat ketika tes AI menyatakan hasil "Negatif".
    *   *Alasan Pemilihan:* NPV yang tinggi memberikan kepastian hukum klinis. Dengan NPV mencapai **100.00%** pada threshold Youden kita, klinisi di puskesmas terpencil dapat dengan sangat aman menyatakan bahwa pasien bergejala yang berstatus negatif AI benar-benar sehat dari TBC, sehingga tidak perlu merujuk mereka untuk tes GeneXpert yang mahal.
5.  **AUROC (Area Under the ROC Curve):**
    *   *Definisi:* Kemampuan diskriminatif model untuk membedakan antara subjek Sehat dan TBC secara threshold-independent.
    *   *Alasan Pemilihan:* AUROC sebesar **0.7895** membuktikan secara matematis bahwa model kita memiliki kemampuan pemisahan kelas yang baik dan stabil secara keseluruhan, terlepas dari di mana batas threshold keputusan diletakkan.

---

## 11. Justifikasi Pemilihan Wilson Score Interval untuk Confidence Interval (CI) 95%

Reviewer jurnal menuntut penggantian metode interval kepercayaan (CI) dari pendekatan bootstrap ke metode Wilson Score. Berikut adalah justifikasi matematis di balik keputusan ini:

1.  **Kegagalan Wald Interval (Standard CI):**
    *   Rumus standard interval kepercayaan yang diajarkan di statistika dasar (Wald Interval) adalah:
        $$CI = \hat{p} \pm z \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$
    *   Ketika proporsi $\hat{p}$ bernilai ekstrem dekat 0 atau 1 (misalnya, Sensitivitas dan NPV data uji kita yang bernilai tepat **1.00** atau **100%**), nilai varians $\hat{p}(1-\hat{p})$ menjadi **0**.
    *   Akibatnya, batas Wald akan melaporkan selang kepercayaan lebar nol (`100%–100%`) atau bahkan batas atas melebihi `100%` (misal `98%–102%`). Hal ini secara metodologi salah, tidak realistis, dan akan langsung ditolak oleh reviewer karena jumlah sampel uji kita terbatas ($n = 40$ subjek TB).
2.  **Keterbatasan Bootstrap CI:**
    *   Bootstrap menggunakan resampling stokastik berulang. Jika data berukuran kecil atau hasil proporsi menyentuh batas ekstrem (100%), interval bootstrap cenderung menjadi tidak stabil, sangat lebar, dan menghasilkan angka yang berbeda-beda setiap kali program dijalankan, sehingga sulit direproduksi secara konsisten dalam publikasi ilmiah.
3.  **Keunggulan Wilson Score Interval (Exact Binomial-oriented):**
    *   Wilson Score Interval mengasumsikan distribusi binomial secara langsung tanpa pendekatan normal Wald yang kasar pada batas ekstrem.
    *   Ia secara asimetris menyesuaikan batas kepercayaan di sekitar nilai proporsi sampel $\hat{p}$.
    *   Batas interval tidak akan pernah melebihi 100% atau kurang dari 0%.
    *   Wilson Score tetap memberikan rentang ketidakpastian yang realistis dan valid secara akademis meskipun proporsi bernilai 1.0 (misalnya, Sensitivitas 100.00% dengan $n = 40$ subjek menghasilkan rentang **91.24%–100.00%**). Hal ini menunjukkan kepada reviewer bahwa meskipun performa uji kita sempurna pada data saat ini, batas bawah performa asli dalam populasi yang lebih luas diperkirakan sebesar 91.24%.

---

## 12. Alasan Langkah-Langkah Desain Arsitektur dan Pipeline

Mengapa kita memilih setiap konfigurasi teknis dalam penelitian ini? Dokumen metodologi ini mendokumentasikan keputusan ilmiah tersebut:

1.  **Mengapa Memilih MobileNetV2 (Bukan Model Raksasa seperti ResNet-50 atau Vision Transformer)?**
    *   *Alasan:* Target klinis RespiScan adalah puskesmas terpencil (*low-resource settings*) dengan keterbatasan listrik, koneksi internet, dan infrastruktur komputer. MobileNetV2 dirancang khusus untuk komputasi edge mobile menggunakan operasi *Depthwise Separable Convolution*. Ukuran modelnya sangat kecil (**11.7 MB** dengan **2.2 Juta parameter**). Ini menjadikannya realistis untuk dijalankan secara instan pada ponsel pintar Android kelas menengah ke bawah tanpa membutuhkan server GPU.
2.  **Mengapa Memilih Complex Morlet Wavelet (`cmor1.5-1.0`) sebagai Mother Wavelet?**
    *   *Alasan:* Wavelet Morlet kompleks merupakan fungsi Gaussian yang dimodulasi oleh osilasi sinusoidal kompleks. Parameter bandwidth sebesar **1.5** memberikan resolusi frekuensi yang tajam untuk mengisolasi resonansi vokal batuk, sementara center frequency sebesar **1.0** memberikan temporal resolution yang seimbang untuk menangkap transients berdurasi sangat singkat (seperti letupan udara batuk pertama yang berlangsung $<50$ ms). Komponen kompleksnya juga memisahkan magnitudo dan fase sinyal, memberikan representasi spasial 2D yang lebih kaya dibandingkan wavelet riil biasa.
3.  **Mengapa Noise mixing Hanya Diterapkan pada Set Training (Dilarang pada Val/Test)?**
    *   *Alasan:* Menambahkan noise pada set latih berfungsi sebagai teknik regularisasi (*data augmentation*) agar model belajar mengabaikan kebisingan ruangan dan fokus pada pola fisik batuk. Namun, jika set Val/Test ikut dicampur noise, evaluasi performa model akan terdistorsi dan rentan mengalami kebocoran data buatan (*synthetic performance inflation*). Set Val/Test dibiarkan bersih (*clean-only*) untuk menguji performa model secara murni pada data biologis batuk pasien yang asli.
4.  **Mengapa Menggunakan Weighted Cross-Entropy Loss (Healthy = 2.0, TB = 1.0)?**
    *   *Alasan:* Pada pembagian subjek latihan, terdapat ketidakseimbangan kelas (330 subjek TB vs 119 subjek Healthy). Hal ini menyebabkan model cenderung bias memprediksi semua sampel sebagai TB selama training. Dengan menaikkan bobot kelas Healthy menjadi **2.0**, kita memaksa gradien optimasi memberikan hukuman yang lebih berat jika salah mengklasifikasikan orang sehat. Ini berfungsi sebagai penyeimbang matematis (*regularizer*) agar batas keputusan model tetap stabil dan tidak sembarangan melabeli TBC sebelum ambang keputusan disempurnakan oleh Youden Index.
