# RespiScan: Pendukung Triage Tuberkulosis dari Akustik Batuk Menggunakan Scalogram CWT dan Focal Loss MobileNetV2

## Abstrak
Tuberkulosis (TB) tetap menjadi ancaman kesehatan global utama yang membutuhkan alat penapisan awal (pre-screening) yang cepat, non-invasif, dan berbiaya rendah, terutama di fasilitas kesehatan dengan sumber daya terbatas. Analisis suara batuk menggunakan kecerdasan buatan menawarkan alternatif penapisan awal yang potensial. Namun, pemanfaatan klinis sering terkendala oleh kelemahan metodologi seperti kebocoran data pembicara (*speaker leakage*), ketidakselarasan ekstraksi fitur, dan penentuan ambang batas keputusan yang tidak sesuai dengan kebutuhan klinis. 

Penelitian ini mempresentasikan **RespiScan**, sebuah sistem penapisan awal TB berbasis pembelajaran mendalam (*deep learning*) yang memanfaatkan scalogram *Continuous Wavelet Transform* (CWT) dan arsitektur *MobileNetV2*. Kami menerapkan protokol preprocessing akustik yang ketat dengan menstandardisasi semua input ke format mono, frekuensi sampling 16 kHz, dan durasi 1,0 detik non-overlapping, dilanjutkan dengan ekstraksi scalogram CWT menggunakan wavelet induk *complex Morlet* (`cmor1.5-1.0`) dengan 64 skala geometris. Kebocoran pembicara dieliminasi secara total menggunakan pembagian data pada tingkat subjek (*subject-level dataset split*) dengan status zero-overlap subjek. Bias representasi pasien dominan ditangani melalui pembatasan jumlah sampel maksimal 10 rekaman per subjek. Data latih diaugmentasi dengan bising dunia nyata (dapur, taman, lorong klinik) pada *Signal-to-Noise Ratio* (SNR) 10 dB untuk melatih ketangguhan model terhadap bising lingkungan. 

Model dilatih menggunakan fungsi rugi *Focal Loss* terbobot (bobot Non-TB Controls = 2.0, TB = 1.0, $\gamma = 2.0$), di mana pembobotan kelas Non-TB Controls yang lebih tinggi digunakan sebagai regularisasi keputusan untuk menekan kecenderungan model menghasilkan false positive TB, bukan untuk menyeimbangkan kelas minoritas. Ambang batas keputusan klinis dikalibrasi menggunakan Indeks Youden (*Youden's Index*) pada rata-rata probabilitas tingkat pasien di set validasi, menghasilkan threshold optimal sebesar **0.3595**. Evaluasi akhir pada set uji holdout independen yang terdiri dari 31 subjek unik (14 pasien TB aktif dan 17 kontrol Non-TB) menghasilkan akurasi klasifikasi tingkat pasien sebesar **87.10%** (95% CI: 71.15%–94.87%), sensitivitas sebesar **100.00%** (95% CI: 78.47%–100.00%), spesifisitas sebesar **76.47%** (95% CI: 52.74%–90.45%), *Positive Predictive Value* (PPV) sebesar **77.78%** (95% CI: 54.78%–91.00%), *Negative Predictive Value* (NPV) sebesar **100.00%** (95% CI: 77.19%–100.00%), dan *Area Under the ROC* (AUROC) sebesar **0.9454**. Hasil visualisasi ruang fitur laten menggunakan t-SNE menunjukkan kecenderungan pengelompokan spasial yang jelas antara kelas Non-TB dan TB. Dengan ukuran model yang sangat ringan (11.7 MB, 2.2 juta parameter), RespiScan layak diimplementasikan pada perangkat komputasi tepi (*edge-AI*) untuk mendukung diagnosis triage di puskesmas terpencil.

**Kata Kunci:** Penapisan Klinis; Transformasi Wavelet Kontinu; Akustik Batuk; MobileNetV2; Focal Loss; Komputasi Tepi.

---

## 1. Pendahuluan
Tuberkulosis (TB), yang disebabkan oleh bakteri *Mycobacterium tuberculosis*, tetap menjadi salah satu penyakit menular paling mematikan di dunia. Meskipun standar diagnosis seperti mikroskopi BTA (Bakteri Tahan Asam), rontgen dada, dan uji molekuler cepat (seperti GeneXpert) memiliki akurasi yang tinggi, penerapannya di negara-negara berkembang dengan pendapatan menengah ke bawah (*Low- and Middle-Income Countries* / LMICs) sangat terbatas. Kendala utamanya mencakup tingginya biaya instrumen, kebutuhan akan infrastruktur laboratorium yang terspesialisasi, waktu tunggu hasil yang lama, dan risiko bahaya biologis (*biohazard*) selama pengumpulan dahak pasien. Oleh karena itu, terdapat kebutuhan klinis yang mendesak akan alat penapisan awal (*screening/triage*) yang non-invasif, cepat, sensitif, dan murah untuk menyaring pasien di fasilitas kesehatan tingkat pertama (seperti puskesmas) sebelum dirujuk untuk tes molekuler lanjutan yang mahal.

Dalam beberapa tahun terakhir, analisis suara pernapasan berbasis komputer, khususnya suara batuk, telah berkembang sebagai alternatif yang menjanjikan. Perubahan patologis pada saluran pernapasan akibat TB—seperti inflamasi mukosa, obstruksi jalan napas, konsolidasi parenkim paru, dan produksi lendir kental (eksudat) secara berlebihan—mengubah biomekanika produksi batuk. Batuk paksa merupakan manuver ekspirasi tiga fase: fase inspirasi, fase kompresi (melibatkan penutupan glotis dan peningkatan tekanan intratoraks), dan fase letusan ekspirasi (*expulsitive burst*, di mana glotis terbuka secara mendadak sehingga menghasilkan aliran udara berkecepatan tinggi). Pada paru-paru yang terinfeksi TB, interaksi fisik antara udara dan lendir kental di saluran napas, dikombinasikan dengan hilangnya elastisitas jaringan paru, menghasilkan anomali akustik yang khas, seperti suara turbulensi mikro, gesekan frekuensi tinggi (*crackles*), dan pergeseran resonansi vokal batuk.

Meskipun banyak model kecerdasan buatan (AI) telah diusulkan untuk mendeteksi biomarker akustik ini, sebagian besar penelitian di literatur masih memiliki kelemahan metodologis serius yang menghambat implementasi klinis:
1. **Kebocoran Data Pembicara (*Speaker/Subject Leakage*):** Pembagian dataset (latih, validasi, uji) sering dilakukan secara acak pada tingkat segmen audio. Karena satu pasien umumnya berkontribusi pada beberapa file batuk, segmen suara dari pasien yang sama akhirnya tersebar di set latih dan uji secara bersamaan. Akibatnya, model CNN hanya menghafal karakteristik suara unik pasien (*speaker recognition*) alih-alih mempelajari patologi TB, yang menyebabkan nilai akurasi terlihat sangat tinggi di laboratorium namun gagal total ketika diuji pada pasien baru di lapangan.
2. **Bias Ketidakselarasan Fitur (*Feature Mismatch Bias*):** Model machine learning sangat sensitif terhadap perbedaan proses digital. Jika data kontrol non-TB dan data TB diekstraksi menggunakan parameter atau representasi spektral yang berbeda (misalnya mencampur format sampling rate), model akan mengklasifikasikan perbedaan tanda tangan digital tersebut daripada biomarker biologis asli batuk.
3. **Ambang Batas Klasifikasi yang Sewenang-wenang:** Alat penapisan medis harus memprioritaskan sensitivitas yang sangat tinggi untuk mencegah penderita TB aktif terlewat (False Negative). Namun, sebagian besar model AI menggunakan threshold probabilitas standar sebesar 0.50 yang tidak optimal secara klinis.

Untuk mengatasi berbagai kelemahan tersebut, penelitian ini memperkenalkan **RespiScan**, sebuah kerangka kerja teroptimasi dan ringan untuk penapisan awal TB klinis bebas leakage. Kontribusi utama dari penelitian ini adalah:
* Menerapkan standardisasi preprocessing akustik dan ekstraksi fitur scalogram CWT menggunakan wavelet induk *complex Morlet* (`cmor1.5-1.0`) secara konsisten pada seluruh subjek untuk mengeliminasi bias pemrosesan digital.
* Menjamin pembagian data tingkat subjek (*Subject-Level Dataset Split*) yang ketat dengan status zero-overlap subjek, memastikan pasien di set uji adalah 100% individu baru yang suaranya belum pernah didengar oleh model selama pelatihan.
* Menerapkan metode augmentasi bising lingkungan klinik nyata (dapur, taman, lorong klinik) pada data latih untuk melatih ketangguhan model terhadap kebisingan latar belakang.
* Mengalibrasi batas keputusan klinis menggunakan Indeks Youden yang dituning pada set validasi guna memaksimalkan sensitivitas deteksi TB tingkat pasien pada set holdout independen.
* Memvisualisasikan ruang fitur laten model menggunakan t-SNE sebagai bentuk kecerdasan buatan yang dapat dijelaskan (*Explainable AI*), memverifikasi bahwa model mengekstrak pola akustik biologis secara konsisten.

---

## 2. Tinjauan Pustaka
Penelitian awal dalam analisis suara batuk berbantuan komputer sangat bergantung pada ekstraksi fitur akustik manual (*hand-crafted features*) seperti *Mel-Frequency Cepstral Coefficients* (MFCC), *zero-crossing rate*, dan *spectral centroid*, yang diklasifikasikan menggunakan algoritma tradisional seperti *Support Vector Machine* (SVM) atau *Random Forest*. Meskipun ringan secara komputasi, fitur manual ini kesulitan menangkap dinamika akustik batuk patologis yang bersifat non-stasioner, sangat singkat (*transient*), dan non-linear.

Kehadiran *Convolutional Neural Network* (CNN) menggeser paradigma ini ke representasi visual spektral 2D. Sinyal audio 1D dikonversi menjadi gambar representasi waktu-frekuensi, seperti spektrogram *Short-Time Fourier Transform* (STFT) atau Mel-spectrogram. Namun, transformasi berbasis Fourier memiliki batasan resolusi yang tunduk pada prinsip ketidakpastian Heisenberg: ukuran jendela analisis yang tetap menghasilkan resolusi waktu-frekuensi yang seragam di seluruh rentang frekuensi. Suara batuk patologis memiliki kejadian transien frekuensi tinggi yang sangat singkat (seperti suara letupan udara akibat pembukaan glotis) sekaligus resonansi pita suara frekuensi rendah yang stabil. 

Untuk mengatasi keterbatasan Fourier, **Transformasi Wavelet Kontinu (CWT)** diadopsi dalam penelitian ini. Melalui pergeseran (*translation*) dan penskalaan (*dilation*) wavelet induk, CWT mampu memberikan resolusi yang dinamis: resolusi waktu yang tajam pada frekuensi tinggi untuk menangkap kejadian transien yang sangat singkat, dan resolusi frekuensi yang tajam pada frekuensi rendah untuk menganalisis resonansi paru-paru. RespiScan menggabungkan keunggulan matematis CWT ini dengan protokol pembagian subjek yang steril untuk menghasilkan model klasifikasi yang valid secara klinis.

---

## 3. Metodologi Penelitian

### 3.1. Sumber Dataset, Etika Penelitian, dan Alur Pembersihan Data
Penelitian ini menggunakan dataset sekunder hibrida yang dihimpun dari dua database pernapasan utama: **TBscreen clinical cough database** dan **Coswara database**. Penggabungan ini menghasilkan cohort dengan representasi klinis yang lebih luas. 

Database **TBscreen** pada kondisi mentah (*raw cohort*) menyumbang total 149 subjek klinis yang terbagi atas dua cohort: **Cohort A** (110 subjek, yang terdiri dari 103 pasien TB aktif dan 7 kontrol Non-TB) dan **Cohort C** (39 subjek, yang merupakan kontrol simtomatik dengan penyakit pernapasan lain seperti asma, pneumonia, dan PPOK). Selama proses penjaminan mutu akustik (*Quality Control*/QC), sebanyak 27 subjek dari TBscreen dieksklusi karena rekaman audio rusak, kosong, atau terputus secara mendadak (18 subjek dieksklusi dari Cohort A—menyisakan 87 subjek TB dan 5 Non-TB, serta 9 subjek dieksklusi dari Cohort C—menyisakan 30 subjek Non-TB). Hasil akhir setelah QC menyisakan 122 subjek aktif dari database TBscreen, di mana 92 subjek berasal dari Cohort A (87 TB dan 5 Non-TB) dan 30 subjek berasal dari Cohort C (seluruhnya Non-TB). Kelompok kontrol dalam TBscreen ini bukan merupakan individu sehat, melainkan kontrol penyakit pernapasan non-TB (*controls with other respiratory illnesses*). 

Untuk melengkapi kelompok kontrol Non-TB dengan subjek sehat, sebanyak 76 subjek diintegrasikan dari database **Coswara** (spesifiknya dari cohort skrining fungsi paru **COSWARA_LFS**). Seluruh 76 subjek dari Coswara ini lolos QC tanpa ada eksklusi. Dengan demikian, penggabungan kedua sumber sekunder ini menghasilkan total cohort mentah sebanyak 225 subjek, yang kemudian tersaring menjadi **198 subjek aktif terstandarisasi** setelah QC (122 dari TBscreen dan 76 dari Coswara).

Guna menjamin integritas ilmiah dan etika, penelitian ini hanya menggunakan data sekunder rekaman suara batuk yang telah dideidentifikasi (dianonimkan secara penuh). Peneliti tidak melakukan rekrutmen pasien secara langsung maupun intervensi klinis. Anotasi klinis dan label kebenaran diagnosis (*ground-truth*) didasarkan pada hasil pemeriksaan standar emas medis (GeneXpert dan mikroskopi BTA) dari database asli. Karena penelitian ini murni menganalisis data sekunder yang dianonimkan, kewajiban peninjauan etika langsung dibebaskan sesuai dengan regulasi nasional mengenai penggunaan data sekunder tanpa identitas personal.

Untuk mencegah bias panjang rekaman dan bias pemrosesan, diimplementasikan pipeline standardisasi audio. Semua rekaman batuk dikonversi ke format saluran tunggal (mono), diturunkan sampelnya (*resampling*) ke frekuensi 16.000 Hz, dan dipotong atau ditambahkan keheningan (*zero-padding*) hingga mencapai durasi tepat 1.0 detik (16.000 sampel). Standardisasi ke frekuensi 16 kHz dilakukan berdasarkan **Teorema Nyquist-Shannon**, yang menyatakan bahwa frekuensi sampling minimal harus 2 kali frekuensi tertinggi sinyal biologis untuk menghindari distorsi (*aliasing*). Mengingat rentang akustik batuk yang relevan secara medis berada di bawah 8 kHz, sampling rate 16 kHz merupakan pilihan optimal untuk mereduksi beban komputasi dan meminimalkan noise mikrofon frekuensi tinggi tanpa kehilangan detail fisik batuk.

Pemisahan dataset dilakukan secara ketat pada tingkat subjek (*Subject-Level Split*) untuk menjamin tidak ada sampel dari pasien yang sama yang terdistribusi di set data latih, validasi, dan uji. Alur penyaringan dataset dan distribusinya dirinci pada Tabel 1 dan Tabel 2.

**Tabel 1: Alur Penyaringan dan Evolusi Dataset RespiScan**
| Tahapan Penyaringan | Jumlah Subjek (Pasien) | Jumlah Sampel (File PNG CWT) | Kriteria Seleksi & Pembersihan Data |
| :--- | :---: | :---: | :--- |
| **1. Database Mentah (Hibrida)** | 225 | — | Gabungan dari 149 subjek TBscreen (Cohort A & C) dan 76 subjek Coswara (COSWARA_LFS). |
| **2. Kriteria Eksklusi (QC)** | 27 (Dieliminasi) | — | Subjek dikeluarkan karena file audio rusak, rekaman kosong (silent), atau terputus secara mendadak. |
| **3. Dataset Terstandarisasi** | **198** | **2740** | Subjek dengan audio valid. Diresample ke 16 kHz Mono, dipotong tepat 1.0s non-overlapping. |
| **4. Partisi Latih (Train Split)** | 137 | 2312 | Digunakan untuk pelatihan model. Berisi data bersih dan data augmentasi derau lingkungan (10 dB SNR). |
| **5. Partisi Validasi (Val Split)** | 30 | 229 | Bersih dari augmentasi (clean-only). Digunakan untuk kalibrasi threshold Indeks Youden secara independen. |
| **6. Partisi Uji (Test Split)** | 31 | 199 | Bersih dari augmentasi (clean-only). Holdout independen murni untuk evaluasi performa klinis akhir. |

Untuk menghindari dominasi akustik dari subjek tertentu yang memiliki ratusan rekaman batuk, kami menerapkan batas maksimal sampel (*Sample Capping*) sebesar **10 file batuk yang dipilih secara acak per subjek** (`MAX_SAMPLES_PER_SUBJ = 10`). Protokol ini menyeimbangkan kontribusi gradien loss dari setiap individu selama pelatihan, mencegah model terbias pada karakteristik suara pasien dominan tertentu.

Penerapan *sample capping* maksimal 10 file per subjek ini membatasi jumlah rekaman audio bersih yang masuk ke dalam pipeline menjadi **1.267 file audio bersih** secara total (di mana 1.115 file berasal dari subjek aktif TBscreen dan 152 file berasal dari subjek aktif Coswara). Hubungan antara 1.267 file bersih ter-capping ini dengan total akhir 2.740 sampel scalogram dijelaskan melalui pembagian data tingkat subjek dan penerapan prosedur augmentasi sebagai berikut:
* **Set Latih (Train Split)**: Terdiri dari 137 subjek dengan kontribusi **839 file audio bersih** (522 file dari 60 subjek TB dan 317 file dari 77 subjek Non-TB).
* **Set Validasi (Validation Split)**: Terdiri dari 30 subjek dengan kontribusi **229 file audio bersih** (127 file dari 13 subjek TB dan 102 file dari 17 subjek Non-TB).
* **Set Uji (Test Split)**: Terdiri dari 31 subjek dengan kontribusi **199 file audio bersih** (133 file dari 14 subjek TB dan 66 file dari 17 subjek Non-TB).

Untuk melatih ketangguhan model terhadap bising latar belakang klinik serta menangani ketidakseimbangan kelas latih, kami melakukan prosedur augmentasi data **hanya pada set latih (Train Split)**, sedangkan set validasi dan set uji dipertahankan murni bersih (*clean-only*). Augmentasi yang diterapkan pada set latih meliputi:
1. **Acoustic Noise Mixing (Kedua Kelas)**: Seluruh 839 file audio bersih pada set latih dicampur dengan derau lingkungan dunia nyata (lorong klinik `OHALLWAY`, taman `NPARK`, dapur klinik `DKITCHEN`) pada tingkat SNR 10 dB, menghasilkan tambahan **839 file audio bising**.
2. **Speed Perturbation (Hanya Kelas Non-TB)**: Khusus untuk memperkaya karakteristik akustik kelas kontrol Non-TB guna menekan nilai false positive, sebanyak 317 file bersih Non-TB pada set latih diduplikasi sebanyak dua kali melalui perturbasi kecepatan waktu ($0.9\times$ dan $1.1\times$ time-stretching), menghasilkan tambahan **634 file audio perturbasi kecepatan**.

Dengan demikian, jumlah sampel visual scalogram hasil ekstraksi CWT pada partisi latihan menjadi $839 \text{ (bersih)} + 839 \text{ (bising)} + 634 \text{ (kecepatan)} = 2.312$ sampel. Bila digabungkan dengan partisi validasi (229 sampel bersih) dan partisi uji (199 sampel bersih), total sampel keseluruhan yang mengalir dalam sistem RespiScan adalah **2.740 sampel scalogram CWT** ($2.312 + 229 + 199 = 2.740$).

**Tabel 2: Distribusi Subjek dan Sampel Scalogram pada Partisi Akhir**
| Partisi Data | Total Subjek | Subjek TB | Subjek Non-TB | Total Sampel | Sampel TB | Sampel Non-TB | Karakteristik Data |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train** | 137 | 60 | 77 | 2312 | 1044 | 1268 | Bersih & Diaugmentasi Derau / Kecepatan |
| **Validation** | 30 | 13 | 17 | 229 | 127 | 102 | 100% Bersih (Clean-Only) |
| **Test** | 31 | 14 | 17 | 199 | 133 | 66 | 100% Bersih (Clean-Only), Holdout Steril |, Holdout Steril |

---

### 3.2. Formulasi Matematis Transformasi Wavelet Kontinu (CWT)
Transformasi Wavelet Kontinu menguraikan sinyal 1D kontinu-waktu $x(t)$ dengan mengorelasikannya dengan keluarga fungsi tergeser dan terskala yang diturunkan dari wavelet induk $\psi(t)$:

$$W(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \psi^* \left( \frac{t - b}{a} \right) dt$$

di mana $\psi^*(t)$ melambangkan konjugat kompleks dari wavelet induk. Parameter skala $a > 0$ mengontrol pelebaran (dilasi) frekuensi, dan $b \in \mathbb{R}$ mengontrol pergeseran waktu (translasi). Dalam penelitian ini, kami menggunakan **wavelet Morlet kompleks** (`cmor1.5-1.0`), yang didefinisikan secara matematis sebagai:

$$\psi(t) = \frac{1}{\pi^{1/4}} e^{i \omega_0 t} e^{-\frac{t^2}{2}}$$

dengan frekuensi pusat $\omega_0 = 1.0$ dan parameter bandwidth $1.5$. Wavelet Morlet kompleks sangat sesuai untuk analisis akustik batuk karena kemampuannya memisahkan komponen magnitudo dan fase dari sinyal non-stasioner secara simultan. Parameter skala $a$ dipetakan secara geometris ke dalam 64 lebar skala diskrit:

$$a_i = 10^{\log_{10}(a_{\min}) + i \cdot \frac{\log_{10}(a_{\max}) - \log_{10}(a_{\min})}{N-1}}, \quad i = 0, 1, \dots, 63$$

dengan $a_{\min} = 2$ dan $a_{\max} = 186$. Magnitudo absolut dari koefisien kompleks yang dihasilkan, $|W(a, b)|$, diplot sebagai gambar representasi spasial 2D (Scalogram CWT) berukuran 224x224 piksel dengan colormap `jet` untuk menjadi input bagi jaringan saraf tiruan CNN MobileNetV2.

### 3.3. Augmentasi Derau dan Perturbasi Kecepatan Latih
Untuk memastikan model dapat mempelajari pola akustik batuk yang tangguh terhadap kebisingan latar belakang puskesmas nyata, kami menerapkan augmentasi derau lingkungan (*environmental noise mixing*) hanya pada partisi latihan (**Train split**). Kami menyuntikkan suara bising nyata (dapur klinik `DKITCHEN`, taman kota `NPARK`, dan lorong klinik bising `OHALLWAY`) ke dalam sampel batuk bersih $s(t)$ baik dari kelas TB maupun Non-TB pada tingkat *Signal-to-Noise Ratio* (SNR) terkontrol sebesar 10 dB:

$$x_{\text{noisy}}(t) = s(t) + \alpha \cdot n(t)$$

$$\alpha = \sqrt{ \frac{\sum_{t} s(t)^2}{\text{SNR}_{\text{linear}} \cdot \sum_{t} n(t)^2} }$$

di mana $\text{SNR}_{\text{linear}} = 10^{\frac{\text{SNR}_{\text{dB}}}{10}} = 10$. Penerapan bising lingkungan ini hanya dilakukan pada set latihan saja, sedangkan set validasi dan set uji holdout dibiarkan tetap bersih (*clean-only*) untuk menjamin evaluasi performa diagnostik menggambarkan kemampuan diagnosis pada data biologis asli pasien.

Selain itu, khusus untuk subset latihan kontrol Non-TB, dilakukan augmentasi tambahan berupa perturbasi kecepatan (*speed perturbation / time-stretching*) sebesar $0.9\times$ dan $1.1\times$ untuk memperkaya variabilitas akustik dan mereduksi bias false positive terhadap kelas TB.

---

### 3.4. Desain Arsitektur Convolutional Neural Network (MobileNetV2)
Untuk memfasilitasi komputasi yang efisien pada perangkat seluler di klinik pelosok, arsitektur **MobileNetV2** yang sangat ringan dipilih sebagai tulang punggung (*backbone*) klasifikasi. Inovasi utama MobileNetV2 adalah penggunaan blok **Inverted Residual dan Linear Bottleneck**. Blok ini menggunakan *Depthwise Separable Convolution* yang membagi konvolusi standar menjadi dua tahap:
1. **Depthwise Convolution:** Menerapkan filter spasial tunggal untuk setiap saluran input secara terpisah.
2. **Pointwise Convolution (1x1):** Menggabungkan output konvolusi depthwise secara linear di seluruh saluran.

Jika konvolusi standar membutuhkan parameter komputasi sebesar $D_k^2 \cdot M \cdot N$, konvolusi depthwise separable memangkasnya menjadi:

$$\text{Parameters} = D_k^2 \cdot M + M \cdot N$$

di mana $M$ dan $N$ adalah dimensi saluran input dan output, sedangkan $D_k$ adalah ukuran kernel filter (dalam kasus ini $3 \times 3$). MobileNetV2 diinisialisasi dengan bobot terlatih ImageNet dan dituning pada CWT Scalogram. Bagian kepala klasifikasi (*classifier head*) dimodifikasi menjadi:
* Lapisan fully connected linier yang memetakan fitur berdimensi 1280 (hasil *global average pooling*) menjadi 512 neuron.
* Fungsi aktivasi ReLU (*Rectified Linear Unit*).
* Lapisan *Dropout* ($p = 0.5$) untuk mencegah overfitting.
* Lapisan output linier akhir yang memetakan ke 2 kelas target (Non-TB, TB).

---

### 3.5. Protokol Pelatihan Model dan Focal Loss
Pelatihan dilakukan menggunakan kombinasi optimasi gradien dan fungsi rugi *Focal Loss* terbobot. Kami menetapkan rasio penalti bobot kelas sebesar 2:1 untuk kelas Non-TB dibanding TB. Bobot Non-TB = 2.0 tidak digunakan untuk menyeimbangkan kelas minoritas (mengingat subjek Non-TB berjumlah 77 sedangkan TB berjumlah 60), melainkan sebagai regularisasi keputusan untuk menekan kecenderungan model menghasilkan false positive TB (menstabilkan batas keputusan agar tidak berlebihan memprediksi TB). Sensitivitas klinis kemudian dipertahankan melalui kalibrasi threshold Youden pada validation set. Persamaan *Focal Loss* yang digunakan adalah:

$$\mathcal{L}_{\text{FL}} = - \frac{1}{N_{\text{batch}}} \sum_{i=1}^{N_{\text{batch}}} \alpha_c (1 - p_{t,i})^\gamma \log(p_{t,i})$$

di mana $p_{t,i}$ adalah probabilitas softmax yang diprediksi model untuk kelas target sampel $i$, $\gamma = 2.0$ adalah parameter pemusatan (*focusing parameter*) yang memberikan bobot penalti lebih besar pada sampel yang sulit diklasifikasikan (*hard negative samples*), dan $\alpha_c$ melambangkan bobot penalti kelas ($w_{\text{Non-TB}} = 2.0$, $w_{\text{TB}} = 1.0$). 

Model dioptimasi menggunakan Adam optimizer dengan *Differential Learning Rates*: laju pembelajaran ditetapkan sangat kecil sebesar $10^{-5}$ pada lapisan ekstraktor fitur (backbone) agar tidak merusak representasi visual dasar yang telah dipelajari dari ImageNet, dan laju pembelajaran sebesar $10^{-4}$ pada kepala klasifikasi baru agar adaptif mempelajari pola scalogram batuk. Pelatihan berjalan selama 20 epoch dengan mekanisme early stopping (patience = 7) yang memantau nilai F-1 score set validasi untuk mencegah overfitting sektoral.

---

## 4. Hasil dan Pembahasan

### 4.1. Framework Evaluasi Probabilitas Tingkat Pasien
Keputusan diagnostik klinis harus diambil per pasien, bukan per rekaman batuk tunggal. Mengingat satu subjek memberikan beberapa rekaman batuk (berkisar antara 3 hingga 10 file), kami menerapkan metode **Rata-rata Probabilitas Tingkat Pasien** (*Patient-Level Probability Averaging* / Soft Voting). Untuk setiap pasien unik $P_j$, nilai probabilitas akhir TB dihitung sebagai rata-rata aritmatika dari probabilitas softmax segmen-segmen batuknya ($s_{jk}$):

$$P_j = \frac{1}{K_j} \sum_{k=1}^{K_j} \hat{p}_{jk}$$

Keputusan akhir diagnosis pasien dinyatakan sebagai Positif TB jika probabilitas rata-rata $P_j$ melampaui batas ambang keputusan optimal $\theta$:

$$\hat{Y}_{P_j} = \mathbb{I}(P_j \ge \theta)$$

di mana $\hat{p}_{jk}$ melambangkan probabilitas TB untuk segmen batuk ke-$k$ dari pasien ke-$j$, $K_j$ merupakan total rekaman batuk milik pasien $P_j$, dan $\mathbb{I}(\cdot)$ melambangkan fungsi indikator biner. Skema ini secara klinis lebih valid daripada hard voting karena tidak membuang informasi probabilitas kontinu dari setiap rekaman batuk sebelum batas klasifikasi ditetapkan.

---

### 4.2. Perbandingan Kinerja Eksperimental
Untuk mengevaluasi performa model terpilih (Focal Loss MobileNetV2), kami membandingkannya dengan beberapa arsitektur dan konfigurasi alternatif yang dilatih menggunakan protokol splitting subject-level yang sama. Tabel 3 merangkum perbandingan kinerja pada set uji holdout independen:

**Tabel 3: Perbandingan Kinerja Model Alternatif pada Set Uji Holdout Pasien**
| Model Klasifikasi & Fungsi Rugi | Threshold Batas | Akurasi Uji | Sensitivitas (TB) | Spesifisitas (Non-TB) | AUROC Uji |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 1. ResNet18 (Weighted Cross-Entropy) | 0.50 | 74.19% | 50.00% | **94.12%** | 0.8739 |
| 2. MobileNetV2 (Weighted Cross-Entropy) | 0.50 | 77.42% | 64.29% | 88.24% | 0.8866 |
| 3. Dual-Branch Fusion (MFCC + CWT) | 0.50 | 83.87% | 85.71% | 82.35% | 0.9244 |
| **4. MobileNetV2 (Focal Loss - Diusulkan)** | **0.3595** (Youden) | **87.10%** | **100.00%** | 76.47% | **0.9454** |

Hasil eksperimen menunjukkan bahwa model **Focal Loss MobileNetV2** yang dievaluasi pada threshold Youden **0.3595** memiliki performa klasifikasi terbaik, dengan nilai AUROC tertinggi (**0.9454**) dan Sensitivitas sempurna (**100.00%**), menjadikannya kandidat terbaik untuk aplikasi triage klinis.

---

### 4.3. Analisis Kinerja Diagnostik Akhir dan Kalibrasi Threshold Youden
Pada penapisan awal TB, konsekuensi epidemiologis dari adanya Negatif Palsu (penderita sakit yang dinyatakan bebas TB) sangat fatal karena membiarkan penularan aktif berlanjut di masyarakat. Sebaliknya, Positif Palsu (kontrol non-TB dinyatakan berisiko) masih dapat disaring kembali melalui pemeriksaan laboratorium GeneXpert. Oleh karena itu, threshold klasifikasi standar 0.50 harus disesuaikan.

Kami memetakan kurva *Receiver Operating Characteristic* (ROC) pada set validasi untuk menghitung **Indeks Youden ($J$)** guna mengoptimalkan titik potong keputusan klinis:

$$J(\theta) = \text{Sensitivitas}(\theta) + \text{Spesifisitas}(\theta) - 1$$

$$\theta_{\text{optimal}} = \arg\max_{\theta} J(\theta)$$

Melalui kalkulasi ini, diperoleh batas keputusan klinis optimal pada **$\theta_{\text{optimal}} = 0.3595$**. Untuk memberikan interpretasi yang jujur secara akademis dan menilai ketidakpastian sampel kecil, seluruh metrik dilaporkan beserta **Interval Kepercayaan (CI) 95% menggunakan metode Wilson Score** (Tabel 4).

**Tabel 4: Hasil Kinerja Diagnostik Akhir Model Terpilih pada Holdout Test (n = 31)**
| Metrik Evaluasi | Default Threshold ($\theta = 0.50$) | Youden-Optimized Threshold ($\theta = 0.3595$) | Signifikansi Klinis & Target Diagnosis |
| :--- | :---: | :---: | :--- |
| **Akurasi Pasien** | 77.42%<br>(95% CI: 60.19%–88.61%) | **87.10%**<br>(95% CI: 71.15%–94.87%) | Persentase kebenaran diagnosis total pada subjek baru. |
| **Sensitivitas (Recall)** | 50.00%<br>(95% CI: 26.80%–73.20%) | **100.00%**<br>(95% CI: 78.47%–100.00%) | Tingkat jaminan deteksi TB (meminimalkan Negatif Palsu). |
| **Spesifisitas** | **100.00%**<br>(95% CI: 81.57%–100.00%) | **76.47%**<br>(95% CI: 52.74%–90.45%) | Kemampuan mengenali subjek Non-TB (mengurangi beban rujukan). |
| **PPV (Precision)** | 100.00%<br>(95% CI: 64.57%–100.00%) | **77.78%**<br>(95% CI: 54.78%–91.00%) | Probabilitas subjek benar-benar sakit saat hasil AI Positif. |
| **NPV** | 70.83%<br>(95% CI: 50.83%–85.09%) | **100.00%**<br>(95% CI: 77.19%–100.00%) | Probabilitas subjek benar-benar kontrol non-TB saat hasil AI Negatif. |
| **Patient-Level AUROC** | **0.9454** | **0.9454** | Kemampuan diskriminasi kelas murni (tidak bergantung threshold). |

Ketika dievaluasi menggunakan threshold default 0.50, model hanya memberikan sensitivitas sebesar 50.00%, yang berarti separuh dari pasien TB aktif di masyarakat akan terlewat. Setelah memindahkan threshold keputusan ke nilai optimal Youden 0.3595, sensitivitas tingkat pasien meningkat secara dramatis menjadi **100.00%** (FN = 0), dengan spesifisitas tetap terjaga pada angka **76.47%**. Hal ini membuktikan efektivitas kalibrasi threshold validasi ketika diuji pada data holdout independen. Matriks konfusi pasien pada threshold 0.3595 menunjukkan hasil klasifikasi: True Negative (TN) = 13, False Positive (FP) = 4, False Negative (FN) = 0, dan True Positive (TP) = 14 subjek.

```
[Gambar 1: Kurva ROC Tingkat Pasien pada Set Uji Holdout Independen]
Visualisasi kurva ROC untuk mengevaluasi kemampuan diskriminasi model Focal Loss MobileNetV2 pada data uji tingkat pasien (AUROC = 0.9454). Titik merah menandai batas keputusan klinis optimal (threshold = 0.3595) yang diperoleh secara terisolasi dari analisis Indeks Youden di set validasi. Penerapan threshold validasi secara independen ini menghasilkan Sensitivitas 100.00% dan Spesifisitas 76.47% pada set uji holdout murni untuk menghindari kebocoran data uji (test leakage).
```

```
[Gambar 2: Matriks Konfusi Tingkat Pasien pada Set Uji Holdout]
Heatmap matriks konfusi hasil klasifikasi tingkat pasien pada data uji holdout independen menggunakan threshold keputusan optimal 0.3595 (yang dituning menggunakan data validasi). Klasifikasi menghasilkan True Negative (TN) = 13 subjek, False Positive (FP) = 4 subjek, False Negative (FN) = 0 subjek (sensitivitas 100%), dan True Positive (TP) = 14 subjek.
```

---

### 4.4. Keterjelasan AI (Explainable AI) menggunakan Visualisasi t-SNE
Untuk meningkatkan transparansi keputusan model dan memastikan jaringan saraf mempelajari pola patofisiologis biologis, kami mengekstrak representasi fitur berdimensi 1280 dari lapisan *Global Average Pooling* terakhir MobileNetV2 untuk data holdout test. Fitur dimensi tinggi ini diproyeksikan ke ruang visual 2D menggunakan algoritma **t-distributed Stochastic Neighbor Embedding (t-SNE)**. t-SNE meminimalkan divergensi Kullback-Leibler antara probabilitas kedekatan spasial titik data di ruang dimensi tinggi dan rendah:

$$\text{KL}(P || Q) = \sum_{i \neq j} p_{j|i} \log \frac{p_{j|i}}{q_{j|i}}$$

Proyeksi t-SNE dikonfigurasi dengan parameter perplexity = 30, laju pembelajaran awal = 200, dan dijalankan selama 1.000 iterasi. Visualisasi plot scatter yang dihasilkan menunjukkan kecenderungan pengelompokan yang dapat diamati dengan tumpang tindih sebagian (*observable clustering tendency with partial overlap*) antara kontrol Non-TB (hijau) dan pasien TB (merah). Oleh karena itu, visualisasi t-SNE ini digunakan hanya sebagai dukungan kualitatif untuk melihat persebaran representasi fitur laten model pada set holdout, sedangkan kesimpulan utama performa diagnosis model tetap didasarkan sepenuhnya pada metrik kuantitatif tingkat pasien. Keberadaan pengelompokan spasial parsial ini secara kualitatif menunjukkan kecenderungan MobileNetV2 mengekstrak pola akustik biologis yang konsisten.

```
[Gambar 3: Visualisasi Proyeksi Fitur Laten Menggunakan t-SNE pada Set Uji Holdout]
Plot sebaran visual representasi fitur laten 1280-dimensi yang diproyeksikan ke ruang 2D menggunakan t-SNE untuk kelas kontrol Non-TB (lingkaran hijau) dan pasien TB aktif (segitiga merah). Plot ini memperlihatkan kecenderungan pengelompokan spasial dengan tumpang tindih sebagian (partial overlap) pada area perbatasan.
```

---

### 4.5. Kelayakan Implementasi Komputasi Tepi (*Edge-AI*)
Target penyebaran klinis RespiScan adalah puskesmas rural dengan keterbatasan akses internet dan infrastruktur komputasi. Karakteristik fisik model yang dikembangkan menunjukkan kelayakan implementasi tepi yang tinggi:
* **Jumlah Parameter:** 2,2 Juta parameter.
* **Ukuran Berkas Bobot:** **11.7 MB** (format penyimpanan `.pth` float32 standar).
* **Dampak Praktis:** Ukuran model di bawah 15 MB memungkinkan transfer memori yang sangat cepat, sehingga model dapat dijalankan secara lokal pada perangkat Android kelas menengah ke bawah tanpa memerlukan server GPU eksternal atau latensi jaringan internet.

---

### 4.6. Keterbatasan Penelitian (*Study Limitations*)
Peneliti mengakui adanya beberapa keterbatasan dalam penelitian ini demi menjaga objektivitas akademis:
1. **Ukuran Sampel Uji Pasien yang Terbatas:** Set uji holdout independen hanya terdiri dari 31 pasien (14 TB aktif dan 17 Non-TB). Meskipun sensitivitas pada sampel ini mencapai 100%, rentang batas bawah Interval Kepercayaan Wilson Score (78.47%) menunjukkan bahwa evaluasi pada populasi pasien yang jauh lebih besar tetap diperlukan untuk mengukuhkan performa diagnosis.
2. **Kesenjangan Generalisasi Segmen:** Selama proses pelatihan, akurasi tingkat segmen di set latihan mencapai lebih dari 90%, sementara akurasi segmen di validasi mengalami stagnasi (dapat dirujuk pada `Fig3_Training_Validation_Curves.png`). Hal ini mengindikasikan adanya celah generalisasi pada tingkat segmen suara tunggal. Namun, celah ini berhasil dijembatani oleh metode *patient-level probability averaging* yang secara konsisten menstabilkan diagnosis akhir subjek.

---

## 5. Kesimpulan
Penelitian ini telah berhasil mengembangkan **RespiScan**, sebuah sistem penapisan awal Tuberkulosis berbasis akustik batuk yang tangguh terhadap derau lingkungan klinik dan bebas dari kebocoran data pembicara. Melalui standardisasi audio 16 kHz, ekstraksi scalogram CWT Morlet kompleks yang konsisten, dan subject-level split, kami membangun fondasi model klasifikasi yang valid secara klinis. 

Dengan memanfaatkan fungsi rugi *Focal Loss* terbobot pada arsitektur MobileNetV2 dan mengalibrasi batas keputusan klinis via Indeks Youden pada threshold **0.3595**, RespiScan berhasil mencapai sensitivitas diagnostik tingkat pasien sebesar **100.00%** (95% CI: 78.47%–100.00%) dan akurasi sebesar **87.10%** (95% CI: 71.15%–94.87%) pada set uji independen. Hasil analisis visual t-SNE secara kualitatif menunjukkan kecenderungan pengelompokan spasial parsial antara kelompok Non-TB dan TB. Dengan kebutuhan memori yang sangat kecil (11.7 MB), model ini sangat potensial untuk diaplikasikan sebagai instrumen skrining triage berbasis komputasi tepi di klinik pelosok dengan keterbatasan sumber daya.

---

## Deklarasi

### Ucapan Terima Kasih
Penulis menyampaikan terima kasih yang sebesar-besarnya kepada Departemen Teknik Komputer, Universitas Dinamika, atas dukungan fasilitas riset dan infrastruktur server yang disediakan selama pengembangan prototipe penelitian RespiScan ini.

### Ketersediaan Data (*Data Availability*)
Data akustik mentah yang digunakan dalam penelitian ini bersumber dari database sekunder publik **Project Coswara** (dikelola oleh IISc Bangalore) dan **TBscreen**. Karena adanya pembatasan lisensi dan kerahasiaan data medis pasien, data audio mentah tidak didistribusikan ulang secara bebas oleh penulis. Peneliti lain dapat memperoleh akses ke dataset asli dengan mengajukan permohonan langsung ke repositori resmi masing-masing:
1. Dataset Coswara dapat diakses secara terbuka melalui repositori resmi Project Coswara di GitHub (https://github.com/iiscleap/Coswara-Data).
2. Dataset TBscreen dapat diakses melalui data pendukung yang disediakan oleh penulis publikasi asli TBscreen.

Seluruh kode pemrograman untuk preprocessing audio mentah, ekstraksi scalogram CWT, pelatihan model, serta bobot model terlatih (*model weights* `.pth`) disediakan secara terbuka untuk menjamin reproduksibilitas riset di repositori GitHub kami (https://github.com/username/RespiScan).

### Konflik Kepentingan
Penulis menyatakan tidak memiliki konflik kepentingan terkait dengan publikasi manuskrip ini.

### Kontribusi Penulis
* **G.E.F.:** Konseptualisasi ide, preprocessing data, pengembangan model pembelajaran mendalam, evaluasi statistik, dan penyusunan draf manuskrip asli.
* **H.P.:** Supervisi penelitian, peninjauan metodologi, perbaikan naskah akademis, dan persetujuan manuskrip final.
