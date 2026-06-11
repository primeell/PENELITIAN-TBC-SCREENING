Yang Wajib Diperbaiki
1. Threshold Youden harus jelas berasal dari validation set, bukan test set

Ini koreksi paling penting. Kalau threshold 0.4340 dipilih berdasarkan test set, maka itu termasuk test leakage. Reviewer bisa langsung menilai hasil optimized threshold terlalu optimistis.

Yang benar secara metodologi:

Threshold dipilih dari validation set, dikunci, lalu diterapkan ke holdout test set satu kali.

Kalimat yang wajib masuk ke Method:

The Youden-optimized threshold was selected exclusively from the validation set and fixed before being applied to the unseen holdout test set. The test set was not used for threshold selection.

Namun, kalau kenyataannya threshold 0.4340 memang dihitung dari test set, sebaiknya eksperimen diulang: cari threshold dari validation set, lalu evaluasi ulang di test set.

2. AUROC jangan dibedakan antara threshold 0.50 dan threshold 0.43

Dosen Anda benar. AUROC tidak bergantung pada threshold. Jadi, untuk model dan probabilitas yang sama, AUROC seharusnya tetap sama. Kalau di tabel ada AUROC 0.8205 untuk default threshold dan 0.8718 untuk optimized threshold, itu terlihat mencurigakan secara statistik.

Solusi terbaik:

Tabel utama hanya berisi metrik threshold-dependent:

Metric	Default Threshold 0.50	Youden Threshold 0.4340
Accuracy	83.87%	90.32%
Sensitivity	83.33%	94.44%
Specificity	84.62%	84.62%
PPV	88.24%	89.47%
NPV	78.57%	91.67%

Lalu AUROC ditulis terpisah:

Patient-level AUROC was calculated using aggregated patient-level probabilities and reported as a threshold-independent metric.

Jika AUROC 0.8205 dan 0.8718 berasal dari level berbeda, misalnya segment-level dan patient-level, maka harus ditulis jelas. Jangan ditaruh seolah-olah AUROC berubah karena threshold.

3. Dataset source dan ethical consideration wajib ditambahkan

Untuk AI medis, reviewer akan bertanya: data dari mana, label TB ditentukan bagaimana, apakah pasien TB dikonfirmasi klinis, apakah data anonim, dan apakah ada izin etik. Ini belum cukup kuat di draft sebelumnya.

Tambahkan subbagian:

Dataset Source and Ethical Considerations

Isi aman jika datanya sekunder/publik:

This study used anonymized secondary cough audio data. No direct patient interaction was conducted by the authors. The diagnostic labels followed the original dataset annotation protocol. Because the dataset was anonymized, no personally identifiable information was processed in this study.

Tapi jangan mengarang nomor etik. Kalau belum ada ethical clearance, jangan tulis nomor palsu.

4. Noise augmentation hanya pada healthy harus ditangani

Ini juga penting. Kalau noise hanya diberikan ke Healthy, model bisa belajar pola noise sebagai ciri Healthy. Itu bisa menjadi augmentation-induced artifact.

Ada dua opsi:

Opsi terbaik: ulang eksperimen dengan noise augmentation pada Healthy dan TB di training set saja.

Kalimatnya:

Environmental noise augmentation was applied to both Healthy and TB samples in the training set only to avoid class-specific augmentation artifacts.

Kalau eksperimen belum bisa diulang, tulis sebagai limitation:

Because environmental noise augmentation was primarily applied to healthy samples, future work should evaluate class-balanced augmentation to reduce the risk of augmentation-induced class artifacts.

Untuk submit yang kuat, saya lebih menyarankan perbaiki eksperimennya, bukan hanya ditulis sebagai limitation.

5. Training curve jangan diklaim “without overfitting”

Dosen Anda benar. Dari kurva yang Anda kirim, training accuracy naik tinggi, sedangkan validation accuracy stagnan dan validation loss naik. Itu indikasi segment-level overfitting/generalization gap.

Narasi yang aman:

The training curves indicate that the model learned the training distribution; however, the divergence between training and validation loss suggests potential segment-level overfitting. Therefore, the primary performance interpretation was based on patient-level evaluation using unseen holdout subjects rather than training accuracy alone.

Ini bukan melemahkan paper. Justru membuat paper terlihat jujur dan matang.

Yang Sangat Disarankan
6. Tambahkan confusion matrix patient-level

Ini sangat bagus karena test set hanya 31 subjek. Dari metrik optimized threshold:

	Predicted TB	Predicted Healthy
Actual TB	17	1
Actual Healthy	2	11

Ini membuat hasil lebih transparan. Sensitivity 94.44% berarti 17/18 TB terdeteksi, specificity 84.62% berarti 11/13 healthy benar dikenali.

7. Tambahkan confidence interval

Karena test patient hanya 31, satu pasien salah/benar bisa mengubah angka besar. Tambahkan 95% CI agar jurnal terlihat lebih kuat.

Kalimat Method:

Confidence intervals were estimated using patient-level bootstrap resampling with 1,000 iterations.

Kalau belum dihitung, nanti saya bisa bantu hitungkan dari confusion matrix untuk accuracy, sensitivity, specificity, PPV, dan NPV.

8. Weighted loss harus dijelaskan

Reviewer bisa bertanya: kenapa Healthy weight = 2.0, bukan TB? Untuk screening TB, biasanya TB diberi bobot lebih besar.

Penjelasan yang masuk akal:

The Healthy class was assigned a higher loss weight to reduce excessive false-positive screening observed in the preliminary baseline model, while TB sensitivity was controlled through validation-based threshold optimization.

Ini membuat keputusan metodologinya terlihat sadar, bukan asal.

9. Baseline harus dijelaskan atau dikurangi penekanannya

Kalau baseline tidak jelas, reviewer akan menyerang. Baseline harus menjelaskan:

input baseline apa,
model baseline apa,
split sama atau tidak,
apakah ada speaker leakage,
apakah preprocessing berbeda.

Kalau baseline lama memang tidak sepenuhnya fair, jangan terlalu menjual “kami naik dari baseline”. Lebih aman menjadikan baseline sebagai preliminary configuration, bukan pembanding utama.

10. t-SNE harus hati-hati

t-SNE bagus untuk visualisasi, tapi bukan bukti klinis final. Tambahkan kalimat:

Because t-SNE is sensitive to parameter settings and may exaggerate local structures, this visualization was used only as qualitative support and not as primary evidence of diagnostic performance.

Juga sebutkan parameter t-SNE jika ada: perplexity, learning rate, random state, iterations, dan apakah fitur berasal dari test set saja.

Yang Perlu Dirapikan untuk Format JoTI
11. Abstract perlu dipadatkan

Dosen Anda benar. Abstract jangan terlalu penuh angka split. Cukup masukkan total dataset, subject-level split, model, hasil utama, dan limitation. Ending harus hati-hati:

These findings suggest that RespiScan may serve as a lightweight triage-support prototype for TB pre-screening, although larger prospective and multi-center validation is required before clinical deployment.

12. Keywords urut alfabetis

Gunakan:

Clinical Pre-Screening; Continuous Wavelet Transform; Cough Acoustics; Edge AI; MobileNetV2; Tuberculosis

Ini lebih rapi dan sesuai saran dosen.

13. Acknowledgments jangan placeholder

Kalau tidak ada funding:

The authors would like to thank the Department of Computer Engineering, Universitas Dinamika, for academic support during the development of this research prototype.

14. Tambahkan Data Availability, Conflict of Interest, dan Author Contributions

Walaupun mungkin tidak wajib di template JoTI, untuk AI medis ini membuat naskah terlihat lebih profesional.

Contoh:

Data Availability

The data used in this study are available from the original dataset source or from the corresponding author upon reasonable request, subject to privacy and ethical restrictions.

Conflict of Interest

The authors declare no conflict of interest.

Author Contributions

G.E.F. contributed to conceptualization, preprocessing, modeling, evaluation, and manuscript drafting. H.P. contributed to supervision, methodology review, and manuscript revision.