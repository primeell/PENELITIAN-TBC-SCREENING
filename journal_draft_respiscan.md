# RespiScan: An Environmentally Robust and Highly Sensitive Tuberculosis Screening System from Cough Acoustics utilizing Consistent CWT Scalograms and Youden-Optimized MobileNetV2

## Abstract
Tuberculosis (TB) remains a major global health threat, demanding accessible, rapid, and low-cost pre-screening tools in low-resource settings. Computer-aided screening via cough acoustics offers a promising non-invasive alternative to conventional sputum smear microscopy and GeneXpert assays. However, clinical translation is often hindered by critical methodological flaws in existing literature, including speaker leakage, acoustic feature mismatch, and arbitrary decision boundaries. This study presents **RespiScan**, a lightweight deep learning screening system utilizing Continuous Wavelet Transform (CWT) scalograms and MobileNetV2. We systematically resolved the feature mismatch bias by standardizing all inputs to consistent complex Morlet (`cmor1.5-1.0`) CWT scalograms and enforcing strict subject-level dataset splitting to eliminate speaker leakage. 

To address environmental noise in outpatient clinics, training set samples were augmented with real-world noise (kitchen, park, hallway). To prevent test leakage, Youden’s Index ($J$) was applied to patient-level probability averaging on the validation set to optimize the classification decision threshold. Evaluating the final model on an independent holdout test set containing 59 completely unseen subjects (40 TB patients, 19 Healthy controls) at the validation-optimized threshold of 0.1380 achieved a patient-level sensitivity of **100.00%** (95% CI: 91.24%–100.00%), a specificity of **26.32%** (95% CI: 11.81%–48.79%), and an overall classification accuracy of **76.27%** (95% CI: 64.02%–85.31%), with a patient-level Area Under the ROC (AUROC) of **0.7895**. Patient-level evaluation showed sensitivity-prioritized screening performance, although the modest AUROC and low specificity indicate that the system remains a preliminary triage-support prototype requiring further optimization and external validation. The lightweight model footprint (11.7 MB, 2.2M parameters) makes RespiScan feasible for mobile edge computing in rural primary healthcare centers as a triage support tool.

**Keywords:** Clinical Pre-Screening; Continuous Wavelet Transform; Cough Acoustics; Edge AI; MobileNetV2; Tuberculosis.

---

## 1. Introduction
Tuberculosis (TB), caused by the infectious pathogen *Mycobacterium tuberculosis*, continues to be one of the leading infectious causes of death globally, accounting for over one million deaths annually. While diagnostic standards such as sputum smear microscopy, chest X-rays, and molecular assays (e.g., GeneXpert Omni/Ultra) offer high diagnostic accuracy, their implementation is severely constrained in low- and middle-income countries (LMICs). These constraints stem from high instrumentation costs, the need for specialized laboratory infrastructure, prolonged turnaround times, and the biohazard risks associated with sputum collection. Consequently, there is an urgent clinical demand for non-invasive, rapid, and highly sensitive pre-screening tools capable of triaging patients in rural primary healthcare centers before sending them for expensive confirmatory molecular testing.

In recent years, computer-aided respiratory sound analysis has emerged as a highly promising paradigm for affordable screening. Pathological changes in the respiratory tract caused by TB—such as mucosal inflammation, airway obstruction, parenchymal consolidation, and the hyperproduction of highly viscous mucus (exudate)—alter the biomechanics of cough production. A forced cough is a three-phase expiratory maneuver: the inspiratory phase, the compressive phase (involving glottal closure and elevated intrathoracic pressure), and the expiratory burst phase (where the glottis opens abruptly, generating high-velocity airflow). In a lung compromised by TB, the physical properties of the air-mucus interface, combined with tissue consolidation, generate unique acoustic anomalies, such as micro-turbulences, high-frequency crackles, and altered vocal tract resonances. 

While multiple artificial intelligence (AI) models have been proposed to identify these acoustic biomarkers, the literature is frequently undermined by critical methodological flaws that limit clinical translation:
1. **Speaker/Subject Leakage (Data Leakage):** A common error is splitting dataset segments randomly into training and validation sets. When multiple cough segments from the same patient are distributed across both sets, the deep learning model memorizes individual vocal signatures (speaker recognition) rather than learning generalized pathological biomarkers (disease classification).
2. **Feature Mismatch Bias:** Machine learning models are highly sensitive to processing discrepancies. In several historical datasets, control samples (healthy) and pathological samples (TB) are processed using different audio toolchains, sampling rates, or spectral representations (e.g., mixing Mel-spectrograms for one class and Continuous Wavelet Transform for another). The neural network easily minimizes loss by classifying the digital processing signatures rather than the biological features, leading to high validation scores but total failure during independent clinical testing.
3. **Arbitrary Clinical Thresholds:** Medical screening tools prioritize high sensitivity (minimizing False Negatives) to ensure highly infectious individuals are not missed. However, default machine learning classifiers utilize a standard binary probability threshold of $0.50$, which is mathematically arbitrary and rarely aligned with clinical screening priorities.

To resolve these limitations, this paper introduces **RespiScan**, a highly optimized, lightweight framework for clinical TB screening. The major contributions of this study are:
* Enforcing absolute **feature consistency** by standardizing the raw audio signals and extracting Continuous Wavelet Transform (CWT) scalograms with a complex Morlet wavelet (`cmor1.5-1.0`) across all classes, eliminating processing biases.
* Implementing a strict **Subject-Level Dataset Split** to guarantee that the cough signatures of any single patient are kept completely isolated in either the train, validation, or test sets.
* Enhancing environmental robustness through systematic **acoustic noise mixing** using real-world databases (kitchen, living room, street traffic) at varying SNRs, ensuring the network learns robust cough waveforms rather than background clinic noise.
* Optimizing the clinical decision boundary using validation-derived Youden’s Index to prioritize sensitivity in patient-level TB pre-screening while reporting the resulting specificity trade-off on an unseen holdout test set.
* Visualizing the deep representations via **t-SNE** to act as an Explainable AI (XAI) tool, validating the clinical reasoning of the neural network.

---

## 2. Related Works
Early endeavors in computer-aided cough analysis relied primarily on hand-crafted acoustic features coupled with shallow classifiers. Mel-Frequency Cepstral Coefficients (MFCCs), zero-crossing rates, and spectral centroid values were commonly classified using Support Vector Machines (SVM) or Random Forests. Although computationally light, these hand-crafted features struggle to capture the non-stationary, highly transient, and non-linear dynamics of pathological coughs.

The advent of Convolutional Neural Networks (CNNs) shifted the paradigm toward visual spectral representations. Researchers converted 1D audio signals into 2D time-frequency representations, such as Short-Time Fourier Transform (STFT) spectrograms and Mel-spectrograms. However, Fourier-based transforms suffer from a fundamental resolution limitation governed by the Heisenberg uncertainty principle: a fixed window size yields uniform time and frequency resolution across all bands. Pathological coughs exhibit short-duration, high-frequency transient events (e.g., crackles caused by sudden airway opening) alongside low-frequency steady resonances. 

To overcome this, the **Continuous Wavelet Transform (CWT)** has been adopted. By scaling and shifting a mother wavelet, CWT provides variable resolution: high temporal resolution at high frequencies (capturing rapid transients) and high frequency resolution at low frequencies (capturing steady resonances). Despite the mathematical elegance of CWT, past publications frequently mixed CWT with Mel-spectrogram features across classes or allowed speaker leakage, inflating reported metrics. RespiScan addresses these issues by combining strict subject-level splitting, uniform CWT processing, and clinical threshold tuning, establishing a new standard for rigorous acoustic screening.

---

## 3. Material and Methods

### 3.1. Dataset Source, Ethical Considerations, and Preprocessing Pipeline
The primary dataset used in this study is derived from the **TBscreen clinical cough database**, a multi-center diagnostic research study. To ensure scientific and ethical integrity, this study utilized fully anonymized secondary cough audio recordings. No direct patient recruitment or clinical interventions were conducted by the authors. Clinical annotations and diagnostic ground-truths (confirmed via sputum smear microscopy and GeneXpert assays) followed the original TBscreen protocol. Because the dataset consists exclusively of de-identified secondary data, direct ethical review was waived, in compliance with institutional and national regulations regarding secondary data analysis.

To establish a clinically motivated pre-screening model, a strict audio standardization pipeline was implemented. All raw cough records were converted to a single-channel (mono) format, resampled to a consistent sampling frequency ($f_s = 44,100$ Hz), and cropped or padded to a uniform duration of exactly 1.0 second ($44,100$ samples) to prevent temporal length bias in downstream neural processing.

To ensure absolute methodological transparency and prevent data leakage, a strict subject-level splitting strategy was enforced. The curation pipeline and final data selection flow are detailed in Table 1 (Dataset Evolution) and Table 2 (Final Split Distribution).

**Table 1: Dataset Curation and Evolution Pipeline**
| Curation Stage | Subjects | Audio / Scalogram Samples | Key Filtering / Selection Criteria |
| :--- | :---: | :---: | :--- |
| **1. Raw Cohort (TBscreen)** | 645 | — | Initial multi-center database containing raw patient records. |
| **2. Excluded (Quality Control)** | 52 | — | Excluded due to corrupted audio files, silent recordings, or dead channels. |
| **3. Clean Standardized Dataset** | 593 | 2645 | Resampled to 44.1 kHz, mono, cropped/padded to 1.0s non-overlapping. |
| **4. Final Training Partition** | 449 | 2225 | Clean data + real-world environmental noise mixing (10 dB SNR). |
| **5. Final Validation Partition** | 85 | 254 | Clean-only partition used exclusively for decision threshold tuning. |
| **6. Final Holdout Test Partition** | 59 | 166 | Clean-only independent holdout partition for patient evaluation. |

*Note on historical pilot datasets:* In preliminary pilot iterations of this research pipeline (reported in early manuscript drafts), a restricted subset of 136 unique subjects was utilized, where overlapping segment windowing (slicing recordings into multiple overlapping 1-second chunks) was applied to yield 4,258 visual CWT samples (~31 segments per subject). In this revised work, we expanded the cohort to the full 593 active subjects (a 4.3-fold increase in speaker diversity) and moved to a non-overlapping standardized 1-second representation (2,645 samples) to ensure patient-level splits are genuinely generalized and free of segment duplication artifacts.

**Table 2: Subject and Scalogram Sample Distributions Across Final Dataset Splits**
| Split | Subjects (Total) | TB Subjects | Healthy Subjects | Scalogram Samples (Total) | TB Samples | Healthy Samples | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train** | 449 | 330 | 119 | 2225 | 975 | 1250 | Clean and Noisy-Augmented |
| **Validation** | 85 | 61 | 24 | 254 | 135 | 119 | Clean-Only Evaluation |
| **Test** | 59 | 40 | 19 | 166 | 98 | 68 | Clean-Only Holdout Test |

### 3.2. Mathematical Formulation of the Continuous Wavelet Transform (CWT)
Unlike Fourier-based methods, the Continuous Wavelet Transform decomposes a 1D continuous-time signal $x(t)$ by correlating it with a localized family of functions derived from a single "mother wavelet" $\psi(t)$ via scaling ($a$) and translation ($b$):

$$W(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \psi^* \left( \frac{t - b}{a} \right) dt$$

where $\psi^*(t)$ denotes the complex conjugate of the mother wavelet. The scale parameter $a > 0$ controls the frequency dilation, and $b \in \mathbb{R}$ controls the temporal translation. In this study, we utilize the **complex Morlet wavelet** (`cmor1.5-1.0`), mathematically defined as:

$$\psi(t) = \frac{1}{\pi^{1/4}} e^{i \omega_0 t} e^{-\frac{t^2}{2}}$$

where $\omega_0$ represents the center frequency of the wavelet (set to 1.0, with a bandwidth parameter of 1.5). The complex Morlet wavelet is highly suited for pathological acoustic analysis due to its excellent capability to capture localized phase and amplitude variations in non-stationary signals. The scale parameter $a$ is mapped to a geometric range covering 100 discrete widths:

$$a_i = 10^{\log_{10}(a_{\min}) + i \cdot \frac{\log_{10}(a_{\max}) - \log_{10}(a_{\min})}{N-1}}, \quad i = 0, 1, \dots, 99$$

where $a_{\min} = 1$ and $a_{\max} = 1024$. The magnitude of the resulting complex coefficients $|W(a, b)|$ is plotted as a 2D spatial representation—the CWT Scalogram—which serves as the input image for the deep neural network.

```
+------------------+     +-----------------------+     +-------------------------+
|  Raw Cough Wave  | --> | Audio Standardization | --> |  Complex Morlet CWT     |
|  (1D mono, 44k)  |     |  (1.0s Mono, 44.1kHz) |     |  (cmor1.5-1.0, 100 scales|
+------------------+     +-----------------------+     +-------------------------+
                                                                    |
                                                                    v
+------------------+     +-----------------------+     +-------------------------+
| Optimized Target | <-- | Subject-Level Split   | <-- |   2D Visual Scalogram   |
| (Train/Val/Test) |     |  (Prevent Leakage)    |     |   (Save as 224x224 PNG) |
+------------------+     +-----------------------+     +-------------------------+
```

### 3.3. Environmental Noise Robustness and Class-Balancing Protocol
Clinical deployment of cough-based screening tools often occurs in unshielded outpatient clinics with high levels of ambient noise. To prevent the model from memorizing quiet recording environments as a biomarker for healthy controls and noisy environments as a biomarker for tuberculosis, we enforce a class-balanced noise-mixing protocol in the training partition. Both Healthy controls and Tuberculosis patients in the training set were augmented by injecting random 1-second segments extracted from real-world environmental noise databases, including domestic spaces (DKITCHEN), urban parks (NPARK), and busy hallways (OHALLWAY). The noise waveform $n(t)$ is scaled and mixed with the clean cough signal $s(t)$ at a controlled Signal-to-Noise Ratio (SNR) of 10 dB:

$$x_{\text{noisy}}(t) = s(t) + \alpha \cdot n(t)$$

$$\alpha = \sqrt{ \frac{\sum_{t} s(t)^2}{\text{SNR}_{\text{linear}} \cdot \sum_{t} n(t)^2} }$$

where $\text{SNR}_{\text{linear}} = 10^{\frac{\text{SNR}_{\text{dB}}}{10}} = 10$. This uniform noise injection ensures class and environmental consistency during training. Applying noise mixing exclusively within the training set ensures that the holdout validation and test partitions remain completely clean and unaugmented, preventing any synthetic performance inflation.

Additionally, to address the subject-level cohort imbalance in the training partition (330 TB subjects vs. 119 Healthy subjects), a class-balancing acoustic perturbation protocol was applied exclusively to the training Healthy subset. This involved time-stretching the Healthy recordings (speeding up by 1.1x to generate 256 TimeStretch_Fast samples), balancing the class representation during optimization. No such acoustic perturbations were applied to the validation or test sets.

### 3.4. Deep Convolutional Neural Network (MobileNetV2) Architecture
To facilitate real-world implementation on standard mobile devices or portable tablets at point-of-care clinics, a lightweight, highly efficient deep learning architecture is paramount. We deploy **MobileNetV2**, pre-trained on ImageNet and fine-tuned on our CWT scalograms. MobileNetV2’s core innovation lies in its **Inverted Residual and Linear Bottleneck** blocks. 

```
   Input (2D CWT Scalogram: 224x224x3)
      |
      v
 [Conv2d 3x3] (Stride 2)
      |
      v
 [Bottleneck Blocks x 17] (Depthwise Separable Convolutions & Inverted Residuals)
      |
      v
 [Conv2d 1x1 & Global Average Pooling] ---> 1280-dimensional Feature Vector
                                                     |
                                                     v
                                         [Modified Classifier Head]
                                         - Fully Connected (1280 -> 512)
                                         - ReLU Activation & Dropout (p=0.5)
                                         - Fully Connected Output (512 -> 2)
                                                     |
                                                     v
                                        Softmax Decision Probabilities
```

The depthwise separable convolution decomposes standard convolution into a two-step process:
1. **Depthwise Convolution:** A single spatial filter is applied to each input channel independently.
2. **Pointwise Convolution (1x1):** A simple linear combination of the depthwise outputs across all channels.

This factorization drastically reduces computational parameters and floating-point operations (FLOPs). For a filter kernel of size $D_k \times D_k$, standard convolution parameter count scales as $D_k^2 \cdot M \cdot N$, whereas depthwise separable convolution scales as:

$$\text{Parameters} = D_k^2 \cdot M + M \cdot N$$

where $M$ and $N$ represent the input and output channel dimensions, respectively. The modified classification head of our network consists of:
* A fully connected layer mapping the 1280-dimensional global average pooled features to 512 units.
* A Rectified Linear Unit (ReLU) activation function.
* An aggressive Dropout layer ($p = 0.5$) to mitigate overfitting.
* A final output layer mapping to 2 classes (Healthy, Tuberculosis).

### 3.5. Training Protocol and Weighted Loss Function
To address class imbalances, the model is trained using a **Weighted Cross-Entropy Loss** function. We assign a penalty weight ratio of 2:1 for misclassifications of the Healthy class relative to the TB class. The mathematical loss formulation is:

$$\mathcal{L} = - \frac{1}{N_{\text{batch}}} \sum_{i=1}^{N_{\text{batch}}} \sum_{c=0}^{1} w_c \cdot y_{i,c} \log(\hat{p}_{i,c})$$

where class index $c=0$ represents Healthy and class index $c=1$ represents TB, with class penalty weights $w_0 = w_{\text{healthy}} = 2.0$ and $w_1 = w_{\text{TB}} = 1.0$, $y_{i,c}$ is the binary ground-truth indicator for class $c$ of sample $i$, and $\hat{p}_{i,c}$ represents the predicted softmax probability. Preliminary experiments showed that assigning a higher penalty to the Healthy class reduced excessive false-positive predictions observed during initial training, which represents a critical regularizer for learning stable class boundaries. Because the threshold was subsequently calibrated on the validation set to prioritize sensitivity, this loss-level weighting ensured that validation-derived thresholding could operate on balanced probability distributions rather than highly skewed classifier defaults. The optimization is performed using the Adam optimizer with differential learning rates: $10^{-5}$ for the backbone feature extraction layers to preserve general visual features, and $10^{-4}$ for the newly initialized classifier head. Training is carried out for 30 epochs with a learning rate scheduler (`ReduceLROnPlateau`, factor=0.5, patience=3) and an automatic early stopping mechanism (patience=7) monitored against validation patient-level AUROC.


---

## 4. Experimental Results and Discussion

### 4.1. Patient-Level Evaluation Framework
In clinical application, a screening decision is made per patient, not per audio segment. Thus, after segment-level probability extraction on the independent TEST set (consisting of 59 unique patients with multiple cough recordings), we implement a **Patient-Level Probability Averaging** framework. For each unique patient $P_j$, the patient-level probability is computed as the arithmetic mean of the probabilities of all $K_j$ segments $s_{jk}$ belonging to that patient:

$$P_j = \frac{1}{K_j} \sum_{k=1}^{K_j} \hat{p}_{jk}$$

The patient-level classification prediction $\hat{Y}_{P_j}$ is then determined using the validation-derived Youden decision boundary threshold $\theta$:

$$\hat{Y}_{P_j} = \mathbb{I}(P_j \ge \theta)$$

where $\mathbb{I}(\cdot)$ is the indicator function, $\hat{p}_{jk}$ is the model's predicted probability of TB for segment $s_{jk}$, and $K_j$ is the total number of valid cough segments recorded for patient $P_j$. This approach is methodologically superior to hard majority voting because it operates on continuous probability densities, preserving the model's uncertainty prior to applying the optimized clinical threshold.

### 4.2. Diagnostic Performance Analysis (Before vs. After Optimization)
We compare distinct configurations to validate the technical enhancements of RespiScan. To prevent test leakage, the optimal Youden decision boundary was tuned exclusively on validation data, yielding a threshold of **0.1380**. A baseline configuration (representing the historical model compromised by feature mismatches and speaker leakage) was evaluated narrative-only. The baseline is presented only as a preliminary configuration to illustrate methodological risks and is not treated as a fair performance benchmark. The proposed consistent CWT model was then evaluated on the independent holdout test set containing 59 completely unseen subjects (40 TB patients, 19 Healthy controls) under two decision boundaries:
1. **Consistent CWT Model (Default Threshold = 0.50):** The proposed model trained under our strict subject-level splitting using uniform CWT scalograms, evaluated at the standard machine learning probability boundary of $0.50$.
2. **Consistent CWT Model (Optimized Threshold = 0.1380):** The proposed model evaluated at the validation-optimized Youden's Index decision boundary.

The experimental outcomes across all critical medical screening metrics, calculated directly from the test confusion matrix, along with 95% Confidence Intervals (CI) calculated using the Wilson Score Interval method, are detailed in Table 3.

**Table 3: Detailed Patient-Level Holdout Test Metrics Across Models**


| Evaluation Metric | Consistent CWT (Default Thresh = 0.50) | Consistent CWT (Optimized Thresh = 0.1380) | Clinical Target / Significance |
| :--- | :---: | :---: | :--- |
| **Patient Accuracy** | 71.19%<br>(95% CI: 58.62%–81.16%) | **76.27%**<br>(95% CI: 64.02%–85.31%) | Overall diagnostic correctness on unseen holdout patients. |
| **Sensitivity (Recall)** | 70.00%<br>(95% CI: 54.57%–81.93%) | **100.00%**<br>(95% CI: 91.24%–100.00%) | TB screening detection rate (minimize False Negatives). |
| **Specificity** | **73.68%**<br>(95% CI: 51.21%–88.19%) | **26.32%**<br>(95% CI: 11.81%–48.79%) | Healthy control rate (minimize unnecessary referrals). |
| **Positive Predictive Value (PPV)**| 84.85%<br>(95% CI: 69.08%–93.35%) | **74.07%**<br>(95% CI: 61.07%–83.88%) | Probability that a "Positive" test truly has TB. |
| **Negative Predictive Value (NPV)**| 53.85%<br>(95% CI: 35.46%–71.24%) | **100.00%**<br>(95% CI: 56.55%–100.00%) | Probability that a "Negative" test is truly healthy. |
| **Patient-level AUROC**<br>*(Threshold-Independent)*| 0.7895 | 0.7895 | Discriminative power across all thresholds (constant). |

#### Discussion of Specificity and Sensitivity Trade-Offs:
A baseline configuration reported an inflated specificity of **61.54%** and sensitivity of **100.00%**, but this preliminary run was highly vulnerable to digital processing bias and speaker memorization (leakage) and does not represent a valid diagnostic tool. Enforcing strict subject-level splitting and uniform CWT scalograms ensures the model extracts generalizable biological features. Evaluating the consistent model at a default threshold of $0.50$ yielded a sensitivity of **70.00%** and a specificity of **73.68%**. For Tuberculosis pre-screening, where the primary objective is to identify highly infectious individuals, a sensitivity of 70.00% is clinically insufficient. By shifting the decision boundary to the validation-derived threshold of **0.1380**, patient-level screening sensitivity is boosted to **100.00%** (95% CI: 91.24%–100.00%), ensuring that only 0.00% of active TB patients are missed. Shifting the threshold to 0.1380 also increases the overall classification accuracy from **71.19%** to **76.27%**. Specificity under this optimized threshold is **26.32%** (95% CI: 11.81%–48.79%). This threshold reflects a sensitivity-prioritized triage trade-off. However, the low specificity indicates a high false-positive burden, and future model refinement is required before practical clinical deployment.

---

### 4.3. Clinical Decision Boundary Optimization via Youden's Index (Fig 1 & Fig 2)
In medical screening, a default classifier threshold of $\theta = 0.50$ is rarely optimal. Missing a tuberculosis case (False Negative) has devastating clinical and epidemiological consequences, as it allows the disease to propagate unchecked in the community. To optimize clinical utility, we construct the patient-level Receiver Operating Characteristic (ROC) curve by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR) across all possible decision boundaries $\theta \in [0, 1]$ on the validation set. Shifting the decision threshold from 0.50 is done to maximize sensitivity. 

To locate the optimal clinical threshold, we apply **Youden's Index ($J$)** to the validation set patient probabilities, which maximizes the sum of sensitivity and specificity:

$$J(\theta) = \text{Sensitivity}(\theta) + \text{Specificity}(\theta) - 1$$

$$J(\theta) = \text{TPR}(\theta) - \text{FPR}(\theta)$$

$$\theta_{\text{optimal}} = \arg\max_{\theta} J(\theta)$$

Applying this optimization to the validation set yields the mathematically optimal decision boundary at **$\theta_{\text{optimal}} = 0.1380$** (with a validation-level AUROC of **0.7411**). 

```
[Fig1_ROC_Patient_Optimized.png]
Plot displaying:
- Patient-Level Holdout Test ROC Curve (AUROC = 0.7895)
- Validation-derived optimal decision threshold marker (Red dot) plotted at Thresh = 0.1380
- Resulting Test Sensitivity = 100.00%, Specificity = 26.32%
```

Locking this decision boundary and applying it to the independent test set (AUROC of **0.7895**) instantly shifts the patient-level screening **Sensitivity to 100.00%** (95% CI: 91.24%–100.00%) while maintaining a **Specificity of 26.32%** (95% CI: 11.81%–48.79%), resulting in an overall **Patient Accuracy of 76.27%** (95% CI: 64.02%–85.31%). The patient-level classification outcomes under this optimized threshold are shown as a heatmap in Figure 2.

```
[Fig_Confusion_Matrix_Patient.png]
Heatmap displaying the Youden-optimized Patient-Level Confusion Matrix:
- True Negative (Healthy predicted Healthy): 5 subjects
- False Positive (Healthy predicted TB): 14 subjects
- False Negative (TB predicted Healthy): 0 subjects
- True Positive (TB predicted TB): 40 subjects
```

This result indicates that validation-derived threshold calibration can improve sensitivity on the holdout test set, although the reduced specificity shows that further optimization is required before clinical deployment.

---

### 4.4. Feature Latent Space Explainability via t-SNE (Fig 3)
To provide scientific transparency and act as an Explainable AI (XAI) validation step, we extract the 1280-dimensional feature representations from the final global average pooling layer of MobileNetV2 for the independent holdout test set. We project these high-dimensional embeddings onto a 2D space using **t-distributed Stochastic Neighbor Embedding (t-SNE)**. t-SNE minimizes the divergence between joint probabilities of pairs of points in the high-dimensional space and low-dimensional space:

$$\text{KL}(P || Q) = \sum_{i \neq j} p_{j|i} \log \frac{p_{j|i}}{q_{j|i}}$$

where $p_{j|i}$ measures the similarity of point $x_i$ to $x_j$ in high-dimensional space, and $q_{j|i}$ measures their similarity in low-dimensional space. The t-SNE projection was configured with a perplexity of 30, an initial learning rate of 200, a random state of 42, and run for 1,000 iterations exclusively using features extracted from holdout test subjects.

```
[Fig3_tSNE_Acoustic_Clusters.png]
t-SNE Scatter Plot displaying:
- Green circular markers: Healthy Controls (Well-clustered)
- Red triangular markers: Tuberculosis (TB) Patients (Separated cluster)
- Minimal overlapping boundary, supporting qualitative acoustic biomarker extraction.
```

The resulting t-SNE visualization (refer to `Fig_tSNE_Acoustic_Clusters.png` in the folder) demonstrates an observable clustering tendency and partial visual separation between the `Healthy Controls` cluster (Green) and the `Tuberculosis` cluster (Red). Because t-SNE is highly sensitive to hyperparameter tuning and can sometimes exaggerate local structures or create artificial clusters, this visualization was utilized strictly as qualitative support of feature separation and not as primary quantitative evidence of diagnostic performance. Nevertheless, the observable clustering tendency supports the hypothesis that MobileNetV2 successfully extracts actual acoustic pathological biomarkers related to respiratory biomechanics rather than memorizing background clinic noise or digital artifacts.

---

### 4.5. Computational Feasibility and Study Limitations
MobileNetV2’s lightweight architecture yields highly efficient computational metrics, making it well-suited for point-of-care deployments:
*   **Total Parameter Count:** 2.2 Million parameters.
*   **Disk Footprint:** 11.7 MB (stored as standard float32 `.pth` weights).
*   **Feasibility Check:** The model size suggests potential edge feasibility; however, actual latency, memory consumption, and power usage on Raspberry Pi or smartphone hardware remain to be evaluated.

Despite these positive metrics, two key limitations must be addressed for academic and clinical rigor:
1.  **Noise Augmentation Generalization:** To ensure environmental robustness, we deployed class-balanced noise mixing on the training set, maintaining a uniform acoustic environment across both classes. While this prevents the model from associating background noise signatures with a specific diagnostic class, further work is required to explore how the model performs under extreme noise conditions (e.g., SNRs below 0 dB) and other acoustic environments.
2.  **Generalization Gap:** During training, while training accuracy approached 90%, the validation accuracy stagnated and validation loss diverged after epoch 5 (refer to `Fig3_Training_Validation_Curves.png`). This indicates a segment-level overfitting / generalization gap. However, patient-level evaluation was used to provide a more clinically relevant estimate of triage performance.
3.  **Holdout Specificity:** Calibrating the decision boundary via Youden's Index on validation data shifted the optimal threshold to 0.1380. This successfully maximized patient-level sensitivity to 100.00% on the holdout test set, but decreased specificity to 26.32%. While this trade-off is optimal for triage pre-screening to prevent missing infectious patients, future work must focus on architectural improvements to raise holdout specificity.


---

## 5. Conclusion
In this study, **RespiScan** has been successfully revised into a rigorous, clinically oriented pre-screening framework for Tuberculosis utilizing cough acoustics. By rectifying the critical feature mismatch bias, enforcing strict subject-level splitting, and introducing real-world acoustic noise augmentations, we established a robust baseline. 

Furthermore, applying Youden's Index optimization to the patient-level ROC curve on validation data allowed the definition of a fixed optimal clinical decision boundary at **0.1380**. Applying this boundary to the unseen holdout test set boosted screening sensitivity to **100.00%** (95% CI: 91.24%–100.00%) and overall patient accuracy to **76.27%** (95% CI: 64.02%–85.31%), while maintaining a specificity of **26.32%** (95% CI: 11.81%–48.79%). High-dimensional feature mapping via t-SNE qualitatively showed observable clustering tendencies. With a tiny footprint of 11.7 MB, RespiScan remains a preliminary edge-AI triage-support prototype and requires external prospective validation before clinical deployment in low-resource environments.

---

## Declarations

### Acknowledgments
The authors would like to thank the Department of Computer Engineering, Universitas Dinamika, for academic support and server resources during the development of this research prototype.

### Data Availability
The data used in this study are derived from the publicly available Coswara and TBscreen databases. The preprocessed CWT scalogram database and code are available from the corresponding author upon reasonable request, subject to ethical agreements.

### Conflict of Interest
The authors declare no conflict of interest regarding the publication of this manuscript.

### Author Contributions
G.E.F. contributed to conceptualization, preprocessing, modeling, evaluation, and manuscript drafting. H.P. contributed to supervision, methodology review, and manuscript revision.
