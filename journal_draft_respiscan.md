# RespiScan: Tuberculosis Triage Support from Cough Acoustics Using CWT Scalograms and Focal Loss MobileNetV2

## Abstract
Tuberculosis (TB) remains a major global health threat, demanding accessible, rapid, and low-cost pre-screening tools in low-resource settings. Computer-aided screening via cough acoustics offers a promising non-invasive alternative to conventional sputum smear microscopy and GeneXpert assays. However, clinical translation is often hindered by critical methodological flaws in existing literature, including speaker leakage, acoustic feature mismatch, and arbitrary decision boundaries. This study presents **RespiScan**, a lightweight deep learning screening system utilizing Continuous Wavelet Transform (CWT) scalograms and MobileNetV2. We systematically resolved the feature mismatch bias by standardizing all inputs to consistent complex Morlet (`cmor1.5-1.0`) CWT scalograms and enforcing strict subject-level dataset splitting to eliminate speaker leakage. 

To address environmental noise in outpatient clinics, training set samples were augmented with real-world noise (kitchen, park, hallway). To prevent test leakage, Youden’s Index ($J$) was applied to patient-level probability averaging on the validation set to optimize the classification decision threshold. Evaluating the final model on an independent holdout test set containing 31 completely unseen subjects (14 TB patients, 17 Non-TB controls) at the validation-optimized threshold of 0.3595 achieved a patient-level sensitivity of **100.00%** (95% CI: 78.47%–100.00%), a specificity of **76.47%** (95% CI: 52.74%–90.45%), and an overall classification accuracy of **87.10%** (95% CI: 71.15%–94.87%), with a patient-level Area Under the ROC (AUROC) of **0.9454**. Patient-level evaluation showed sensitivity-prioritized screening performance, although the validation-derived thresholding indicates that the system remains a triage-support prototype requiring further optimization and external validation. The lightweight model footprint (11.7 MB, 2.2M parameters) makes RespiScan feasible for mobile edge computing in rural primary healthcare centers as a triage support tool.

**Keywords:** Clinical Pre-Screening; Continuous Wavelet Transform; Cough Acoustics; Edge AI; MobileNetV2; Tuberculosis.

---

## 1. Introduction
Tuberculosis (TB), caused by the infectious pathogen *Mycobacterium tuberculosis*, continues to be one of the leading infectious causes of death globally, accounting for over one million deaths annually. While diagnostic standards such as sputum smear microscopy, chest X-rays, and molecular assays (e.g., GeneXpert Omni/Ultra) offer high diagnostic accuracy, their implementation is severely constrained in low- and middle-income countries (LMICs). These constraints stem from high instrumentation costs, the need for specialized laboratory infrastructure, prolonged turnaround times, and the biohazard risks associated with sputum collection. Consequently, there is an urgent clinical demand for non-invasive, rapid, and highly sensitive pre-screening tools capable of triaging patients in rural primary healthcare centers before sending them for expensive confirmatory molecular testing.

In recent years, computer-aided respiratory sound analysis has emerged as a highly promising paradigm for affordable screening. Pathological changes in the respiratory tract caused by TB—such as mucosal inflammation, airway obstruction, parenchymal consolidation, and the hyperproduction of highly viscous mucus (exudate)—alter the biomechanics of cough production. A forced cough is a three-phase expiratory maneuver: the inspiratory phase, the compressive phase (involving glottal closure and elevated intrathoracic pressure), and the expiratory burst phase (where the glottis opens abruptly, generating high-velocity airflow). In a lung compromised by TB, the physical properties of the air-mucus interface, combined with tissue consolidation, generate unique acoustic anomalies, such as micro-turbulences, high-frequency crackles, and altered vocal tract resonances. 

While multiple artificial intelligence (AI) models have been proposed to identify these acoustic biomarkers, the literature is frequently undermined by critical methodological flaws that limit clinical translation:
1. **Speaker/Subject Leakage (Data Leakage):** A common error is splitting dataset segments randomly into training and validation sets. When multiple cough segments from the same patient are distributed across both sets, the deep learning model memorizes individual vocal signatures (speaker recognition) rather than learning generalized pathological biomarkers (disease classification).
2. **Feature Mismatch Bias:** Machine learning models are highly sensitive to processing discrepancies. In several historical datasets, control samples (Non-TB) and pathological samples (TB) are processed using different audio toolchains, sampling rates, or spectral representations (e.g., mixing Mel-spectrograms for one class and Continuous Wavelet Transform for another). The neural network easily minimizes loss by classifying the digital processing signatures rather than the biological features, leading to high validation scores but total failure during independent clinical testing.
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

To overcome this, the **Continuous Wavelet Transform (CWT)** has been adopted. By scaling and shifting a mother wavelet, CWT provides variable resolution: high temporal resolution at high frequencies (capturing rapid transients) and high frequency resolution at low frequencies (capturing steady resonances). Despite the mathematical elegance of CWT, past publications frequently mixed CWT with Mel-spectrogram features across classes or allowed speaker leakage, inflating reported metrics. RespiScan addresses these issues by combining strict subject-level splitting, uniform CWT processing, and clinical threshold tuning, establishing a new standard for rigorous acoustic screening.

---

## 3. Material and Methods

### 3.1. Dataset Source, Ethical Considerations, and Preprocessing Pipeline
This study utilizes a hybrid secondary dataset compiled from two prominent respiratory sound databases: the **TBscreen clinical cough database** and the **Coswara database**. Combining these datasets establishes a cohort with a wider clinical representation. 

The **TBscreen** clinical database in its raw condition (*raw cohort*) contributes a total of 149 clinical subjects divided into two cohorts: **Cohort A** (110 subjects, consisting of 103 active TB patients and 7 Non-TB controls) and **Cohort C** (39 subjects, representing symptomatic respiratory controls with other illnesses such as asthma, pneumonia, and COPD). During the acoustic Quality Control (QC) screening process, 27 subjects from TBscreen were excluded due to corrupted audio files, silent recordings, or dead channels (18 subjects from Cohort A—leaving 87 TB and 5 Non-TB, and 9 subjects from Cohort C—leaving 30 Non-TB). The final active cohort after QC consists of 122 subjects from TBscreen, where 92 subjects are from Cohort A (87 TB and 5 Non-TB) and 30 subjects are from Cohort C (consisting entirely of Non-TB). The control group in TBscreen consists of controls with other respiratory illnesses rather than healthy individuals. 

To complement the Non-TB controls with healthy subjects, 76 subjects were integrated from the **Coswara** database (specifically from the lung function screening cohort, **COSWARA_LFS**). All 76 Coswara subjects passed the QC process with zero exclusions. Consequently, combining these secondary databases yields a raw cohort of 225 subjects, which filters down to **198 active standardized subjects** after QC (122 from TBscreen and 76 from Coswara).

To ensure scientific and ethical integrity, this study utilized fully anonymized secondary cough audio recordings. No direct patient recruitment or clinical interventions were conducted by the authors. Clinical annotations and diagnostic ground-truths (confirmed via sputum smear microscopy and GeneXpert assays) followed the original database protocols. Because the dataset consists exclusively of de-identified secondary data, direct ethical review was waived, in compliance with institutional and national regulations regarding secondary data analysis.

To establish a clinically motivated pre-screening model, a strict audio standardization pipeline was implemented. All raw cough records were converted to a single-channel (mono) format, resampled to a consistent sampling frequency ($f_s = 16,000$ Hz), and cropped or padded to a uniform duration of exactly 1.0 second ($16,000$ samples) to prevent temporal length bias in downstream neural processing.

To ensure absolute methodological transparency and prevent data leakage, a strict subject-level splitting strategy was enforced. The curation pipeline and final data selection flow are detailed in Table 1 (Dataset Evolution) and Table 2 (Final Split Distribution).

**Table 1: Dataset Curation and Evolution Pipeline**
| Curation Stage | Subjects | Audio / Scalogram Samples | Key Filtering / Selection Criteria |
| :--- | :---: | :---: | :--- |
| **1. Raw Cohort (Hybrid)** | 225 | — | Combined database from 149 TBscreen clinical subjects and 76 Coswara healthy controls. |
| **2. Excluded (Quality Control)** | 27 | — | Excluded due to corrupted audio files, silent recordings, or dead channels. |
| **3. Clean Standardized Dataset** | **198** | **2740** | Resampled to 16 kHz, mono, cropped/padded to 1.0s non-overlapping. |
| **4. Final Training Partition** | 137 | 2312 | Clean data + real-world environmental noise mixing (10 dB SNR). |
| **5. Final Validation Partition** | 30 | 229 | Clean-only partition used exclusively for decision threshold tuning. |
| **6. Final Holdout Test Partition** | 31 | 199 | Clean-only independent holdout partition for patient evaluation. |

*Note on historical pilot datasets:* In preliminary pilot iterations of this research pipeline (reported in early manuscript drafts), a restricted subset of 136 unique subjects was utilized, where overlapping segment windowing (slicing recordings into multiple overlapping 1-second chunks) was applied to yield 4,258 visual CWT samples (~31 segments per subject). In this revised work, we expanded the cohort to the full 198 active subjects (a 4.3-fold increase in speaker diversity) and moved to a non-overlapping standardized 1-second representation (2740 samples) to ensure patient-level splits are genuinely generalized and free of segment duplication artifacts.

To prevent acoustic dominance from specific subjects who contributed hundreds of recordings, we enforced a strict sample capping policy of **10 randomly selected cough records per subject** (`MAX_SAMPLES_PER_SUBJ = 10`). This limits the clean audio subset to **1,267 clean files** (1,115 from TBscreen, 152 from Coswara). The relationship between these 1,267 clean files and the grand total of 2,740 scalograms is mathematically explained through subject-level partitioning and training set augmentations as follows:
* **Train Split** (137 subjects): Contributes **839 clean audio files** (522 files from 60 TB subjects, 317 files from 77 Non-TB subjects).
* **Validation Split** (30 subjects): Contributes **229 clean audio files** (127 files from 13 TB subjects, 102 files from 17 Non-TB subjects).
* **Test Split** (31 subjects): Contributes **199 clean audio files** (133 files from 14 TB subjects, 66 files from 17 Non-TB subjects).

To establish robust representations and address class imbalance, we applied data augmentation **exclusively to the Train Split**, leaving validation and test splits clean-only:
1. **Acoustic Noise Mixing (Both Classes)**: All 839 clean training files were mixed with clinic background noises (`OHALLWAY`, `NPARK`, `DKITCHEN`) at 10 dB SNR, yielding an additional **839 noisy training files**.
2. **Speed Perturbation (Non-TB Controls Only)**: To enrich control acoustics and prevent false positives, the 317 clean Non-TB training files were duplicated twice via speed-stretching ($0.9\times$ and $1.1\times$), yielding an additional **634 speed-perturbed training files**.

This increases the training partition to $839 \text{ (clean)} + 839 \text{ (noisy)} + 634 \text{ (speed)} = 2,312$ samples. Combined with the 229 clean validation samples and 199 clean test samples, the overall pipeline utilizes exactly **2,740 CWT scalograms** ($2,312 + 229 + 199 = 2,740$).

**Table 2: Subject and Scalogram Sample Distributions Across Final Dataset Splits**
| Split | Subjects (Total) | TB Subjects | Non-TB Subjects | Scalogram Samples (Total) | TB Samples | Non-TB Samples | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Train** | 137 | 60 | 77 | 2312 | 1044 | 1268 | Clean and Noisy-Augmented |
| **Validation** | 30 | 13 | 17 | 229 | 127 | 102 | Clean-Only Evaluation |
| **Test** | 31 | 14 | 17 | 199 | 133 | 66 | Clean-Only Holdout Test |

### 3.2. Mathematical Formulation of the Continuous Wavelet Transform (CWT)
Unlike Fourier-based methods, the Continuous Wavelet Transform decomposes a 1D continuous-time signal $x(t)$ by correlating it with a localized family of functions derived from a single "mother wavelet" $\psi(t)$ via scaling ($a$) and translation ($b$):

$$W(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \psi^* \left( \frac{t - b}{a} \right) dt$$

where $\psi^*(t)$ denotes the complex conjugate of the mother wavelet. In this study, we utilize the **complex Morlet wavelet** (`cmor1.5-1.0`), mathematically defined as:

$$\psi(t) = \frac{1}{\pi^{1/4}} e^{i \omega_0 t} e^{-\frac{t^2}{2}}$$

where $\omega_0$ represents the center frequency of the wavelet (set to 1.0, with a bandwidth parameter of 1.5). The complex Morlet wavelet is highly suited for pathological acoustic analysis due to its excellent capability to capture localized phase and amplitude variations in non-stationary signals. The scale parameter $a$ is mapped to a geometric range covering 64 discrete widths:

$$a_i = 10^{\log_{10}(a_{\min}) + i \cdot \frac{\log_{10}(a_{\max}) - \log_{10}(a_{\min})}{N-1}}, \quad i = 0, 1, \dots, 63$$

where $a_{\min} = 2$ and $a_{\max} = 186$. The magnitude of the resulting complex coefficients $|W(a, b)|$ is plotted as a 2D spatial representation—the CWT Scalogram—which serves as the input image for the deep neural network.

```
+------------------+     +-----------------------+     +-------------------------+
|  Raw Cough Wave  | --> | Audio Standardization | --> |  Complex Morlet CWT     |
|  (1D mono, 16k)  |     |  (1.0s Mono, 16kHz)   |     |  (cmor1.5-1.0, 64 scales|
+------------------+     +-----------------------+     +-------------------------+
                                                                    |
                                                                    v
+------------------+     +-----------------------+     +-------------------------+
| Optimized Target | <-- | Subject-Level Split   | <-- |   2D Visual Scalogram   |
| (Train/Val/Test) |     |  (Prevent Leakage)    |     |   (Save as 224x224 PNG) |
+------------------+     +-----------------------+     +-------------------------+
```

### 3.3. Noise Augmentation and Speed Perturbation Protocols
To train the model to be robust against outpatient clinic background noises, we apply environmental noise mixing exclusively to the training set (**Train split**). Environmental noise mixing was applied to both TB and Non-TB training samples only, while validation and test sets remained clean. We scale and mix the clean cough signal $s(t)$ with real-world noise signals $n(t)$ (DKITCHEN, NPARK, and OHALLWAY) at a controlled Signal-to-Noise Ratio (SNR) of 10 dB:

$$x_{\text{noisy}}(t) = s(t) + \alpha \cdot n(t)$$

$$\alpha = \sqrt{ \frac{\sum_{t} s(t)^2}{\text{SNR}_{\text{linear}} \cdot \sum_{t} n(t)^2} }$$

where $\text{SNR}_{\text{linear}} = 10^{\frac{\text{SNR}_{\text{dB}}}{10}} = 10$. Noise mixing is restricted entirely to the training split, ensuring that the validation and test sets remain clean-only for clinical validation.

Additionally, to enrich the acoustic variability of the Non-TB controls and prevent model bias on clinic-specific acoustic signatures, speed perturbation (time-stretching at $0.9\times$ and $1.1\times$ factors) was applied exclusively to the Non-TB training subset. No acoustic perturbations were applied to the validation or test sets.

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
* A final output layer mapping to 2 classes (Non-TB, Tuberculosis).

### 3.5. Training Protocol and Weighted Focal Loss Function
The model is trained using a **Weighted Focal Loss** function. We assign a penalty weight ratio of 2:1 for misclassifications of the Non-TB controls class relative to the TB class. The penalty weight $w_{\text{Non-TB}} = 2.0$ is not used to balance a minority class (since Non-TB controls represent the majority with 77 subjects in the training set compared to 60 TB subjects), but as a decision regularization mechanism to suppress the model's tendency to produce false-positive TB classifications. The mathematical loss formulation is:

$$\mathcal{L}_{\text{FL}} = - \frac{1}{N_{\text{batch}}} \sum_{i=1}^{N_{\text{batch}}} \alpha_c (1 - p_{t,i})^\gamma \log(p_{t,i})$$

where $\alpha_c$ represents the class penalty weights ($w_{\text{Non-TB}} = 2.0$ and $w_{\text{TB}} = 1.0$), $p_{t,i}$ is the predicted softmax probability for the ground-truth class, and $\gamma = 2.0$ is the focusing parameter that down-weights the loss assigned to easy-to-classify samples, forcing the network to focus on hard samples. Assigning a higher penalty weight to the Non-TB class regularizes the decision boundary, reducing excessive false-positive classifications during training. Clinical sensitivity is subsequently optimized by calibrating the decision threshold via Youden's Index on the validation set. Optimization is performed using the Adam optimizer with differential learning rates: $10^{-5}$ for the backbone feature extraction layers to preserve general visual features, and $10^{-4}$ for the newly initialized classifier head. Training is carried out for 20 epochs with a learning rate scheduler (`ReduceLROnPlateau`, factor=0.5, patience=3) and an automatic early stopping mechanism (patience=7) monitored against validation patient-level AUROC.


---

## 4. Experimental Results and Discussion

### 4.1. Patient-Level Evaluation Framework
In clinical application, a screening decision is made per patient, not per audio segment. Thus, after segment-level probability extraction on the independent TEST set (consisting of 31 unique subjects with multiple cough recordings), we implement a **Patient-Level Probability Averaging** framework. For each unique patient $P_j$, the patient-level probability is computed as the arithmetic mean of the probabilities of all $K_j$ segments $s_{jk}$ belonging to that patient:

$$P_j = \frac{1}{K_j} \sum_{k=1}^{K_j} \hat{p}_{jk}$$

The patient-level classification prediction $\hat{Y}_{P_j}$ is then determined using the validation-derived Youden decision boundary threshold $\theta$:

$$\hat{Y}_{P_j} = \mathbb{I}(P_j \ge \theta)$$

where $\mathbb{I}(\cdot)$ is the indicator function, $\hat{p}_{jk}$ is the model's predicted probability of TB for segment $s_{jk}$, and $K_j$ is the total number of valid cough segments recorded for patient $P_j$. This approach is methodologically superior to hard majority voting because it operates on continuous probability densities, preserving the model's uncertainty prior to applying the optimized clinical threshold.

### 4.2. Diagnostic Performance Analysis (Before vs. After Optimization)
We compare distinct configurations to validate the technical enhancements of RespiScan. To prevent test leakage, the optimal Youden decision boundary was tuned exclusively on validation data, yielding a threshold of **0.3595**. A baseline configuration (representing the historical model compromised by feature mismatches and speaker leakage) was evaluated narrative-only. The baseline is presented only as a preliminary configuration to illustrate methodological risks and is not treated as a fair performance benchmark. The proposed consistent CWT model was then evaluated on the independent holdout test set containing 31 completely unseen subjects (14 TB patients, 17 Non-TB controls) under two decision boundaries:
1. **Consistent CWT Model (Default Threshold = 0.50):** The proposed model trained under our strict subject-level splitting using uniform CWT scalograms, evaluated at the standard machine learning probability boundary of $0.50$.
2. **Consistent CWT Model (Optimized Threshold = 0.3595):** The proposed model evaluated at the validation-optimized Youden's Index decision boundary.

**Table 3: Detailed Patient-Level Holdout Test Metrics Across Models**

| Evaluation Metric | Consistent CWT (Default Thresh = 0.50) | Consistent CWT (Optimized Thresh = 0.3595) | Clinical Target / Significance |
| :--- | :---: | :---: | :--- |
| **Patient Accuracy** | 77.42%<br>(95% CI: 60.19%–88.61%) | **87.10%**<br>(95% CI: 71.15%–94.87%) | Overall diagnostic correctness on unseen holdout patients. |
| **Sensitivity (Recall)** | 50.00%<br>(95% CI: 26.80%–73.20%) | **100.00%**<br>(95% CI: 78.47%–100.00%) | TB screening detection rate (minimize False Negatives). |
| **Specificity** | **100.00%**<br>(95% CI: 81.57%–100.00%) | **76.47%**<br>(95% CI: 52.74%–90.45%) | Non-TB control rate (minimize unnecessary referrals). |
| **Positive Predictive Value (PPV)**| 100.00%<br>(95% CI: 64.57%–100.00%) | **77.78%**<br>(95% CI: 54.78%–91.00%) | Probability that a "Positive" test truly has TB. |
| **Negative Predictive Value (NPV)**| 70.83%<br>(95% CI: 50.83%–85.09%) | **100.00%**<br>(95% CI: 77.19%–100.00%) | Probability that a "Negative" test is truly Non-TB. |
| **Patient-level AUROC**<br>*(Threshold-Independent)*| 0.9454 | 0.9454 | Discriminative power across all thresholds (constant). |

#### Discussion of Specificity and Sensitivity Trade-Offs:
Enforcing strict subject-level splitting and uniform CWT scalograms ensures the model extracts generalizable biological features. Evaluating the consistent model at a default threshold of $0.50$ yielded a sensitivity of **50.00%** and a specificity of **100.00%**. For Tuberculosis pre-screening, where the primary objective is to identify highly infectious individuals, a sensitivity of 50.00% is clinically insufficient. By shifting the decision boundary to the validation-derived threshold of **0.3595**, patient-level screening sensitivity is boosted to **100.00%** (95% CI: 95% CI: 78.47%–100.00%), ensuring that only 0 active TB patients are missed. Shifting the threshold to 0.3595 also increases the overall classification accuracy from **77.42%** to **87.10%**. Specificity under this optimized threshold is **76.47%** (95% CI: 95% CI: 52.74%–90.45%). This threshold reflects a sensitivity-prioritized triage trade-off. However, the low specificity indicates a false-positive burden, and future model refinement is required before practical clinical deployment.

---

### 4.3. Clinical Decision Boundary Optimization via Youden's Index (Fig 1 & Fig 2)
In medical screening, a default classifier threshold of $\theta = 0.50$ is rarely optimal. Missing a tuberculosis case (False Negative) has devastating clinical and epidemiological consequences, as it allows the disease to propagate unchecked in the community. To optimize clinical utility, we construct the patient-level Receiver Operating Characteristic (ROC) curve by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR) across all possible decision boundaries $\theta \in [0, 1]$ on the validation set. Shifting the decision threshold from 0.50 is done to maximize sensitivity. 

To locate the optimal clinical threshold, we apply **Youden's Index ($J$)** to the validation set patient probabilities, which maximizes the sum of sensitivity and specificity:

$$J(\theta) = \text{Sensitivity}(\theta) + \text{Specificity}(\theta) - 1$$

$$J(\theta) = \text{TPR}(\theta) - \text{FPR}(\theta)$$

$$\theta_{\text{optimal}} = \arg\max_{\theta} J(\theta)$$

Applying this optimization to the validation set yields the mathematically optimal decision boundary at **$\theta_{\text{optimal}} = 0.3595$** (with a validation-level AUROC of **0.7019**). 

```
[Figure 1: Patient-Level ROC Curve on Independent Holdout Test Set]
The ROC curve demonstrates the discriminative power of the Focal Loss MobileNetV2 model at the patient level (AUROC = 0.9454). The red dot indicates the clinical decision threshold (0.3595) which was derived using Youden's Index optimization exclusively on the validation set. Applying this validation-derived threshold to the independent holdout test set yields a sensitivity of 100.00% and a specificity of 76.47%, preventing test leakage.
```

Locking this decision boundary and applying it to the independent test set (AUROC of **0.9454**) instantly shifts the patient-level screening **Sensitivity to 100.00%** (95% CI: 78.47%–100.00%) while maintaining a **Specificity of 76.47%** (95% CI: 52.74%–90.45%), resulting in an overall **Patient Accuracy of 87.10%** (95% CI: 71.15%–94.87%). The patient-level classification outcomes under this optimized threshold are shown as a heatmap in Figure 2.

```
[Figure 2: Heatmap of Patient-Level Confusion Matrix on Holdout Test Set]
Confusion matrix showing patient-level classification outcomes on the holdout test set using the fixed validation-derived decision threshold of 0.3595: 13 True Negatives, 4 False Positives, 0 False Negatives (100% sensitivity), and 14 True Positives.
```

This result indicates that validation-derived threshold calibration can improve sensitivity on the holdout test set, although the reduced specificity shows that further optimization is required before clinical deployment.

---

### 4.4. Feature Latent Space Explainability via t-SNE (Fig 3)
To provide scientific transparency and act as an Explainable AI (XAI) validation step, we extract the 1280-dimensional feature representations from the final global average pooling layer of MobileNetV2 for the independent holdout test set. We project these high-dimensional embeddings onto a 2D space using **t-distributed Stochastic Neighbor Embedding (t-SNE)**. t-SNE minimizes the divergence between joint probabilities of pairs of points in the high-dimensional space and low-dimensional space:

$$\text{KL}(P || Q) = \sum_{i \neq j} p_{j|i} \log \frac{p_{j|i}}{q_{j|i}}$$

where $p_{j|i}$ measures the similarity of point $x_i$ to $x_j$ in high-dimensional space, and $q_{j|i}$ measures their similarity in low-dimensional space. The t-SNE projection was configured with a perplexity of 30, an initial learning rate of 200, a random state of 42, and run for 1,000 iterations exclusively using features extracted from holdout test subjects.

```
[Figure 3: t-SNE Visualization of Latent Feature Projections on Holdout Test Set]
t-SNE scatter plot projecting the 1280-dimensional feature representations from the final global average pooling layer of MobileNetV2 onto a 2D space. Green circular markers represent Non-TB Controls, and red triangular markers represent Tuberculosis (TB) Patients, showing an observable clustering tendency with partial overlap.
```

The resulting t-SNE visualization demonstrates an observable clustering tendency with partial overlap between the `Non-TB Controls` (Green) and `Tuberculosis` patients (Red). Because t-SNE is highly sensitive to hyperparameter tuning and can sometimes exaggerate local structures or create artificial clusters, this visualization was utilized strictly as qualitative support of feature distribution and not as primary quantitative evidence of diagnostic performance. Nevertheless, the observable clustering tendency supports the hypothesis that MobileNetV2 extracts consistent acoustic pathological biomarkers related to respiratory biomechanics rather than memorizing clinic-specific noise or digital artifacts.

---

### 4.5. Computational Feasibility and Study Limitations
MobileNetV2’s lightweight architecture yields highly efficient computational metrics, making it well-suited for point-of-care deployments:
*   **Total Parameter Count:** 2.2 Million parameters.
*   **Disk Footprint:** 11.7 MB (stored as standard float32 `.pth` weights).
*   **Feasibility Check:** The model size suggests potential edge feasibility; however, actual latency, memory consumption, and power usage on Raspberry Pi or smartphone hardware remain to be evaluated.

Despite these positive metrics, two key limitations must be addressed for academic and clinical rigor:
1.  **Noise Augmentation Generalization:** To ensure environmental robustness, we mixed real-world noise on the training set, maintaining a uniform acoustic environment across both classes. While this prevents the model from associating background noise signatures with a specific diagnostic class, further work is required to explore how the model performs under extreme noise conditions (e.g., SNRs below 0 dB) and other acoustic environments.
2.  **Generalization Gap:** During training, while training accuracy approached 90%, the validation accuracy stagnated and validation loss diverged after epoch 5 (refer to `Fig3_Training_Validation_Curves.png`). This indicates a segment-level overfitting / generalization gap. However, patient-level evaluation was used to provide a more clinically relevant estimate of triage performance.
3.  **Holdout Specificity:** Calibrating the decision boundary via Youden's Index on validation data shifted the optimal threshold to 0.3595. This successfully maximized patient-level sensitivity to 100.00% on the holdout test set, but decreased specificity to 76.47%. While this trade-off is optimal for triage pre-screening to prevent missing infectious patients, future work must focus on architectural improvements to raise holdout specificity.


---

## 5. Conclusion
In this study, **RespiScan** has been successfully revised into a rigorous, clinically oriented pre-screening framework for Tuberculosis utilizing cough acoustics. By rectifying the critical feature mismatch bias, enforcing strict subject-level splitting, and introducing real-world acoustic noise augmentations, we established a robust baseline. 

Furthermore, applying Youden's Index optimization to the patient-level ROC curve on validation data allowed the definition of a fixed optimal clinical decision boundary at **0.3595**. Applying this boundary to the unseen holdout test set boosted screening sensitivity to **100.00%** (95% CI: 78.47%–100.00%) and overall patient accuracy to **87.10%** (95% CI: 71.15%–94.87%), while maintaining a specificity of **76.47%** (95% CI: 52.74%–90.45%). High-dimensional feature mapping via t-SNE qualitatively showed observable clustering tendencies. With a tiny footprint of 11.7 MB, RespiScan remains a preliminary edge-AI triage-support prototype and requires external prospective validation before clinical deployment in low-resource environments.

---

## Declarations

### Acknowledgments
The authors would like to thank the Department of Computer Engineering, Universitas Dinamika, for academic support and server resources during the development of this research prototype.

### Data Availability
The raw acoustic datasets analyzed in this study are sourced from the **Project Coswara** database (maintained by the Indian Institute of Science, Bangalore) and the **TBscreen** database. Due to licensing terms and patient privacy restrictions, the raw audio recordings cannot be directly redistributed by the authors. Researchers can obtain access to the primary data by applying directly to the official repositories:
1. The Coswara dataset is publicly accessible through the Project Coswara GitHub repository (https://github.com/iiscleap/Coswara-Data).
2. The TBscreen dataset is available through the materials provided by the original TBscreen publication authors.

To ensure scientific reproducibility, the complete python source code for raw audio preprocessing, CWT scalogram extraction, model training pipelines, and the pre-trained neural network weights (`.pth` files) are made openly available in our repository at GitHub (https://github.com/username/RespiScan).

### Conflict of Interest
The authors declare no conflict of interest regarding the publication of this manuscript.

### Author Contributions
G.E.F. contributed to conceptualization, preprocessing, modeling, evaluation, and manuscript drafting. H.P. contributed to supervision, methodology review, and manuscript revision.
