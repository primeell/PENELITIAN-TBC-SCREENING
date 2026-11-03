# Reproducibility and audit guide

## Scope

This repository contains both development history and the final manuscript-revision evidence. Reproduction should begin from the locked configuration and its audit trail, rather than from the historical exploratory notebooks.

## Recommended reading order

1. [`RESEARCH_WORKFLOW.md`](RESEARCH_WORKFLOW.md) — final question, workflow, result, and limitations.
2. [`PAKET_REVISI_JURNAL_RESPISCAN/REPORT_TAHAP_1_Audit_Data_dan_Baseline.md`](../PAKET_REVISI_JURNAL_RESPISCAN/REPORT_TAHAP_1_Audit_Data_dan_Baseline.md) — data, split, preprocessing, augmentation, and baseline correction.
3. [`PAKET_REVISI_JURNAL_RESPISCAN/REPORT_TAHAP_2_Class_Weight_dan_Hasil_Final.md`](../PAKET_REVISI_JURNAL_RESPISCAN/REPORT_TAHAP_2_Class_Weight_dan_Hasil_Final.md) — validation-only selection and one-time internal-holdout evaluation.
4. [`PAKET_REVISI_JURNAL_RESPISCAN/REPORT_TAHAP_3_Statistik_Robustness_dan_Bias.md`](../PAKET_REVISI_JURNAL_RESPISCAN/REPORT_TAHAP_3_Statistik_Robustness_dan_Bias.md) — uncertainty, stability, source bias, and calibration diagnostics.
5. [`PAKET_REVISI_JURNAL_RESPISCAN/REPORT_TAHAP_4_Revisi_Manuskrip_dan_Reviewer.md`](../PAKET_REVISI_JURNAL_RESPISCAN/REPORT_TAHAP_4_Revisi_Manuskrip_dan_Reviewer.md) — manuscript and reviewer-response reconciliation.

## Locks that must not be silently changed

| Lock | Location |
|---|---|
| Subject split | `reviewer_revision_outputs/final_split_subject_ids.json` |
| Class-weight decision | `reviewer_revision_outputs/class_weight_selection_lock.json` |
| Selected configuration | `reviewer_revision_outputs/class_weight_ablation_final_config.json` |
| Model export and threshold parity | `RespiScanMobile/TARA_MODEL_EXPORT_VERIFICATION.json` |
| Android model contract | `RespiScanMobile/app/src/main/java/com/example/respiscanmobile/OnnxClassifier.kt` |

## Core checks

The audit artifacts include:

- patient-disjoint split and subject manifests;
- final configuration, checkpoint, and threshold locks;
- validation and test patient-level predictions;
- training histories and ROC data;
- class-weight comparison and selection proof;
- bootstrap confidence intervals;
- multi-seed validation stability;
- source-domain, calibration, false-positive, and false-negative analyses.

Use the verification scripts under `reviewer_revision_outputs/` only after placing the associated local data and environment in the expected locations. Raw audio and several datasets are intentionally excluded from Git because of size, provenance, and privacy/licensing considerations.

## Environment

The historical Python environment is recorded in:

```text
reviewer_revision_outputs/environment_versions.txt
```

The application/edge requirements are separate:

```text
RespiScanMobile/requirements.txt
```

## Interpretation rules

- Do not choose a configuration based on the internal-holdout test result.
- Do not reuse the historical threshold 0.3595.
- Do not combine segment-level and patient-level metrics.
- Do not treat calibration diagnostics as clinical risk calibration.
- Report the internal-holdout size and uncertainty alongside point estimates.
