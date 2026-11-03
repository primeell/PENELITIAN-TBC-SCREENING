# Repository map

## Purpose of this map

The project evolved from exploratory notebooks into a documented research and deployment package. This map separates final evidence from historical development files so reviewers can navigate the repository confidently.

## Primary paths

| Path | Use it for | Status |
|---|---|---|
| `README.md` | High-level overview, final results, and navigation | Start here |
| `docs/` | Curated reading paths and app/release guidance | Current |
| `PAKET_REVISI_JURNAL_RESPISCAN/` | Final manuscript-revision bundle and publication figures | Final evidence package |
| `reviewer_revision_outputs/` | Scripts, locks, predictions, plots, and audit reports | Technical evidence |
| `RespiScanMobile/` | Android app, TARA UI, Radxa edge node, ESP32, dashboard | Engineering implementation |
| `1_*.ipynb` … `5_*.ipynb` | Development and exploratory notebooks | Historical; do not cite as final output |

## Research files in the repository root

| File or group | Role |
|---|---|
| `run_pipeline_leak_free.py` | Subject-disjoint pipeline implementation |
| `prepare_kaggle_dataset.py` | Dataset preparation helper |
| `modelbuild.py` | Earlier model-building code |
| `Metadata.csv`, CSV files | Local metadata / development support; verify provenance before reuse |
| `Fig*.png` at root | Historical figures; do not use for the final paper or final-result claims |

## Files intentionally excluded from version control

Raw audio, derived datasets, large local backups, virtual environments, Android build directories, secrets, and local SQLite databases are not source code and should not be committed. The root `.gitignore` is the policy reference.

## Before creating a public commit

1. Confirm no raw audio, participant metadata, passwords, `.env` files, or local database is staged.
2. Use the four final figures from `PAKET_REVISI_JURNAL_RESPISCAN/`, not historical root-level figures.
3. Review `git status` carefully: this workspace contains existing research changes and generated artifacts.
4. Publish APKs as GitHub Release assets; avoid placing them in normal Git history.
5. Confirm that manuscript author, affiliation, and conflict-of-interest fields have been approved by all authors.
