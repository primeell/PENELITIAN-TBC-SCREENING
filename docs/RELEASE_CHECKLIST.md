# Release checklist for RespiScan Mobile

## Why use GitHub Releases?

An APK is a distributable binary, not source code. GitHub Releases provides a stable download page, release notes, and integrity information without inflating the repository history. This is particularly important because the universal APK exceeds GitHub’s normal 100 MB source-file limit.

## Recommended first release

| Field | Suggested value |
|---|---|
| Tag | `v2.3.0-cough-gate` |
| Release title | `RespiScan Mobile v2.3.0 — Cough Gate` |
| Primary asset | `RespiScan-v2.3.0-CoughGate-arm64-v8a-debug.apk` |
| Optional asset | `RespiScan-v2.3.0-CoughGate-universal-debug.apk` |
| Release type | Pre-release / research build |

## Pre-publish checks

- [ ] Confirm Android app version: `2.3.0-respiscan-cough-gate`.
- [ ] Confirm the model asset is `tara_mobilenetv2_cw1p5_seed42.onnx`.
- [ ] Confirm Android threshold is `0.42408434`.
- [ ] Run the available unit tests/build checks on a machine with accepted Android SDK licences.
- [ ] Install once on a representative ARM64 phone and verify microphone permission, recording, quality rejection, and result screen.
- [ ] Include the SHA-256 of each uploaded APK in the release notes.
- [ ] State clearly: research build, triage support only, not diagnosis or clinical deployment.

## Suggested release notes

```markdown
## RespiScan Mobile v2.3.0 — Cough Gate

Research-stage Android build for cough-audio TB triage experiments.

### Included
- locked MobileNetV2 ONNX model (CW 1.5:1.0, seed 42);
- validation-derived threshold 0.42408434;
- multi-cough arithmetic averaging;
- pre-inference recording-quality gate;
- TARA user interface.

### Important
This application is for research and technical demonstration only. It is not a diagnostic medical device and does not replace clinical assessment.
```

## Do not publish

- `.env`, Wi-Fi credentials, broker passwords, private API keys, or database files;
- raw participant audio or identifiable metadata;
- unreviewed checkpoints that conflict with the locked final configuration;
- historical figures labelled as final results.
