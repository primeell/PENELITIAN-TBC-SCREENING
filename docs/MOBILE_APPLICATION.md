# RespiScan Mobile application

## Purpose

**RespiScan v2.3.0 — Cough Gate** is the latest Android engineering build. It packages the locked ONNX model and provides an on-device, non-diagnostic triage demonstration.

It includes:

- the final `tara_mobilenetv2_cw1p5_seed42.onnx` model;
- validation-derived threshold `0.42408434` in float32;
- arithmetic averaging of valid cough-level scores;
- a pre-inference quality gate for weak, clipped, stationary, continuous, or obviously non-event recordings;
- the TARA mobile interface.

The quality gate is a signal-quality screen, not a clinically validated cough detector. Its exact validation scope and limitations are documented in [`RespiScanMobile/RESPISCAN_COUGH_GATE_VALIDATION.md`](../RespiScanMobile/RESPISCAN_COUGH_GATE_VALIDATION.md).

## Download for Android users

For a professional public download, publish the APK as a **GitHub Release asset**. Do not commit large APK binaries to the normal source tree.

Recommended release asset:

```text
RespiScan-v2.3.0-CoughGate-arm64-v8a-debug.apk
```

The ARM64 build is appropriate for most current Android phones. The universal APK is useful for compatibility testing but is too large for normal GitHub source commits and should also be attached only as a release asset when needed.

Once a release is published, users should:

1. Open the repository’s **Releases** page.
2. Download the `arm64-v8a` APK from the latest `v2.3.0-cough-gate` release.
3. Open the downloaded file on Android.
4. Permit installation from the browser or file manager when Android asks.
5. Open RespiScan and grant microphone permission.

> Android may warn because this is a research/debug build and is not distributed through Google Play. That warning does not turn the application into a clinical product; it remains research-stage software.

## Local integrity check

The verified universal APK hash is:

```text
SHA-256: D0A6598210559929A6C15F72E5673B1E318FF6F5D9607C76560385FF3833B949
```

On Windows, compare a downloaded file with the intended build:

```powershell
Get-FileHash -LiteralPath .\RespiScan-v2.3.0-CoughGate-universal-debug.apk -Algorithm SHA256
```

Only compare this hash with the **universal** APK, not the ARM64 APK.

## Build from source

Prerequisites:

- Android Studio with Android SDK Platform 36 and accepted SDK licences;
- JDK 17;
- an Android device running Android 7.0 / API 24 or later.

```bash
git clone https://github.com/primeell/PENELITIAN-TBC-SCREENING.git
cd PENELITIAN-TBC-SCREENING/RespiScanMobile
./gradlew assembleDebug
```

On Windows, use:

```powershell
.\gradlew.bat assembleDebug
```

Generated APKs are placed under `app/build/outputs/apk/debug/`. Do not copy the build directory into Git.

## Responsible use

- Use only for research, technical demonstration, or planned validation.
- Do not interpret the result as a diagnosis, treatment recommendation, or proof of absence of TB.
- Follow the in-app recording instructions and repeat the recording when the quality gate requests it.
- Do not change model, aggregation, or threshold without a new pre-specified validation protocol.
