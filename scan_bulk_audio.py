import os
import re

bulk_dir = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\bulk_audio"
missing_subjects = [
    'PID_4A', 'PID_113A', 'PID_133A', 'PID_136A', 'PID_17C', 'PID_183A', 'PID_192A', 'PID_198A', 
    'PID_199A_now_C', 'PID_207A', 'PID_218A', 'PID_21C', 'PID_224A', 'PID_22A', 'PID_23C', 'PID_24C', 
    'PID_25C', 'PID_32A_now_C', 'PID_35C', 'PID_59A', 'PID_5C', 'PID_60A', 'PID_64A', 'PID_68A', 
    'PID_6A', 'PID_70A', 'PID_71C', 'PID_76A', 'PID_78C'
]

print("Scanning bulk_audio for missing subjects...")
bulk_files_count = 0
found_missing = {}

for root, dirs, files in os.walk(bulk_dir):
    wav_files = [f for f in files if f.endswith('.wav')]
    bulk_files_count += len(wav_files)
    for f in wav_files:
        for ms in missing_subjects:
            # Check clean names (e.g. PID_32A for PID_32A_now_C)
            clean_ms = ms.replace('_now_C', '')
            if f.startswith(clean_ms):
                if ms not in found_missing:
                    found_missing[ms] = []
                found_missing[ms].append(os.path.join(root, f))

print(f"Total .wav files in bulk_audio: {bulk_files_count}")
print(f"Missing subjects found in bulk_audio: {len(found_missing)} out of {len(missing_subjects)}")
for ms, paths in found_missing.items():
    print(f" - {ms}: {len(paths)} files (e.g. {os.path.basename(paths[0])})")
