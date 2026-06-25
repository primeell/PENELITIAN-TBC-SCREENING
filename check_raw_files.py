import os

base_dirs = {
    "Nairobi_New_Forced": r'c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\data_baru_8000\TBscreen_Dataset\Forced_coughs\Audio_files',
    "Nairobi_New_Passive": r'c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\data_baru_8000\TBscreen_Dataset\Passive_coughs\Audio_files',
    "Nairobi_Old": r'c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\dataset_tbc',
    "NTB_Bulk": r'c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\ntb_bulk_10k\TBscreen_Dataset\Passive_coughs\Audio_files'
}

noise_dirs = [
    r'c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\DKITCHEN',
    r'c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\DLIVING',
    r'c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\NPARK',
    r'c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\OHALLWAY',
    r'c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\OOFFICE',
    r'c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\STRAFFIC'
]

print("--- Checking Raw Audio Directories ---")
for name, path in base_dirs.items():
    exists = os.path.exists(path)
    print(f"{name}: {'Exists' if exists else 'Does NOT exist'}")
    if exists:
        try:
            files = os.listdir(path)
            print(f"  Count: {len(files)} files")
            print(f"  Examples: {files[:3]}")
        except Exception as e:
            print(f"  Error listing: {e}")

print("\n--- Checking Noise Directories ---")
for path in noise_dirs:
    exists = os.path.exists(path)
    name = os.path.basename(path)
    print(f"{name}: {'Exists' if exists else 'Does NOT exist'}")
    if exists:
        try:
            files = os.listdir(path)
            print(f"  Count: {len(files)} files")
        except Exception as e:
            print(f"  Error listing: {e}")
