import os

for root, dirs, files in os.walk(r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING"):
    if ".git" in root:
        continue
    for f in files:
        if f.endswith(('.py', '.ipynb')):
            path = os.path.join(root, f)
            try:
                with open(path, 'r', encoding='utf-8') as file:
                    content = file.read()
                if "extract_features_stable" in content:
                    print(f"Found in: {path}")
                    # Print context of occurrences
                    lines = content.splitlines()
                    for idx, line in enumerate(lines):
                        if "extract_features_stable" in line:
                            start = max(0, idx - 5)
                            end = min(len(lines), idx + 20)
                            print(f"\n--- Context in {f} lines {start}-{end} ---")
                            for i in range(start, end):
                                print(f"{i}: {lines[i]}")
            except Exception as e:
                print(f"Error reading {path}: {e}")
