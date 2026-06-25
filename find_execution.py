import json

notebook_path = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\1_Data_Standardization_and_Acoustic_Preprocessing.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for idx, cell in enumerate(nb['cells']):
    source = "".join(cell['source'])
    if "extract_features_stable(" in source and "def extract_features_stable(" not in source:
        print(f"Found call to extract_features_stable in Cell {idx}:")
        print(source)
    if "standardize_audio(" in source and "def standardize_audio(" not in source:
        print(f"Found call to standardize_audio in Cell {idx}:")
        print(source)
    if "train_test_split" in source:
        print(f"Found train_test_split in Cell {idx}:")
        print(source)
