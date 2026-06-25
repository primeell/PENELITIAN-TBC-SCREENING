import pandas as pd
import os

excel_path = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\metadata_databaru\TBscreen_Dataset\Dataset_key.xlsx"
csv_path = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\metadata_databaru\TBscreen_Dataset\Metadata.csv"

print("--- Checking Excel File ---")
if os.path.exists(excel_path):
    try:
        xls = pd.ExcelFile(excel_path)
        print("Sheet names:", xls.sheet_names)
        for sheet in xls.sheet_names:
            df = pd.read_excel(excel_path, sheet_name=sheet)
            print(f"\nSheet: {sheet}")
            print("Columns:", list(df.columns))
            print("Shape:", df.shape)
            print("First 3 rows:\n", df.head(3))
    except Exception as e:
        print("Error reading Excel:", e)
else:
    print("Excel file not found!")

print("\n--- Checking CSV Metadata ---")
if os.path.exists(csv_path):
    df_csv = pd.read_csv(csv_path)
    print("CSV Shape:", df_csv.shape)
    print("Class distribution:")
    print(df_csv['Label'].value_counts())
    print("\nFirst 5 subjects:\n", df_csv.head(5))
    print("\nLast 5 subjects:\n", df_csv.tail(5))
else:
    print("CSV file not found!")
