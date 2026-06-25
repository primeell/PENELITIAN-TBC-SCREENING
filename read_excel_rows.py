import pandas as pd

file_path = r"c:\TBC SCREENING PENELITIAN\PENELITIAN-TBC-SCREENING\dataset_respiscan_local\Dataset_key.xlsx"
df = pd.read_excel(file_path)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 1000)

print("Rows 0 to 20:")
print(df.iloc[0:20])

print("\nValue counts of Unnamed: 3 (which seems to contain coding keys):")
print(df['Keys'].value_counts())
