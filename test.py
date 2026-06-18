import pandas as pd
df_raw = pd.read_csv('Data/Data Ots OD(1).csv', encoding='latin1', header=None)
header_row = None
for i, row in df_raw.iterrows():
    if row.astype(str).str.strip().str.upper().isin(['PIUTANG', 'NONOTA']).any():
        header_row = i
        break

df = pd.read_csv('Data/Data Ots OD(1).csv', encoding='latin1', skiprows=header_row)
df.columns = df.columns.str.strip()
print(df['DUE DATE'].head(10).tolist())