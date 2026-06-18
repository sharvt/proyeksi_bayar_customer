import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

print("1. Memproses dan Membersihkan Data...")
df_pay = pd.read_csv('Data/Data Payment OD.csv', low_memory=False)
df_ots = pd.read_csv('Data/Data Ots OD.csv', low_memory=False)

# Cleansing Data Ots OD (PIUTANG)
if df_ots['PIUTANG'].dtype == 'object':
    df_ots['PIUTANG'] = df_ots['PIUTANG'].astype(str).str.replace(',', '').astype(float)

# FILTER: Mengecualikan piutang yang nilainya minus
df_ots = df_ots[df_ots['PIUTANG'] >= 0]

# Cleansing Data Payment OD
df_pay['INVOICE DATE'] = pd.to_datetime(df_pay['INVOICE DATE'], dayfirst=True)
df_pay['TOP'] = pd.to_datetime(df_pay['TOP'], dayfirst=True)

# =========================================================================
# 2. AGREGASI DATA PAYMENT & MERGING DENGAN OTS OD
# =========================================================================
# Mengambil titik pelunasan terakhir (max TOP DAYS) dari setiap invoice
df_pay_agg = df_pay.groupby('INVOICE NO').agg({
    'Kode Cust': 'first',      
    'INVOICE DATE': 'first',   
    'TOP': 'first',            
    'Jenis': 'first',          
    'TOP DAYS': 'max'          # Ambil keterlambatan terlama (pelunasan)
}).reset_index()

# Melakukan Matching/Merge dengan Data Ots berdasarkan Nomor Nota
# Kita hanya mengambil kolom NONOTA dan PIUTANG dari Ots
df = pd.merge(df_pay_agg, df_ots[['NONOTA', 'PIUTANG']], left_on='INVOICE NO', right_on='NONOTA', how='inner')

# =========================================================================
# 3. FEATURE ENGINEERING & DEFINISI TARGET
# =========================================================================
y = df['TOP DAYS'] # Target: Hari Pelunasan Final

df['Durasi_Kredit_Hari'] = (df['TOP'] - df['INVOICE DATE']).dt.days
df['Invoice_Month'] = df['INVOICE DATE'].dt.month

le = LabelEncoder()
df['Jenis_Encoded'] = le.fit_transform(df['Jenis'].astype(str))

# Profil Customer (Seberapa sering mereka telat)
rata2_telat = df.groupby('Kode Cust')['TOP DAYS'].mean().to_dict()
df['Histori_Rata_Telat'] = df['Kode Cust'].map(rata2_telat)

# MENGGUNAKAN 'PIUTANG' SEBAGAI FITUR (Menggantikan PAYMENT)
fitur_yang_dipakai = ['Durasi_Kredit_Hari', 'PIUTANG', 'Invoice_Month', 'Jenis_Encoded', 'Histori_Rata_Telat']
X = df[fitur_yang_dipakai]

# =========================================================================
# 4. PELATIHAN MODEL & EVALUASI
# =========================================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nMelatih Model Random Forest Regressor...")
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

y_pred = model_rf.predict(X_test)

print("\n--- HASIL EVALUASI MODEL REGRESI ---")
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE) : {mae:.2f} Hari")
r2 = r2_score(y_test, y_pred)
print(f"R-Squared (R2)            : {r2:.2f}")


# =========================================================================
# 5. MENYIMPAN HASIL KE CSV
# =========================================================================
print("\nMenyimpan hasil ke CSV...")

hasil_df = X_test.copy()
hasil_df['Kode Cust'] = df.loc[X_test.index, 'Kode Cust'] 
hasil_df['INVOICE NO'] = df.loc[X_test.index, 'INVOICE NO']
hasil_df['Jatuh_Tempo (TOP)'] = df.loc[X_test.index, 'TOP']

hasil_df['Aktual_TOP_DAYS'] = y_test
hasil_df['Prediksi_TOP_DAYS'] = np.round(y_pred, 0).astype(int)

# Menghitung Prediksi SETTLEMENT DATE
hasil_df['Prediksi_Tanggal_Bayar'] = hasil_df['Jatuh_Tempo (TOP)'] + pd.to_timedelta(hasil_df['Prediksi_TOP_DAYS'], unit='d')

# Rapikan urutan kolomnya (Menampilkan 'PIUTANG' agar lebih informatif)
kolom_final = ['INVOICE NO', 'Kode Cust', 'PIUTANG', 'Jatuh_Tempo (TOP)', 'Aktual_TOP_DAYS', 'Prediksi_TOP_DAYS', 'Prediksi_Tanggal_Bayar']
hasil_df = hasil_df[kolom_final]

hasil_df.to_csv('Data/Hasil_Prediksi_Settlement_2.csv', index=False, sep=',')
print("Selesai! File berhasil disimpan di 'Data/Hasil_Prediksi_Settlement.csv'")