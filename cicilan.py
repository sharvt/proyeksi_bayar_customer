import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

print("1. Membaca dan Memproses Data...")
df_pay = pd.read_csv('Data/Data Payment OD.csv', low_memory=False)
df_ots = pd.read_csv('Data/Data Ots OD.csv', low_memory=False)

# Standarisasi tipe data angka
if df_ots['PIUTANG'].dtype == 'object':
    df_ots['PIUTANG'] = df_ots['PIUTANG'].astype(str).str.replace(',', '').astype(float)

# ==========================================
# FILTER PIUTANG MINUS
# ==========================================
# Menyaring data agar hanya mengambil piutang yang nilainya 0 atau lebih (positif)
df_ots = df_ots[df_ots['PIUTANG'] >= 0]

if df_pay['PAYMENT'].dtype == 'object':
    df_pay['PAYMENT'] = df_pay['PAYMENT'].astype(str).str.replace(',', '').astype(float)

# Pastikan urutan tanggal pembayaran benar (agar cicilan 1, 2, 3 akurat)
df_pay['SETTLEMENT DATE'] = pd.to_datetime(df_pay['SETTLEMENT DATE'], format='%d-%m-%Y', errors='coerce')
df_pay = df_pay.sort_values(by=['INVOICE NO', 'SETTLEMENT DATE'])

print("2. Merestrukturisasi Data (Pivoting ke Bentuk Sekuensial)...")
df_pay['Cicilan_Ke'] = df_pay.groupby('INVOICE NO').cumcount() + 1
MAX_CICILAN = 5
df_pay_filtered = df_pay[df_pay['Cicilan_Ke'] <= MAX_CICILAN].copy()

df_pivot = df_pay_filtered.pivot(index='INVOICE NO', columns='Cicilan_Ke', values='PAYMENT').fillna(0)
df_pivot.columns = [f'Bayar_{i}' for i in df_pivot.columns]

df_model = pd.merge(df_ots, df_pivot, left_on='NONOTA', right_on='INVOICE NO', how='inner')
df_jenis = df_pay[['INVOICE NO', 'Jenis', 'Kode Cust']].drop_duplicates(subset=['INVOICE NO'])
df_model = pd.merge(df_model, df_jenis, left_on='NONOTA', right_on='INVOICE NO')

# Jadikan Nomor Invoice sebagai Index agar bisa dilacak di hasil Test
df_model.set_index('INVOICE NO', inplace=True)

print("3. Menghitung Target Proporsi (Persentase)...")
target_cols = []
for i in range(1, MAX_CICILAN + 1):
    col_name = f'Bayar_{i}'
    pct_col_name = f'Pct_Bayar_{i}'
    target_cols.append(pct_col_name)
    
    if col_name not in df_model.columns:
        df_model[col_name] = 0
        
    df_model[pct_col_name] = df_model[col_name] / df_model['PIUTANG']

le = LabelEncoder()
df_model['Jenis_Encoded'] = le.fit_transform(df_model['Jenis'].astype(str))

print("4. Melakukan Train-Test Split & Melatih Model...")
X = df_model[['PIUTANG', 'Jenis_Encoded']]
Y = df_model[target_cols]
Y = Y.replace([np.inf, -np.inf], 0).fillna(0)

# Split 80:20
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_nominal = MultiOutputRegressor(model_rf)
model_nominal.fit(X_train, Y_train)


print("\n=======================================================")
print("5. HASIL PENGUJIAN PADA DATA TEST (Asli vs Prediksi)")
print("=======================================================")

# Tebak persentase di data test
Y_pred_pct = model_nominal.predict(X_test)

# Bersihkan prediksi persentase (hapus < 1% dan normalisasi jadi 100%)
Y_pred_pct = np.where(Y_pred_pct < 0.01, 0, Y_pred_pct)
row_sums = Y_pred_pct.sum(axis=1, keepdims=True)
Y_pred_pct = np.divide(Y_pred_pct, row_sums, out=np.zeros_like(Y_pred_pct), where=row_sums!=0)

# Jika model menebak 0 untuk semua kolom, asumsikan lunas 1x di termin pertama
for i in range(len(Y_pred_pct)):
    if row_sums[i] == 0:
        Y_pred_pct[i, 0] = 1.0

# Kembalikan persentase menjadi nominal uang (Kalikan dengan PIUTANG test)
piutang_test = X_test['PIUTANG'].values
pred_nominals = Y_pred_pct * piutang_test[:, np.newaxis]
actual_nominals = Y_test.values * piutang_test[:, np.newaxis]

# Tampilkan 5 data pertama secara rapi
invoice_list = X_test.index.tolist()

for i in range(min(5, len(invoice_list))): # Menampilkan max 5 contoh
    inv_no = invoice_list[i]
    total_piutang = piutang_test[i]
    
    print(f"\n[Invoice: {inv_no}] | Total Piutang: Rp {total_piutang:,.0f}")
    print("-" * 65)
    print(f"{'Termin':<10} | {'Aktual (Kenyataan)':<22} | {'Prediksi Model':<22}")
    print("-" * 65)
    
    for termin in range(MAX_CICILAN):
        aktual_val = actual_nominals[i, termin]
        pred_val = pred_nominals[i, termin]
        
        # Hanya tampilkan baris jika aktual atau prediksinya ada nilainya (>0)
        if aktual_val > 0 or pred_val > 0:
            print(f"Bayar {termin+1:<4} | Rp {aktual_val:>18,.0f} | Rp {pred_val:>18,.0f}")

print("\n=======================================================")
print("6. SKOR EVALUASI MODEL KESELURUHAN")
print("=======================================================")

# 1. Menghitung MAE dan RMSE untuk Nominal Uang (Rupiah)
# Flatten matriks agar dievaluasi secara global (semua termin sekaligus)
mae_nominal = mean_absolute_error(actual_nominals.flatten(), pred_nominals.flatten())
rmse_nominal = np.sqrt(mean_squared_error(actual_nominals.flatten(), pred_nominals.flatten()))

print(f"Mean Absolute Error (MAE)  : Meleset rata-rata Rp {mae_nominal:,.0f} per termin")
print(f"Root Mean Squared Error    : Rp {rmse_nominal:,.0f} (Penalti error besar)")

# 2. Menghitung Akurasi Frekuensi (Jumlah Termin)
# Menghitung berapa termin yang nominalnya > 0
aktual_freq = (actual_nominals > 0).sum(axis=1)
pred_freq = (pred_nominals > 0).sum(axis=1)

# Menghitung selisih jumlah termin (Kenyataan vs Prediksi)
selisih_termin = np.abs(aktual_freq - pred_freq)
akurasi_sempurna = (selisih_termin == 0).mean() * 100
meleset_1_termin = (selisih_termin <= 1).mean() * 100

print("\nSkor Ketepatan Jumlah Cicilan:")
print(f"Tepat 100% Sesuai Kenyataan : {akurasi_sempurna:.1f}% dari total tagihan")
print(f"Toleransi Meleset 1 Termin  : {meleset_1_termin:.1f}% dari total tagihan")


print("\n=======================================================")
print("7. EKSPOR HASIL PENGUJIAN KE CSV (Dengan Persentase Simpangan)")
print("=======================================================")

# 1. Menyiapkan List untuk menampung data baris demi baris
export_data = []

for i in range(len(invoice_list)):
    inv_no = invoice_list[i]
    total_piutang = piutang_test[i]
    
    # Membuat dictionary untuk setiap baris invoice
    row = {
        'No_Invoice': inv_no,
        'Total_Piutang': total_piutang,
        'Jumlah_Termin_Aktual': aktual_freq[i],
        'Jumlah_Termin_Prediksi': pred_freq[i]
    }
    
    total_error_invoice = 0
    
    # Menambahkan detail nominal per termin sekaligus menghitung selisih errornya
    for termin in range(MAX_CICILAN):
        aktual_val = actual_nominals[i, termin]
        pred_val = pred_nominals[i, termin]
        
        row[f'Aktual_Bayar_{termin+1}'] = aktual_val
        row[f'Prediksi_Bayar_{termin+1}'] = pred_val
        
        # Menghitung selisih absolut untuk termin ini
        total_error_invoice += abs(aktual_val - pred_val)
    
    # Menghitung nominal yang "salah kamar" (dibagi 2 agar tidak double counting)
    nominal_salah_prediksi = total_error_invoice / 2
    
    # Menghitung Persentase Simpangan
    if total_piutang > 0:
        persentase_simpangan = (nominal_salah_prediksi / total_piutang) * 100
    else:
        persentase_simpangan = 0
        
    row['Total_Nominal_Meleset'] = nominal_salah_prediksi
    row['Persentase_Simpangan (%)'] = np.round(persentase_simpangan, 2)
    
    export_data.append(row)

# 2. Mengubah list menjadi DataFrame
df_export = pd.DataFrame(export_data)

# 3. Ekspor ke CSV
# Menggunakan sep=',' untuk format pemisah CSV internasional
nama_file = 'Hasil_Uji_Prediksi_Cicilan_2.csv'
df_export.to_csv(nama_file, index=False, sep=',')

print(f"Selesai! Hasil pengujian telah disimpan dalam file: {nama_file}")
print("Anda bisa membuka file ini di Excel. Kolom paling kanan menunjukkan seberapa besar persentase errornya.")