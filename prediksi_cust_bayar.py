import pandas as pd
import numpy as np

print("1. Membaca Data...")
df_pay = pd.read_csv('Data/Data Payment OD(1).csv', low_memory=False)

# 2. CLEANSING & FORMATTING DATA
print("2. Melakukan Cleansing Data...")
# Mengubah Payment menjadi angka
if df_pay['PAYMENT'].dtype == 'object':
    df_pay['PAYMENT'] = df_pay['PAYMENT'].astype(str).str.replace(',', '').astype(float)

# Memastikan format tanggal benar
df_pay['INVOICE DATE'] = pd.to_datetime(df_pay['INVOICE DATE'], dayfirst=True, errors='coerce')
df_pay['SETTLEMENT DATE'] = pd.to_datetime(df_pay['SETTLEMENT DATE'], dayfirst=True, errors='coerce')

# Jika kolom Year kosong/bermasalah, ambil langsung dari INVOICE DATE
df_pay['Year'] = df_pay['INVOICE DATE'].dt.year


# 3. FILTERING SESUAI ATURAN MITRA
print("3. Memfilter Data (Mulai 2025, Payment >= 0)...")
# Abaikan data sebelum 2025 dan abaikan payment minus
df_filtered = df_pay[(df_pay['Year'] >= 2025) & (df_pay['PAYMENT'] >= 0)].copy()

# Abaikan juga baris yang belum ada tanggal settlement-nya (jika ada)
df_filtered = df_filtered.dropna(subset=['SETTLEMENT DATE'])


# 4. MENGHITUNG DURASI & BOBOT
print("4. Menghitung Proyeksi Durasi (Mingguan) dengan Bobot Nominal...")
# Durasi dalam Hari = Settlement Date - Invoice Date
df_filtered['Durasi_Hari'] = (df_filtered['SETTLEMENT DATE'] - df_filtered['INVOICE DATE']).dt.days

# Konversi Durasi Hari ke Minggu
df_filtered['Durasi_Minggu'] = df_filtered['Durasi_Hari'] / 7.0

# Membuat fitur Bobot (Durasi * Nominal Pembayaran)
df_filtered['Bobot_Durasi'] = df_filtered['Durasi_Minggu'] * df_filtered['PAYMENT']


# 5. AGREGASI PER CUSTOMER
# Mengelompokkan data berdasarkan Kode Cust
df_proyeksi = df_filtered.groupby('Kode Cust').agg(
    Jumlah_Invoice=('INVOICE NO', 'nunique'),
    Total_Payment=('PAYMENT', 'sum'),
    Total_Bobot_Durasi=('Bobot_Durasi', 'sum')
).reset_index()

# Menghitung Proyeksi Durasi Final (Minggu) = Total Bobot / Total Payment
df_proyeksi['Proyeksi_Lama_Bayar_Minggu'] = df_proyeksi['Total_Bobot_Durasi'] / df_proyeksi['Total_Payment']

# Dibulatkan agar enak dibaca (misal 2.45 minggu)
df_proyeksi['Proyeksi_Lama_Bayar_Minggu'] = df_proyeksi['Proyeksi_Lama_Bayar_Minggu'].round(2)

# Mengurutkan dari customer dengan nominal payment terbesar
df_proyeksi = df_proyeksi.sort_values(by='Total_Payment', ascending=False)

# Membuang kolom bantuan yang tidak perlu ditampilkan
df_proyeksi = df_proyeksi.drop(columns=['Total_Bobot_Durasi'])


# 6. MENAMPILKAN & MENYIMPAN HASIL
print("\n--- 5 CUSTOMER DENGAN TOTAL PAYMENT TERBESAR ---")
print(df_proyeksi.head(5).to_string(index=False))

# Ekspor ke CSV
nama_file = 'Data/Proyeksi_Durasi_Per_Customer.csv'
df_proyeksi.to_csv(nama_file, index=False, sep=',')
print(f"\nSelesai! File berhasil disimpan di '{nama_file}'")