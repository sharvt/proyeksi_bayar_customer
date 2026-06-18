import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="SPIL Finance Dashboard", layout="wide", page_icon="🚢")
st.title("🚢 Dashboard Proyeksi & Peringatan Arus Kas (A/R)")
st.markdown("Memantau tenggat waktu pembayaran dan mendeteksi anomali pelunasan customer secara real-time.")

# ==========================================
# 2. FUNGSI LOAD DATA (Membaca File Asli & Kalkulasi Live)
# ==========================================
@st.cache_data
def load_data():
    try:
        # 1. Load Data Master
        df_ots = pd.read_csv('Data/Data Ots OD.csv', low_memory=False, encoding='latin1')
        df_pay = pd.read_csv('Data/Data Payment OD(1).csv', low_memory=False, encoding='latin1')
        
        # --- Format Tanggal pada Data Payment ---
        df_pay['INVOICE DATE'] = pd.to_datetime(df_pay['INVOICE DATE'], dayfirst=True, errors='coerce')
        df_pay['SETTLEMENT DATE'] = pd.to_datetime(df_pay['SETTLEMENT DATE'], dayfirst=True, errors='coerce')
        df_pay['TOP'] = pd.to_datetime(df_pay['TOP'], dayfirst=True, errors='coerce')
        df_pay['Year'] = df_pay['INVOICE DATE'].dt.year
        
        if df_pay['PAYMENT'].dtype == 'object':
            df_pay['PAYMENT'] = df_pay['PAYMENT'].astype(str).str.replace(',', '').astype(float)

        # ========================================================
        # PERBAIKAN A: KALKULASI PROYEKSI DARI PAYMENT OD (>= 2025)
        # ========================================================
        # Hanya ambil data tahun 2025 ke atas, yang sudah lunas, dan payment positif
        df_pay_proj = df_pay[(df_pay['Year'] >= 2025) & (df_pay['PAYMENT'] > 0)].copy()
        df_pay_proj = df_pay_proj.dropna(subset=['SETTLEMENT DATE'])
        
        # Hitung durasi mingguan dan bobot
        df_pay_proj['Durasi_Hari'] = (df_pay_proj['SETTLEMENT DATE'] - df_pay_proj['INVOICE DATE']).dt.days
        df_pay_proj['Durasi_Minggu'] = df_pay_proj['Durasi_Hari'] / 7.0
        df_pay_proj['Bobot_Durasi'] = df_pay_proj['Durasi_Minggu'] * df_pay_proj['PAYMENT']
        
        # Agregasi untuk mendapatkan Proyeksi Mingguan per Customer
        df_proyeksi = df_pay_proj.groupby('Kode Cust').agg(
            Total_Payment=('PAYMENT', 'sum'),
            Total_Bobot=('Bobot_Durasi', 'sum')
        ).reset_index()
        df_proyeksi['Proyeksi_Lama_Bayar_Minggu'] = (df_proyeksi['Total_Bobot'] / df_proyeksi['Total_Payment']).round(2)


        # ========================================================
        # PERBAIKAN B: INVOICE MENUNGGAK DARI OTS OD (YANG BELUM ADA PAYMENT)
        # ========================================================
        # Cleansing Data Ots (Mencari Piutang Aktif)
        if df_ots['PIUTANG'].dtype == 'object':
            df_ots['PIUTANG'] = df_ots['PIUTANG'].astype(str).str.replace(',', '').astype(float)
        
        # Hanya ambil yang masih menunggak (Piutang > 0)
        df_ots = df_ots[df_ots['PIUTANG'] > 0]
        df_ots_clean = df_ots[['NONOTA', 'PIUTANG']]
        
        # Mengambil Info Tanggal dan Customer dari Data Payment
        df_pay_info = df_pay[['INVOICE NO', 'Kode Cust', 'INVOICE DATE', 'TOP', 'Year']].drop_duplicates(subset=['INVOICE NO'])
        
        # Menggunakan LEFT JOIN agar Ots OD yang belum pernah ada riwayat di Payment OD tidak hilang
        df_invoice = pd.merge(df_ots_clean, df_pay_info, left_on='NONOTA', right_on='INVOICE NO', how='left')
        
        # PERBAIKAN C: Hanya pakai Invoice dari tahun 2025 ke atas
        df_invoice = df_invoice[df_invoice['Year'] >= 2025]
        
        return df_proyeksi, df_invoice

    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e}. Pastikan folder 'Data' tersedia.")
        return pd.DataFrame(), pd.DataFrame()

df_proyeksi, df_invoice = load_data()

# ==========================================
# 3. PROSES LOGIKA PERINGATAN (WARNING SYSTEM)
# ==========================================
if not df_proyeksi.empty and not df_invoice.empty:
    
    # Menggabungkan Invoice Outstanding dengan Proyeksi Lama Bayarnya
    df_gabungan = pd.merge(df_invoice, df_proyeksi[['Kode Cust', 'Proyeksi_Lama_Bayar_Minggu']], on='Kode Cust', how='left')

    # Waktu Real-Time
    hari_ini = pd.Timestamp.now().normalize()
    
    # Hitung umur invoice dari tanggal cetak sampai hari ini
    df_gabungan['Umur_Invoice_Hari'] = (hari_ini - df_gabungan['INVOICE DATE']).dt.days

    # Konversi ekspektasi durasi ke hari (Default 4 minggu jika customer baru/belum ada proyeksi)
    df_gabungan['Proyeksi_Lama_Bayar_Minggu'] = df_gabungan['Proyeksi_Lama_Bayar_Minggu'].fillna(4.0)
    df_gabungan['Ekspektasi_Bayar_Hari'] = df_gabungan['Proyeksi_Lama_Bayar_Minggu'] * 7

    # Peringatan Anomali
    df_gabungan['Status_Peringatan'] = df_gabungan.apply(
        lambda x: "⚠️ Anomali: Telat dari Kebiasaan" if x['Umur_Invoice_Hari'] > x['Ekspektasi_Bayar_Hari'] else "✅ Aman",
        axis=1
    )

    # ==========================================
    # 4. MEMBUAT FILTER TENGGAT WAKTU (DEADLINE) REAL-TIME
    # ==========================================
    batas_minggu_ini = hari_ini + pd.Timedelta(days=7)
    batas_minggu_depan = hari_ini + pd.Timedelta(days=14)

    df_lewat_top = df_gabungan[df_gabungan['TOP'] < hari_ini].sort_values(by='TOP')
    df_minggu_ini = df_gabungan[(df_gabungan['TOP'] >= hari_ini) & (df_gabungan['TOP'] <= batas_minggu_ini)].sort_values(by='TOP')
    df_minggu_depan = df_gabungan[(df_gabungan['TOP'] > batas_minggu_ini) & (df_gabungan['TOP'] <= batas_minggu_depan)].sort_values(by='TOP')
    df_anomali = df_gabungan[df_gabungan['Status_Peringatan'] == "⚠️ Anomali: Telat dari Kebiasaan"].sort_values(by='PIUTANG', ascending=False)


    # ==========================================
    # 5. TAMPILAN DASHBOARD STREAMLIT
    # ==========================================
    # --- Sidebar Filter ---
    st.sidebar.header("🔍 Filter Data")
    pilihan_cust = st.sidebar.multiselect("Cari Customer Spesifik:", options=df_gabungan['Kode Cust'].dropna().unique())
    
    if pilihan_cust:
        df_lewat_top = df_lewat_top[df_lewat_top['Kode Cust'].isin(pilihan_cust)]
        df_minggu_ini = df_minggu_ini[df_minggu_ini['Kode Cust'].isin(pilihan_cust)]
        df_minggu_depan = df_minggu_depan[df_minggu_depan['Kode Cust'].isin(pilihan_cust)]
        df_anomali = df_anomali[df_anomali['Kode Cust'].isin(pilihan_cust)]

    # --- Ringkasan Metrik (Top Kanan) ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Outstanding (Tahun 2025+)", f"Rp {df_gabungan['PIUTANG'].sum():,.0f}")
    col2.metric("Lewat Jatuh Tempo", f"{len(df_lewat_top)} Invoice", delta="- Segera Tagih!", delta_color="inverse")
    col3.metric("Jatuh Tempo Minggu Ini", f"{len(df_minggu_ini)} Invoice")
    col4.metric("Anomali Pembayaran (Red Flag)", f"{len(df_anomali)} Invoice")

    st.markdown("---")

    # --- Tab Layout ---
    tab1, tab2 = st.tabs(["📅 Tenggat Waktu (Jatuh Tempo)", "🚨 Peringatan Perilaku Customer"])

    # --- TAB 1: JADWAL JATUH TEMPO ---
    with tab1:
        kolom_tampil_1 = ['NONOTA', 'Kode Cust', 'PIUTANG', 'INVOICE DATE', 'TOP', 'Proyeksi_Lama_Bayar_Minggu']
        
        st.subheader("🚨 Sudah Lewat Jatuh Tempo (Menunggak)")
        if not df_lewat_top.empty:
            df_lewat_top['INVOICE DATE'] = df_lewat_top['INVOICE DATE'].dt.strftime('%d-%m-%Y')
            df_lewat_top['TOP'] = df_lewat_top['TOP'].dt.strftime('%d-%m-%Y')
            st.dataframe(df_lewat_top[kolom_tampil_1], use_container_width=True, hide_index=True)
        else:
            st.success("Tidak ada tagihan yang menunggak!")

        st.subheader("🔴 Jatuh Tempo Minggu Ini (0 - 7 Hari Kedepan)")
        if not df_minggu_ini.empty:
            df_minggu_ini['INVOICE DATE'] = df_minggu_ini['INVOICE DATE'].dt.strftime('%d-%m-%Y')
            df_minggu_ini['TOP'] = df_minggu_ini['TOP'].dt.strftime('%d-%m-%Y')
            st.dataframe(df_minggu_ini[kolom_tampil_1], use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada invoice yang jatuh tempo minggu ini.")
        
        st.subheader("🟠 Jatuh Tempo Minggu Depan (8 - 14 Hari Kedepan)")
        if not df_minggu_depan.empty:
            df_minggu_depan['INVOICE DATE'] = df_minggu_depan['INVOICE DATE'].dt.strftime('%d-%m-%Y')
            df_minggu_depan['TOP'] = df_minggu_depan['TOP'].dt.strftime('%d-%m-%Y')
            st.dataframe(df_minggu_depan[kolom_tampil_1], use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada invoice yang jatuh tempo minggu depan.")

    # --- TAB 2: PERINGATAN ANOMALI ---
    with tab2:
        st.subheader("⚠️ Customer yang Belum Bayar Padahal Biasanya Sudah Lunas")
        st.info("Tabel ini mendeteksi anomali. Invoice di bawah ini mungkin belum melewati batas Jatuh Tempo (TOP), tetapi berdasarkan histori historisnya, tagihan ini sudah melebihi **rata-rata ekspektasi durasi normal** mereka membayar.")
        
        kolom_tampil_2 = ['NONOTA', 'Kode Cust', 'PIUTANG', 'INVOICE DATE', 'TOP', 'Umur_Invoice_Hari', 'Ekspektasi_Bayar_Hari', 'Status_Peringatan']
        
        if not df_anomali.empty:
            df_anomali['INVOICE DATE'] = df_anomali['INVOICE DATE'].dt.strftime('%d-%m-%Y')
            df_anomali['TOP'] = df_anomali['TOP'].dt.strftime('%d-%m-%Y')
            
            # Tanpa style warna agar tidak terjadi error batasan cell saat jumlah anomali sangat besar
            st.dataframe(
                df_anomali[kolom_tampil_2],
                use_container_width=True, hide_index=True
            )
        else:
            st.success("Luar biasa! Semua customer membayar sesuai dengan ekspektasi waktu atau lebih cepat.")