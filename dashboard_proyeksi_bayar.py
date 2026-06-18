import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(page_title="SPIL Finance Dashboard", layout="wide", page_icon="🚢")
st.title("🚢 Dashboard Proyeksi & Peringatan Arus Kas (A/R)")
st.markdown("Memantau tenggat waktu pembayaran dan mendeteksi anomali pelunasan customer secara real-time.")

# ==========================================
# 2. UPLOAD FILE
# ==========================================
with st.expander("📂 Upload Data", expanded=True):
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        file_ots = st.file_uploader("Upload **Data OTS OD** (.csv)", type="csv", key="ots")
        st.caption("Kolom wajib: NONOTA, Kode Cust 1, NOTA, DUE DATE, PIUTANG, Aging Group")
    with col_up2:
        file_pay = st.file_uploader("Upload **Data Payment OD** (.csv)", type="csv", key="pay")
        st.caption("Kolom wajib: INVOICE NO, Kode Cust, INVOICE DATE, TOP, SETTLEMENT DATE, PAYMENT")

if not file_ots or not file_pay:
    st.info("⬆️ Silakan upload kedua file di atas untuk menampilkan dashboard.")
    st.stop()

# ==========================================
# 3. FUNGSI LOAD DATA
# ==========================================
def read_ots(file) -> pd.DataFrame:
    """Baca OTS OD — deteksi otomatis header, strip spasi kolom."""
    raw = pd.read_csv(file, low_memory=False, encoding='latin1', header=None)
    # Cari baris header: baris pertama yang mengandung 'PIUTANG' atau 'NONOTA'
    header_row = None
    for i, row in raw.iterrows():
        if row.astype(str).str.strip().str.upper().isin(['PIUTANG', 'NONOTA']).any():
            header_row = i
            break
    if header_row is None:
        st.error("OTS OD: tidak dapat menemukan header. Pastikan file mengandung kolom 'PIUTANG' dan 'NONOTA'.")
        st.stop()
    # Reset pointer lalu baca ulang dengan skiprows
    file.seek(0)
    df = pd.read_csv(file, low_memory=False, encoding='latin1', skiprows=header_row)
    df.columns = df.columns.str.strip()
    return df

def read_pay(file) -> pd.DataFrame:
    """Baca Payment OD — header selalu di baris pertama."""
    df = pd.read_csv(file, low_memory=False, encoding='latin1')
    df.columns = df.columns.str.strip()
    return df

@st.cache_data
def load_data(ots_bytes: bytes, pay_bytes: bytes):
    import io
    try:
        df_ots = read_ots(io.BytesIO(ots_bytes))
        df_pay = read_pay(io.BytesIO(pay_bytes))

        # -----------------------------------------------
        # CLEANSING PAYMENT OD
        # -----------------------------------------------
        df_pay['INVOICE DATE']    = pd.to_datetime(df_pay['INVOICE DATE'],    dayfirst=True, errors='coerce')
        df_pay['SETTLEMENT DATE'] = pd.to_datetime(df_pay['SETTLEMENT DATE'], dayfirst=True, errors='coerce')
        # TOP di Payment OD formatnya campur: '/' (D/M/YYYY) dan '-' (DD-MM-YYYY)
        df_pay['TOP'] = df_pay['TOP'].astype(str).str.strip().str.replace('/', '-', regex=False)
        df_pay['TOP'] = pd.to_datetime(df_pay['TOP'], dayfirst=True, errors='coerce')
        df_pay['Year'] = df_pay['INVOICE DATE'].dt.year

        if df_pay['PAYMENT'].dtype == 'object':
            df_pay['PAYMENT'] = df_pay['PAYMENT'].astype(str).str.replace(',', '').astype(float)

        # -----------------------------------------------
        # KALKULASI PROYEKSI LAMA BAYAR PER CUSTOMER
        # Sumber: Payment OD tahun >= 2025 yang sudah lunas (ada SETTLEMENT DATE)
        # -----------------------------------------------
        df_pay_proj = df_pay[
            (df_pay['Year'] >= 2025) &
            (df_pay['PAYMENT'] > 0)
        ].copy()
        df_pay_proj = df_pay_proj.dropna(subset=['SETTLEMENT DATE'])

        df_pay_proj['Durasi_Hari']   = (df_pay_proj['SETTLEMENT DATE'] - df_pay_proj['INVOICE DATE']).dt.days
        df_pay_proj['Durasi_Minggu'] = df_pay_proj['Durasi_Hari'] / 7.0
        df_pay_proj['Bobot_Durasi']  = df_pay_proj['Durasi_Minggu'] * df_pay_proj['PAYMENT']

        df_proyeksi = df_pay_proj.groupby('Kode Cust').agg(
            Total_Payment=('PAYMENT',      'sum'),
            Total_Bobot  =('Bobot_Durasi', 'sum')
        ).reset_index()
        df_proyeksi['Proyeksi_Lama_Bayar_Minggu'] = (
            df_proyeksi['Total_Bobot'] / df_proyeksi['Total_Payment']
        ).round(2)

        # -----------------------------------------------
        # CLEANSING OTS OD
        # Hanya ambil nota yang masih punya sisa piutang > 0
        # -----------------------------------------------
        if df_ots['PIUTANG'].dtype == 'object':
            df_ots['PIUTANG'] = df_ots['PIUTANG'].astype(str).str.replace(',', '').astype(float)

        df_ots = df_ots[df_ots['PIUTANG'] > 0].copy()

        # NOTA dan DUE DATE di OTS keduanya MM/DD/YYYY
        df_ots['NOTA']     = df_ots['NOTA'].astype(str).str.strip().str.replace('/', '-', regex=False)
        df_ots['NOTA']     = pd.to_datetime(df_ots['NOTA'],     dayfirst=False, errors='coerce')
        df_ots['DUE DATE'] = df_ots['DUE DATE'].astype(str).str.strip().str.replace('/', '-', regex=False)
        df_ots['DUE DATE'] = pd.to_datetime(df_ots['DUE DATE'], dayfirst=False, errors='coerce')

        # Ambil kolom yang dibutuhkan dari OTS; gunakan "Kode Cust 1" sebagai kode customer
        df_ots_clean = df_ots[['NONOTA', 'Kode Cust 1', 'NOTA', 'DUE DATE', 'PIUTANG', 'Aging Group']].copy()
        df_ots_clean.rename(columns={
            'Kode Cust 1': 'Kode Cust',
            'NOTA':        'INVOICE DATE_ots',
            'DUE DATE':    'TOP_ots'
        }, inplace=True)

        # -----------------------------------------------
        # AMBIL INFO INVOICE DATE & TOP DARI PAYMENT OD
        # -----------------------------------------------
        df_pay_info = (
            df_pay[['INVOICE NO', 'INVOICE DATE', 'TOP', 'Year']]
            .drop_duplicates(subset=['INVOICE NO'])
        )

        # -----------------------------------------------
        # LEFT JOIN: semua nota di OTS tetap muncul
        # -----------------------------------------------
        df_invoice = pd.merge(
            df_ots_clean,
            df_pay_info,
            left_on='NONOTA', right_on='INVOICE NO',
            how='left'
        )

        # Fallback: gunakan data OTS jika Payment OD tidak punya info
        df_invoice['TOP']          = df_invoice['TOP'].combine_first(df_invoice['TOP_ots'])
        df_invoice['INVOICE DATE'] = df_invoice['INVOICE DATE'].combine_first(df_invoice['INVOICE DATE_ots'])
        df_invoice.drop(columns=['TOP_ots', 'INVOICE DATE_ots', 'INVOICE NO'], inplace=True)

        df_invoice['INVOICE DATE'] = pd.to_datetime(df_invoice['INVOICE DATE'], errors='coerce')
        df_invoice['TOP']          = pd.to_datetime(df_invoice['TOP'],          errors='coerce')

        df_invoice['Year'] = df_invoice['INVOICE DATE'].dt.year
        df_invoice = df_invoice[(df_invoice['Year'] >= 2025) | (df_invoice['Year'].isna())]

        return df_proyeksi, df_invoice

    except Exception as e:
        st.error(f"Error saat memproses data: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Baca bytes sekali, pass ke cached function
ots_bytes = file_ots.read()
pay_bytes = file_pay.read()
df_proyeksi, df_invoice = load_data(ots_bytes, pay_bytes)

# ==========================================
# 3. PROSES LOGIKA PERINGATAN (WARNING SYSTEM)
# ==========================================
if not df_invoice.empty:
    
    # Menggabungkan Invoice Outstanding dengan Proyeksi Lama Bayarnya
    df_gabungan = pd.merge(df_invoice, df_proyeksi[['Kode Cust', 'Proyeksi_Lama_Bayar_Minggu']], on='Kode Cust', how='left')

    # ⚠️ TESTING MODE: tanggal di-hardcode ke awal Februari untuk cek data OTS
    # Ganti kembali ke pd.Timestamp.now().normalize() saat production
    hari_ini = pd.Timestamp('03-02-2026').normalize()
    
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

    # Countdown: sisa hari menuju proyeksi bayar (negatif = sudah melewati proyeksi)
    # Contoh: +5 = "5 hari lagi menuju proyeksi", -3 = "3 hari melewati proyeksi"
    df_gabungan['Sisa_Hari_Proyeksi'] = (df_gabungan['Ekspektasi_Bayar_Hari'] - df_gabungan['Umur_Invoice_Hari']).astype(int)

    def format_countdown(sisa):
        if pd.isna(sisa):
            return '-'
        elif sisa > 0:
            return f"🟡 {sisa} hari lagi"
        elif sisa == 0:
            return f"🔴 Hari ini batas proyeksi!"
        else:
            return f"🔴 Lewat {abs(sisa)} hari"

    df_gabungan['Countdown_Proyeksi'] = df_gabungan['Sisa_Hari_Proyeksi'].apply(format_countdown)

    # ==========================================
    # 4. MEMBUAT FILTER TENGGAT WAKTU (DEADLINE) REAL-TIME
    # ==========================================
    batas_minggu_ini = hari_ini + pd.Timedelta(days=7)
    batas_minggu_depan = hari_ini + pd.Timedelta(days=14)

    df_lewat_top    = df_gabungan[df_gabungan['TOP'] < hari_ini].sort_values(by='TOP', ascending=True)
    df_minggu_ini   = df_gabungan[(df_gabungan['TOP'] >= hari_ini) & (df_gabungan['TOP'] <= batas_minggu_ini)].sort_values(by='TOP', ascending=True)
    df_minggu_depan = df_gabungan[(df_gabungan['TOP'] > batas_minggu_ini) & (df_gabungan['TOP'] <= batas_minggu_depan)].sort_values(by='TOP', ascending=True)
    df_anomali      = df_gabungan[df_gabungan['Status_Peringatan'] == "⚠️ Anomali: Telat dari Kebiasaan"].sort_values(by='TOP', ascending=True)


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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        f"🚨 Lewat Jatuh Tempo ({len(df_lewat_top)})",
        f"🔴 Minggu Ini ({len(df_minggu_ini)})",
        f"🟠 Minggu Depan ({len(df_minggu_depan)})",
        "⚠️ Peringatan Perilaku Customer",
        "🏷️ Kategori Customer",
        "📊 Visualisasi Overall"
    ])

    kolom_tampil_1 = ['NONOTA', 'Kode Cust', 'PIUTANG', 'INVOICE DATE', 'TOP', 'Proyeksi_Lama_Bayar_Minggu', 'Countdown_Proyeksi']

    # --- TAB 1: SUDAH LEWAT JATUH TEMPO ---
    with tab1:
        st.subheader("🚨 Sudah Lewat Jatuh Tempo (Menunggak)")
        if not df_lewat_top.empty:
            df_lewat_top['INVOICE DATE'] = df_lewat_top['INVOICE DATE'].dt.strftime('%d-%m-%Y')
            df_lewat_top['TOP'] = df_lewat_top['TOP'].dt.strftime('%d-%m-%Y')
            st.dataframe(df_lewat_top[kolom_tampil_1], use_container_width=True, hide_index=True)
        else:
            st.success("Tidak ada tagihan yang menunggak!")

    # --- TAB 2: JATUH TEMPO MINGGU INI ---
    with tab2:
        st.subheader("🔴 Jatuh Tempo Minggu Ini (0 - 7 Hari Kedepan)")
        if not df_minggu_ini.empty:
            df_minggu_ini['INVOICE DATE'] = df_minggu_ini['INVOICE DATE'].dt.strftime('%d-%m-%Y')
            df_minggu_ini['TOP'] = df_minggu_ini['TOP'].dt.strftime('%d-%m-%Y')
            st.dataframe(df_minggu_ini[kolom_tampil_1], use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada invoice yang jatuh tempo minggu ini.")

    # --- TAB 3: JATUH TEMPO MINGGU DEPAN ---
    with tab3:
        st.subheader("🟠 Jatuh Tempo Minggu Depan (8 - 14 Hari Kedepan)")
        if not df_minggu_depan.empty:
            df_minggu_depan['INVOICE DATE'] = df_minggu_depan['INVOICE DATE'].dt.strftime('%d-%m-%Y')
            df_minggu_depan['TOP'] = df_minggu_depan['TOP'].dt.strftime('%d-%m-%Y')
            st.dataframe(df_minggu_depan[kolom_tampil_1], use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada invoice yang jatuh tempo minggu depan.")

    # --- TAB 4: PERINGATAN ANOMALI ---
    with tab4:
        st.subheader("⚠️ Customer yang Belum Bayar Padahal Biasanya Sudah Lunas")
        st.info("Tabel ini mendeteksi anomali. Invoice di bawah ini mungkin belum melewati batas Jatuh Tempo (TOP), tetapi berdasarkan histori historisnya, tagihan ini sudah melebihi **rata-rata ekspektasi durasi normal** mereka membayar.")

        kolom_tampil_2 = ['NONOTA', 'Kode Cust', 'PIUTANG', 'INVOICE DATE', 'TOP', 'Umur_Invoice_Hari', 'Ekspektasi_Bayar_Hari', 'Countdown_Proyeksi', 'Status_Peringatan']

        if not df_anomali.empty:
            df_anomali['INVOICE DATE'] = df_anomali['INVOICE DATE'].dt.strftime('%d-%m-%Y')
            df_anomali['TOP'] = df_anomali['TOP'].dt.strftime('%d-%m-%Y')
            st.dataframe(
                df_anomali[kolom_tampil_2],
                use_container_width=True, hide_index=True
            )
        else:
            st.success("Luar biasa! Semua customer membayar sesuai dengan ekspektasi waktu atau lebih cepat.")

    # --- TAB 5: KATEGORI CUSTOMER BERDASARKAN HABIT ---
    with tab5:
        st.subheader("🏷️ Kategorisasi Customer Berdasarkan Kebiasaan Bayar")
        st.info("Kategori dihitung dari **rata-rata durasi bayar historis** (Payment OD) per customer.")

        if not df_proyeksi.empty:
            # Kategorisasi berdasarkan proyeksi lama bayar
            def kategorikan(minggu):
                if minggu <= 2:
                    return "🟢 Cepat (≤ 2 minggu)"
                elif minggu <= 4:
                    return "🔵 Normal (2–4 minggu)"
                elif minggu <= 6:
                    return "🟡 Lambat (4–6 minggu)"
                else:
                    return "🔴 Sangat Lambat (> 6 minggu)"

            df_cat = df_proyeksi.copy()
            df_cat['Kategori'] = df_cat['Proyeksi_Lama_Bayar_Minggu'].apply(kategorikan)
            df_cat = df_cat.sort_values('Proyeksi_Lama_Bayar_Minggu')

            # Ringkasan per kategori
            ringkasan_cat = df_cat.groupby('Kategori').agg(
                Jumlah_Customer=('Kode Cust', 'count'),
                Rata_Rata_Minggu=('Proyeksi_Lama_Bayar_Minggu', 'mean')
            ).reset_index()
            ringkasan_cat['Rata_Rata_Minggu'] = ringkasan_cat['Rata_Rata_Minggu'].round(2)

            c1, c2 = st.columns([1, 2])
            with c1:
                st.markdown("**Ringkasan Kategori**")
                st.dataframe(ringkasan_cat, use_container_width=True, hide_index=True)

            with c2:
                fig_pie = px.pie(
                    ringkasan_cat,
                    names='Kategori',
                    values='Jumlah_Customer',
                    title='Distribusi Customer per Kategori Bayar',
                    color='Kategori',
                    color_discrete_map={
                        '🟢 Cepat (≤ 2 minggu)':       '#2ecc71',
                        '🔵 Normal (2–4 minggu)':       '#3498db',
                        '🟡 Lambat (4–6 minggu)':       '#f1c40f',
                        '🔴 Sangat Lambat (> 6 minggu)':'#e74c3c',
                    }
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)

            st.markdown("---")
            st.markdown("**Detail per Customer**")

            # Filter kategori
            pilihan_cat = st.multiselect(
                "Filter Kategori:",
                options=df_cat['Kategori'].unique().tolist(),
                default=df_cat['Kategori'].unique().tolist(),
                key="filter_cat"
            )
            df_cat_filtered = df_cat[df_cat['Kategori'].isin(pilihan_cat)]

            # Bar chart: lama bayar per customer
            fig_bar = px.bar(
                df_cat_filtered.assign(**{'Kode Cust': df_cat_filtered['Kode Cust'].astype(str)})
                .sort_values('Proyeksi_Lama_Bayar_Minggu', ascending=False),
                x='Kode Cust',
                y='Proyeksi_Lama_Bayar_Minggu',
                color='Kategori',
                color_discrete_map={
                    '🟢 Cepat (≤ 2 minggu)':       '#2ecc71',
                    '🔵 Normal (2–4 minggu)':       '#3498db',
                    '🟡 Lambat (4–6 minggu)':       '#f1c40f',
                    '🔴 Sangat Lambat (> 6 minggu)':'#e74c3c',
                },
                labels={'Proyeksi_Lama_Bayar_Minggu': 'Rata-rata Lama Bayar (Minggu)', 'Kode Cust': 'Customer'},
                title='Rata-rata Lama Bayar per Customer'
            )
            fig_bar.update_layout(xaxis_tickangle=-45, xaxis_type='category')
            st.plotly_chart(fig_bar, use_container_width=True)

            # Tabel detail
            st.dataframe(
                df_cat_filtered[['Kode Cust', 'Proyeksi_Lama_Bayar_Minggu', 'Kategori', 'Total_Payment']]
                .rename(columns={'Proyeksi_Lama_Bayar_Minggu': 'Rata-rata Bayar (Minggu)', 'Total_Payment': 'Total Payment Historis'}),
                use_container_width=True, hide_index=True
            )
        else:
            st.warning("Tidak ada data proyeksi customer yang tersedia.")

    # --- TAB 6: VISUALISASI OVERALL ---
    with tab6:
        st.subheader("📊 Visualisasi Overall Piutang")

        # ---- Baris 1: Distribusi piutang per Aging Group & status ----
        c1, c2 = st.columns(2)

        with c1:
            if 'Aging Group' in df_gabungan.columns:
                aging_summary = df_gabungan.groupby('Aging Group').size().reset_index(name='Jumlah_Invoice')
                aging_summary = aging_summary.sort_values('Jumlah_Invoice', ascending=False)
                fig_aging = px.bar(
                    aging_summary,
                    x='Aging Group', y='Jumlah_Invoice',
                    title='Jumlah Invoice per Aging Group',
                    labels={'Jumlah_Invoice': 'Jumlah Invoice', 'Aging Group': 'Aging Group'},
                    color='Jumlah_Invoice',
                    color_continuous_scale='Reds'
                )
                fig_aging.update_layout(coloraxis_showscale=False, xaxis_type='category')
                st.plotly_chart(fig_aging, use_container_width=True)

        with c2:
            status_counts = df_gabungan['Status_Peringatan'].value_counts().reset_index()
            status_counts.columns = ['Status', 'Jumlah']
            fig_status = px.pie(
                status_counts,
                names='Status', values='Jumlah',
                title='Proporsi Invoice: Aman vs Anomali',
                color='Status',
                color_discrete_map={
                    '✅ Aman':                    '#2ecc71',
                    '⚠️ Anomali: Telat dari Kebiasaan': '#e74c3c'
                }
            )
            fig_status.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_status, use_container_width=True)

        # ---- Baris 2: Top 10 customer by piutang & tren jatuh tempo ----
        c3, c4 = st.columns(2)

        with c3:
            # Distribusi jatuh tempo per minggu ke depan (semua invoice dengan TOP)
            df_tren = df_gabungan.dropna(subset=['TOP']).copy()
            df_tren['Minggu_TOP'] = df_tren['TOP'].dt.to_period('W').astype(str)
            tren_summary = df_tren.groupby('Minggu_TOP').agg(
                Jumlah_Invoice=('NONOTA', 'count'),
                Total_Piutang=('PIUTANG', 'sum')
            ).reset_index().sort_values('Minggu_TOP')

            fig_tren = go.Figure()
            fig_tren.add_bar(
                x=tren_summary['Minggu_TOP'],
                y=tren_summary['Total_Piutang'],
                name='Total Piutang',
                marker_color='#3498db'
            )
            fig_tren.add_scatter(
                x=tren_summary['Minggu_TOP'],
                y=tren_summary['Jumlah_Invoice'],
                name='Jumlah Invoice',
                yaxis='y2',
                mode='lines+markers',
                marker_color='#e67e22'
            )
            fig_tren.update_layout(
                title='Tren Jatuh Tempo per Minggu',
                xaxis_title='Minggu',
                yaxis=dict(title='Total Piutang (Rp)'),
                yaxis2=dict(title='Jumlah Invoice', overlaying='y', side='right'),
                legend=dict(orientation='h', y=1.1),
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig_tren, use_container_width=True)