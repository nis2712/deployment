import streamlit as st
import pandas as pd
import plotly.express as px

# Konfigurasi Halaman - Menjadikan UI lebih lega
st.set_page_config(page_title="BPJS Sentiment Analytics", layout="wide", page_icon="🏥")

# --- UI STYLE: MODERN & CLEAN ---
st.markdown("""
    <style>
    /* Mengatur agar setiap kartu memiliki shadow tipis dan sudut melengkung */
    .css-1r6slp0, .stMetric, .stMarkdown, .stPlotlyChart {
        padding: 10px;
    }
    .card {
        background-color: #FFFFFF;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- SIDEBAR: NAVIGATION ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d0/BPJS_Kesehatan_logo.svg/1200px-BPJS_Kesehatan_logo.svg.png", width=150)
    st.title("Control Panel")
    pilih_data = st.selectbox("Fokus Analisis:", ["Keseluruhan", "Sentimen Negatif saja"])
    st.divider()
    st.write("### Tentang")
    st.caption("Dashboard ini merupakan hasil implementasi deployment model NLP (IndoBERTweet + LDA) untuk evaluasi layanan BPJS Kesehatan.")

# --- HEADER ---
st.title("🏥 BPJS Health Command Center")
st.markdown("Analisis strategis sentimen publik untuk optimalisasi layanan kesehatan berbasis *data-driven*.")

# --- LOGIKA DATA ---
data = {'Sentimen': ['Negatif', 'Netral', 'Positif'], 'Jumlah': [2618, 2227, 491]}
df = pd.DataFrame(data)
df_plot = df[df['Sentimen'] == 'Negatif'] if pilih_data == "Sentimen Negatif saja" else df

# --- KPI SECTION (Spaced out) ---
c1, c2, c3 = st.columns(3)
c1.metric("Total Sampel", "5.336", help="Jumlah total cuitan yang dianalisis")
c2.metric("Sentimen Negatif", "2.618", "-49,06%", delta_color="inverse")
c3.metric("Akurasi Model", "92%", help="Skor performa model IndoBERTweet")

st.markdown("<br>", unsafe_allow_html=True)

# --- TABS ---
tab1, tab2 = st.tabs(["📊 Distribusi Sentimen", "💡 Analisis Isu Strategis"])

with tab1:
    col_chart, col_insight = st.columns([2, 1])
    with col_chart:
        fig = px.bar(df_plot, x='Sentimen', y='Jumlah', color='Sentimen',
                     color_discrete_map={'Negatif': '#FF6B6B', 'Netral': '#FFD43B', 'Positif': '#51CF66'},
                     text_auto=True)
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col_insight:
        st.markdown("### Insight")
        st.write("Data menunjukkan dominasi sentimen negatif. Hal ini mengindikasikan adanya celah pada kualitas layanan yang perlu dioptimalkan oleh pihak manajemen.")

with tab2:
    st.markdown("### 🎯 Prioritas Perbaikan Layanan")
    # Menggunakan layout yang sangat bersih dan teratur
    cols = st.columns(3)
    items = [
        ("🏥 Fasilitas Kesehatan", "Fokus pada IGD & RS Mitra", "#FF6B6B"),
        ("📋 Administrasi", "Simplifikasi birokrasi & durasi", "#FFD43B"),
        ("💰 Biaya & Tagihan", "Transparansi prosedur kelas", "#51CF66")
    ]
    for i, col in enumerate(cols):
        with col:
            st.markdown(f"**{items[i][0]}**")
            st.info(f"{items[i][1]}")

st.markdown("<br><br>", unsafe_allow_html=True)
st.caption("© 2026 Skripsi Deployment - Nur Insan Subekti | UMMI")