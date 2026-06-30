import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
import pickle
import random

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="BPJS Command Center", layout="wide", page_icon="🏥")

# --- LOAD MODEL (DI-CACHE AGAR CEPAT) ---
@st.cache_resource
def load_models():
    # 1. Load Model Sentimen IndoBERTweet
    sentiment_model = pipeline(
        "sentiment-analysis", 
        model="Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis"
    )
    
    # 2. Load Model LDA (Berbasis Gensim)
    try:
        with open("lda_model.pkl", "rb") as f:
            # dictionary_gensim adalah pengganti vectorizer di library Gensim
            lda_model, dictionary_gensim = pickle.load(f)
    except FileNotFoundError:
        lda_model, dictionary_gensim = None, None
        
    return sentiment_model, lda_model, dictionary_gensim

classifier, lda_model, dictionary = load_models()

# --- HEADER ---
st.title("🏥 BPJS Health Command Center")
st.markdown("Sistem Analisis Sentimen dan Deteksi Isu Kritis Cuitan Masyarakat secara *Real-Time*.")
st.divider()

# --- TABS ---
tab1, tab2 = st.tabs(["📝 Analisis Teks Tunggal", "📂 Analisis Batch & Isu Kritis (CSV)"])

# ==========================================
# TAB 1: TEKS TUNGGAL
# ==========================================
with tab1:
    st.subheader("Uji Coba Sentimen Komentar")
    
    # 1. Daftar kalimat acak
    daftar_kalimat = [
        "Antrean di RS Mitra sangat lama dan adminnya kurang ramah, tolong diperbaiki.",
        "Pelayanan BPJS sekarang makin cepat dan sangat mudah karena ada JKN Mobile.",
        "Bagaimana prosedur pindah faskes tingkat 1 untuk bulan depan?",
        "Kecewa banget, tagihan iuran bulan ini tiba-tiba naik padahal saya kelas 3.",
        "Terima kasih BPJS, biaya operasi ayah saya ditanggung full tanpa biaya tambahan sedikitpun.",
        "Aplikasi mobile JKN sering error kalau mau ambil nomor antrean online, capek deh."
    ]

    # 2. Fungsi Session State
    if 'teks_input' not in st.session_state:
        st.session_state['teks_input'] = ""

    def pilih_kalimat_acak():
        st.session_state['teks_input'] = random.choice(daftar_kalimat)

    # 3. Kotak Teks Input
    user_input = st.text_area("Masukkan teks keluhan atau komentar:", key='teks_input')
    
    st.markdown("<br>", unsafe_allow_html=True) # Jarak pemisah agar tidak terlalu mepet

    # 4. Tata Letak Tombol (Saling Bersebelahan)
    col_analisis, col_dadu = st.columns([6, 1]) 
    
    with col_analisis:
        # Tombol utama ditaruh di kolom kiri (lebar)
        btn_analisis = st.button("🚀 Analisis Sentimen", type="primary", use_container_width=True)
        
    with col_dadu:
        # Tombol dadu ditaruh di kolom kanan (kecil), teks dihilangkan
        st.button("🎲", on_click=pilih_kalimat_acak, use_container_width=True, help="Munculkan kalimat acak")

    # 5. Logika Eksekusi Analisis
    if btn_analisis:
        if user_input.strip() == "":
            st.warning("Teks tidak boleh kosong! Silakan ketik atau klik tombol 🎲 di sebelah kanan.")
        else:
            with st.spinner("Sistem Menganalisis..."):
                hasil = classifier(user_input)[0]
                label = hasil['label'].capitalize()
                skor = hasil['score'] * 100
                
                if label == 'Negative':
                    st.error(f"**Sentimen: NEGATIF** (Keyakinan: {skor:.2f}%)")
                elif label == 'Positive':
                    st.success(f"**Sentimen: POSITIF** (Keyakinan: {skor:.2f}%)")
                else:
                    st.warning(f"**Sentimen: NETRAL** (Keyakinan: {skor:.2f}%)")

# ==========================================
# TAB 2: UPLOAD CSV (BATCH & GENSIM LDA)
# ==========================================
with tab2:
    st.subheader("Unggah Dataset Cuitan Baru")
    st.info("Sistem akan memproses analisis sentimen untuk seluruh data dan mengekstraksi isu kritis khusus pada sentimen negatif.")
    
    uploaded_file = st.file_uploader("Pilih file CSV (Pastikan memiliki kolom bernama 'tweet_clean')", type=["csv"])
    
    if uploaded_file is not None:
        if lda_model is None or dictionary is None:
            st.error("File 'lda_model.pkl' tidak ditemukan. Fitur deteksi isu dinonaktifkan.")
        else:
            # Sesuaikan separator CSV ('sep') jika data Anda menggunakan titik koma (;) atau koma (,)
            df_baru = pd.read_csv(uploaded_file, sep=";") 
            
            if 'tweet_clean' not in df_baru.columns:
                st.error("Gagal! File CSV harus memiliki kolom dengan nama 'tweet_clean'.")
            else:
                with st.spinner("Sistem sedang memproses data. Mohon tunggu..."):
                    # 1. Prediksi Sentimen
                    texts = df_baru['tweet_clean'].fillna('').tolist()
                    results = classifier(texts, truncation=True, max_length=512)
                    df_baru['Sentimen'] = [res['label'].capitalize() for res in results]
                    
                    # 2. Hitung Statistik Sentimen
                    df_stats = df_baru['Sentimen'].value_counts().reset_index()
                    df_stats.columns = ['Sentimen', 'Jumlah']
                    
                    # 3. Ekstraksi Isu LDA (Menggunakan GENSIM)
                    df_negatif = df_baru[df_baru['Sentimen'] == 'Negative'].copy()
                    
                    if not df_negatif.empty:
                        # Fungsi pembersih teks baru (Stopwords)
                        stopwords_tambahan = ['pakai', 'juga', 'buat', 'sama', 'pada', 'bukan', 'lagi', 'saja', 'jadi', 'tapi', 'tidak', 'ini', 'itu', 'atau', 'dengan', 'untuk', 'yang', 'dan', 'di', 'ke']
                        
                        def clean_new_text(text):
                            words = str(text).split()
                            return [w for w in words if w.lower() not in stopwords_tambahan and len(w) > 2]
                            
                        texts_baru = df_negatif['tweet_clean'].apply(clean_new_text).tolist()
                        
                        # Konversi teks ke format Bag of Words (BoW) Gensim
                        corpus_baru = [dictionary.doc2bow(text) for text in texts_baru]
                        
                        # Prediksi Topik untuk setiap dokumen
                        topik_prediksi = []
                        for doc in corpus_baru:
                            if len(doc) > 0:
                                hasil_topik = lda_model[doc]
                                # Ambil topik dengan probabilitas (skor) tertinggi
                                topik_tertinggi = max(hasil_topik, key=lambda x: x[1])[0]
                                topik_prediksi.append(topik_tertinggi)
                            else:
                                topik_prediksi.append(-1) # Jika teks kosong setelah dibersihkan
                                
                        df_negatif['Topik_ID'] = topik_prediksi
                        
                        # Hitung jumlah per topik (abaikan yang -1)
                        topik_counts = df_negatif[df_negatif['Topik_ID'] != -1]['Topik_ID'].value_counts().to_dict()
                    else:
                        topik_counts = {}

                # 4. Tampilkan Hasil Dasbor
                st.success("✅ Proses Batch Selesai!")
                col_chart, col_isu = st.columns([1, 1])
                
                with col_chart:
                    st.markdown("**Distribusi Sentimen**")
                    fig = px.pie(df_stats, values='Jumlah', names='Sentimen', hole=0.6,
                                 color='Sentimen', color_discrete_map={'Negative': '#FF6B6B', 'Neutral': '#FFD43B', 'Positive': '#51CF66'})
                    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col_isu:
                    st.markdown("**Isu Kritis Dominan (Dari Sentimen Negatif)**")
                    
                    # PENTING: Ubah nama isu ini sesuai dengan hasil kesimpulan topik di skripsi Anda!
                    nama_isu = {
                        0: "Topik 1: Kendala Fasilitas & Layanan Faskes", 
                        1: "Topik 2: Birokrasi Administrasi & Pendaftaran", 
                        2: "Topik 3: Isu Biaya & Transparansi Tagihan"
                    }
                    
                    if topik_counts:
                        for topik_id, count in topik_counts.items():
                            isu_name = nama_isu.get(topik_id, f"Isu Lainnya (Topik {topik_id + 1})")
                            st.error(f"⚠️ **{isu_name}**: Ditemukan pada {count} cuitan.")
                    else:
                        st.info("Tidak ada cuitan sentimen negatif yang signifikan untuk diekstrak isunya.")
