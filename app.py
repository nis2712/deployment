import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline
import random
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="BPJS Command Center", layout="wide", page_icon="🏥")

# --- LOAD MODEL ---
@st.cache_resource
def load_models():
    # 1. Load Model Sentimen
    sentiment_model = pipeline("sentiment-analysis", model="Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis")
    
    # 2. Load Model LDA (Format Native Gensim)
    try:
        lda_model = LdaModel.load('lda_model.model')
        dictionary_gensim = Dictionary.load('dictionary.dict')
    except Exception as e:
        st.error(f"Error memuat model LDA: {e}")
        lda_model, dictionary_gensim = None, None
        
    return sentiment_model, lda_model, dictionary_gensim

classifier, lda_model, dictionary = load_models()

# --- ANTARMUKA ---
st.title("🏥 BPJS Health Command Center")
st.markdown("Sistem Analisis Sentimen dan Deteksi Isu Kritis.")
st.divider()

tab1, tab2 = st.tabs(["📝 Analisis Teks Tunggal", "📂 Analisis Batch & Isu Kritis (CSV)"])

# --- TAB 1: TEKS TUNGGAL ---
with tab1:
    st.subheader("Uji Coba Sentimen Komentar")
    
    daftar_kalimat = [
        "Antrean di RS Mitra sangat lama dan adminnya kurang ramah.",
        "Pelayanan BPJS sekarang makin cepat dan sangat mudah.",
        "Kecewa banget, tagihan iuran bulan ini tiba-tiba naik.",
        "Terima kasih BPJS, biaya operasi ayah saya ditanggung full.",
        "Aplikasi mobile JKN sering error saat ambil antrean."
    ]

    if 'teks_input' not in st.session_state:
        st.session_state['teks_input'] = ""

    def pilih_kalimat_acak():
        st.session_state['teks_input'] = random.choice(daftar_kalimat)

    user_input = st.text_area("Masukkan teks keluhan atau komentar:", key='teks_input')
    
    st.markdown("<br>", unsafe_allow_html=True)

    col_analisis, col_dadu = st.columns([6, 1]) 
    with col_analisis:
        btn_analisis = st.button("🚀 Analisis Sentimen", type="primary", use_container_width=True)
    with col_dadu:
        st.button("🎲", on_click=pilih_kalimat_acak, use_container_width=True, help="Munculkan kalimat acak")

    if btn_analisis:
        if df_negatif['Topik_ID'] != -1:
            st.warning("Teks tidak boleh kosong!")
        else:
            with st.spinner("Menganalisis..."):
                hasil = classifier(user_input)[0]
                label = hasil['label'].capitalize()
                st.write(f"**Hasil Sentimen:** {label}")

# --- TAB 2: UPLOAD CSV ---
with tab2:
    st.subheader("Unggah Dataset Cuitan Baru")
    uploaded_file = st.file_uploader("Pilih file CSV (kolom: 'tweet_clean')", type=["csv"])
    
    if uploaded_file is not None and lda_model is not None:
        df_baru = pd.read_csv(uploaded_file, sep=";") 
        
        with st.spinner("Memproses data..."):
            texts = df_baru['tweet_clean'].fillna('').tolist()
            results = classifier(texts, truncation=True, max_length=512)
            df_baru['Sentimen'] = [res['label'].capitalize() for res in results]
            
            df_negatif = df_baru[df_baru['Sentimen'] == 'Negative'].copy()
            
            # Deteksi Isu
            if not df_negatif.empty:
                stopwords_tambahan = ['pakai', 'juga', 'buat', 'sama', 'pada', 'bukan', 'lagi', 'saja']
                def clean_text(text):
                    words = str(text).split()
                    return [w for w in words if w.lower() not in stopwords_tambahan and len(w) > 2]
                
                texts_baru = df_negatif['tweet_clean'].apply(clean_text).tolist()
                corpus_baru = [dictionary.doc2bow(text) for text in texts_baru]
                
                topik_prediksi = []
                for doc in corpus_baru:
                    hasil_topik = lda_model.get_document_topics(doc)
                    if hasil_topik:
                        topik_tertinggi = max(hasil_topik, key=lambda x: x[1])[0]
                        topik_prediksi.append(topik_tertinggi)
                    else:
                        topik_prediksi.append(-1)
                
                df_negatif['Topik_ID'] = topik_prediksi
                topik_counts = df_negatif[df_negatif['Topik_ID'] != -1]['Topik_ID'].value_counts().to_dict()
                
                st.success("Proses Selesai!")
                # Tampilkan hasil topik
                for tid, count in topik_counts.items():
                    st.error(f"⚠️ **Isu {tid+1}**: Ditemukan {count} keluhan.")
