import streamlit as st
import pandas as pd
from transformers import pipeline
from gensim.models import LdaModel
from gensim.corpora import Dictionary

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="BPJS Health Command Center", layout="wide", page_icon="🏥")

# --- LOAD MODEL (OPTIMASI RAM) ---
@st.cache_resource
def load_models():
    # Menghindari reload model saat interaksi user
    try:
        classifier = pipeline("sentiment-analysis", model="Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis")
        lda_model = LdaModel.load('lda_model.model')
        dictionary = Dictionary.load('dictionary.dict')
        return classifier, lda_model, dictionary
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None, None, None

classifier, lda_model, dictionary = load_models()

# --- UI APP ---
st.title("🏥 BPJS Health Command Center")
tab1, tab2 = st.tabs(["📝 Analisis Sentimen Tunggal", "📂 Analisis Isu Kritis (Batch)"])

# TAB 1: ANALISIS TEKS
with tab1:
    st.subheader("Analisis Sentimen IndoBERTweet")
    user_input = st.text_area("Masukkan komentar keluhan:")
    if st.button("🚀 Analisis Sekarang", type="primary"):
        if classifier and user_input:
            with st.spinner("Menganalisis..."):
                hasil = classifier(user_input)[0]
                st.info(f"**Sentimen:** {hasil['label'].capitalize()} (Skor: {hasil['score']:.2f})")
        else:
            st.warning("Pastikan model sudah termuat dan teks tidak kosong.")

# TAB 2: ANALISIS BATCH LDA
with tab2:
    st.subheader("Analisis Isu Kritis (LDA)")
    uploaded_file = st.file_uploader("Upload CSV (kolom: 'tweet_clean')", type=["csv"])
    
    if uploaded_file and lda_model and dictionary:
        df = pd.read_csv(uploaded_file, sep=";")
        if 'tweet_clean' in df.columns:
            with st.spinner("Memproses topik..."):
                # Preprocessing
                texts = [str(t).split() for t in df['tweet_clean'].fillna('')]
                corpus = [dictionary.doc2bow(t) for t in texts]
                
                # Prediksi Topik
                topik_ids = []
                for doc in corpus:
                    dist = lda_model.get_document_topics(doc)
                    topik_ids.append(max(dist, key=lambda x: x[1])[0] if dist else -1)
                
                df['Isu_ID'] = topik_ids
                
                st.success("✅ Analisis Selesai")
                st.dataframe(df[['tweet_clean', 'Isu_ID']])
        else:
            st.error("Kolom 'tweet_clean' tidak ditemukan!")
