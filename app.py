import streamlit as st
import pandas as pd
import random
from transformers import pipeline
from gensim.models import LdaModel
from gensim.corpora import Dictionary

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="BPJS Command Center", layout="wide", page_icon="🏥")

# --- LOAD MODEL ---
@st.cache_resource
def load_models():
    # Model sentiment di-load sekali saja
    sentiment_model = pipeline("sentiment-analysis", model="Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis")
    try:
        lda_model = LdaModel.load('lda_model.model')
        dictionary_gensim = Dictionary.load('dictionary.dict')
    except Exception as e:
        st.error(f"Error memuat model LDA: {e}")
        lda_model, dictionary_gensim = None, None
    return sentiment_model, lda_model, dictionary_gensim

classifier, lda_model, dictionary = load_models()

# --- UI ---
st.title("🏥 BPJS Health Command Center")
tab1, tab2 = st.tabs(["📝 Analisis Teks Tunggal", "📂 Analisis Batch (CSV)"])

with tab1:
    st.subheader("Uji Coba Sentimen Komentar")
    daftar_kalimat = ["Antrean di RS Mitra sangat lama.", "Pelayanan BPJS cepat.", "Kecewa dengan birokrasi."]
    if 'teks_input' not in st.session_state: st.session_state['teks_input'] = ""
    
    user_input = st.text_area("Masukkan teks:", key='teks_input')
    col1, col2 = st.columns([6, 1])
    with col1: btn = st.button("🚀 Analisis Sentimen", type="primary", use_container_width=True)
    with col2: st.button("🎲", on_click=lambda: st.session_state.update({'teks_input': random.choice(daftar_kalimat)}), use_container_width=True)

    if btn and user_input:
        hasil = classifier(user_input)[0]
        st.success(f"**Hasil:** {hasil['label'].capitalize()}")

with tab2:
    st.subheader("Analisis Batch")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file and lda_model:
        df = pd.read_csv(uploaded_file, sep=";")
        if 'tweet_clean' in df.columns:
            # Prediksi sentimen
            texts = df['tweet_clean'].fillna('').tolist()
            res = classifier(texts, truncation=True, max_length=512)
            df['Sentimen'] = [r['label'].capitalize() for r in res]
            
            # Deteksi Isu Negatif
            neg = df[df['Sentimen'] == 'Negative']
            if not neg.empty:
                corpus = [dictionary.doc2bow(str(t).split()) for t in neg['tweet_clean']]
                df.loc[neg.index, 'Isu_ID'] = [max(lda_model.get_document_topics(c), key=lambda x: x[1])[0] if c else -1 for c in corpus]
                st.write("### Analisis Isu Kritis:")
                st.write(df[df['Isu_ID'].notnull()]['Isu_ID'].value_counts())
