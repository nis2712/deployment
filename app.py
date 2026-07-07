import streamlit as st
import pandas as pd
import os
from gensim.models import LdaModel
from gensim import corpora

# 1. Pemuatan Model (Menggunakan cache agar memori lebih efisien)
@st.cache_resource
def load_models():
    # Mendapatkan lokasi direktori tempat app.py ini berada
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Menggabungkan path direktori dengan nama file model Gensim
    lda_path = os.path.join(current_dir, 'lda_model.model')
    dict_path = os.path.join(current_dir, 'dictionary.dict')
    
    # Memuat model menggunakan modul bawaan Gensim (bukan joblib)
    lda_model = LdaModel.load(lda_path)
    dictionary = corpora.Dictionary.load(dict_path)
    
    return lda_model, dictionary

# 2. Eksekusi Pemuatan Model
lda_model, dictionary = load_models()

# Pengaturan Konfigurasi Halaman Web
st.set_page_config(page_title="BPJS Health Insight Tracker", page_icon="🏥", layout="centered")

st.title("BPJS Health Insight Tracker 🏥")
st.subheader("Sistem Deteksi Isu Kritis Layanan Publik (LDA)")

# 3. Mode Input Data
mode = st.radio("Pilih Mode Analisis:", ["Keluhan Tunggal", "Keluhan Batch"])

# --- MODE 1: KELUHAN TUNGGAL ---
if mode == "Keluhan Tunggal":
    st.info("💡 Masukkan teks keluhan bersentimen negatif untuk mendeteksi kategori masalahnya.")
    text = st.text_area("Tulis keluhan di sini:")
    
    if st.button("Analisis Isu Kritis"):
        if text.strip():
            # Preprocessing sederhana (huruf kecil dan split kata)
            bow = dictionary.doc2bow(text.lower().split())
            
            # Mendapatkan distribusi topik dari teks input
            topics = lda_model.get_document_topics(bow)
            
            st.write("### 🔍 Hasil Deteksi Isu Kritis:")
            if topics:
                # Mengurutkan topik berdasarkan probabilitas tertinggi (paling relevan)
                topics = sorted(topics, key=lambda x: x[1], reverse=True)
                
                for topic_id, prob in topics:
                    # Mengambil kata kunci dari masing-masing topik
                    topic_words = lda_model.show_topic(topic_id)
                    words = ", ".join([word for word, weight in topic_words])
                    
                    st.success(f"**Topik {topic_id + 1}** (Tingkat Kesesuaian: {prob:.2%})")
                    st.write(f"**Kata Kunci Dominan:** {words}")
            else:
                st.warning("Kata-kata dalam keluhan ini tidak dikenali oleh kamus model data latih.")
        else:
            st.error("Harap masukkan teks keluhan terlebih dahulu sebelum menekan tombol.")

# --- MODE 2: KELUHAN BATCH (VIA CSV) ---
elif mode == "Keluhan Batch":
    st.info("💡 Unggah file dataset CSV Anda. Pastikan file memiliki kolom dengan nama **'keluhan'**.")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    
    if uploaded_file:
        try:
            # Membaca file CSV
            df = pd.read_csv(uploaded_file, sep=';') # Sesuaikan separator (; atau ,) jika perlu
            
            st.write("### 📊 Pratinjau Data Keluhan")
            st.dataframe(df.head())
            
            if 'keluhan' in df.columns:
                st.write("### 📌 Daftar Topik Isu Terdeteksi (Dari Keseluruhan Model)")
                
                # Mengambil dan memformat definisi topik dari model LDA
                global_topics = lda_model.show_topics(formatted=False)
                
                for topic_id, words_weights in global_topics:
                    # Mengekstrak hanya kata-katanya saja
                    words = ", ".join([word for word, weight in words_weights])
                    st.markdown(f"- **Isu Kritis {topic_id + 1}:** {words}")
                    
                st.markdown("---")
                st.caption("*(Catatan: Kata-kata di atas mewakili klaster masalah utama yang berhasil dipetakan oleh model dari keseluruhan data latih)*")
                
            else:
                st.error("❌ Gagal memproses: File CSV harus memiliki header kolom bernama 'keluhan'.")
                st.write("Kolom yang terdeteksi di file Anda:", df.columns.tolist())
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")
