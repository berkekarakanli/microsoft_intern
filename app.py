import streamlit as st
import os
from src.config import DATA_DIR
from src.data_ingestion import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.llm_chain import RAGPipeline
from langchain_community.llms import Ollama

# Arayüz Ayarları
st.set_page_config(page_title="Yapay Zeka Asistanı", page_icon="🤖", layout="centered")

st.title("🤖 Çift Motorlu AI Asistanı")
st.markdown("""
Bu asistan, **Foundry Local / Ollama** altyapısını kullanarak tamamen yerel (local) çalışır.
İsterseniz belgelerinize (PDF) soru sorun, isterseniz serbest sohbet edin!
""")

st.divider()

# Mod Seçimi (Butonlar)
mod_secimi = st.radio(
    "Asistan nasıl çalışsın?",
    ["📄 Belgeye Göre Cevapla (PDF RAG)", "💬 Serbest Sohbet (Normal Yapay Zeka)"],
    horizontal=True
)

st.divider()

# Dosya Yükleme Alanı (Sadece RAG modu seçiliyse ekranda görünür)
uploaded_file = None
if mod_secimi == "📄 Belgeye Göre Cevapla (PDF RAG)":
    uploaded_file = st.file_uploader("İncelemek istediğiniz PDF dosyasını buraya sürükleyin veya seçin", type="pdf")

# Soru girdisi alanı
query = st.text_input("Ne öğrenmek istersiniz?", placeholder="Sorunuzu buraya yazın...")

if st.button("Cevapla", type="primary"):
    if query:
        with st.spinner("Yapay zeka yanıt üretiyor (Biraz sürebilir)..."):
            try:
                if mod_secimi == "📄 Belgeye Göre Cevapla (PDF RAG)":

                    # Kullanıcı PDF yüklemeden soru sorarsa uyar
                    if uploaded_file is None:
                        st.error("Lütfen soruyu sormadan önce bir PDF dosyası yükleyin!")
                        st.stop()

                    # 1. Yüklenen dosyayı arka planda 'data' klasörüne kaydet
                    file_path = os.path.join(DATA_DIR, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

                    # 2. RAG MİMARİSİ (Belge Okuma ve Soru Cevaplama)
                    doc_processor = DocumentProcessor()
                    texts = doc_processor.load_and_split(file_path)

                    vs_manager = VectorStoreManager()
                    vectorstore = vs_manager.create_vectorstore(texts)

                    pipeline = RAGPipeline()
                    qa_chain = pipeline.create_chain(vectorstore)

                    result = qa_chain({"query": query})
                    cevap = result["result"]

                else:
                    # 3. NORMAL SOHBET MİMARİSİ (PDF'siz)
                    llm = Ollama(model="llama3", temperature=0.7)
                    cevap = llm.invoke(query)

                # Ekrana Yazdırma
                st.success("Yanıt Başarıyla Üretildi!")
                st.markdown("### Cevap:")
                st.info(cevap)

            except Exception as e:
                st.error(f"Sistem bir hatayla karşılaştı: {str(e)}")
    else:
        st.warning("Lütfen cevaplamak için bir soru girin.")