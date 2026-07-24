import os
import argparse
from src.config import DATA_DIR
from src.data_ingestion import DocumentProcessor
from src.vector_store import VectorStoreManager
from src.llm_chain import RAGPipeline

def main(query):
    file_path = os.path.join(DATA_DIR, "sample_document.txt")
    
    print("[1] Veriler işleniyor...")
    doc_processor = DocumentProcessor()
    texts = doc_processor.load_and_split(file_path)
    
    print("[2] Vektör veritabanı oluşturuluyor...")
    vs_manager = VectorStoreManager()
    vectorstore = vs_manager.create_vectorstore(texts)
    
    print("[3] LLM zinciri kuruluyor ve soru soruluyor...")
    pipeline = RAGPipeline()
    qa_chain = pipeline.create_chain(vectorstore)
    
    result = qa_chain({"query": query})
    
    print("\n--- CEVAP ---")
    print(result["result"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, default="Bu belgenin ana konusu nedir?")
    args = parser.parse_args()
    main(args.query)