import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHUNK_SIZE, CHUNK_OVERLAP


class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

    def load_and_split(self, file_path):
       
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Sistem Hatası: Belirtilen konumda dosya bulunamadı -> {file_path}")

       
        file_path_lower = file_path.lower()

        if file_path_lower.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path_lower.endswith('.txt'):
            loader = TextLoader(file_path, encoding='utf-8')
        else:
            raise ValueError(f"Desteklenmeyen format: {file_path}. Lütfen sadece .txt veya .pdf kullanın.")

        
        documents = loader.load()
        return self.text_splitter.split_documents(documents)
