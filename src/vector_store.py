from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import EMBEDDING_MODEL, DB_DIR

class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    def create_vectorstore(self, texts):
        return Chroma.from_documents(
            documents=texts, 
            embedding=self.embeddings,
            persist_directory=DB_DIR
        )