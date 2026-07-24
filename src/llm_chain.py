from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from src.config import LLM_MODEL_NAME

class RAGPipeline:
    def __init__(self):
        self.llm = Ollama(model=LLM_MODEL_NAME, temperature=0.1)
        self.prompt = self._build_prompt()

    def _build_prompt(self):
        template = """
        Aşağıdaki bağlamı (context) kullanarak kullanıcının sorusuna cevap ver. 
        Cevabı bağlamda bulamıyorsan uydurma, 'Bilgi bulunamadı' de.
        
        Bağlam: {context}
        Soru: {question}
        Cevap:"""
        return PromptTemplate(template=template, input_variables=["context", "question"])

    def create_chain(self, vectorstore):
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": self.prompt}
        )