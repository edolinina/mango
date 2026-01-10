import os
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


KNOWLEDGE_BASE_PATH = "knowledge_base"

def load_knowledge():
    loader = DirectoryLoader(
        path=KNOWLEDGE_BASE_PATH,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
    )
    return loader.load()

def get_knowledge_retriever():
    docs = load_knowledge()
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    embedding_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

    embeddings = OllamaEmbeddings(
        base_url=ollama_url,
        model=embedding_model,
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(k=5)