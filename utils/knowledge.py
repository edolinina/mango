import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.vectorstores import FAISS
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
    embeddings = HuggingFaceEmbeddings(
        model_name=os.getenv("EMBEDDING_MODEL")
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    chunks = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(k=5)