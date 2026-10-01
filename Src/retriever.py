import os
import re
import glob

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()  # loads GROQ_API_KEY from .env

DATA_DIR = "Data"
DB_DIR = "chroma_store"


# 1. LOAD ---- read each transcript, throw away the VTT timestamps
def load_transcripts():

    docs = []
    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []
        for line in open(path):
            line = line.strip()
            if not line or line == "WEBVTT" or "-->" in line:
                continue
            lines.append(line)
        text = " ".join(lines)

        session = re.search(r"Session[ _]*(\d+)", path).group(1)

        docs.append(Document(page_content=text, metadata={"session": session}))

    return docs


import torch

# 2. BUILD ---- chunk, embed once, and keep it on disk so we don't re-embed
def load_store(rebuild: bool = False):
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(4)
    device = "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 32, "normalize_embeddings": True},
    )


    if rebuild and os.path.exists(DB_DIR):
        import shutil
        shutil.rmtree(DB_DIR)

    if os.path.exists(DB_DIR):
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    docs = load_transcripts()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)


    return Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)



def build_retriever(use_reranker: bool = False, fetch_k: int = 25, top_k: int = 6):
    if use_reranker:
        from src.reranker import build_reranking_retriever
        return build_reranking_retriever(fetch_k=fetch_k, top_k=top_k)
    return load_store().as_retriever(search_kwargs={"k": top_k})


# 3. TRY IT ---- python src/retriever.py
if __name__ == "__main__":

    retriever = build_retriever()

    results = retriever.invoke("what is regression testing?")
    
    for r in results:
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")