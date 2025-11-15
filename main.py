from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma 
from langchain_ollama import ChatOllama
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from typing import List, Dict
import os
import shutil

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"  
DEFAULT_PERSIST = "./chroma_db"

def load_text(path: str) -> str:
    """Return raw text from path (raises FileNotFoundError if missing)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No file at {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def split_text(text: str) -> List:
    splitter = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=800,
        chunk_overlap=120,
        length_function=len,
    )
    return splitter.split_text(text)

def build_or_load_vectorstore(text, rebuild: bool = False) -> Chroma:
    """
    Create or load a Chroma vectorstore for the given docs.
    If rebuild True, the persist directory is removed and recreated.
    """
    if rebuild and os.path.exists(DEFAULT_PERSIST):
        shutil.rmtree(DEFAULT_PERSIST)

    os.makedirs(DEFAULT_PERSIST, exist_ok=True)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # If there's already persisted data, we try to load it
    if os.path.isdir(DEFAULT_PERSIST) and os.listdir(DEFAULT_PERSIST):
        try:
            return Chroma(persist_directory=DEFAULT_PERSIST, embedding_function=embeddings)
        except Exception:
            pass

    db = Chroma.from_texts(text, embedding=embeddings, persist_directory=DEFAULT_PERSIST)
    db.persist()
    return db
    
def build_qa_chain(vectordb: Chroma):
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})
    llm = ChatOllama(model=OLLAMA_MODEL)
    from langchain_classic import hub
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    qa = create_retrieval_chain(retriever, combine_docs_chain)
    return qa


if __name__ == "__main__":
    speech_path = "speech.txt"
    text = load_text(speech_path)
    text = split_text(text)
    vectordb = build_or_load_vectorstore(text, rebuild=False)
    qa = build_qa_chain(vectordb)

    q = "What does the author say is the 'real remedy' for caste?"
    res = qa.invoke({"input": q})
    answer = res.get("answer")
    print("Question:", q)
    print("Answer:", answer)