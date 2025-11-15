
# **AmbedkarGPT — RAG Q&A System (LangChain + ChromaDB + Ollama)**

This project is a **simple Retrieval-Augmented Generation (RAG)** system built using:
Deployed it using Streamlit on Lightning AI:
Visit [AmbedkarGPT](https://8501-01ka31jr2wwdsnpdma6rcc3s21.cloudspaces.litng.ai)


* **LangChain**
* **ChromaDB**
* **HuggingFace Embeddings** (`all-MiniLM-L6-v2`)
* **Ollama (Mistral 7B)** — runs fully locally
* **Added Streamlit UI**

It loads a speech by **Dr. B. R. Ambedkar**, splits it into chunks, creates embeddings, stores them in ChromaDB, and answers user questions based on the speech only.

<img width="1912" height="903" alt="image" src="https://github.com/user-attachments/assets/166afd44-10e1-4dc5-aea2-b852d5f8989c" />

---

## 📦 Installation

### 1. Create virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```
pip install -r requirements.txt
```

### 3. Install Ollama

Linux / Mac:

```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

Windows:
Download from [https://ollama.ai/download](https://ollama.ai/download)

### 4. Pull the Mistral model

```
ollama pull mistral
```

Ensure the Ollama daemon is running:

```
ollama run mistral
```

---

## 📁 Project Structure

```
/AmbedkarGPT
│── main.py              
│── app.py               # Minimal Streamlit UI
│── speech.txt           # Provided Ambedkar speech excerpt
│── requirements.txt
│── README.md
```

---

## ▶️ Running the CLI version

```
python main.py
```

It will:

* Load `speech.txt`
* Split into chunks
* Create embeddings
* Build / load ChromaDB
* Run a sample question

---

## 🖥 Running the Streamlit App

```
streamlit run app.py
```

The app lets you:

* Upload your own `speech.txt` (optional)
* Build the vectorstore
* Ask questions interactively

## 📌 Notes

* All components run **locally** — no cloud usage.
* ChromaDB persists inside the `./chroma_db` directory.
* Delete the directory to rebuild embeddings:

```
rm -rf chroma_db
```

---

## ✔️ Assignment Requirements Covered

* LangChain ✓
* HuggingFace embeddings ✓
* ChromaDB ✓
* Ollama (Mistral 7B) ✓
* Functional RAG pipeline ✓
* Streamlit optional UI ✓

---

