# AI Video Assistant

## 🚀 What this does
This project processes video or audio files and allows you to ask questions about the content.

It:
- Extracts audio from video
- Transcribes speech into text
- Converts text into embeddings
- Uses Retrieval-Augmented Generation (RAG) to answer questions or summarize content

---

## 🧠 How it works (Architecture)

1. **Audio Extraction**  
   Converts video into audio format

2. **Transcription**  
   Uses a speech-to-text model to generate text

3. **Text Chunking**  
   Splits transcript into smaller chunks

4. **Embedding (MiniLM)**  
   Converts chunks into vector representations

5. **Vector Store (ChromaDB)**  
   Stores embeddings for retrieval

6. **Retrieval + QA**  
   Finds relevant chunks and answers user queries

---

## 🛠 Tech Stack
- Python
- LangChain
- ChromaDB
- HuggingFace Embeddings

---

## ▶️ How to run

```bash
# 1. Clone the repository
git clone https://github.com/venom312004/AI-video-assistant.git
cd ai-video-assistant

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python app.py
