# 🎬 AI Video Assistant

**Meeting Intelligence — Transcribe · Summarise · Chat with your meetings**

AI Video Assistant is an end-to-end meeting/video intelligence tool. Upload an audio/video file and it automatically transcribes it (English or Hinglish), generates a summary, extracts action items, key decisions, and open questions, and lets you chat with the transcript using a RAG-powered assistant.

---

## 🔗 Links

- **Live App:** [ai-video-assistant-pranjal-pandey.streamlit.app](https://ai-video-assistant-pranjal-pandey-0301.streamlit.app/)

---

## 📌 Description

This project combines a dual transcription engine (OpenAI Whisper for English, Sarvam AI for Hinglish), a LangChain + Qwen3 (served by Groq) summarization and extraction pipeline, and a ChromaDB-backed RAG chat system with local embeddings, all wrapped in a custom-styled Streamlit interface.

It's built for anyone who wants to turn a raw meeting recording, podcast, or video into a structured, searchable, and conversational summary.

---

## ✨ Features

- 🎙️ **Dual Transcription Engine** — OpenAI Whisper (English) and Sarvam AI (Hinglish)
- 📋 **Auto-Summarization** — map-reduce meeting summaries powered by Qwen3 on Groq
- ✅ **Action Item Extraction** — tasks, owners, and deadlines pulled automatically
- 🔑 **Key Decision & Open Question Detection** — extracted in a single call per transcript chunk
- 💬 **RAG Chat** — ask follow-up questions grounded in the transcript
- 🧠 **Local Embeddings** — multilingual HuggingFace embeddings, so no API quota is spent on retrieval
- 📁 **Flexible Input** — upload audio/video files directly (`.mp3`, `.wav`, `.mp4`, `.m4a`, `.webm`)
- 🎨 **Custom dark-themed UI** built with Streamlit
- 🔁 **Automatic retry with exponential backoff** on LLM API rate limits / transient errors
- 🛡️ **Friendly error handling** — rate-limit errors show a clear message instead of a raw API dump

> **Note:** Direct YouTube link processing is unreliable on cloud hosting (Streamlit Cloud / Render free tier) due to YouTube's bot-detection blocking data-center IPs. **File upload is the recommended and fully supported input method.**

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM Orchestration | LangChain (LCEL) |
| LLM Provider | Qwen 3.8 27B (`qwen/qwen3.8-27b`) via Groq |
| Speech-to-Text | OpenAI Whisper, Sarvam AI |
| Vector Store | ChromaDB (`langchain-chroma`) |
| Embeddings | HuggingFace Sentence-Transformers (`paraphrase-multilingual-MiniLM-L12-v2`, runs locally) |
| Audio Processing | `pydub`, `ffmpeg`, `yt-dlp` |
| Deployment | Streamlit Community Cloud |

> The LLM is configurable. Set `LLM_MODEL` in your environment (for example `openai/gpt-oss-120b`) to switch models without changing code. Groq's model list changes often, so check the current list at [console.groq.com/docs/models](https://console.groq.com/docs/models).

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10, 3.11, or 3.12** (Python 3.13+ is not recommended — several dependencies like `torch`, `pillow`, and `chromadb` don't yet ship prebuilt Windows wheels for the newest Python releases, which can cause install failures)
- `ffmpeg` installed on your system
- Groq API key — get one at [console.groq.com/keys](https://console.groq.com/keys)
- Sarvam AI API key

### Installation

```bash
git clone https://github.com/venom312004/AI-video-assistant.git
cd AI-video-assistant

# create and activate a virtual environment (use Python 3.11 or 3.12)
py -3.11 -m venv venv311
.\venv311\Scripts\Activate.ps1     # Windows (PowerShell)
# source venv311/bin/activate      # macOS/Linux

# install dependencies
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
SARVAM_API_KEY=your_sarvam_api_key

# optional
# LLM_MODEL=qwen/qwen3.8-27b
# EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
```

On Streamlit Cloud, add the same keys under **App settings → Secrets**.

### Run Locally

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
AI video Assistant/
├── app.py                  # Streamlit UI and pipeline orchestration
├── main.py                 # CLI entry point
├── core/
│   ├── extractor.py        # Action items, decisions, questions extraction
│   ├── rag_engine.py       # RAG chain builder and Q&A
│   ├── summary.py          # Summarization and title generation
│   ├── transcriber.py      # Whisper / Sarvam transcription logic
│   └── vector_store.py     # ChromaDB vector store setup (local embeddings)
├── utils/
│   └── audio_processor.py  # Audio download, conversion, chunking
├── requirements.txt
├── packages.txt            # System-level dependencies (ffmpeg)
└── .gitignore
```

---

## ⚠️ Known Limitations

- **YouTube URL input** may fail on cloud deployments with `HTTP 403` or `format not available` errors due to YouTube's bot-detection on data-center IPs. Use the **file upload** option for reliable results.
- **Groq free-tier limits** apply per model (requests and tokens per minute/day). Very long videos may hit them; the app retries automatically and shows a friendly message if the limit is still exceeded.
- Free-tier cloud hosting has limited CPU/RAM, so large files or long videos may take longer to process or hit resource throttling.
- **Python version:** Avoid the newest Python releases (3.13+) until upstream libraries publish prebuilt wheels for them — otherwise `pip install` may attempt to build packages like Pillow from source and fail.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Sarvam AI](https://www.sarvam.ai/)
- [Qwen](https://github.com/QwenLM/Qwen3) by Alibaba Cloud
- [Groq](https://groq.com/)
- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)