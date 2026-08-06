# 🎬 AI Video Assistant

**Meeting Intelligence — Transcribe · Summarise · Chat with your meetings**

AI Video Assistant is an end-to-end meeting/video intelligence tool. Upload an audio/video file  and it automatically transcribes it (English or Hinglish), generates a summary, extracts action items, key decisions, and open questions, and lets you chat with the transcript using a RAG-powered assistant.

---

## 🔗 Links

- **Live App:** [ai-video-assistant-pranjal-pandey.streamlit.app](https://ai-video-assistant-pranjal-pandey-0301.streamlit.app/)


---

## 📌 Description

This project combines a dual transcription engine (OpenAI Whisper for English, Sarvam AI for Hinglish), a LangChain + Mistral-powered summarization and extraction pipeline, and a ChromaDB-backed RAG chat system — all wrapped in a custom-styled Streamlit interface.

It's built for anyone who wants to turn a raw meeting recording, podcast, or video into a structured, searchable, and conversational summary.

---

## ✨ Features

- 🎙️ **Dual Transcription Engine** — OpenAI Whisper (English) and Sarvam AI (Hinglish)
- 📋 **Auto-Summarization** — concise meeting/video summaries via Mistral AI
- ✅ **Action Item Extraction** — tasks, owners, and deadlines pulled automatically
- 🔑 **Key Decision & Open Question Detection**
- 💬 **RAG Chat** — ask follow-up questions grounded in the transcript
- 📁 **Flexible Input** — upload audio/video files directly (`.mp3`, `.wav`, `.mp4`, `.m4a`, `.webm`)
- 🎨 **Custom dark-themed UI** built with Streamlit

> **Note:** Direct YouTube link processing is unreliable on cloud hosting (Streamlit Cloud / Render free tier) due to YouTube's bot-detection blocking data-center IPs. **File upload is the recommended and fully supported input method.**

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM Orchestration | LangChain (LCEL) |
| LLM Provider | Mistral AI (`mistral-small-latest`) |
| Speech-to-Text | OpenAI Whisper, Sarvam AI |
| Vector Store | ChromaDB (`langchain-chroma`) |
| Embeddings | HuggingFace / Sentence-Transformers |
| Audio Processing | `pydub`, `ffmpeg`, `yt-dlp` |
| Deployment | Streamlit Community Cloud |

---

## 🚀 Getting Started

### Prerequisites
- Python ≥ 3.10
- `ffmpeg` installed on your system
- Mistral AI API key
- Sarvam AI API key

### Installation

```bash
git clone https://github.com/venom312004/AI-video-assistant.git
cd AI-video-assistant

# create and activate a virtual environment
uv venv
.venv\Scripts\activate     # Windows
# source .venv/bin/activate  # macOS/Linux

# install dependencies
uv pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key
SARVAM_API_KEY=your_sarvam_api_key
```

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
│   ├── rag_engine.py        # RAG chain builder and Q&A
│   ├── summary.py           # Summarization and title generation
│   ├── transcriber.py       # Whisper / Sarvam transcription logic
│   └── vector_store.py      # ChromaDB vector store setup
├── utils/
│   └── audio_processor.py  # Audio download, conversion, chunking
├── requirements.txt
├── packages.txt             # System-level dependencies (ffmpeg)
└── .gitignore
```

---

## ⚠️ Known Limitations

- **YouTube URL input** may fail on cloud deployments with `HTTP 403` or `format not available` errors due to YouTube's bot-detection on data-center IPs. Use the **file upload** option for reliable results.
- Free-tier cloud hosting has limited CPU/RAM, so large files or long videos may take longer to process or hit resource throttling.

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper)
- [Sarvam AI](https://www.sarvam.ai/)
- [Mistral AI](https://mistral.ai/)
- [LangChain](https://www.langchain.com/)
- [Streamlit](https://streamlit.io/)
