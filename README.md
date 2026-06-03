# 🎙️ Whisper Transcriber

Local audio/video transcription powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and FastAPI.  
Segments stream to the browser in real time as Whisper processes your file — no waiting for the whole job to finish.

---

## Why FastAPI instead of Streamlit?

Streamlit re-runs the entire script on every interaction, which makes long-running tasks (2–5 min transcription) awkward and blocks the UI. FastAPI with Server-Sent Events (SSE) lets Whisper segments stream to the browser token-by-token as they are produced, giving a much better experience.

---

## Features

- Chunked real-time streaming via SSE — long files show text as chunks complete
- CPU speed presets — fast draft, balanced, and accurate modes
- Model cache — model loads once, reused across requests (no 10-second reload penalty)
- Silero VAD filter — silences automatically skipped
- Language auto-detection with confidence score
- Word-level timestamps (optional)
- Export to TXT, SRT, and/or JSON
- Drag-and-drop file upload
- Supports MP4, MKV, MOV, AVI, MP3, WAV, M4A, WEBM, FLAC, OGG

---

## Setup

```bash
# 1. Clone / copy the project
cd transcriber

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Run
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in your browser.

---

## Model guide (CPU INT8)

| Model | RAM | Speed (13 min audio) | Best for |
|---|---|---|---|
| tiny | ~500 MB | ~1 min | Quick drafts |
| **base** | **~700 MB** | **~1.5 min** | **Default fast draft** |
| small | ~1.5 GB | ~2 min | Good daily balance |
| medium | ~2.5 GB | ~4 min | Accurate preset |
| large-v3 | ~4+ GB | ~6 min | Maximum accuracy |

The browser defaults to **Fast draft** mode for long CPU-only files. It uses the `base` model, INT8 quantization, beam size 1, batched inference, 8 CPU threads, and 5-minute chunks with 5-second overlap so text appears progressively. Choose **Balanced** for the `small` model or **Accurate** for `medium`.

---

## Project structure

```
transcriber/
├── main.py          # FastAPI backend + SSE transcription stream
├── index.html       # Frontend UI (served at /)
├── requirements.txt
└── README.md
```

---

## Notes

- No GPU required — runs on CPU with INT8 quantization
- No FFmpeg install needed — faster-whisper bundles PyAV
- Files are processed in a temporary directory and deleted immediately after transcription
- Long files are decoded once, split into overlapping chunks, and streamed chunk-by-chunk
- The app allows one transcription job at a time so CPU threads do not fight each other
- The model is loaded once and kept in memory; changing speed presets reloads the matching model automatically
