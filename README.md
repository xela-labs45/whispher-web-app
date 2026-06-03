# Whisper Transcriber

Local audio/video transcription powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) and FastAPI. The app is optimized for CPU-only machines: long files are decoded once, split into overlapping chunks, transcribed with batched inference, and streamed to the browser as text becomes available.

## Features

- Chunked Server-Sent Events (SSE) streaming so long files show progress and text incrementally
- CPU speed presets: Fast draft, Balanced, and Accurate
- Batched faster-whisper inference with INT8 CPU quantization by default
- Model and batched pipeline cache to avoid reloading the same model per request
- One active transcription job at a time so CPU threads do not fight each other
- Language auto-detection, optional word-level timestamps, and export to TXT, SRT, or JSON
- Safe upload handling: extension validation, temp-file cleanup, empty-file rejection, and upload size cap

## Setup

```bash
# 1. Create and activate a local virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open `http://localhost:8000` in your browser. The first transcription with a new model downloads model files from Hugging Face and caches them locally.

## CPU Presets

| Preset | Model | Beam | Batch | Chunking | Best for |
|---|---|---:|---:|---|---|
| Fast draft | `base` | 1 | 4 | 5 min + 5 sec overlap | Long CPU-only files and quick drafts |
| Balanced | `small` | 3 | 4 | 5 min + 5 sec overlap | Better accuracy without jumping to medium |
| Accurate | `medium` | 5 | 2 | 4 min + 5 sec overlap | Slower, higher-quality transcripts |

All presets use `int8`, up to 8 CPU threads, VAD silence filtering, and `faster_whisper.BatchedInferencePipeline`. The browser defaults to Fast draft.

## API

`POST /transcribe` accepts multipart form data:

- `file`: audio/video upload. Supported extensions: MP4, MKV, MOV, AVI, MP3, WAV, M4A, WEBM, OGG, FLAC.
- `speed_preset`: `fast_draft`, `balanced`, or `accurate`. Defaults to `fast_draft`.
- `language`: `auto` or a language code such as `en`, `fr`, `de`, `es`, `pt`, `zh`, `ar`.
- `word_timestamps`: `true` or `false`.

For compatibility, the backend also accepts custom fields when `speed_preset` is omitted: `model_size`, `compute_type`, `beam_size`, `batch_size`, `cpu_threads`, `num_workers`, `chunk_seconds`, and `overlap_seconds`.

The response is `text/event-stream` and may emit `status`, `info`, `chunk_start`, `segment`, `chunk_done`, `done`, or `error` events. Check service status with `GET /health`.

## Project Structure

```text
whispher-web-app/
├── main.py            # FastAPI backend, upload validation, chunking, SSE stream
├── index.html         # Browser UI served at /
├── requirements.txt   # Python runtime dependencies
├── README.md          # User-facing setup and operation guide
├── AGENTS.md          # Contributor/agent guidelines
└── .gitignore         # Ignores venvs, pycache, env files, logs, generated outputs
```

## Operational Notes

- No GPU is required, but CPU transcription can still be slow for very long files.
- The upload is copied to a temp directory and removed after the stream ends or fails.
- `MAX_UPLOAD_BYTES` controls the upload cap and defaults to 2 GB.
- Only one transcription runs at a time by design. Extra requests wait and receive a queued status event.
- Extremely long files are still decoded into memory before chunking. A future improvement would stream/decode directly into disk-backed chunks.
