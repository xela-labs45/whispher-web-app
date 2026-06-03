"""
Audio/Video Transcription Server
Powered by faster-whisper + FastAPI

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from faster_whisper import WhisperModel

# ─── App setup ───────────────────────────────────────────────────────────────

app = FastAPI(title="Whisper Transcription Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Model cache ─────────────────────────────────────────────────────────────
# Model is loaded once and reused — avoids the ~10s reload penalty per request

_model: WhisperModel | None = None
_model_name: str = ""


def get_model(model_size: str, compute_type: str) -> WhisperModel:
    global _model, _model_name
    key = f"{model_size}:{compute_type}"
    if _model is None or _model_name != key:
        print(f"[whisper] Loading model: {model_size} ({compute_type}) …")
        _model = WhisperModel(
            model_size,
            device="cpu",
            compute_type=compute_type,
            num_workers=1,
        )
        _model_name = key
        print("[whisper] Model ready.")
    return _model


# ─── SSE helper ──────────────────────────────────────────────────────────────

def sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ─── Transcription stream ─────────────────────────────────────────────────────

async def transcribe_stream(
    audio_path: str,
    model_size: str,
    compute_type: str,
    language: str | None,
    beam_size: int,
    word_timestamps: bool,
) -> AsyncGenerator[str, None]:
    """
    Yields SSE events as transcription progresses.
    Runs the blocking Whisper call in a thread pool so the event loop stays free.
    """
    loop = asyncio.get_event_loop()

    yield sse("status", {"message": "Loading model…", "phase": "loading"})
    await asyncio.sleep(0)

    try:
        model = await loop.run_in_executor(None, get_model, model_size, compute_type)
    except Exception as e:
        yield sse("error", {"message": f"Model load failed: {e}"})
        return

    yield sse("status", {"message": "Detecting language…", "phase": "detecting"})
    await asyncio.sleep(0)

    transcribe_kwargs = {
        "beam_size": beam_size,
        "word_timestamps": word_timestamps,
        "vad_filter": True,            # Silero VAD — skips silence automatically
        "vad_parameters": {"min_silence_duration_ms": 500},
    }
    if language and language != "auto":
        transcribe_kwargs["language"] = language

    try:
        segments_gen, info = await loop.run_in_executor(
            None,
            lambda: model.transcribe(audio_path, **transcribe_kwargs),
        )
    except Exception as e:
        yield sse("error", {"message": f"Transcription failed: {e}"})
        return

    yield sse("info", {
        "language": info.language,
        "language_probability": round(info.language_probability * 100, 1),
        "duration": round(info.duration, 1),
    })

    full_text_parts = []
    segment_index = 0

    def iter_segments():
        return list(segments_gen)   # materialize generator in thread

    try:
        segments = await loop.run_in_executor(None, iter_segments)
    except Exception as e:
        yield sse("error", {"message": f"Segment iteration failed: {e}"})
        return

    total = len(segments)

    for seg in segments:
        segment_index += 1
        text = seg.text.strip()
        full_text_parts.append(text)

        payload = {
            "index": segment_index,
            "total": total,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": text,
            "progress": round((segment_index / total) * 100, 1) if total else 0,
        }

        if word_timestamps and seg.words:
            payload["words"] = [
                {"word": w.word, "start": round(w.start, 2), "end": round(w.end, 2)}
                for w in seg.words
            ]

        yield sse("segment", payload)
        await asyncio.sleep(0)   # yield control so client receives event

    yield sse("done", {
        "full_text": " ".join(full_text_parts),
        "segment_count": segment_index,
        "duration": round(info.duration, 1),
    })


# ─── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model_size: str = Form("medium"),
    compute_type: str = Form("int8"),
    language: str = Form("auto"),
    beam_size: int = Form(5),
    word_timestamps: bool = Form(False),
):
    """
    Upload a file and receive a transcription stream via SSE.
    """
    tmpdir = tempfile.mkdtemp()
    audio_path = os.path.join(tmpdir, file.filename)

    try:
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
    finally:
        file.file.close()

    lang = None if language == "auto" else language

    async def event_stream():
        try:
            async for event in transcribe_stream(
                audio_path, model_size, compute_type, lang, beam_size, word_timestamps
            ):
                yield event
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # important for Nginx proxies
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _model is not None, "model": _model_name}


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
