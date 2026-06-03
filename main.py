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
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from faster_whisper import BatchedInferencePipeline, WhisperModel
from faster_whisper.audio import decode_audio

app = FastAPI(title="Whisper Transcription Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_MODEL_SIZES = {"tiny", "base", "small", "medium", "large-v3"}
ALLOWED_COMPUTE_TYPES = {"int8", "float32"}
ALLOWED_EXTENSIONS = {
    ".avi",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".wav",
    ".webm",
}
SAMPLE_RATE = 16000
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
COPY_CHUNK_SIZE = 1024 * 1024
DEFAULT_CPU_THREADS = min(8, os.cpu_count() or 4)
DEFAULT_NUM_WORKERS = 1


@dataclass(frozen=True)
class SpeedPreset:
    model_size: str
    compute_type: str
    beam_size: int
    batch_size: int
    cpu_threads: int
    chunk_seconds: int
    overlap_seconds: int


@dataclass(frozen=True)
class TranscriptionConfig:
    preset: str
    model_size: str
    compute_type: str
    language: str | None
    beam_size: int
    batch_size: int
    cpu_threads: int
    num_workers: int
    chunk_seconds: int
    overlap_seconds: int


SPEED_PRESETS = {
    "fast_draft": SpeedPreset(
        model_size="base",
        compute_type="int8",
        beam_size=1,
        batch_size=4,
        cpu_threads=DEFAULT_CPU_THREADS,
        chunk_seconds=300,
        overlap_seconds=5,
    ),
    "balanced": SpeedPreset(
        model_size="small",
        compute_type="int8",
        beam_size=3,
        batch_size=4,
        cpu_threads=DEFAULT_CPU_THREADS,
        chunk_seconds=300,
        overlap_seconds=5,
    ),
    "accurate": SpeedPreset(
        model_size="medium",
        compute_type="int8",
        beam_size=5,
        batch_size=2,
        cpu_threads=DEFAULT_CPU_THREADS,
        chunk_seconds=240,
        overlap_seconds=5,
    ),
}

_model: WhisperModel | None = None
_pipeline: BatchedInferencePipeline | None = None
_model_key: str = ""
_model_lock = threading.Lock()
_transcription_semaphore = asyncio.Semaphore(1)


def sse(event: str, data: dict) -> str:
    """Format a Server-Sent Event message."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def get_pipeline(config: TranscriptionConfig) -> BatchedInferencePipeline:
    """Load and cache the CPU-threaded model and batched inference wrapper."""
    global _model, _pipeline, _model_key
    key = ":".join(
        [
            config.model_size,
            config.compute_type,
            str(config.cpu_threads),
            str(config.num_workers),
        ]
    )
    with _model_lock:
        if _model is None or _pipeline is None or _model_key != key:
            print(
                "[whisper] Loading model: "
                f"{config.model_size} ({config.compute_type}, "
                f"threads={config.cpu_threads}, workers={config.num_workers})"
            )
            _model = WhisperModel(
                config.model_size,
                device="cpu",
                compute_type=config.compute_type,
                cpu_threads=config.cpu_threads,
                num_workers=config.num_workers,
            )
            _pipeline = BatchedInferencePipeline(model=_model)
            _model_key = key
            print("[whisper] Model ready.")
        return _pipeline


def clean_language(language: str) -> str | None:
    language = language.strip().lower()
    if language == "auto":
        return None
    if not (2 <= len(language) <= 8 and language.replace("-", "").isalpha()):
        raise HTTPException(status_code=400, detail="Unsupported language code.")
    return language


def validate_positive_int(name: str, value: int, minimum: int, maximum: int) -> int:
    if not minimum <= value <= maximum:
        raise HTTPException(status_code=400, detail=f"{name} must be between {minimum} and {maximum}.")
    return value


def validate_settings(
    speed_preset: str | None,
    model_size: str,
    compute_type: str,
    language: str,
    beam_size: int,
    batch_size: int,
    cpu_threads: int,
    num_workers: int,
    chunk_seconds: int,
    overlap_seconds: int,
) -> TranscriptionConfig:
    lang = clean_language(language)
    preset_name = (speed_preset or "").strip()

    if preset_name:
        preset = SPEED_PRESETS.get(preset_name)
        if preset is None:
            raise HTTPException(status_code=400, detail="Unsupported speed preset.")
        return TranscriptionConfig(
            preset=preset_name,
            model_size=preset.model_size,
            compute_type=preset.compute_type,
            language=lang,
            beam_size=preset.beam_size,
            batch_size=preset.batch_size,
            cpu_threads=preset.cpu_threads,
            num_workers=DEFAULT_NUM_WORKERS,
            chunk_seconds=preset.chunk_seconds,
            overlap_seconds=preset.overlap_seconds,
        )

    model_size = model_size.strip()
    compute_type = compute_type.strip()
    if model_size not in ALLOWED_MODEL_SIZES:
        raise HTTPException(status_code=400, detail="Unsupported model size.")
    if compute_type not in ALLOWED_COMPUTE_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported compute type.")

    beam_size = validate_positive_int("Beam size", beam_size, 1, 10)
    batch_size = validate_positive_int("Batch size", batch_size, 1, 16)
    cpu_threads = validate_positive_int("CPU threads", cpu_threads, 1, max(1, os.cpu_count() or 1))
    num_workers = validate_positive_int("Workers", num_workers, 1, 4)
    chunk_seconds = validate_positive_int("Chunk seconds", chunk_seconds, 60, 900)
    overlap_seconds = validate_positive_int("Overlap seconds", overlap_seconds, 0, 30)
    if overlap_seconds >= chunk_seconds:
        raise HTTPException(status_code=400, detail="Overlap must be shorter than chunk length.")

    return TranscriptionConfig(
        preset="custom",
        model_size=model_size,
        compute_type=compute_type,
        language=lang,
        beam_size=beam_size,
        batch_size=batch_size,
        cpu_threads=cpu_threads,
        num_workers=num_workers,
        chunk_seconds=chunk_seconds,
        overlap_seconds=overlap_seconds,
    )


def validate_upload_name(filename: str | None) -> str:
    safe_name = Path(filename or "").name
    if not safe_name:
        raise HTTPException(status_code=400, detail="Uploaded file must have a name.")
    if Path(safe_name).suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    return safe_name


async def save_upload(file: UploadFile, destination: Path) -> int:
    total = 0
    try:
        with destination.open("wb") as output:
            while True:
                chunk = await file.read(COPY_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Uploaded file is too large.")
                output.write(chunk)
    finally:
        await file.close()
    return total


def build_chunk_windows(duration: float, chunk_seconds: int, overlap_seconds: int) -> list[dict]:
    windows = []
    start = 0.0
    index = 1
    while start < duration:
        end = min(start + chunk_seconds, duration)
        emit_after = 0.0 if index == 1 else start + overlap_seconds
        windows.append({"index": index, "start": start, "end": end, "emit_after": emit_after})
        if end >= duration:
            break
        start = max(0.0, end - overlap_seconds)
        index += 1
    return windows


def audio_slice(audio, start_seconds: float, end_seconds: float):
    start_sample = int(start_seconds * SAMPLE_RATE)
    end_sample = int(end_seconds * SAMPLE_RATE)
    return audio[start_sample:end_sample]


def text_tail(parts: list[str], max_chars: int = 240) -> str | None:
    tail = " ".join(parts).strip()
    if not tail:
        return None
    return tail[-max_chars:]


def segment_words(words, offset: float) -> list[dict]:
    return [
        {
            "word": word.word,
            "start": round(word.start + offset, 2),
            "end": round(word.end + offset, 2),
        }
        for word in words or []
    ]


async def next_segment(loop: asyncio.AbstractEventLoop, iterator):
    def read_next():
        try:
            return next(iterator)
        except StopIteration:
            return None

    return await loop.run_in_executor(None, read_next)


async def transcribe_stream(
    audio_path: str,
    config: TranscriptionConfig,
    word_timestamps: bool,
) -> AsyncGenerator[str, None]:
    """Yield SSE events for decoded, overlapped chunks so long CPU jobs show text early."""
    loop = asyncio.get_running_loop()

    if _transcription_semaphore.locked():
        yield sse("status", {"message": "Waiting for current transcription to finish...", "phase": "queued"})

    async with _transcription_semaphore:
        yield sse(
            "status",
            {
                "message": "Loading model...",
                "phase": "loading",
                "preset": config.preset,
                "model": config.model_size,
                "cpu_threads": config.cpu_threads,
                "batch_size": config.batch_size,
            },
        )
        await asyncio.sleep(0)

        try:
            pipeline = await loop.run_in_executor(None, get_pipeline, config)
        except Exception as e:
            yield sse("error", {"message": f"Model load failed: {e}"})
            return

        yield sse("status", {"message": "Decoding audio...", "phase": "decoding"})
        await asyncio.sleep(0)

        try:
            audio = await loop.run_in_executor(None, decode_audio, audio_path, SAMPLE_RATE)
        except Exception as e:
            yield sse("error", {"message": f"Audio decode failed: {e}"})
            return

        duration = round(float(audio.shape[0]) / SAMPLE_RATE, 1)
        if duration <= 0:
            yield sse("error", {"message": "Decoded audio is empty."})
            return

        windows = build_chunk_windows(duration, config.chunk_seconds, config.overlap_seconds)
        yield sse(
            "info",
            {
                "language": config.language or "detecting",
                "language_probability": None,
                "duration": duration,
                "preset": config.preset,
                "model": config.model_size,
                "batch_size": config.batch_size,
                "cpu_threads": config.cpu_threads,
                "chunk_count": len(windows),
            },
        )

        full_text_parts: list[str] = []
        segment_index = 0
        detected_language = config.language
        detected_probability = None

        for window in windows:
            chunk_index = window["index"]
            chunk_start = float(window["start"])
            chunk_end = float(window["end"])
            emit_after = float(window["emit_after"])
            chunk_audio = audio_slice(audio, chunk_start, chunk_end)
            progress = round(min((chunk_start / duration) * 100, 100), 1)

            yield sse(
                "chunk_start",
                {
                    "index": chunk_index,
                    "total": len(windows),
                    "start": round(chunk_start, 2),
                    "end": round(chunk_end, 2),
                    "progress": progress,
                },
            )
            await asyncio.sleep(0)

            kwargs = {
                "language": detected_language,
                "beam_size": config.beam_size,
                "best_of": max(config.beam_size, 1),
                "word_timestamps": word_timestamps,
                "vad_filter": True,
                "vad_parameters": {"min_silence_duration_ms": 500},
                "chunk_length": 30,
                "batch_size": config.batch_size,
                "condition_on_previous_text": False,
                "initial_prompt": text_tail(full_text_parts),
                "language_detection_segments": 1,
            }

            try:
                segments_gen, info = await loop.run_in_executor(
                    None,
                    lambda: pipeline.transcribe(chunk_audio, **kwargs),
                )
            except Exception as e:
                yield sse("error", {"message": f"Chunk {chunk_index} failed: {e}"})
                return

            if detected_language is None:
                detected_language = info.language
                detected_probability = (
                    round(info.language_probability * 100, 1)
                    if info.language_probability is not None
                    else None
                )
                yield sse(
                    "info",
                    {
                        "language": detected_language,
                        "language_probability": detected_probability,
                        "duration": duration,
                        "preset": config.preset,
                        "model": config.model_size,
                        "batch_size": config.batch_size,
                        "cpu_threads": config.cpu_threads,
                        "chunk_count": len(windows),
                    },
                )

            while True:
                try:
                    seg = await next_segment(loop, segments_gen)
                except Exception as e:
                    yield sse("error", {"message": f"Chunk {chunk_index} iteration failed: {e}"})
                    return
                if seg is None:
                    break

                abs_start = round(seg.start + chunk_start, 2)
                abs_end = round(seg.end + chunk_start, 2)
                if abs_end <= emit_after:
                    continue

                text = seg.text.strip()
                if not text:
                    continue

                segment_index += 1
                full_text_parts.append(text)
                progress = round(min((abs_end / duration) * 100, 100), 1)
                payload = {
                    "index": segment_index,
                    "chunk_index": chunk_index,
                    "chunk_total": len(windows),
                    "start": abs_start,
                    "end": abs_end,
                    "text": text,
                    "progress": progress,
                }

                if word_timestamps and seg.words:
                    payload["words"] = segment_words(seg.words, chunk_start)

                yield sse("segment", payload)
                await asyncio.sleep(0)

            yield sse(
                "chunk_done",
                {
                    "index": chunk_index,
                    "total": len(windows),
                    "progress": round(min((chunk_end / duration) * 100, 100), 1),
                },
            )
            await asyncio.sleep(0)

        yield sse(
            "done",
            {
                "full_text": " ".join(full_text_parts),
                "segment_count": segment_index,
                "duration": duration,
                "language": detected_language,
                "language_probability": detected_probability,
                "preset": config.preset,
            },
        )


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    speed_preset: str | None = Form("fast_draft"),
    model_size: str = Form("medium"),
    compute_type: str = Form("int8"),
    language: str = Form("auto"),
    beam_size: int = Form(5),
    batch_size: int = Form(4),
    cpu_threads: int = Form(DEFAULT_CPU_THREADS),
    num_workers: int = Form(DEFAULT_NUM_WORKERS),
    chunk_seconds: int = Form(300),
    overlap_seconds: int = Form(5),
    word_timestamps: bool = Form(False),
):
    """Upload a media file and receive a chunked transcription stream via SSE."""
    config = validate_settings(
        speed_preset,
        model_size,
        compute_type,
        language,
        beam_size,
        batch_size,
        cpu_threads,
        num_workers,
        chunk_seconds,
        overlap_seconds,
    )
    safe_name = validate_upload_name(file.filename)
    tmpdir = Path(tempfile.mkdtemp(prefix="whisper-upload-"))
    audio_path = tmpdir / safe_name

    try:
        upload_size = await save_upload(file, audio_path)
        if upload_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise

    async def event_stream():
        try:
            async for event in transcribe_stream(str(audio_path), config, word_timestamps):
                yield event
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model": _model_key,
        "presets": list(SPEED_PRESETS.keys()),
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
