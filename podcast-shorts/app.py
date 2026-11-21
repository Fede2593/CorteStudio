from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import unicodedata
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel


ROOT = Path(__file__).resolve().parent
STATIC = ROOT / "static"
MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "1800"))
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi"}
MODEL: WhisperModel | None = None
MODEL_LOCK = threading.Lock()
TRANSCRIBE_LOCK = threading.Lock()

app = FastAPI(title="CorteStudio", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC), name="static")

STOP_WORDS = set("""
para como pero porque cuando donde quien quienes cual cuales que los las el la un una unos unas de del al y o u en con por sin sobre entre tras desde hasta es son fue eran ser estar esta este estos estas eso esa ese esos esas
yo tu tus mi mis su sus nos me te se le les lo ya no si sí muy mas más menos todo toda todos todas algo nada hay era tiene tener hacer hizo hacen puede pueden quiero quieres quieren dijo dice dicen digo vamos van voy esta también tambien porque entonces así asi aqui aquí ahí ahora esto aquello cada otro otra otros otras parte vez veces manera día dias días año años cosa cosas gente mundo vida
the a an and or of to is in on with for from it this that was were are be been i we you they he she not but so if have has had do did does as at by up out about into over after before your his their its then there when who what why how all any more less than can will would should could
""".split())

HOOK_WORDS = {
    "verdad", "mentira", "error", "secreto", "nadie", "siempre", "nunca", "fracaso",
    "éxito", "exito", "dinero", "miedo", "problema", "solución", "solucion", "cambio",
    "decisión", "decision", "aprendí", "aprendi", "descubrí", "descubri", "ojo", "clave",
    "increíble", "increible", "imposible", "peor", "mejor", "realmente", "imagina",
    "prueba", "funciona", "perdí", "perdi", "gané", "gane", "porqué", "porqué",
}


def get_model() -> WhisperModel:
    global MODEL
    if MODEL is None:
        with MODEL_LOCK:
            if MODEL is None:
                MODEL = WhisperModel(
                    MODEL_NAME,
                    device=os.getenv("WHISPER_DEVICE", "cpu"),
                    compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
                    cpu_threads=max(1, int(os.getenv("WHISPER_CPU_THREADS", "4"))),
                )
    return MODEL


def clean_temp(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


async def save_upload(upload: UploadFile) -> tuple[str, Path]:
    name = Path(upload.filename or "podcast.mp4")
    if name.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Formato no compatible. Usa MP4, MOV, MKV, WEBM, M4V o AVI.")
    temp_dir = tempfile.mkdtemp(prefix="cortestudio-")
    target = Path(temp_dir) / ("source" + name.suffix.lower())
    total = 0
    try:
        with target.open("wb") as out:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_MB * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"El archivo supera el máximo de {MAX_UPLOAD_MB} MB.")
                out.write(chunk)
    except Exception:
        clean_temp(temp_dir)
        raise
    finally:
        await upload.close()
    if total == 0:
        clean_temp(temp_dir)
        raise HTTPException(status_code=400, detail="El archivo está vacío.")
    return temp_dir, target


def probe_duration(path: Path) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
            check=True, capture_output=True, text=True, timeout=60,
        )
        return float(json.loads(result.stdout)["format"]["duration"])
    except Exception as exc:
        raise ValueError("No pude leer la duración del video. Comprueba que el archivo no esté dañado.") from exc


def normalized_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.findall(r"[a-z0-9]{3,}", text)


def pick_tags(text: str) -> tuple[list[str], list[str]]:
    words = normalized_words(text)
    counts: dict[str, int] = {}
    for word in words:
        if word not in STOP_WORDS and not word.isdigit() and len(word) >= 4:
            counts[word] = counts.get(word, 0) + 1
    ranked = sorted(counts, key=lambda w: (counts[w], len(w)), reverse=True)
    keywords = ranked[:5]
    hashtags = ["#" + word for word in keywords[:3]]
    if "#podcast" not in hashtags:
        hashtags.append("#podcast")
    return keywords, hashtags[:4]


def candidate_score(text: str, duration: float, start: float, end: float) -> tuple[int, str]:
    words = normalized_words(text)
    raw = text.strip()
    if not words or duration <= 0:
        return 0, "Poco diálogo reconocible"

    density = len(words) / duration
    score = 0
    reasons: list[str] = []
    low = raw.lower()

    if "?" in raw or "¿" in raw:
        score += 18
        reasons.append("abre con pregunta o plantea una duda")
    elif re.match(r"^(mira|imagina|ojo|sabes|sabías|sabias|te cuento|la verdad|nunca|nadie)\b", low):
        score += 16
        reasons.append("abre con una frase que puede captar atención")
    else:
        score += 7

    signal_count = sum(1 for word in words if word in HOOK_WORDS)
    if signal_count:
        score += min(18, signal_count * 7)
        reasons.append("incluye palabras de contraste, decisión o aprendizaje")

    if any(token in low for token in ("pero", "sin embargo", "en cambio", "resulta que", "lo que pasó")):
        score += 12
        reasons.append("presenta contraste o giro")
    if raw.rstrip().endswith((".", "!", "?", "…", "”", '"')):
        score += 10
        reasons.append("cierra una idea completa")
    if 1.8 <= density <= 3.5:
        score += 18
        reasons.append("mantiene una densidad de diálogo adecuada")
    elif 1.1 <= density <= 4.2:
        score += 11
    else:
        score += 5

    if 24 <= duration <= 48:
        score += 17
    elif 18 <= duration <= 60:
        score += 13
    else:
        score += 7

    if len(set(words)) / max(1, len(words)) > 0.48:
        score += 7
    reason = "; ".join(reasons[:3]) if reasons else "fragmento breve y autónomo"
    return min(100, score), reason


def make_title(text: str) -> str:
    candidates = [part.strip(" \n\t\"“”") for part in re.split(r"(?<=[.!?])\s+", text) if len(part.strip()) >= 12]
    if not candidates:
        candidates = [text.strip()]
    chosen = max(candidates[:5], key=lambda item: ("?" in item or "¿" in item, len(normalized_words(item))))
    chosen = re.sub(r"\s+", " ", chosen)
    if len(chosen) > 72:
        chosen = chosen[:69].rsplit(" ", 1)[0] + "…"
    return chosen or "Una idea para conversar"


def find_candidates(segments: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    raw_candidates: list[dict[str, Any]] = []
    min_len, max_len = 18.0, 58.0
    for i in range(len(segments)):
        start = float(segments[i]["start"])
        parts: list[str] = []
        for j in range(i, len(segments)):
            end = float(segments[j]["end"])
            span = end - start
            if span > max_len:
                break
            parts.append(str(segments[j]["text"]).strip())
            if span < min_len:
                continue
            text = " ".join(part for part in parts if part)
            score, reason = candidate_score(text, span, start, end)
            keywords, hashtags = pick_tags(text)
            raw_candidates.append({
                "start": round(start, 2), "end": round(end, 2), "duration": round(span, 1),
                "score": score, "reason": reason, "transcript": text,
                "title": make_title(text), "hashtags": hashtags,
                "tags": keywords[:5], "words_per_minute": round(len(normalized_words(text)) / span * 60),
            })

    raw_candidates.sort(key=lambda item: item["score"], reverse=True)
    selected: list[dict[str, Any]] = []
    for item in raw_candidates:
        if all(item["end"] <= old["start"] + 4 or item["start"] >= old["end"] - 4 for old in selected):
            item["id"] = f"clip-{len(selected)+1}"
            selected.append(item)
        if len(selected) >= 5:
            break
    selected.sort(key=lambda item: item["start"])
    for idx, item in enumerate(selected, 1):
        item["id"] = f"clip-{idx}"
    return selected


def analyze_video(path: Path) -> dict[str, Any]:
    try:
        duration = probe_duration(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    try:
        with TRANSCRIBE_LOCK:
            model = get_model()
            segment_iter, info = model.transcribe(
                str(path), beam_size=3, vad_filter=True, word_timestamps=False,
                condition_on_previous_text=False,
            )
            segments = [
                {"start": round(float(seg.start), 2), "end": round(float(seg.end), 2), "text": seg.text.strip()}
                for seg in segment_iter if seg.text.strip()
            ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Falló la transcripción: {exc}") from exc

    if not segments:
        raise HTTPException(status_code=422, detail="No encontré voz clara. Prueba con un archivo que tenga diálogo audible.")
    candidates = find_candidates(segments, duration)
    if not candidates:
        raise HTTPException(status_code=422, detail="No encontré fragmentos de diálogo de 18 segundos o más.")
    keywords, hashtags = pick_tags(" ".join(seg["text"] for seg in segments))
    return {
        "duration": round(duration, 2), "language": getattr(info, "language", "unknown"),
        "language_probability": round(float(getattr(info, "language_probability", 0.0)), 2),
        "segments": segments, "candidates": candidates,
        "keywords": keywords[:8], "hashtags": hashtags,
        "notice": "Puntuación editorial heurística; no predice viralidad ni rendimiento.",
    }


def srt_time(value: float) -> str:
    ms = max(0, round(value * 1000))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    seconds, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"


def write_srt(path: Path, captions: list[dict[str, Any]], clip_start: float, clip_end: float) -> bool:
    lines: list[str] = []
    idx = 1
    for caption in captions:
        start = max(clip_start, float(caption.get("start", 0)))
        end = min(clip_end, float(caption.get("end", 0)))
        text = str(caption.get("text", "")).strip()
        if end <= start or not text:
            continue
        lines.extend([str(idx), f"{srt_time(start-clip_start)} --> {srt_time(end-clip_start)}", text, ""])
        idx += 1
    if not lines:
        return False
    path.write_text("\n".join(lines), encoding="utf-8")
    return True


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "model": MODEL_NAME, "model_loaded": MODEL is not None}


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> JSONResponse:
    temp_dir, path = await save_upload(file)
    try:
        result = await asyncio.to_thread(analyze_video, path)
        return JSONResponse(result)
    finally:
        clean_temp(temp_dir)


@app.post("/api/clip")
async def export_clip(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    start: float = Form(...),
    end: float = Form(...),
    crop_position: str = Form("center"),
    captions_json: str = Form("[]"),
) -> FileResponse:
    if start < 0 or end <= start or end - start > 180:
        raise HTTPException(status_code=400, detail="El recorte debe durar entre 0 y 180 segundos.")
    if crop_position not in {"center", "left", "right"}:
        crop_position = "center"
    try:
        captions = json.loads(captions_json)
        if not isinstance(captions, list):
            captions = []
    except json.JSONDecodeError:
        captions = []

    temp_dir, source = await save_upload(file)
    work = Path(temp_dir)
    output = work / "cortestudio-short.mp4"
    srt = work / "captions.srt"
    has_captions = write_srt(srt, captions, start, end)
    x_position = {"center": "(in_w-1080)/2", "left": "0", "right": "in_w-1080"}[crop_position]
    video_filter = f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920:{x_position}:0,setsar=1"
    if has_captions:
        subtitle_path = str(srt).replace("\\", "/").replace(":", r"\:")
        video_filter += f",subtitles='{subtitle_path}':force_style='FontName=Arial,FontSize=21,Outline=2,Shadow=1,Alignment=2,MarginV=150'"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(start), "-i", str(source),
        "-t", str(end - start), "-map", "0:v:0", "-map", "0:a?", "-vf", video_filter,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", str(output),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        _, stderr = await proc.communicate()
        if proc.returncode != 0 or not output.exists():
            clean_temp(temp_dir)
            detail = stderr.decode("utf-8", "replace")[-1000:] or "FFmpeg no pudo generar el recorte."
            raise HTTPException(status_code=500, detail=detail)
    except FileNotFoundError as exc:
        clean_temp(temp_dir)
        raise HTTPException(status_code=500, detail="FFmpeg no está instalado en el servidor.") from exc

    background_tasks.add_task(clean_temp, temp_dir)
    return FileResponse(output, media_type="video/mp4", filename="cortestudio-short.mp4")


@app.get("/")
def home() -> FileResponse:
    return FileResponse(STATIC / "index.html")
