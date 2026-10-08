"""
Kashmiri Voice (کٲشُر آواز) - Interactive Web Application Server.
Provides REST APIs for:
1. Neural Text-to-Speech synthesis with regional Kashmiri accent conditioning.
2. Regional Kashmiri accent classification (Kupwara, Bandipora, Shopian, Islamabad).
3. Speech corpora explorer and audio streaming.
"""

import os
import sys
import shutil
import uuid
from pathlib import Path

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import (
    SAMPLES_DATA_DIR,
    KASHMIRI_DIALECTS,
    SPEAKER_ID_MAP
)
from scripts.inference import synthesize_kashmiri_speech, BENCHMARK_SENTENCES, load_kashmiri_tts_model
from scripts.accent_classifier import predict_kashmiri_accent

from contextlib import asynccontextmanager

tts_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model
    print("[+] Initializing Kashmiri Neural TTS Model on startup...")
    tts_model = load_kashmiri_tts_model()
    print("[OK] Kashmiri Voice Server ready.")
    yield

# Initialize FastAPI app
app = FastAPI(
    title="Kashmiri Voice (کٲشُر آواز)",
    description="Neural Speech Synthesis & Regional Accent Classification for Kashmiri",
    version="1.0.0",
    lifespan=lifespan
)

# Ensure samples directory exists
SAMPLES_DATA_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = PROJECT_ROOT / "static"

# Mount static web assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/samples", StaticFiles(directory=str(SAMPLES_DATA_DIR)), name="samples")


class SynthesizeRequest(BaseModel):
    text: str
    dialect: str = "kupwara"


@app.get("/")
def serve_index():
    """Serves the main interactive web dashboard."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Index file not found")
    return FileResponse(str(index_file))


@app.get("/api/dialects")
def get_dialects():
    """Returns available Kashmiri dialect / accent options."""
    return {"dialects": KASHMIRI_DIALECTS}


@app.get("/api/benchmark-samples")
def get_benchmark_samples():
    """Returns preset authentic Kashmiri benchmark sentences."""
    return {
        "benchmarks": [
            {"id": tag, "text": text, "translation": trans}
            for tag, text, trans in BENCHMARK_SENTENCES
        ]
    }


@app.post("/api/synthesize")
def api_synthesize(req: SynthesizeRequest):
    """Synthesizes speech from written Kashmiri text in the chosen regional dialect."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    dialect_key = req.dialect.lower()
    dialect_info = KASHMIRI_DIALECTS.get(dialect_key, {
        "name_en": dialect_key.capitalize(),
        "dialect_zone": "General Kashmiri",
        "speaker_id": "spk_kupwara"
    })

    speaker_spk = dialect_info.get("speaker_id", "spk_kupwara")
    speaker_idx = SPEAKER_ID_MAP.get(speaker_spk, 0)

    # Unique output filename
    out_id = uuid.uuid4().hex[:8]
    filename = f"synth_{dialect_key}_{out_id}.wav"
    out_path = SAMPLES_DATA_DIR / filename

    try:
        synthesize_kashmiri_speech(
            text=req.text,
            output_path=out_path,
            model=tts_model,
            speaker_id=speaker_idx
        )

        import soundfile as sf
        info = sf.info(str(out_path))
        duration = info.duration

        return {
            "status": "success",
            "audio_url": f"/samples/{filename}",
            "filename": filename,
            "duration": round(duration, 2),
            "dialect_name": dialect_info.get("name_en", dialect_key),
            "dialect_zone": dialect_info.get("dialect_zone", "Kashmiri")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis error: {str(e)}")


@app.post("/api/classify")
async def api_classify(file: UploadFile = File(...)):
    """Classifies an uploaded Kashmiri audio recording into regional dialect clusters."""
    if not file.filename.lower().endswith((".wav", ".mp3", ".ogg", ".flac")):
        raise HTTPException(status_code=400, detail="Invalid audio format. Please upload WAV, MP3, or OGG.")

    temp_id = uuid.uuid4().hex[:8]
    temp_path = SAMPLES_DATA_DIR / f"temp_upload_{temp_id}_{file.filename}"

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        result = predict_kashmiri_accent(temp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print(" 🎙️ STARTING KASHMIRI VOICE (کٲشُر آواز) WEB APPLICATION")
    print(" Local URL: http://localhost:8000")
    print("="*70 + "\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
