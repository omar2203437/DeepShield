import csv
import uuid
import datetime
import shutil
import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from video_detector import VideoDeepfakeDetector
from image_detector import ImageDeepfakeDetector
from audio_detector import AudioDeepfakeDetector

# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_POOL    = PROJECT_ROOT / "data_pool"
INCOMING     = DATA_POOL / "incoming"
LABELED_REAL = DATA_POOL / "labeled" / "real"
LABELED_FAKE = DATA_POOL / "labeled" / "fake"
META_CSV     = DATA_POOL / "metadata.csv"
FRONTEND     = PROJECT_ROOT / "frontend" / "index.html"

INCOMING.mkdir(parents=True, exist_ok=True)
LABELED_REAL.mkdir(parents=True, exist_ok=True)
LABELED_FAKE.mkdir(parents=True, exist_ok=True)


def log_row(row: dict):
    file_exists = META_CSV.exists()
    with open(META_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            w.writeheader()
        w.writerow(row)


# -------------------------
# App
# -------------------------
app = FastAPI(title="DeepShield")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load models once at startup
# -------------------------
image_detector = ImageDeepfakeDetector(
    (PROJECT_ROOT / "models" / "image_deepfake_detector_best").as_posix()
)
audio_detector = AudioDeepfakeDetector(
    (PROJECT_ROOT / "models" / "wavlm-deepfake-detector-final").as_posix()
)

# Standard model: 0.827 AUC, face-crop enabled — best for face-swap deepfakes
video_detector_standard = VideoDeepfakeDetector(
    model_path=(PROJECT_ROOT / "models" / "clip_finetuned_0827.torchscript").as_posix(),
    use_face_crop=True,
)
# Sensitive model: 0.799 AUC, no face-crop — better generalization for other deepfake types
video_detector_sensitive = VideoDeepfakeDetector(
    model_path=(PROJECT_ROOT / "models" / "clip_finetuned.torchscript").as_posix(),
    use_face_crop=False,
)


# -------------------------
# UI
# -------------------------
@app.get("/", response_class=FileResponse)
def serve_ui():
    if not FRONTEND.exists():
        return JSONResponse({"error": f"Missing frontend at: {FRONTEND}"}, status_code=500)
    return FileResponse(FRONTEND)


# -------------------------
# Video prediction
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form("standard")):
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")):
        return JSONResponse({"error": "Please upload a video file."}, status_code=400)

    detector = video_detector_sensitive if model == "sensitive" else video_detector_standard

    upload_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    save_path = INCOMING / f"{upload_id}{ext}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = detector.predict(str(save_path))
    except Exception as e:
        return JSONResponse({"error": f"Video prediction failed: {str(e)}"}, status_code=500)

    log_row({
        "timestamp":  datetime.datetime.now().isoformat(),
        "upload_id":  upload_id,
        "filename":   file.filename,
        "stored_path": str(save_path),
        "pred_label": result.get("label", ""),
        "confidence": result.get("confidence", ""),
        "prob_fake":  result.get("prob_fake", ""),
        "prob_real":  result.get("prob_real", ""),
        "user_label": "",
    })

    result["upload_id"] = upload_id
    return result


# -------------------------
# Video feedback
# -------------------------
class FeedbackRequest(BaseModel):
    upload_id: str
    label: str

@app.post("/feedback")
def feedback(payload: FeedbackRequest):
    label = payload.label.strip().lower()
    if label not in ("real", "fake"):
        return JSONResponse({"error": "label must be 'real' or 'fake'."}, status_code=400)

    matches = list(INCOMING.glob(f"{payload.upload_id}.*"))
    if not matches:
        return JSONResponse({"error": "upload_id not found."}, status_code=404)

    src = matches[0]
    dst = (LABELED_REAL if label == "real" else LABELED_FAKE) / src.name
    src.rename(dst)

    log_row({
        "timestamp":   datetime.datetime.now().isoformat(),
        "upload_id":   payload.upload_id,
        "filename":    "",
        "stored_path": str(dst),
        "pred_label":  "",
        "confidence":  "",
        "prob_fake":   "",
        "prob_real":   "",
        "user_label":  label,
    })

    return {"status": "ok", "moved_to": str(dst)}


# -------------------------
# Image prediction
# -------------------------
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        return JSONResponse({"error": "Please upload an image file."}, status_code=400)

    content = await file.read()
    try:
        return image_detector.predict_bytes(content)
    except Exception as e:
        return JSONResponse({"error": f"Image prediction failed: {str(e)}"}, status_code=500)


# -------------------------
# Audio prediction
# -------------------------
@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
        return JSONResponse({"error": "Please upload an audio file."}, status_code=400)

    upload_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    save_path = INCOMING / f"{upload_id}{ext}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = audio_detector.predict_audio(str(save_path))
        result["upload_id"] = upload_id

        log_row({
            "timestamp":   datetime.datetime.now().isoformat(),
            "upload_id":   upload_id,
            "filename":    file.filename,
            "stored_path": str(save_path),
            "pred_label":  result.get("label", ""),
            "confidence":  result.get("confidence", ""),
            "prob_fake":   result.get("probs", {}).get("FAKE", ""),
            "prob_real":   result.get("probs", {}).get("REAL", ""),
            "user_label":  "",
        })

        return result
    except Exception as e:
        return JSONResponse({"error": f"Audio prediction failed: {str(e)}"}, status_code=500)
