import csv, uuid, datetime
from pathlib import Path
import os
import shutil
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
from pydantic import BaseModel

from model import predict_video
from image_detector import ImageDeepfakeDetector


# -------------------------
# Paths
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root (one level above backend/)
DATA_POOL = PROJECT_ROOT / "data_pool"
INCOMING = DATA_POOL / "incoming"
LABELED_REAL = DATA_POOL / "labeled" / "real"
LABELED_FAKE = DATA_POOL / "labeled" / "fake"
META_CSV = DATA_POOL / "metadata.csv"

FRONTEND_INDEX = PROJECT_ROOT / "frontend" / "index.html"

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

# Allow frontend to call backend (dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # for dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Image model (loads once)
# Put your fine-tuned model folder here:
# backend/models/image_deepfake_detector_best/
# -------------------------


IMAGE_MODEL_DIR = (PROJECT_ROOT / "models" / "image_deepfake_detector_best").as_posix()
image_detector = ImageDeepfakeDetector(IMAGE_MODEL_DIR)


# -------------------------
# UI
# -------------------------
@app.get("/", response_class=FileResponse)
def serve_ui():
    if not FRONTEND_INDEX.exists():
        return JSONResponse({"error": f"Missing frontend file at: {FRONTEND_INDEX}"}, status_code=500)
    return FileResponse(FRONTEND_INDEX)


# -------------------------
# Video prediction
# -------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")):
        return JSONResponse({"error": "Please upload a video file."}, status_code=400)

    upload_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    save_path = INCOMING / f"{upload_id}{ext}"

    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = predict_video(str(save_path))
    if "error" in result:
        return JSONResponse(result, status_code=400)

    log_row({
        "timestamp": datetime.datetime.now().isoformat(),
        "upload_id": upload_id,
        "filename": file.filename,
        "stored_path": str(save_path),
        "pred_label": result.get("label", ""),
        "confidence": result.get("confidence", ""),
        "prob_fake": result.get("prob_fake", ""),
        "prob_real": result.get("prob_real", ""),
        "user_label": "",   # filled later
    })

    result["upload_id"] = upload_id
    return result


# -------------------------
# Feedback (FIXED to accept JSON object {upload_id, label})
# -------------------------
class FeedbackRequest(BaseModel):
    upload_id: str
    label: str

@app.post("/feedback")
def feedback(payload: FeedbackRequest):
    upload_id = payload.upload_id
    label = payload.label.strip().lower()

    if label not in ["real", "fake"]:
        return JSONResponse({"error": "label must be 'real' or 'fake'."}, status_code=400)

    matches = list(INCOMING.glob(f"{upload_id}.*"))
    if not matches:
        return JSONResponse({"error": "upload_id not found in incoming."}, status_code=404)

    src = matches[0]
    dst_dir = LABELED_REAL if label == "real" else LABELED_FAKE
    dst = dst_dir / src.name
    src.rename(dst)

    log_row({
        "timestamp": datetime.datetime.now().isoformat(),
        "upload_id": upload_id,
        "filename": "",
        "stored_path": str(dst),
        "pred_label": "",
        "confidence": "",
        "prob_fake": "",
        "prob_real": "",
        "user_label": label,
    })

    return {"status": "ok", "moved_to": str(dst)}


# -------------------------
# Image prediction (NEW)
# -------------------------
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp")):
        return JSONResponse({"error": "Please upload an image file (jpg/png/webp/bmp)."}, status_code=400)

    content = await file.read()
    try:
        return image_detector.predict_bytes(content)
    except Exception as e:
        return JSONResponse({"error": f"Image prediction failed: {str(e)}"}, status_code=500)
