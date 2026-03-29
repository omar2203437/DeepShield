import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

# -------- CONFIG (MUST MATCH TRAINING) --------
FRAMES = 16   # MUST match training
SIZE = 112
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "models",
    "best_model_auc.pt"
)
# ---------------------------------------------


def build_model():
    m = torchvision.models.video.r2plus1d_18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()

MODEL = build_model().to(DEVICE)
state = torch.load(MODEL_PATH, map_location=DEVICE)

# handle different checkpoint formats
if isinstance(state, dict) and "model_state_dict" in state:
    state = state["model_state_dict"]

MODEL.load_state_dict(state, strict=True)
MODEL.eval()

print("✅ Loaded model from:", MODEL_PATH)
print("✅ Using FRAMES =", FRAMES)
print("✅ Using device =", DEVICE)


def read_video_cv2(path, num_frames):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError("Video has no frames")

    idxs = np.linspace(0, total - 1, num_frames).astype(int)

    frames = []
    cur = 0
    grab_idx = set(idxs.tolist())

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if cur in grab_idx:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (SIZE, SIZE))
            frames.append(frame)

        cur += 1

    cap.release()

    if len(frames) == 0:
        raise RuntimeError("No frames extracted")

    while len(frames) < num_frames:
        frames.append(frames[-1])

    frames = np.stack(frames, axis=0)
    return frames


@torch.no_grad()
def predict_video(video_path: str):
    frames = read_video_cv2(video_path, FRAMES)

    x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
    x = x.permute(1, 0, 2, 3).unsqueeze(0)
    x = x.to(DEVICE)

    logits = MODEL(x)
    probs = torch.softmax(logits, dim=1).squeeze(0)

    prob_real = float(probs[0])
    prob_fake = float(probs[1])

    label = "FAKE" if prob_fake >= 0.5 else "REAL"
    confidence = prob_fake if label == "FAKE" else prob_real

    return {
        "label": label,
        "confidence": round(confidence * 100, 2),
        "prob_fake": round(prob_fake * 100, 2),
        "prob_real": round(prob_real * 100, 2),
        "device": str(DEVICE),
        "frames_used": FRAMES,
        "size": SIZE
    }