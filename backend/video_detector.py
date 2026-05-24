import os
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor

CLIP_REPO   = "openai/clip-vit-large-patch14"
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "..", "models")
NUM_FRAMES  = 8
FACE_MARGIN = 0.35


class VideoDeepfakeDetector:
    def __init__(self, model_path: str = None, use_face_crop: bool = True):
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "clip_finetuned.torchscript")

        print(f"[VideoDetector] Loading model from {model_path} ...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Fine-tuned model not found at {model_path}\n"
                "Download clip_finetuned.torchscript from Kaggle and place it in models/"
            )

        self.use_face_crop = use_face_crop

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        try:
            self.model = torch.jit.load(model_path, map_location=self.device)
        except Exception:
            self.device = "cpu"
            self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()

        print(f"[VideoDetector] Loading CLIP preprocessor from {CLIP_REPO} ...")
        self.processor = CLIPProcessor.from_pretrained(CLIP_REPO)

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        print(f"[VideoDetector] Ready on {self.device} | face_crop={self.use_face_crop}.")

    def _crop_face(self, frame_bgr: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        if len(faces) == 0:
            return frame_bgr
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        ih, iw = frame_bgr.shape[:2]
        mx, my = int(w * FACE_MARGIN), int(h * FACE_MARGIN)
        x1, y1 = max(0, x - mx), max(0, y - my)
        x2, y2 = min(iw, x + w + mx), min(ih, y + h + my)
        return frame_bgr[y1:y2, x1:x2]

    def _extract_frames(self, video_path: str) -> list:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            raise RuntimeError("Video has no frames")

        idxs = set(np.linspace(0, max(total - 1, 0), NUM_FRAMES).astype(int).tolist())
        frames, cur = [], 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if cur in idxs:
                crop = self._crop_face(frame) if self.use_face_crop else frame
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
            cur += 1
        cap.release()

        if not frames:
            raise RuntimeError("No frames could be extracted")
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1])
        return frames

    @torch.inference_mode()
    def predict(self, video_path: str) -> dict:
        frames = self._extract_frames(video_path)

        tensors = torch.stack([
            self.processor(images=img, return_tensors="pt")["pixel_values"][0]
            for img in frames
        ]).to(device=self.device, dtype=torch.float32)

        output = self.model(tensors)
        # output shape: (N, 2) — index 0 = real, index 1 = fake
        probs = output.softmax(dim=1).float().numpy()
        fake_probs = probs[:, 1]

        mean_fake = float(np.mean(fake_probs))
        max_fake = float(np.max(fake_probs))
        combined = 0.7 * mean_fake + 0.3 * max_fake
        real_prob = 1.0 - combined

        label = "FAKE" if combined >= 0.5 else "REAL"
        conf = combined if label == "FAKE" else real_prob

        return {
            "label": label,
            "confidence": round(conf * 100, 2),
            "prob_fake": round(combined * 100, 2),
            "prob_real": round(real_prob * 100, 2),
            "frames_used": len(frames),
            "device": str(self.device),
            "size": "face_crop",
        }
