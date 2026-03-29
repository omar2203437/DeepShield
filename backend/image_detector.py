import io
from typing import Optional
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipForImageClassification


class ImageDeepfakeDetector:
    def __init__(self, model_dir: str, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_dir)
        self.model = SiglipForImageClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.id2label = getattr(self.model.config, "id2label", None) or {}

    @torch.inference_mode()
    def predict_pil(self, img: Image.Image) -> dict:
        img = img.convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, interpolate_pos_encoding=True)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

        pred_id = int(torch.argmax(probs).item())
        conf = float(probs[pred_id].item())
        label = self.id2label.get(pred_id, str(pred_id))

        per_class = {self.id2label.get(i, str(i)): float(probs[i].item()) for i in range(probs.numel())}
        return {"label": label, "confidence": conf, "probs": per_class}

    def predict_bytes(self, data: bytes) -> dict:
        img = Image.open(io.BytesIO(data))
        return self.predict_pil(img)
