# DeepShield — Multi-Modal Deepfake Detection System

DeepShield is a unified forensic platform that detects deepfakes across **Video, Image, and Audio** in a single interface. It returns a clear **REAL / FAKE** verdict with a confidence score for every prediction.

---

## Features

- **Video Detection** — CLIP ViT-L/14 fine-tuned on Celeb-DF v2 (AUC 0.827)
- **Image Detection** — SiGLIP ViT fine-tuned on Deepfake & Real Images dataset (Val AUC 99.76%)
- **Audio Detection** — WavLM Base+ fine-tuned on the Fake-or-Real (FoR) dataset (Val Acc 99.81%)
- **Dual-mode video detection** — Standard (with face crop) and Sensitive (full frame)
- **Confidence scoring** on every prediction
- **Human-in-the-loop feedback** — user corrections are stored and used for periodic video model fine-tuning

---

## Project Structure

```
DeepShield/
├── backend/
│   ├── app.py               # FastAPI backend
│   └── video_detector.py    # Video detection pipeline
├── frontend/
│   └── index.html           # Web interface
└── models/                  # Not included — see below
```

---

## Models

Model files are not included in this repository due to their size.

| Model | File | Download |
|-------|------|----------|
| CLIP ViT-L/14 (Standard) | `clip_finetuned_0827.torchscript` | [Google Drive](https://drive.google.com/drive/folders/1ydk68Nf_q0oiQ3I4q9_QXJKotolKryQy?usp=sharing) |
| CLIP ViT-L/14 (Sensitive) | `clip_finetuned.torchscript` | [Google Drive](https://drive.google.com/drive/folders/1ydk68Nf_q0oiQ3I4q9_QXJKotolKryQy?usp=sharing) |
| SiGLIP ViT | `image_deepfake_detector_best/` |  [Googlerive](https://drive.google.com/drive/folders/1lcYQETaYFgzzuZ-ZcUwejUoltepHDuuF?usp=sharing) |
| WavLM Base+ | `wavlm-deepfake-detector-final/` | [Google Drive](https://drive.google.com/drive/folders/1I0RAOHs_9DR2DQBa2Ci-D8VuYfWL36II?usp=sharing) |

Place downloaded models in the `models/` directory before running the backend.

---

## Installation

```bash
pip install fastapi uvicorn torch transformers opencv-python librosa soundfile
```

## Usage

```bash
cd backend
uvicorn app:app --reload
```

Then open `frontend/index.html` in your browser.

---

## Team

- Omar Ashraf
- Andrew Medhat
- Hussein Mostafa
- Ahmed Mahmoud

**Supervisors:** Dr. Mohamed Seif & Dr. Maya Waleed  
**Faculty of Computers and Artificial Intelligence — 2025/2026**
