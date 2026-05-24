import torch
import librosa
from transformers import Wav2Vec2FeatureExtractor, AutoModelForAudioClassification

class AudioDeepfakeDetector:
    def __init__(self, model_dir: str):
        """
        Initializes the WavLM model and feature extractor from the local directory.
        Automatically checks for Apple Silicon (MPS), Nvidia (CUDA), or defaults to CPU.
        """
        # Determine the best available hardware for inference
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Loading Audio Model onto {self.device}...")
        
        # Load the feature extractor and the model from your local models/ folder
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        self.model = AutoModelForAudioClassification.from_pretrained(model_dir).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()

    def predict_audio(self, audio_path: str) -> dict:
        """
        Loads an audio file, extracts features, and runs it through the WavLM model.
        Returns a dictionary formatted for the FastAPI backend.
        """
    
       # 1. Load Audio
        speech, _ = librosa.load(audio_path, sr=16000)

        # 2. Extract Features (FIXED: Added Truncation to 3 seconds)
        inputs = self.feature_extractor(
            speech, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True,
            max_length=16000 * 3  # 16,000 Hz * 3 seconds = 48,000 samples
        )
        
        # Move inputs to the appropriate device (CPU, MPS, or CUDA)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}

        # 3. Model Inference
        with torch.no_grad():
            logits = self.model(**inputs).logits
            
            # --- THE FIX: Temperature Scaling ---
            # Increase this number (e.g., 2.0, 3.0, 5.0) to make the model less confident.
            # 1.0 is default behavior. 3.0 is a good starting point for overfit models.
            temperature = 3.0 
            scaled_logits = logits / temperature
            
            # Convert the softened logits to percentages
            probabilities = torch.nn.functional.softmax(scaled_logits, dim=-1)
            
            # Based on our training data mapping: Index 0 = FAKE, Index 1 = REAL
            prob_fake = probabilities[0][0].item()
            prob_real = probabilities[0][1].item()

        # 4. Determine Final Verdict
        if prob_real > prob_fake:
            label = "REAL"
            confidence = prob_real
        else:
            label = "FAKE"
            confidence = prob_fake

        # 5. Return Frontend-Ready Response
        return {
            "label": label,
            "confidence": round(confidence, 4),  # e.g., 0.9854
            "probs": {
                "FAKE": round(prob_fake, 4),
                "REAL": round(prob_real, 4)
            }
        }