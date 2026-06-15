import os
import sys
import numpy as np
import librosa
import torch
import torch.nn as nn

#Contsants 
N_MFCC  = 40
MAX_LEN = 200
SR      = 16000
MODEL_PATH = "deepfake_audio_cnn.pth"

#Model Definition
class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 5 * 25, 256),       
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.classifier(self.conv_block(x))


#Feature extraction
    def extract_mfcc(file_path):
    audio, _ = librosa.load(file_path, sr=SR, duration=4.0)
    mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
    if mfcc.shape[1] < MAX_LEN:
        mfcc = np.pad(mfcc, ((0, 0), (0, MAX_LEN - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :MAX_LEN]
    return mfcc


# inference
    def predict(file_path, model, device):
    mfcc = extract_mfcc(file_path)
    x = torch.tensor(mfcc[np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        out   = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred  = int(np.argmax(probs))
    label      = "Real (Human)" if pred == 1 else "Deepfake (AI-Generated)"
    confidence = float(probs[pred])
    return label, confidence


#main
    if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_audio.wav>")
        sys.exit(1)

    audio_path = sys.argv[1]
    if not os.path.exists(audio_path):
        print(f"File not found: {audio_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AudioCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    label, confidence = predict(audio_path, model, device)
    print(f"\nFile       : {audio_path}")
    print(f"Prediction : {label}")
    print(f"Confidence : {confidence:.2%}")
