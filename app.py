import streamlit as st
import numpy as np
import librosa
import torch
import torch.nn as nn
import tempfile
import os

#constants
N_MFCC     = 40
MAX_LEN    = 200
SR         = 16000
MODEL_PATH = "deepfake_audio_cnn.pth"

# Model
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
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.classifier(self.conv_block(x))


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = AudioCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model, device


#Feature extraction 
    def extract_mfcc(file_path):
        audio, _ = librosa.load(file_path, sr=SR, duration=4.0)
        mfcc = librosa.feature.mfcc(y=audio, sr=SR, n_mfcc=N_MFCC)
        if mfcc.shape[1] < MAX_LEN:
            mfcc = np.pad(mfcc, ((0, 0), (0, MAX_LEN - mfcc.shape[1])), mode='constant')
        else:
            mfcc = mfcc[:, :MAX_LEN]
        return mfcc


def predict(file_path, model, device):
    mfcc = extract_mfcc(file_path)
    x    = torch.tensor(mfcc[np.newaxis, np.newaxis, :, :], dtype=torch.float32).to(device)
    with torch.no_grad():
        out   = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred  = int(np.argmax(probs))
    label      = "Real (Human)" if pred == 1 else "Deepfake (AI-Generated)"
    confidence = float(probs[pred])
    return label, confidence, probs


#UI
st.set_page_config(page_title="Deepfake Audio Detector", page_icon="🎙️", layout="centered")

st.title("🎙️ Deepfake Audio Detector")
st.caption("Upload a `.wav` file to find out if it's human or AI-generated.")

uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("Analysing audio..."):
        # Save to temp file so librosa can read it
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        model, device = load_model()
        label, confidence, probs = predict(tmp_path, model, device)
        os.unlink(tmp_path)

    # Result
    if "Real" in label:
        st.success(f"✅ **{label}**")
    else:
        st.error(f"⚠️ **{label}**")

    st.metric("Confidence", f"{confidence:.2%}")

    st.divider()
    st.subheader("Class Probabilities")
    col1, col2 = st.columns(2)
    col1.metric("Deepfake", f"{probs[0]:.2%}")
    col2.metric("Real",     f"{probs[1]:.2%}")
