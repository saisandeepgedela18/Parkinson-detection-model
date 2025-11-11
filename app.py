import streamlit as st
import librosa
import numpy as np
import sounddevice as sd
import joblib
import tempfile
import scipy.io.wavfile as wav

# Load trained model
model = joblib.load("parkinsons_model.pkl")

# Streamlit UI
st.set_page_config(page_title="Parkinson’s Voice Detector", page_icon="🧠")
st.title("🧠 AI for Early Detection of Parkinson’s from Voice")
st.write("Speak clearly for 4 seconds — say something like **'aaah'** or **'hello'**.")
st.markdown("---")

duration = 4  # seconds
sample_rate = 22050

if st.button("🎙️ Record Voice"):
    st.info("Recording... Please speak now 🎤")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    st.success("✅ Recording complete!")

    # Save recording temporarily
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    wav.write(temp_wav.name, sample_rate, (recording * 32767).astype(np.int16))

    # Load and preprocess
    y, sr = librosa.load(temp_wav.name, sr=22050)
    y = librosa.util.normalize(y)

    # Trim silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)

    # Check if enough speech detected
    if len(y_trimmed) < 0.5 * sr:
        st.warning("⚠️ No clear voice detected. Please try again and speak louder or closer to the mic.")
    else:
        # Extract MFCC features
        mfcc = np.mean(librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13).T, axis=0)

        # Make sure feature vector matches model input
        features = np.zeros((model.n_features_in_,))
        features[:len(mfcc)] = mfcc[:model.n_features_in_] if len(mfcc) >= model.n_features_in_ else mfcc

        # Predict
        pred = model.predict([features])[0]
        prob = model.predict_proba([features])[0][pred] * 100

        # Display result
        st.subheader("🩺 Prediction Result:")
        if pred == 1:
            st.error(f"⚠️ High risk of Parkinson’s detected (Confidence: {prob:.1f}%)")
        else:
            st.success(f"✅ Voice appears healthy (Confidence: {prob:.1f}%)")

    st.markdown("---")
    st.write("Tip: Background noise or silence can affect accuracy. Try speaking steadily into the mic.")
