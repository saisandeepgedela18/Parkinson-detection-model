
## 🧠 AI for Early Detection of Parkinson’s from Voice

### 🎯 Overview

This project is developed as part of the **CSA Hackathon Challenge: “AI for Early Detection of Parkinson’s from Voice.”**
It aims to detect early signs of **Parkinson’s disease** from subtle vocal biomarkers using **machine learning** and **audio analysis**.

By analyzing a user’s short voice recording, the system predicts whether the voice shows potential signs of Parkinson’s — enabling early awareness and diagnosis support.

---

### ⚙️ Features

* 🎙️ **Live Voice Recording:** Capture audio directly from the microphone.
* 🧹 **Preprocessing:** Silence trimming, normalization, and noise filtering.
* 🎧 **Feature Extraction:** MFCCs (Mel-Frequency Cepstral Coefficients).
* 🤖 **Prediction Model:** Random Forest Classifier trained on the UCI Parkinson’s Dataset.
* 💬 **Real-Time Result:** Instantly display Parkinson’s risk prediction with confidence score.
* 🧩 **Simple Interface:** Built using Streamlit for fast and intuitive interaction.

---

### 🧾 Dataset

We used the publicly available **[UCI Parkinson’s Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)**,
which includes voice features such as jitter, shimmer, fundamental frequency (Fo), and harmonic-to-noise ratio (HNR).

Each record in the dataset corresponds to one patient’s sustained vowel phonation.

| Feature | Description                         |
| ------- | ----------------------------------- |
| Fo (Hz) | Average vocal fundamental frequency |
| Jitter  | Frequency variation                 |
| Shimmer | Amplitude variation                 |
| HNR     | Harmonics-to-noise ratio            |
| status  | 1 = Parkinson’s, 0 = Healthy        |

---

### 🧩 Tech Stack

* **Python 3.10+**
* **Libraries:**

  * `scikit-learn` – Model training
  * `librosa` – Audio processing
  * `streamlit` – Web interface
  * `sounddevice`, `scipy` – Voice recording and saving
  * `pandas`, `numpy`, `joblib` – Data handling & model storage

---

### 🧠 Model Training (`train_model.py`)

```bash
python train_model.py
```

This script:

1. Loads the UCI Parkinson’s dataset
2. Trains a **Random Forest Classifier**
3. Evaluates the accuracy
4. Saves the trained model as `parkinsons_model.pkl`

You can replace the RandomForest with an SVM or XGBoost for experimentation.

---

### 🧮 Streamlit Web App (`app.py`)

To launch the app locally:

```bash
run app.py
```

Then open:
👉 [http://localhost:8501](http://localhost:8501)

**How it works:**

1. Click “🎙️ Record Voice” and speak clearly (“aaah” for 4 seconds).
2. The app preprocesses and extracts MFCC features.
3. The model predicts Parkinson’s risk in real-time.

---

### 📊 Example Output

| Input                      | Predicted Result        | Confidence |
| -------------------------- | ----------------------- | ---------- |
| Healthy voice              | ✅ Voice appears healthy | 92.3%      |
| Parkinson’s-affected voice | ⚠️ High risk detected   | 88.7%      |

---

### 🧱 Folder Structure

```
parkinsons-voice-detector/
│
├── train_model.py
├── app.py
├── parkinsons_model.pkl
├── requirements.txt
└── README.md
```

---

### 🚀 Installation

1. Clone the repository

   ```bash
   git clone https://github.com/<your-username>/parkinsons-voice-detector.git
   cd parkinsons-voice-detector
   ```
2. Create and activate a virtual environment

   ```bash
   python -m venv venv
   venv\Scripts\activate   # (Windows)
   source venv/bin/activate  # (Mac/Linux)
   ```
3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

---

### 📈 Results

The Random Forest model achieved **~90% accuracy** on the UCI dataset, demonstrating that simple acoustic features can effectively indicate Parkinson’s risk.

---

### 💡 Future Improvements

* 🗣️ Multi-language voice support
* 📱 Android app integration using Streamlit Cloud or Flask API
* 🧬 Use pretrained audio embeddings (YAMNet, wav2vec2) for higher accuracy
* ⏱️ Continuous monitoring and progress tracking

---

### 🏁 Acknowledgements

Dataset source: UCI Machine Learning Repository
Hackathon hosted by: CSA Hackathon Committee

