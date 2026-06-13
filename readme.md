# Speech Emotion Recognition System

A deep learning system that detects human emotions from speech audio.
Compares multiple models from simple MLP to state-of-the-art Wav2Vec2.
Built as a final year major project for Computer Engineering.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What it does

Takes a speech audio file (or live microphone input) and predicts the
emotion of the speaker. Supports 4 different models for comparison:

- **MLP** — handcrafted MFCC features + neural network
- **LSTM** — sequential modeling of MFCC time series
- **Ensemble** — combined MLP + LSTM predictions
- **Wav2Vec2** — pretrained transformer (state of the art)

---

## Live Demo

```bash
streamlit run src/app.py
```

Open browser at http://localhost:8501

---

## Project Structure
speech_emotion_recognition/
├── data/
│   ├── ravdess/              ← RAVDESS dataset
│   ├── crema_d/              ← CREMA-D dataset
│   ├── tess/                 ← TESS dataset
│   ├── savee/                ← SAVEE dataset
│   ├── all_datasets.csv      ← combined file index
│   ├── features/             ← extracted features
│   └── processed/            ← train/val/test splits
├── models/                   ← saved trained models
├── notebooks/                ← exploration code
│   ├── audio_basics.py
│   ├── dataset_exploration.py
│   ├── feature_exploration.py
│   ├── preprocessing_exploration.py
│   ├── mlp_exploration.py
│   ├── cnn_exploration.py
│   ├── lstm_exploration.py
│   └── evaluation_exploration.py
├── src/                      ← production source code
│   ├── load_datasets.py      ← dataset loader
│   ├── extract_features.py   ← feature extraction
│   ├── preprocess.py         ← preprocessing pipeline
│   ├── train_mlp.py          ← MLP model
│   ├── train_cnn.py          ← CNN model
│   ├── train_lstm.py         ← LSTM and CNN-LSTM models
│   ├── evaluate.py           ← evaluation pipeline
│   └── app.py                ← Streamlit demo app
├── .gitignore
├── requirements.txt
└── README.md

---

## Emotions Detected

| Code | Emotion  | Our Models | Wav2Vec2 |
|------|----------|------------|----------|
| -    | Neutral  | ✅ | ✅ |
| -    | Happy    | ✅ | ✅ |
| -    | Sad      | ✅ | ✅ |
| -    | Angry    | ✅ | ✅ |
| -    | Fearful  | ✅ | ❌ |
| -    | Disgust  | ✅ | ❌ |

---

## Dataset

Combined from 4 public datasets:

| Dataset | Files | Speakers | Language |
|---------|-------|----------|----------|
| RAVDESS | 2,112 | 24 | English |
| CREMA-D | 7,442 | 91 | English |
| TESS    | 2,400 | 2  | English |
| SAVEE   |   420 | 4  | English |
| **Total** | **12,374** | **121** | English |

### Download Instructions

1. RAVDESS: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio
2. CREMA-D: https://www.kaggle.com/datasets/ejlok1/cremad
3. TESS: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess
4. SAVEE: https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee

Extract each into the corresponding `data/` subfolder.

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd speech_emotion_recognition
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download datasets
Download all 4 datasets from links above and place in `data/` folder.

### 5. Run full pipeline
```bash
# Load and combine all datasets
python src/load_datasets.py

# Extract audio features
python src/extract_features.py

# Preprocess and split data
python src/preprocess.py

# Train models
python src/train_mlp.py
python src/train_cnn.py
python src/train_lstm.py

# Evaluate all models
python src/evaluate.py

# Run demo app
streamlit run src/app.py
```

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Librosa | Audio processing and feature extraction |
| NumPy & Pandas | Data handling |
| Matplotlib & Seaborn | Visualization |
| Scikit-learn | Preprocessing and baseline models |
| PyTorch | Deep learning (MLP, CNN, LSTM) |
| HuggingFace Transformers | Wav2Vec2 pretrained model |
| Streamlit | Live demo web app |
| Noisereduce | Real-time noise reduction |

---

## Features Extracted

| Feature | Shape | Used By |
|---------|-------|---------|
| MFCC mean + std | (94,) | MLP model |
| Mel Spectrogram | (1, 128, 130) | CNN model |
| MFCC Sequence | (130, 40) | LSTM model |

---

## Data Split

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | 8,666 | 70% |
| Validation | 1,851 | 15% |
| Test | 1,857 | 15% |

---

## Results

### Our Trained Models (RAVDESS + CREMA-D + TESS + SAVEE)

| Model | Accuracy | F1 Score | Parameters |
|-------|----------|----------|------------|
| MLP Baseline | 73.67% | 0.736 | 66,758 |
| CNN | 65.86% | 0.660 | 430,726 |
| LSTM | 76.90% | 0.769 | 348,615 |
| CNN-LSTM | 73.88% | 0.740 | 982,791 |

**Best trained model: LSTM with 76.90% test accuracy**

### Pretrained Model

| Model | Approach | Emotions |
|-------|----------|----------|
| Wav2Vec2 (superb/wav2vec2-base-superb-er) | Fine-tuned transformer | 4 (neu, hap, ang, sad) |

Wav2Vec2 provides significantly better real-world performance
due to pretraining on 960 hours of speech data.

---

## Key Findings

1. **LSTM outperformed all custom models** — speech emotion
   is inherently sequential, LSTM captures this naturally

2. **CNN consistently overfitted** — dataset too small for
   CNN to learn generalizable spatial features

3. **Low-order MFCC coefficients most important** — MFCC 1-9
   carry the most emotion-relevant information

4. **RMS energy is key** — loudness is a strong emotion signal
   (angry/happy = loud, sad/neutral = quiet)

5. **Fearful hardest to classify** — acoustically similar
   to sad, both have low energy and high pitch

6. **More data improves generalization** — 4 datasets with
   121 speakers generalizes far better than 24 speakers

7. **Pretrained models outperform trained from scratch** —
   Wav2Vec2 pretrained on 960hrs beats our models trained
   on 12,374 samples

---

## Limitations

1. **Dataset size** — 12,374 samples is small for deep learning.
   IEMOCAP (10,000) or MSP-Podcast (100,000+) would improve results

2. **Acted vs natural speech** — all datasets use professional
   actors. Real spontaneous speech is harder to classify

3. **Language dependency** — models trained on English only.
   Cross-lingual performance not tested

4. **4 vs 6 emotions** — Wav2Vec2 only supports 4 emotions
   vs our 6-emotion custom models

---

## Future Work

1. Add CREMA-D Song + more diverse datasets
2. Fine-tune Wav2Vec2 on our combined dataset
3. Add real-time continuous emotion tracking
4. Support multiple languages
5. Deploy to cloud (Streamlit Cloud / HuggingFace Spaces)

---

## Author

**Om Joshi**