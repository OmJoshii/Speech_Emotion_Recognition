# Speech Emotion Recognition System

A deep learning system that detects human emotions from speech audio using
CNN and LSTM models. Built as a final year major project for Computer Engineering.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What it does

Takes a speech audio file (or live microphone input) and predicts the
emotion of the speaker — Happy, Sad, Angry, Neutral, Calm, Fearful,
Disgust, or Surprised.

---

## Emotions Detected

| Code | Emotion   |
|------|-----------|
| 0    | Neutral   |
| 1    | Calm      |
| 2    | Happy     |
| 3    | Sad       |
| 4    | Angry     |
| 5    | Fearful   |
| 6    | Disgust   |
| 7    | Surprised |

---

## Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song

- 2880 speech audio files
- 24 actors (12 male, 12 female)
- 8 emotions × 2 intensities × 2 statements × 2 repetitions
- Neutral has 192 files, all others have 384 files

Download from:
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

Extract and place inside `data/ravdess/` folder.

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd speech_emotion_recognition
```

### 2. Create and activate virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dataset
Download RAVDESS from the link above and place in `data/ravdess/`

### 5. Run feature extraction
```bash
python src/extract_features.py
```

### 6. Run preprocessing
```bash
python src/preprocess.py
```

--- 

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Librosa | Audio processing |
| NumPy & Pandas | Data handling |
| Matplotlib & Seaborn | Visualization |
| Scikit-learn | Baseline ML models |
| PyTorch | Deep learning |
| Streamlit | Live demo app |

---

## Features Extracted

| Feature | Shape | Used By |
|---------|-------|---------|
| MFCC (mean + std) | (94,) | Baseline MLP model |
| Mel Spectrogram | (128, 130) | CNN model |
| MFCC Sequence | (130, 40) | LSTM model |

---

## Data Split

| Split | Samples | Percentage |
|-------|---------|------------|
| Training | 2017 | 70% |
| Validation | 431 | 15% |
| Test | 432 | 15% |

---

## Class Weights (handling imbalance)

| Emotion | Files | Weight |
|---------|-------|--------|
| Neutral | 192 | 1.8815 |
| All others | 384 each | 0.9373 |

Neutral has fewer samples so it gets a higher weight during training
to ensure the model pays equal attention to all emotions.

---

## Results

| Model | Accuracy | F1 Score | Parameters |
|-------|----------|----------|------------|
| MLP Baseline | 93.75% | 0.93 | 66,888 |
| CNN | 90.97% | 0.91 | 430,856 |
| LSTM | 96.53% | 0.96 | 348,745 |
| CNN-LSTM | 92.36% | 0.93 | 982,921 |

**Best Model: LSTM with 96.53% test accuracy**

---

### Key Findings
- LSTM outperformed all models — speech emotion is inherently sequential
- CNN overfitted due to limited dataset size (2017 training samples)
- Handcrafted MFCC features proved more effective than learned representations on this dataset size
- Neutral emotion was consistently hardest to classify across all models

---
