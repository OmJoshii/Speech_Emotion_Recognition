# phase3_feature_exploration.py
# Phase 3: Understanding and visualizing audio features
# This is our exploration/learning file

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

print("=" * 50)
print("  Feature Exploration")
print("=" * 50)

# ── Load sample files for different emotions ───────────
# We'll pick one file each from angry, sad, happy, neutral
# to visually compare their features

# Load our CSV from Phase 2
df = pd.read_csv('data/ravdess_files.csv')

# Pick one sample file for each of these 4 emotions
sample_emotions = ['angry', 'sad', 'happy', 'neutral']
samples = {}

for emotion in sample_emotions:
    # Get first file for this emotion
    row = df[df['emotion'] == emotion].iloc[0]
    # Load audio — sr=22050 standardizes sample rate
    y, sr = librosa.load(row['file_path'], sr=22050)
    samples[emotion] = {'y': y, 'sr': sr, 'path': row['file_path']}
    print(f"Loaded {emotion}: {row['filename']}")

print("\nAll samples loaded!")

# ── MFCC Visualization ─────────────────────────────────
print("\nExtracting MFCCs...")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('MFCC Comparison Across Emotions\n'
             '(Each row = 1 coefficient, brighter = stronger)',
             fontsize=12, fontweight='bold')

for idx, (emotion, data) in enumerate(samples.items()):
    row = idx // 2   # 0,0,1,1
    col = idx % 2    # 0,1,0,1
    ax = axes[row, col]

    # Extract MFCC
    # n_mfcc=40 → extract 40 coefficients
    # Each coefficient captures different aspect of voice texture
    mfcc = librosa.feature.mfcc(y=data['y'],
                                  sr=data['sr'],
                                  n_mfcc=40)

    # Display as heatmap
    librosa.display.specshow(mfcc,
                             sr=data['sr'],
                             x_axis='time',
                             ax=ax,
                             cmap='coolwarm')

    ax.set_title(f'{emotion.upper()} — MFCC')
    ax.set_ylabel('MFCC Coefficient')
    fig.colorbar(ax.collections[0], ax=ax)

    # Print the shape
    print(f"  {emotion}: MFCC shape = {mfcc.shape} "
          f"({mfcc.shape[0]} coefficients × {mfcc.shape[1]} frames)")

plt.tight_layout()
plt.savefig('notebooks/mfcc_comparison.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("MFCC comparison saved!")

# ── Mel Spectrogram Comparison ─────────────────────────
print("\nExtracting Mel Spectrograms...")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('Mel Spectrogram Comparison Across Emotions\n'
             '(This is what the CNN will SEE)',
             fontsize=12, fontweight='bold')

for idx, (emotion, data) in enumerate(samples.items()):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    # Extract Mel Spectrogram
    # n_mels=128 → 128 frequency bands
    mel = librosa.feature.melspectrogram(y=data['y'],
                                          sr=data['sr'],
                                          n_mels=128)

    # Convert to dB scale for better visualization
    mel_db = librosa.amplitude_to_db(mel, ref=np.max)

    librosa.display.specshow(mel_db,
                             sr=data['sr'],
                             x_axis='time',
                             y_axis='mel',
                             ax=ax,
                             cmap='magma')

    ax.set_title(f'{emotion.upper()} — Mel Spectrogram')
    fig.colorbar(ax.collections[0], ax=ax, format='%+2.0f dB')

    print(f"  {emotion}: Mel shape = {mel.shape} "
          f"({mel.shape[0]} mel bands × {mel.shape[1]} frames)")

plt.tight_layout()
plt.savefig('notebooks/mel_comparison.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("Mel Spectrogram comparison saved!")

# ── ZCR and RMS Comparison ─────────────────────────────
print("\nExtracting ZCR and RMS Energy...")

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
fig.suptitle('ZCR & RMS Energy Across Emotions',
             fontsize=12, fontweight='bold')

for idx, (emotion, data) in enumerate(samples.items()):
    ax = axes[idx // 2, idx % 2]

    # ZCR — how often signal crosses zero
    zcr = librosa.feature.zero_crossing_rate(data['y'])[0]

    # RMS — root mean square energy (loudness)
    rms = librosa.feature.rms(y=data['y'])[0]

    # Time axis for plotting
    frames = range(len(zcr))
    time = librosa.frames_to_time(frames, sr=data['sr'])

    # Plot both on same axes
    ax.plot(time, zcr, color='steelblue',
            label='ZCR', alpha=0.7, linewidth=0.8)
    ax.plot(time, rms, color='coral',
            label='RMS Energy', alpha=0.7, linewidth=0.8)

    ax.set_title(f'{emotion.upper()}')
    ax.set_xlabel('Time (s)')
    ax.legend()

    print(f"  {emotion}: Mean ZCR={zcr.mean():.4f}  "
          f"Mean RMS={rms.mean():.4f}")

plt.tight_layout()
plt.savefig('notebooks/zcr_rms_comparison.png',
            dpi=150, bbox_inches='tight')
plt.show()
print("ZCR/RMS comparison saved!")

# ── Fixed Size Feature Vector ──────────────────────────
# This is what we'll feed into our baseline ML model
# One audio file → one flat array of numbers

print("\n── Extracting Fixed-Size Feature Vector ──")

def extract_fixed_features(y, sr):
    """
    Takes audio signal and returns a fixed-size
    feature vector of 54 numbers.

    Parameters:
        y  : audio signal (numpy array)
        sr : sample rate (int)

    Returns:
        features : numpy array of 54 numbers
    """

    # MFCC — 40 coefficients, take mean across time
    # Shape: (40, T) → mean → (40,)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    # axis=1 means take mean across columns (time frames)
    # Result: one mean value per coefficient = 40 numbers

    # Chroma — 12 pitch classes, take mean across time
    # Shape: (12, T) → mean → (12,)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # ZCR — take mean across time → 1 number
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # RMS — take mean across time → 1 number
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)

    # Concatenate all features into one flat array
    # np.hstack joins arrays horizontally
    features = np.hstack([
        mfcc_mean,          # 40 numbers
        chroma_mean,        # 12 numbers
        [zcr_mean],         # 1 number
        [rms_mean]          # 1 number
    ])
    # Total: 40 + 12 + 1 + 1 = 54 numbers

    return features

# Test on one sample
test_emotion = 'angry'
y_test = samples[test_emotion]['y']
sr_test = samples[test_emotion]['sr']

features = extract_fixed_features(y_test, sr_test)

print(f"\nEmotion: {test_emotion}")
print(f"Raw audio length : {len(y_test)} samples")
print(f"Feature vector   : {features.shape[0]} numbers")
print(f"First 10 values  : {features[:10].round(3)}")
print(f"\nWe compressed {len(y_test)} numbers → {len(features)} numbers!")
print("This is what gets fed into the baseline model.")