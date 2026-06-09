import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

SAMPLE_RATE   = 22050   # standard sample rate for all files
DURATION      = 3.0     # seconds — we'll use first 3s of each file
N_MFCC        = 40      # number of MFCC coefficients
N_MELS        = 128     # number of mel bands for spectrogram
N_CHROMA      = 12      # number of chroma features
HOP_LENGTH    = 512     # frames between each analysis window
N_FFT         = 2048    # size of FFT window

FIXED_LENGTH  = int(SAMPLE_RATE * DURATION)

def load_audio(file_path):
    """
    Loads an audio file and standardizes it to
    fixed sample rate and duration.

    Parameters:
        file_path : path to .wav file

    Returns:
        y  : audio signal as numpy array
        sr : sample rate (always SAMPLE_RATE)
    """

    y, sr = librosa.load(file_path,
                         sr=SAMPLE_RATE,
                         duration=DURATION)
    
    if len(y) < FIXED_LENGTH:
        # Pad with zeros at the end (silence)
        y = np.pad(y, (0, FIXED_LENGTH - len(y)))
    else:
        # Trim to exactly FIXED_LENGTH
        y = y[:FIXED_LENGTH]

    return y, sr


def extract_fixed_features(y, sr):
    """
    Extracts a fixed-size feature vector from audio.
    Used for baseline ML models (MLP, SVM).

    Parameters:
        y  : audio signal
        sr : sample rate

    Returns:
        features : numpy array of shape (54,)
    """
    # MFCC — captures voice texture
    mfcc = librosa.feature.mfcc(y=y, sr=sr,
                                  n_mfcc=N_MFCC,
                                  hop_length=HOP_LENGTH,
                                  n_fft=N_FFT)
    mfcc_mean = np.mean(mfcc, axis=1)    
    mfcc_std  = np.std(mfcc, axis=1)     

    # Chroma — captures pitch patterns
    chroma = librosa.feature.chroma_stft(y=y, sr=sr,
                                          n_chroma=N_CHROMA,
                                          hop_length=HOP_LENGTH,
                                          n_fft=N_FFT)
    chroma_mean = np.mean(chroma, axis=1)  

    # ZCR — captures signal smoothness
    zcr = librosa.feature.zero_crossing_rate(y,
                                              hop_length=HOP_LENGTH)
    zcr_mean = np.mean(zcr)               

    # RMS — captures loudness/energy
    rms = librosa.feature.rms(y=y,
                               hop_length=HOP_LENGTH)
    rms_mean = np.mean(rms)               

    # Combine everything into one flat vector
    features = np.hstack([
        mfcc_mean,      # 40 numbers
        mfcc_std,       # 40 numbers (added std for richer info)
        chroma_mean,    # 12 numbers
        [zcr_mean],     # 1 number
        [rms_mean]      # 1 number
    ])
    # Total: 40 + 40 + 12 + 1 + 1 = 94 numbers

    return features


def extract_mel_spectrogram(y, sr):
    """
    Extracts Mel Spectrogram as 2D matrix.
    Used for CNN model.

    Parameters:
        y  : audio signal
        sr : sample rate

    Returns:
        mel_db : numpy array of shape (128, T)
    """
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT
    )
    # Convert to dB scale
    mel_db = librosa.amplitude_to_db(mel, ref=np.max)
    return mel_db


def extract_mfcc_sequence(y, sr):
    """
    Extracts MFCC as 2D sequence matrix.
    Used for LSTM model.

    Parameters:
        y  : audio signal
        sr : sample rate

    Returns:
        mfcc : numpy array of shape (T, 40)
               T = time steps, 40 = features per step
    """
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT
    )
    # Transpose: (40, T) → (T, 40)
    # LSTM expects (time_steps, features) format
    return mfcc.T


def process_all_files(csv_path='data/ravdess_files.csv'):
    """
    Loads all audio files from CSV and extracts
    all three types of features. Saves to disk.

    Parameters:
        csv_path : path to the CSV from Phase 2
    """
    print("=" * 55)
    print("   Feature Extraction — All Files")
    print("=" * 55)

    # Load the CSV
    df = pd.read_csv(csv_path)
    print(f"\nTotal files to process: {len(df)}")

    # Create output directory for features
    os.makedirs('data/features', exist_ok=True)

    # ── Storage lists ──────────────────────────────────
    fixed_features  = []   # for baseline model
    mel_features    = []   # for CNN model
    mfcc_sequences  = []   # for LSTM model
    labels          = []   # emotion labels (numbers)
    label_names     = []   # emotion labels (names)

    # ── Label encoding ─────────────────────────────────
    # Models need numbers not strings
    # We convert emotion names to numbers
    emotion_to_int = {
        'neutral'  : 0,
        'calm'     : 1,
        'happy'    : 2,
        'sad'      : 3,
        'angry'    : 4,
        'fearful'  : 5,
        'disgust'  : 6,
        'surprised': 7
    }

    # ── Process each file ──────────────────────────────
    # tqdm() adds a progress bar so we can see progress
    failed = 0

    for idx, row in tqdm(df.iterrows(),
                          total=len(df),
                          desc="Extracting features"):
        try:
            # Load and standardize audio
            y, sr = load_audio(row['file_path'])

            # Extract all feature types
            fixed   = extract_fixed_features(y, sr)
            mel     = extract_mel_spectrogram(y, sr)
            mfcc_sq = extract_mfcc_sequence(y, sr)

            # Store features
            fixed_features.append(fixed)
            mel_features.append(mel)
            mfcc_sequences.append(mfcc_sq)

            # Store label as number
            labels.append(emotion_to_int[row['emotion']])
            label_names.append(row['emotion'])

        except Exception as e:
            # If a file fails, skip it and keep going
            print(f"\n⚠️  Failed: {row['filename']} — {e}")
            failed += 1
            continue

    print(f"\n\nProcessed: {len(labels)} files")
    print(f"Failed:    {failed} files")

    # ── Convert to numpy arrays ────────────────────────

    X_fixed = np.array(fixed_features)
    # Shape: (2880, 94) — 2880 files, 94 features each

    X_mel   = np.array(mel_features)
    # Shape: (2880, 128, T) — 2880 files, 128 mel bands, T frames

    X_mfcc  = np.array(mfcc_sequences)
    # Shape: (2880, T, 40) — 2880 files, T time steps, 40 coefficients

    y_labels = np.array(labels)
    # Shape: (2880,) — one label per file

    print(f"\n── Feature Shapes ──────────────────────")
    print(f"Fixed features  : {X_fixed.shape}")
    print(f"Mel spectrogram : {X_mel.shape}")
    print(f"MFCC sequence   : {X_mfcc.shape}")
    print(f"Labels          : {y_labels.shape}")

    # ── Save to disk ───────────────────────────────────

    print(f"\nSaving features to data/features/...")

    np.save('data/features/X_fixed.npy',  X_fixed)
    np.save('data/features/X_mel.npy',    X_mel)
    np.save('data/features/X_mfcc.npy',   X_mfcc)
    np.save('data/features/y_labels.npy', y_labels)

    # Also save label names for reference
    label_df = pd.DataFrame({
        'label_int' : list(emotion_to_int.values()),
        'label_name': list(emotion_to_int.keys())
    })
    label_df.to_csv('data/features/label_mapping.csv', index=False)

    print("✅ X_fixed.npy   saved")
    print("✅ X_mel.npy     saved")
    print("✅ X_mfcc.npy    saved")
    print("✅ y_labels.npy  saved")
    print("✅ label_mapping.csv saved")

    print(f"\nThese .npy files will be used in ALL future phases.")

    return X_fixed, X_mel, X_mfcc, y_labels



def process_all_datasets(csv_path='data/all_datasets.csv'):
    """
    Processes all datasets from the combined CSV.
    Uses the unified 6-emotion label mapping.
    """
    print("=" * 55)
    print("   Feature Extraction — All Datasets")
    print("=" * 55)

    # Check if combined CSV exists
    if not os.path.exists(csv_path):
        print(f"\n❌ {csv_path} not found!")
        print("Run src/load_datasets.py first!")
        return

    df = pd.read_csv(csv_path)
    print(f"\nTotal files to process: {len(df)}")
    print("This will take 15-30 minutes on CPU...")
    print("Please wait...\n")

    os.makedirs('data/features', exist_ok=True)

    fixed_features = []
    mel_features   = []
    mfcc_sequences = []
    labels         = []
    failed         = 0

    for idx, row in tqdm(df.iterrows(),
                          total=len(df),
                          desc="Extracting features"):
        try:
            y, sr = load_audio(row['file_path'])

            fixed   = extract_fixed_features(y, sr)
            mel     = extract_mel_spectrogram(y, sr)
            mfcc_sq = extract_mfcc_sequence(y, sr)

            fixed_features.append(fixed)
            mel_features.append(mel)
            mfcc_sequences.append(mfcc_sq)
            labels.append(int(row['label']))

        except Exception as e:
            failed += 1
            continue

    print(f"\nProcessed : {len(labels)} files")
    print(f"Failed    : {failed} files")

    # Convert to numpy arrays
    X_fixed  = np.array(fixed_features)
    X_mel    = np.array(mel_features)
    X_mfcc   = np.array(mfcc_sequences)
    y_labels = np.array(labels)

    print(f"\n── Feature Shapes ──────────────────────")
    print(f"Fixed features  : {X_fixed.shape}")
    print(f"Mel spectrogram : {X_mel.shape}")
    print(f"MFCC sequence   : {X_mfcc.shape}")
    print(f"Labels          : {y_labels.shape}")

    # Save features
    print(f"\nSaving features to data/features/...")
    np.save('data/features/X_fixed.npy',  X_fixed)
    np.save('data/features/X_mel.npy',    X_mel)
    np.save('data/features/X_mfcc.npy',   X_mfcc)
    np.save('data/features/y_labels.npy', y_labels)

    # Save updated label mapping (6 emotions)
    label_df = pd.DataFrame({
        'label_int' : [0, 1, 2, 3, 4, 5],
        'label_name': ['neutral', 'happy', 'sad',
                       'angry',   'fearful', 'disgust']
    })
    label_df.to_csv(
        'data/features/label_mapping.csv', index=False
    )

    print("✅ X_fixed.npy   saved")
    print("✅ X_mel.npy     saved")
    print("✅ X_mfcc.npy    saved")
    print("✅ y_labels.npy  saved")
    print("✅ label_mapping.csv saved")

    print(f"\n{'='*55}")
    print(f"   Feature Extraction Complete! 🎉")
    print(f"{'='*55}")

    return X_fixed, X_mel, X_mfcc, y_labels


# ── Run the extraction ─────────────────────────────────

if __name__ == '__main__':
    process_all_datasets()