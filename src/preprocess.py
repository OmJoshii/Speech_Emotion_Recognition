# Loads extracted features, preprocesses them,
# and saves train/val/test splits ready for training

import numpy as np
import pandas as pd
import os
import joblib   # for saving the scaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# ── Constants ──────────────────────────────────────────
RANDOM_STATE = 42     # ensures same split every time
TEST_SIZE    = 0.15   # 15% for test
VAL_SIZE     = 0.176  # ~15% of total for validation

# Emotion label mapping
EMOTION_NAMES = {
    0: 'neutral',  1: 'happy',   2: 'sad',
    3: 'angry',    4: 'fearful', 5: 'disgust'
}
NUM_CLASSES = len(EMOTION_NAMES)


def load_features():
    """
    Loads all extracted feature files from disk.

    Returns:
        X_fixed  : fixed feature vectors (2880, 94)
        X_mel    : mel spectrograms      (2880, 128, T)
        X_mfcc   : mfcc sequences        (2880, T, 40)
        y_labels : emotion labels        (2880,)
    """
    print("Loading extracted features...")

    features_path = 'data/features'

    # Check if features exist
    required_files = ['X_fixed.npy', 'X_mel.npy',
                      'X_mfcc.npy',  'y_labels.npy']

    for f in required_files:
        path = os.path.join(features_path, f)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"❌ {path} not found!\n"
                f"Run src/extract_features.py first."
            )

    X_fixed  = np.load(f'{features_path}/X_fixed.npy')
    X_mel    = np.load(f'{features_path}/X_mel.npy')
    X_mfcc   = np.load(f'{features_path}/X_mfcc.npy')
    y_labels = np.load(f'{features_path}/y_labels.npy')

    print(f"  X_fixed  : {X_fixed.shape}")
    print(f"  X_mel    : {X_mel.shape}")
    print(f"  X_mfcc   : {X_mfcc.shape}")
    print(f"  y_labels : {y_labels.shape}")

    return X_fixed, X_mel, X_mfcc, y_labels


def split_data(X, y, feature_type='fixed'):
    """
    Splits data into train, validation and test sets.
    Uses stratification to maintain class balance.

    Parameters:
        X            : feature array
        y            : labels array
        feature_type : string label for logging

    Returns:
        X_train, X_val, X_test : feature splits
        y_train, y_val, y_test : label splits
    """
    print(f"\nSplitting {feature_type} features...")

    # Step 1 — separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
        # stratify ensures proportional class distribution
        # in each split — crucial for imbalanced data
    )

    # Step 2 — separate validation from remaining
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    total = len(X)
    print(f"  Train : {len(X_train):>5} "
          f"({len(X_train)/total*100:.1f}%)")
    print(f"  Val   : {len(X_val):>5} "
          f"({len(X_val)/total*100:.1f}%)")
    print(f"  Test  : {len(X_test):>5} "
          f"({len(X_test)/total*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def normalize_fixed_features(X_train, X_val, X_test):
    """
    Applies StandardScaler to fixed feature vectors.

    IMPORTANT: We fit the scaler ONLY on training data
    and apply (transform) to val and test.

    Why? Because in real life we won't have test data
    when training. Fitting on test data would be
    'data leakage' — cheating!

    Parameters:
        X_train, X_val, X_test : feature splits

    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled
        scaler : fitted scaler (saved for inference later)
    """
    print("\nNormalizing fixed features...")

    scaler = StandardScaler()

    # fit_transform on TRAIN only
    # fit   → learns mean and std from training data
    # transform → applies standardization
    X_train_scaled = scaler.fit_transform(X_train)

    # transform ONLY on val and test
    # uses the SAME mean and std from training
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    print(f"  Before scaling — mean: {X_train.mean():.2f}, "
          f"std: {X_train.std():.2f}")
    print(f"  After  scaling — mean: "
          f"{X_train_scaled.mean():.4f}, "
          f"std: {X_train_scaled.std():.4f}")

    # Save scaler — we'll need it for the demo app
    # When a user speaks in the app, we must scale
    # their audio the same way we scaled training data
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')
    print("  Scaler saved to models/scaler.pkl ✅")

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def normalize_mel_features(X_train, X_val, X_test):
    """
    Normalizes Mel Spectrogram features for CNN.

    For 2D features we use global normalization:
    subtract mean, divide by std across entire dataset.

    Parameters:
        X_train, X_val, X_test : mel spectrogram splits

    Returns:
        normalized splits + normalization stats
    """
    print("\nNormalizing mel spectrogram features...")

    # Calculate mean and std from training data only
    train_mean = X_train.mean()
    train_std  = X_train.std()

    # Apply same normalization to all splits
    X_train_norm = (X_train - train_mean) / train_std
    X_val_norm   = (X_val   - train_mean) / train_std
    X_test_norm  = (X_test  - train_mean) / train_std

    print(f"  Train mean: {train_mean:.4f}, "
          f"std: {train_std:.4f}")
    print(f"  After norm — mean: "
          f"{X_train_norm.mean():.4f}, "
          f"std: {X_train_norm.std():.4f}")

    # Save normalization stats for demo app
    np.save('models/mel_mean.npy', np.array([train_mean]))
    np.save('models/mel_std.npy',  np.array([train_std]))
    print("  Mel norm stats saved ✅")

    return X_train_norm, X_val_norm, X_test_norm


def normalize_mfcc_features(X_train, X_val, X_test):
    """
    Normalizes MFCC sequence features for LSTM.

    Parameters:
        X_train, X_val, X_test : mfcc sequence splits

    Returns:
        normalized splits
    """
    print("\nNormalizing MFCC sequence features...")

    train_mean = X_train.mean()
    train_std  = X_train.std()

    X_train_norm = (X_train - train_mean) / train_std
    X_val_norm   = (X_val   - train_mean) / train_std
    X_test_norm  = (X_test  - train_mean) / train_std

    print(f"  After norm — mean: "
          f"{X_train_norm.mean():.4f}, "
          f"std: {X_train_norm.std():.4f}")

    np.save('models/mfcc_mean.npy', np.array([train_mean]))
    np.save('models/mfcc_std.npy',  np.array([train_std]))
    print("  MFCC norm stats saved ✅")

    return X_train_norm, X_val_norm, X_test_norm


def compute_class_weights(y_train):
    """
    Computes class weights to handle class imbalance.
    Neutral has fewer samples so it gets higher weight.

    Parameters:
        y_train : training labels

    Returns:
        class_weights_dict : {class_int: weight_float}
        class_weights_array: weights as numpy array
    """
    print("\nComputing class weights...")

    classes = np.unique(y_train)
    weights = compute_class_weight(
        'balanced',      # auto-compute balanced weights
        classes=classes,
        y=y_train
    )

    # Dictionary format for PyTorch loss function
    class_weights_dict = dict(zip(classes, weights))

    print("  Class weights:")
    for cls, w in class_weights_dict.items():
        name = EMOTION_NAMES[cls]
        bar  = '█' * int(w * 10)
        print(f"    {name:<10}: {w:.4f}  {bar}")

    # Array format — index = class number
    class_weights_array = np.array(
        [class_weights_dict[i] for i in range(NUM_CLASSES)]
    )

    return class_weights_dict, class_weights_array


def reshape_for_cnn(X):
    """
    Reshapes mel spectrogram for CNN input.

    CNN expects: (samples, channels, height, width)
    We add a channel dimension (like grayscale image)

    (2880, 128, 130) → (2880, 1, 128, 130)

    Parameters:
        X : mel spectrogram array

    Returns:
        X reshaped for CNN
    """
    # np.expand_dims adds a new dimension at position 1
    return np.expand_dims(X, axis=1)


def preprocess_all():
    """
    Master function — runs the complete preprocessing
    pipeline and saves all processed data to disk.
    """
    print("=" * 55)
    print("   Phase 4: Data Preprocessing")
    print("=" * 55)

    # ── Step 1: Load features ──────────────────────────
    X_fixed, X_mel, X_mfcc, y = load_features()

    # ── Step 2: Split all feature types ───────────────
    (X_fixed_train, X_fixed_val, X_fixed_test,
     y_train, y_val, y_test) = split_data(
        X_fixed, y, 'fixed'
    )

    (X_mel_train, X_mel_val, X_mel_test,
     _, _, _) = split_data(X_mel, y, 'mel')

    (X_mfcc_train, X_mfcc_val, X_mfcc_test,
     _, _, _) = split_data(X_mfcc, y, 'mfcc')

    # ── Step 3: Normalize each feature type ───────────
    (X_fixed_train, X_fixed_val,
     X_fixed_test, _) = normalize_fixed_features(
        X_fixed_train, X_fixed_val, X_fixed_test
    )

    (X_mel_train, X_mel_val,
     X_mel_test) = normalize_mel_features(
        X_mel_train, X_mel_val, X_mel_test
    )

    (X_mfcc_train, X_mfcc_val,
     X_mfcc_test) = normalize_mfcc_features(
        X_mfcc_train, X_mfcc_val, X_mfcc_test
    )

    # ── Step 4: Reshape mel for CNN ────────────────────
    print("\nReshaping mel spectrograms for CNN...")
    X_mel_train = reshape_for_cnn(X_mel_train)
    X_mel_val   = reshape_for_cnn(X_mel_val)
    X_mel_test  = reshape_for_cnn(X_mel_test)
    print(f"  CNN input shape: {X_mel_train.shape}")
    # (samples, 1, 128, 130) — 1 = grayscale channel

    # ── Step 5: Compute class weights ─────────────────
    class_weights_dict, class_weights_array = \
        compute_class_weights(y_train)

    # ── Step 6: Save everything ────────────────────────
    print("\nSaving preprocessed data...")
    save_path = 'data/processed'
    os.makedirs(save_path, exist_ok=True)

    # Fixed features (for baseline MLP)
    np.save(f'{save_path}/X_fixed_train.npy', X_fixed_train)
    np.save(f'{save_path}/X_fixed_val.npy',   X_fixed_val)
    np.save(f'{save_path}/X_fixed_test.npy',  X_fixed_test)

    # Mel spectrograms (for CNN)
    np.save(f'{save_path}/X_mel_train.npy',   X_mel_train)
    np.save(f'{save_path}/X_mel_val.npy',     X_mel_val)
    np.save(f'{save_path}/X_mel_test.npy',    X_mel_test)

    # MFCC sequences (for LSTM)
    np.save(f'{save_path}/X_mfcc_train.npy',  X_mfcc_train)
    np.save(f'{save_path}/X_mfcc_val.npy',    X_mfcc_val)
    np.save(f'{save_path}/X_mfcc_test.npy',   X_mfcc_test)

    # Labels (shared across all models)
    np.save(f'{save_path}/y_train.npy',        y_train)
    np.save(f'{save_path}/y_val.npy',          y_val)
    np.save(f'{save_path}/y_test.npy',         y_test)

    # Class weights
    np.save(f'{save_path}/class_weights.npy',
            class_weights_array)

    print("\n✅ All files saved to data/processed/:")
    for f in sorted(os.listdir(save_path)):
        size = os.path.getsize(f'{save_path}/{f}')
        print(f"   {f:<30} "
              f"{size/1024/1024:.1f} MB")

    # ── Step 7: Summary ────────────────────────────────
    print(f"\n{'='*55}")
    print(f"   Preprocessing Complete! ")
    print(f"{'='*55}")
    print(f"\n── Final Data Summary ──────────────────")
    print(f"  Training samples   : {len(y_train)}")
    print(f"  Validation samples : {len(y_val)}")
    print(f"  Test samples       : {len(y_test)}")
    print(f"\n── Shapes ready for models ─────────────")
    print(f"  MLP  input : {X_fixed_train.shape}")
    print(f"  CNN  input : {X_mel_train.shape}")
    print(f"  LSTM input : {X_mfcc_train.shape}")

    return {
        'fixed' : (X_fixed_train, X_fixed_val, X_fixed_test),
        'mel'   : (X_mel_train,   X_mel_val,   X_mel_test),
        'mfcc'  : (X_mfcc_train,  X_mfcc_val,  X_mfcc_test),
        'labels': (y_train,       y_val,        y_test),
        'class_weights': class_weights_array
    }


# ── Run preprocessing ──────────────────────────────────
if __name__ == '__main__':
    preprocess_all()