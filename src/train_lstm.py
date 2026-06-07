# Contains two models:
#   1. Standalone LSTM (on MFCC sequences)
#   2. CNN-LSTM Hybrid (CNN features + LSTM temporal)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (accuracy_score,
                              classification_report,
                              confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# ── Device ─────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available()
                       else 'cpu')
print(f"Using device: {device}")

# ── Constants ──────────────────────────────────────────
NUM_CLASSES   = 8
BATCH_SIZE    = 32
EPOCHS        = 100
LEARNING_RATE = 0.001
DROPOUT_RATE  = 0.3

# MFCC sequence dimensions
TIME_STEPS    = 130   # number of time frames
N_MFCC        = 40    # features per time step

# Mel spectrogram dimensions (for CNN-LSTM)
MEL_BANDS     = 128
MEL_FRAMES    = 130

EMOTION_NAMES = ['neutral', 'calm',    'happy',    'sad',
                 'angry',   'fearful', 'disgust',  'surprised']


# ══════════════════════════════════════════════════════
# MODEL 1 — STANDALONE LSTM
# ══════════════════════════════════════════════════════

class EmotionLSTM(nn.Module):
    """
    Bidirectional stacked LSTM for Speech Emotion
    Recognition on MFCC sequences.

    Input  : (batch, 130, 40) — MFCC sequence
    Output : (batch, 8)       — emotion scores

    Architecture:
        BiLSTM Layer 1 (128 units each direction)
        BiLSTM Layer 2 (64 units each direction)
        Attention pooling
        FC layers → 8 emotions
    """

    def __init__(self):
        super(EmotionLSTM, self).__init__()

        # ── Bidirectional LSTM layers ──────────────────
        # bidirectional=True → reads forward AND backward
        # hidden_size=128 → 128 units per direction
        # total output = 128 × 2 = 256 per time step

        self.lstm1 = nn.LSTM(
            input_size=N_MFCC,    # 40 features per step
            hidden_size=128,
            num_layers=1,
            batch_first=True,     # (batch, time, features)
            bidirectional=True,   # forward + backward
            dropout=0.0
        )
        # Output shape: (batch, 130, 256)

        self.lstm2 = nn.LSTM(
            input_size=256,       # 128×2 from lstm1
            hidden_size=64,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )
        # Output shape: (batch, 130, 128)

        # ── Batch normalization after LSTM ─────────────
        self.bn1 = nn.BatchNorm1d(256)  # after lstm1
        self.bn2 = nn.BatchNorm1d(128)  # after lstm2

        # ── Attention mechanism ────────────────────────
        # Learns which time steps are most important
        # Linear layer: 128 → 1 (attention score per step)
        self.attention = nn.Linear(128, 1)

        # ── Classifier ────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(64, NUM_CLASSES)
        )

    def attention_pool(self, lstm_output):
        """
        Applies attention pooling over time steps.

        Instead of just taking the last hidden state,
        attention weighs ALL time steps and takes
        a weighted average — focusing on important moments.

        Parameters:
            lstm_output : (batch, time_steps, features)

        Returns:
            context : (batch, features) — weighted summary
        """
        # Calculate attention score for each time step
        # scores shape: (batch, time_steps, 1)
        scores = self.attention(lstm_output)

        # Softmax converts scores to probabilities
        # that sum to 1 across time dimension
        weights = torch.softmax(scores, dim=1)
        # weights shape: (batch, time_steps, 1)

        # Weighted sum across time steps
        # (batch, time, features) × (batch, time, 1)
        # → sum over time → (batch, features)
        context = (lstm_output * weights).sum(dim=1)

        return context

    def forward(self, x):
        """
        Forward pass.

        Parameters:
            x : (batch, 130, 40) MFCC sequence

        Returns:
            output : (batch, 8) emotion scores
        """
        # ── LSTM Layer 1 ───────────────────────────────
        # lstm_out: (batch, 130, 256) — all time steps
        # hidden:   (2, batch, 128)  — final hidden state
        lstm_out1, _ = self.lstm1(x)

        # Apply batch norm — needs (batch, features, time)
        # so we permute, normalize, then permute back
        lstm_out1 = self.bn1(
            lstm_out1.permute(0, 2, 1)
        ).permute(0, 2, 1)

        # ── LSTM Layer 2 ───────────────────────────────
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.bn2(
            lstm_out2.permute(0, 2, 1)
        ).permute(0, 2, 1)

        # ── Attention pooling ──────────────────────────
        # Instead of using only the last time step,
        # use weighted average of ALL time steps
        context = self.attention_pool(lstm_out2)
        # context shape: (batch, 128)

        # ── Classification ─────────────────────────────
        output = self.classifier(context)
        return output


# ══════════════════════════════════════════════════════
# MODEL 2 — CNN-LSTM HYBRID
# ══════════════════════════════════════════════════════

class EmotionCNNLSTM(nn.Module):
    """
    CNN-LSTM Hybrid for Speech Emotion Recognition.

    CNN extracts local frequency features from
    mel spectrogram slices, then LSTM learns
    temporal patterns across those features.

    Input  : (batch, 1, 128, 130) — mel spectrogram
    Output : (batch, 8)           — emotion scores

    Architecture:
        CNN Feature Extractor (2 lightweight conv blocks)
        → Reshape to sequence
        → BiLSTM layers
        → Attention pooling
        → FC → 8 emotions
    """

    def __init__(self):
        super(EmotionCNNLSTM, self).__init__()

        # ── Lightweight CNN Feature Extractor ─────────
        # Only 2 blocks — lightweight to prevent overfit
        # Extracts frequency patterns from spectrogram

        self.cnn = nn.Sequential(
            # Block 1: 1 → 32 channels
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            # Pool only height (frequency), not width (time)
            # → (batch, 32, 64, 130)
            # We keep time dimension intact for LSTM!

            # Block 2: 32 → 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            # → (batch, 64, 32, 130)

            nn.Dropout2d(0.2)
        )

        # After CNN: (batch, 64, 32, 130)
        # CNN feature size per time step = 64 × 32 = 2048

        # ── Projection layer ──────────────────────────
        # Reduce 2048 → 128 before feeding to LSTM
        # This prevents LSTM from being overwhelmed
        self.projection = nn.Sequential(
            nn.Linear(64 * 32, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # ── Bidirectional LSTM ─────────────────────────
        self.lstm = nn.LSTM(
            input_size=128,     # projected CNN features
            hidden_size=128,
            num_layers=2,       # stacked LSTM
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        # Output: (batch, 130, 256)

        # ── Attention ─────────────────────────────────
        self.attention = nn.Linear(256, 1)

        # ── Classifier ────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(64, NUM_CLASSES)
        )

    def attention_pool(self, lstm_output):
        """Attention pooling over time steps."""
        scores  = self.attention(lstm_output)
        weights = torch.softmax(scores, dim=1)
        context = (lstm_output * weights).sum(dim=1)
        return context

    def forward(self, x):
        """
        Forward pass through CNN-LSTM.

        Parameters:
            x : (batch, 1, 128, 130) mel spectrogram

        Returns:
            output : (batch, 8) emotion scores
        """
        batch_size = x.size(0)

        # ── CNN Feature Extraction ─────────────────────
        cnn_out = self.cnn(x)
        # Shape: (batch, 64, 32, 130)

        # ── Reshape for LSTM ───────────────────────────
        # We want: (batch, time_steps, features)
        # time_steps = 130 (time dimension)
        # features   = 64 × 32 = 2048 (CNN features)

        # Permute: (batch, channels, freq, time)
        #       → (batch, time, channels, freq)
        cnn_out = cnn_out.permute(0, 3, 1, 2)
        # Shape: (batch, 130, 64, 32)

        # Flatten last two dims: (batch, 130, 64×32)
        cnn_out = cnn_out.reshape(batch_size, 130, -1)
        # Shape: (batch, 130, 2048)

        # ── Project to smaller size ────────────────────
        # Apply projection to each time step
        projected = self.projection(cnn_out)
        # Shape: (batch, 130, 128)

        # ── LSTM ──────────────────────────────────────
        lstm_out, _ = self.lstm(projected)
        # Shape: (batch, 130, 256)

        # ── Attention pooling ──────────────────────────
        context = self.attention_pool(lstm_out)
        # Shape: (batch, 256)

        # ── Classification ─────────────────────────────
        output = self.classifier(context)
        return output


# ══════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════

def load_mfcc_data():
    """Loads MFCC sequence data for standalone LSTM."""
    print("\nLoading MFCC sequence data...")

    X_train = np.load('data/processed/X_mfcc_train.npy')
    X_val   = np.load('data/processed/X_mfcc_val.npy')
    X_test  = np.load('data/processed/X_mfcc_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_val   = np.load('data/processed/y_val.npy')
    y_test  = np.load('data/processed/y_test.npy')

    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")

    X_train_t = torch.FloatTensor(X_train)
    X_val_t   = torch.FloatTensor(X_val)
    X_test_t  = torch.FloatTensor(X_test)
    y_train_t = torch.LongTensor(y_train)
    y_val_t   = torch.LongTensor(y_val)
    y_test_t  = torch.LongTensor(y_test)

    class_weights   = np.load(
        'data/processed/class_weights.npy'
    )
    class_weights_t = torch.FloatTensor(
        class_weights
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return (train_loader, val_loader, test_loader,
            y_test, class_weights_t)


def load_mel_data():
    """Loads mel spectrogram data for CNN-LSTM."""
    print("\nLoading mel spectrogram data...")

    X_train = np.load('data/processed/X_mel_train.npy')
    X_val   = np.load('data/processed/X_mel_val.npy')
    X_test  = np.load('data/processed/X_mel_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_val   = np.load('data/processed/y_val.npy')
    y_test  = np.load('data/processed/y_test.npy')

    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")

    X_train_t = torch.FloatTensor(X_train)
    X_val_t   = torch.FloatTensor(X_val)
    X_test_t  = torch.FloatTensor(X_test)
    y_train_t = torch.LongTensor(y_train)
    y_val_t   = torch.LongTensor(y_val)
    y_test_t  = torch.LongTensor(y_test)

    class_weights   = np.load(
        'data/processed/class_weights.npy'
    )
    class_weights_t = torch.FloatTensor(
        class_weights
    ).to(device)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return (train_loader, val_loader, test_loader,
            y_test, class_weights_t)


# ══════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ══════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer):
    """Trains model for one epoch."""
    model.train()
    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        predictions = model(X_batch)
        loss        = criterion(predictions, y_batch)

        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping — prevents exploding gradients
        # which is a common problem with LSTMs
        # Clips gradients to max norm of 1.0
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

        total_loss    += loss.item()
        predicted      = torch.argmax(predictions, dim=1)
        correct       += (predicted == y_batch).sum().item()
        total_samples += len(y_batch)

    return total_loss / len(loader), correct / total_samples


def evaluate(model, loader, criterion):
    """Evaluates model."""
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss        = criterion(predictions, y_batch)

            total_loss += loss.item()
            predicted   = torch.argmax(predictions, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, all_preds, all_labels


def train_model(model, train_loader, val_loader,
                criterion, optimizer, model_name):
    """Complete training loop."""

    print(f"\n{'='*55}")
    print(f"   Training {model_name}")
    print(f"{'='*55}")

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc' : [], 'val_acc' : []
    }

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss  = float('inf')
    best_val_acc   = 0.0
    patience       = 20
    patience_count = 0
    best_epoch     = 0
    save_path      = f'models/{model_name.lower().replace(" ", "_")}_best.pth'

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion
        )

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:>3}/{EPOCHS}]  "
                  f"Train Loss: {train_loss:.4f}  "
                  f"Train Acc: {train_acc*100:.2f}%  │  "
                  f"Val Loss: {val_loss:.4f}  "
                  f"Val Acc: {val_acc*100:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_val_acc   = val_acc
            best_epoch     = epoch
            patience_count = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n⚡ Early stopping at epoch {epoch}")
                print(f"   Best epoch {best_epoch} — "
                      f"val loss {best_val_loss:.4f}")
                break

    print(f"\n✅ Training complete!")
    print(f"   Best epoch   : {best_epoch}")
    print(f"   Best val acc : {best_val_acc*100:.2f}%")

    return history, save_path


# ══════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════

def plot_history(history, model_name):
    """Plots training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} Training History',
                 fontsize=13, fontweight='bold')

    epochs_range = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs_range, history['train_loss'],
                 color='steelblue', label='Train', linewidth=1.5)
    axes[0].plot(epochs_range, history['val_loss'],
                 color='coral', label='Val', linewidth=1.5)
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range,
                 [a*100 for a in history['train_acc']],
                 color='steelblue', label='Train', linewidth=1.5)
    axes[1].plot(epochs_range,
                 [a*100 for a in history['val_acc']],
                 color='coral', label='Val', linewidth=1.5)
    axes[1].set_title('Accuracy (%)')
    axes[1].set_xlabel('Epoch')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fname = model_name.lower().replace(' ', '_')
    plt.savefig(f'models/{fname}_training_history.png',
                dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plots confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype('float') / \
             cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_pct, annot=True, fmt='.1f',
                cmap='Blues',
                xticklabels=EMOTION_NAMES,
                yticklabels=EMOTION_NAMES,
                vmin=0, vmax=100)
    plt.title(f'{model_name} — Confusion Matrix (%)')
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    fname = model_name.lower().replace(' ', '_')
    plt.savefig(f'models/{fname}_confusion_matrix.png',
                dpi=150, bbox_inches='tight')
    plt.show()


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

def run_model(ModelClass, load_fn, model_name):
    """
    Generic function to train and evaluate any model.
    Keeps code DRY (Don't Repeat Yourself).
    """
    print(f"\n{'#'*55}")
    print(f"#  {model_name}")
    print(f"{'#'*55}")

    # Load data
    (train_loader, val_loader, test_loader,
     y_test, class_weights) = load_fn()

    # Build model
    model        = ModelClass().to(device)
    total_params = sum(p.numel()
                       for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           weight_decay=1e-3)

    # Train
    history, save_path = train_model(
        model, train_loader, val_loader,
        criterion, optimizer, model_name
    )

    # Plot history
    plot_history(history, model_name)

    # Load best and evaluate
    print(f"\nLoading best weights from {save_path}...")
    model.load_state_dict(
        torch.load(save_path, map_location=device)
    )

    print(f"\n── Test Set Evaluation ──")
    test_loss, test_acc, y_pred, y_true = evaluate(
        model, test_loader, criterion
    )
    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc*100:.2f}%")

    print(f"\n── Classification Report ──")
    print(classification_report(y_true, y_pred,
                                 target_names=EMOTION_NAMES))

    plot_confusion_matrix(y_true, y_pred, model_name)

    # Save results
    results = {
        'model'         : model_name,
        'test_accuracy' : round(test_acc * 100, 2),
        'test_loss'     : round(test_loss, 4),
        'epochs_trained': len(history['train_loss']),
        'parameters'    : total_params
    }
    fname = model_name.lower().replace(' ', '_')
    with open(f'models/{fname}_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    return test_acc * 100


def main():
    print("=" * 55)
    print("   Speech Emotion Recognition")
    print("   Phase 7: LSTM & CNN-LSTM Models")
    print("=" * 55)

    results = {}

    # ── Train LSTM ─────────────────────────────────────
    results['LSTM'] = run_model(
        EmotionLSTM,
        load_mfcc_data,
        'LSTM'
    )

    # ── Train CNN-LSTM ─────────────────────────────────
    results['CNN-LSTM'] = run_model(
        EmotionCNNLSTM,
        load_mel_data,
        'CNN LSTM'
    )

    # ── Final comparison ───────────────────────────────
    print(f"\n{'='*55}")
    print(f"   Final Model Comparison")
    print(f"{'='*55}")

    # Load previous results
    for model_file, label in [
        ('models/mlp_results.json', 'MLP Baseline'),
        ('models/cnn_results.json', 'CNN')
    ]:
        if os.path.exists(model_file):
            with open(model_file) as f:
                data = json.load(f)
                results[label] = data['test_accuracy']

    # Print comparison table
    print(f"\n  {'Model':<15} {'Accuracy':>10}")
    print(f"  {'-'*27}")
    for model, acc in sorted(results.items(),
                              key=lambda x: x[1],
                              reverse=True):
        marker = ' ← BEST' if acc == max(
            results.values()
        ) else ''
        print(f"  {model:<15} {acc:>9.2f}%{marker}")

    print(f"\n{'='*55}")
    print(f"   Phase 7 Complete! 🎉")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()