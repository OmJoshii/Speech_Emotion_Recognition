# Builds and trains a CNN model on Mel Spectrograms
# for Speech Emotion Recognition

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

# ── Device setup ───────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available()
                       else 'cpu')
print(f"Using device: {device}")

# ── Constants ──────────────────────────────────────────
NUM_CLASSES   = 6
BATCH_SIZE    = 32
EPOCHS        = 100
LEARNING_RATE = 0.0005
DROPOUT_RATE  = 0.5

EMOTION_NAMES = ['neutral', 'happy', 'sad',
                 'angry', 'fearful', 'disgust']


# ═════════════════════════════════════════════════════
# MODEL DEFINITION
# ═════════════════════════════════════════════════════

class EmotionCNN(nn.Module):
    """
    CNN for Speech Emotion Recognition.
    Treats Mel Spectrograms as grayscale images.

    Input shape  : (batch, 1, 128, 130)
                    batch = samples per batch
                    1     = channels (grayscale)
                    128   = mel frequency bands
                    130   = time frames

    Architecture:
        Conv Block 1 → Conv Block 2 → Conv Block 3
        → Global Average Pooling
        → Fully Connected layers
        → Output (8 emotions)
    """

    def __init__(self):
        super(EmotionCNN, self).__init__()

        # ── Convolutional Blocks ───────────────────────
        # Each block: Conv2d → BatchNorm → ReLU → MaxPool

        # Block 1: 1 channel → 32 filters
        # Learns basic patterns (edges, textures)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,    # input: 1 (grayscale)
                out_channels=32,  # 32 different filters
                kernel_size=3,    # 3×3 filter size
                padding=1         # keeps output same size
            ),
            nn.BatchNorm2d(32),   # normalize 32 channels
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,    # 2×2 pooling window
                stride=2          # move 2 steps at a time
            )
            # Output: (batch, 32, 64, 65)
            # 128/2=64 height, 130/2=65 width
        )

        # Block 2: 32 channels → 64 filters
        # Learns more complex patterns
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # Output: (batch, 64, 32, 32)
        )

        # Block 3: 64 channels → 128 filters
        # Learns high-level emotion features
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # Output: (batch, 128, 16, 16)
        )

        # Block 4: 128 channels → 256 filters
        # Deepest feature extraction
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
            # Output: (batch, 256, 8, 8)
        )

        # ── Global Average Pooling ─────────────────────
        # Instead of flattening all values, take the
        # average of each feature map
        # (batch, 256, 8, 8) → (batch, 256)
        # Much better than flatten — reduces overfitting
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        # AdaptiveAvgPool2d(1) → output is always (1,1)
        # regardless of input size

        # ── Classifier ────────────────────────────────
        # Takes the 256 pooled features → 8 emotions
        self.classifier = nn.Sequential(
            nn.Flatten(),           # (batch, 256, 1, 1) → (batch, 256)
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        """
        Forward pass through the CNN.

        Parameters:
            x : input tensor (batch, 1, 128, 130)

        Returns:
            output : (batch, 8) emotion scores
        """
        # Pass through conv blocks
        x = self.conv_block1(x)   # (batch, 32,  64, 65)
        x = self.conv_block2(x)   # (batch, 64,  32, 32)
        x = self.conv_block3(x)   # (batch, 128, 16, 16)
        x = self.conv_block4(x)   # (batch, 256,  8,  8)

        # Global average pooling
        x = self.global_avg_pool(x)  # (batch, 256, 1, 1)

        # Classification
        x = self.classifier(x)    # (batch, 8)

        return x


# ══════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════

def load_data():
    """
    Loads preprocessed mel spectrogram data.
    Returns DataLoaders ready for CNN training.
    """
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

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    X_val_t   = torch.FloatTensor(X_val)
    X_test_t  = torch.FloatTensor(X_test)
    y_train_t = torch.LongTensor(y_train)
    y_val_t   = torch.LongTensor(y_val)
    y_test_t  = torch.LongTensor(y_test)

    # Load class weights
    class_weights   = np.load('data/processed/class_weights.npy')
    class_weights_t = torch.FloatTensor(class_weights).to(device)

    # Create datasets
    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset   = TensorDataset(X_val_t,   y_val_t)
    test_dataset  = TensorDataset(X_test_t,  y_test_t)

    # Create dataloaders
    train_loader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              drop_last=True)
    val_loader   = DataLoader(val_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False)
    test_loader  = DataLoader(test_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

    return (train_loader, val_loader, test_loader,
            y_test, class_weights_t)


# ══════════════════════════════════════════════════════
# TRAINING & EVALUATION FUNCTIONS
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
        optimizer.step()

        total_loss    += loss.item()
        predicted      = torch.argmax(predictions, dim=1)
        correct       += (predicted == y_batch).sum().item()
        total_samples += len(y_batch)

    return total_loss / len(loader), correct / total_samples


def evaluate(model, loader, criterion):
    """Evaluates model on val or test data."""
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


# ══════════════════════════════════════════════════════
# LEARNING RATE SCHEDULER
# ══════════════════════════════════════════════════════

# A scheduler reduces the learning rate when the model
# stops improving — like taking smaller steps as you
# get closer to the destination

# ══════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader,
                criterion, optimizer):
    """Complete training loop with LR scheduling."""

    print(f"\n{'='*55}")
    print(f"   Training CNN Model")
    print(f"{'='*55}")
    print(f"Epochs     : {EPOCHS}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Device     : {device}")
    print(f"{'='*55}\n")

    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc' : [], 'val_acc' : []
    }

    # Learning rate scheduler
    # Reduces LR by factor 0.5 if val_loss doesn't
    # improve for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',       # monitor minimum loss
        factor=0.5,       # multiply LR by 0.5
        patience=5,       # wait 5 epochs
    )

    # Early stopping
    best_val_loss  = float('inf')
    best_val_acc   = 0.0
    patience       = 20
    patience_count = 0
    best_epoch     = 0

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion
        )

        # Step scheduler with validation loss
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

        # Early stopping + save best model
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_val_acc   = val_acc
            best_epoch     = epoch
            patience_count = 0
            torch.save(model.state_dict(),
                       'models/cnn_best.pth')
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"\n⚡ Early stopping at epoch {epoch}")
                print(f"   Best was epoch {best_epoch} "
                      f"with val loss {best_val_loss:.4f}")
                break

    print(f"\n✅ Training complete!")
    print(f"   Best epoch    : {best_epoch}")
    print(f"   Best val loss : {best_val_loss:.4f}")
    print(f"   Best val acc  : {best_val_acc*100:.2f}%")

    return history


# ══════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════

def plot_training_history(history):
    """Plots loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('CNN Training History',
                 fontsize=13, fontweight='bold')

    epochs_range = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs_range, history['train_loss'],
                 label='Train Loss',
                 color='steelblue', linewidth=1.5)
    axes[0].plot(epochs_range, history['val_loss'],
                 label='Val Loss',
                 color='coral', linewidth=1.5)
    axes[0].set_title('Loss Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range,
                 [a*100 for a in history['train_acc']],
                 label='Train Accuracy',
                 color='steelblue', linewidth=1.5)
    axes[1].plot(epochs_range,
                 [a*100 for a in history['val_acc']],
                 label='Val Accuracy',
                 color='coral', linewidth=1.5)
    axes[1].set_title('Accuracy Over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('models/cnn_training_history.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Training history saved!")


def plot_confusion_matrix(y_true, y_pred, title):
    """Plots normalized confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / \
                 cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent,
                annot=True, fmt='.1f',
                cmap='Blues',
                xticklabels=EMOTION_NAMES,
                yticklabels=EMOTION_NAMES,
                vmin=0, vmax=100)
    plt.title(f'{title}\n(values are %)')
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('models/cnn_confusion_matrix.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Confusion matrix saved!")


# ══════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("   Speech Emotion Recognition — CNN Model")
    print("=" * 55)

    # ── Step 1: Load data ──────────────────────────────
    (train_loader, val_loader, test_loader,
     y_test, class_weights) = load_data()

    # ── Step 2: Build model ────────────────────────────
    print("\nBuilding CNN model...")
    model = EmotionCNN().to(device)

    total_params = sum(p.numel()
                       for p in model.parameters())
    trainable    = sum(p.numel()
                       for p in model.parameters()
                       if p.requires_grad)
    print(f"  Total parameters    : {total_params:,}")
    print(f"  Trainable parameters: {trainable:,}")

    # ── Step 3: Loss and optimizer ─────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           weight_decay=1e-3)

    # ── Step 4: Train ──────────────────────────────────
    history = train_model(model, train_loader,
                          val_loader, criterion, optimizer)

    # ── Step 5: Plot history ───────────────────────────
    plot_training_history(history)

    # ── Step 6: Load best and evaluate ────────────────
    print("\nLoading best model weights...")
    model.load_state_dict(
        torch.load('models/cnn_best.pth',
                   map_location=device)
    )

    print("\n── Final Evaluation on Test Set ──")
    test_loss, test_acc, y_pred, y_true = evaluate(
        model, test_loader, criterion
    )

    print(f"  Test Loss     : {test_loss:.4f}")
    print(f"  Test Accuracy : {test_acc*100:.2f}%")

    # ── Step 7: Classification report ─────────────────
    print("\n── Classification Report ──────────────")
    print(classification_report(y_true, y_pred,
                                 target_names=EMOTION_NAMES))

    # ── Step 8: Confusion matrix ───────────────────────
    plot_confusion_matrix(y_true, y_pred,
                          'CNN Model — Test Set')

    # ── Step 9: Compare with MLP ───────────────────────
    print("\n── Model Comparison So Far ────────────")

    # Load MLP results if available
    mlp_acc = "N/A"
    if os.path.exists('models/mlp_results.json'):
        with open('models/mlp_results.json') as f:
            mlp_results = json.load(f)
            mlp_acc = f"{mlp_results['test_accuracy']}%"

    print(f"  MLP Baseline : {mlp_acc}")
    print(f"  CNN          : {test_acc*100:.2f}%")

    improvement = test_acc*100 - float(mlp_acc.replace('%','')) \
                  if mlp_acc != "N/A" else 0
    if improvement > 0:
        print(f"  Improvement  : +{improvement:.2f}% ✅")
    else:
        print(f"  Difference   : {improvement:.2f}%")

    # ── Step 10: Save results ──────────────────────────
    results = {
        'model'         : 'CNN',
        'test_accuracy' : round(test_acc * 100, 2),
        'test_loss'     : round(test_loss, 4),
        'epochs_trained': len(history['train_loss']),
        'parameters'    : total_params
    }

    with open('models/cnn_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n{'='*55}")
    print(f"   CNN Training Complete! 🎉")
    print(f"{'='*55}")
    print(f"\n  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Model saved   : models/cnn_best.pth")
    print(f"  Results saved : models/cnn_results.json")


if __name__ == '__main__':
    main()