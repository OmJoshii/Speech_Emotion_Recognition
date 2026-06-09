# Builds and trains an MLP model using PyTorch
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

# ── Device Setup ───────────────────────────────────────
# Check if GPU is available, else use CPU
# GPU trains much faster but CPU works fine for MLP

device = torch.device('cuda' if torch.cuda.is_available()
                       else 'cpu')
print(f"Using device: {device}")

# ── Constants ──────────────────────────────────────────
INPUT_SIZE   = 94      # number of features per sample
NUM_CLASSES  = 6       # number of emotions
BATCH_SIZE   = 32      # samples per batch
EPOCHS       = 100     # training iterations
LEARNING_RATE = 0.001  # how fast model learns
DROPOUT_RATE  = 0.3    # 30% neurons dropped during training

# Emotion label names for display
EMOTION_NAMES = ['neutral', 'happy', 'sad',
                 'angry', 'fearful', 'disgust']


# ══════════════════════════════════════════════════════
# MODEL DEFINITION
# ══════════════════════════════════════════════════════

class EmotionMLP(nn.Module):
    """
    Multi-Layer Perceptron for Speech Emotion Recognition.

    Architecture:
        Input(94) → FC(256) → BN → ReLU → Dropout
                 → FC(128) → BN → ReLU → Dropout
                 → FC(64)  → BN → ReLU → Dropout
                 → FC(8)   → Output

    nn.Module is the base class for all PyTorch models.
    Every model must inherit from it.
    """

    def __init__(self):
        # Always call parent __init__ first
        super(EmotionMLP, self).__init__()

        # ── Layer definitions ──────────────────────────
        # nn.Sequential groups layers in order
        # Data flows through them one by one

        self.network = nn.Sequential(

            # ── Block 1 ────────────────────────────────
            # FC layer: 94 inputs → 256 outputs
            nn.Linear(INPUT_SIZE, 256),
            # BatchNorm stabilizes training
            nn.BatchNorm1d(256),
            # ReLU activation — negative values → 0
            nn.ReLU(),
            # Dropout — randomly zero 30% of neurons
            nn.Dropout(DROPOUT_RATE),

            # ── Block 2 ────────────────────────────────
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),

            # ── Block 3 ────────────────────────────────
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),

            # ── Output Layer ───────────────────────────
            # 64 → 8 (one score per emotion)
            # No activation here — CrossEntropyLoss
            # handles that internally
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        """
        Defines how data flows through the network.
        PyTorch calls this automatically during training.

        Parameters:
            x : input tensor of shape (batch_size, 94)

        Returns:
            output : tensor of shape (batch_size, 8)
                     raw scores for each emotion
        """
        return self.network(x)


# ══════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════

def load_data():
    """
    Loads preprocessed numpy arrays and converts
    them to PyTorch tensors and DataLoaders.

    Returns:
        train_loader, val_loader, test_loader
        y_test : numpy array of test labels
    """
    print("\nLoading preprocessed data...")

    # Load numpy arrays
    X_train = np.load('data/processed/X_fixed_train.npy')
    X_val   = np.load('data/processed/X_fixed_val.npy')
    X_test  = np.load('data/processed/X_fixed_test.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_val   = np.load('data/processed/y_val.npy')
    y_test  = np.load('data/processed/y_test.npy')

    print(f"  Train : {X_train.shape} | {len(y_train)} labels")
    print(f"  Val   : {X_val.shape}   | {len(y_val)} labels")
    print(f"  Test  : {X_test.shape}  | {len(y_test)} labels")

    # ── Convert to PyTorch tensors ─────────────────────
    # PyTorch works with tensors not numpy arrays
    # torch.FloatTensor = 32-bit float (standard for models)
    # torch.LongTensor  = 64-bit int   (required for labels)

    X_train_t = torch.FloatTensor(X_train)
    X_val_t   = torch.FloatTensor(X_val)
    X_test_t  = torch.FloatTensor(X_test)
    y_train_t = torch.LongTensor(y_train)
    y_val_t   = torch.LongTensor(y_val)
    y_test_t  = torch.LongTensor(y_test)

    # ── Load class weights ─────────────────────────────
    class_weights = np.load('data/processed/class_weights.npy')
    class_weights_t = torch.FloatTensor(class_weights).to(device)

    # ── Create TensorDatasets ──────────────────────────
    # TensorDataset pairs features with their labels
    # so they stay together when shuffled

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset   = TensorDataset(X_val_t,   y_val_t)
    test_dataset  = TensorDataset(X_test_t,  y_test_t)

    # ── Create DataLoaders ─────────────────────────────
    # DataLoader automatically:
    # - splits data into batches
    # - shuffles training data each epoch
    # - loads data efficiently

    train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          drop_last=True)
    # shuffle=True randomizes order each epoch
    # prevents model from memorizing order

    val_loader   = DataLoader(val_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

    test_loader  = DataLoader(test_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

    return (train_loader, val_loader, test_loader,
            y_test, class_weights_t)


# ══════════════════════════════════════════════════════
# TRAINING FUNCTION
# ══════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer):
    """
    Trains the model for one complete epoch.

    Parameters:
        model     : our EmotionMLP model
        loader    : training DataLoader
        criterion : loss function
        optimizer : Adam optimizer

    Returns:
        avg_loss : average loss for this epoch
        accuracy : training accuracy for this epoch
    """
    # Set model to training mode
    # This enables dropout and batch normalization
    model.train()

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    # Loop through each batch
    for X_batch, y_batch in loader:

        # Move data to device (GPU/CPU)
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # ── Forward pass ───────────────────────────────
        # Pass input through model to get predictions
        predictions = model(X_batch)
        # Shape: (batch_size, 8) — score for each emotion

        # ── Calculate loss ─────────────────────────────
        # Compare predictions to true labels
        loss = criterion(predictions, y_batch)

        # ── Backward pass ──────────────────────────────
        # Zero gradients from previous batch
        # (PyTorch accumulates gradients by default)
        optimizer.zero_grad()

        # Calculate gradients via backpropagation
        # This computes how much each weight contributed
        # to the loss
        loss.backward()

        # ── Update weights ─────────────────────────────
        # Adam optimizer adjusts weights to reduce loss
        optimizer.step()

        # ── Track metrics ──────────────────────────────
        total_loss += loss.item()

        # Get predicted class (highest score)
        # torch.argmax returns index of max value
        predicted = torch.argmax(predictions, dim=1)
        correct       += (predicted == y_batch).sum().item()
        total_samples += len(y_batch)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total_samples
    return avg_loss, accuracy


# ══════════════════════════════════════════════════════
# VALIDATION FUNCTION
# ══════════════════════════════════════════════════════

def evaluate(model, loader, criterion):
    """
    Evaluates model on validation or test data.
    No weight updates happen here.

    Parameters:
        model     : our EmotionMLP model
        loader    : val or test DataLoader
        criterion : loss function

    Returns:
        avg_loss : average loss
        accuracy : accuracy score
        all_preds: all predictions (for confusion matrix)
        all_labels: all true labels
    """
    # Set model to evaluation mode
    # This disables dropout and uses running stats
    # for batch normalization
    model.eval()

    total_loss  = 0.0
    all_preds   = []
    all_labels  = []

    # torch.no_grad() tells PyTorch not to calculate
    # gradients — saves memory and speeds up evaluation
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(X_batch)
            loss        = criterion(predictions, y_batch)

            total_loss += loss.item()

            predicted = torch.argmax(predictions, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_preds)
    return avg_loss, accuracy, all_preds, all_labels


# ══════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════

def train_model(model, train_loader, val_loader,
                criterion, optimizer):
    """
    Runs the complete training loop for all epochs.
    Implements early stopping to prevent overfitting.

    Parameters:
        model        : EmotionMLP model
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        criterion    : loss function
        optimizer    : Adam optimizer

    Returns:
        history : dict of training metrics per epoch
    """
    print(f"\n{'='*55}")
    print(f"   Training MLP Model")
    print(f"{'='*55}")
    print(f"Epochs     : {EPOCHS}")
    print(f"Batch size : {BATCH_SIZE}")
    print(f"Device     : {device}")
    print(f"{'='*55}\n")

    # ── History tracking ───────────────────────────────
    history = {
        'train_loss': [], 'val_loss'  : [],
        'train_acc' : [], 'val_acc'   : []
    }

    # ── Early stopping ─────────────────────────────────
    # Stop training if validation loss doesn't improve
    # for 'patience' epochs — prevents overfitting

    best_val_loss  = float('inf')  # start with infinity
    best_val_acc   = 0.0
    patience       = 15            # wait 15 epochs
    patience_count = 0
    best_epoch     = 0

    for epoch in range(1, EPOCHS + 1):

        # Train for one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer
        )

        # Evaluate on validation set
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion
        )

        # Store history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # ── Print progress every 10 epochs ────────────
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:>3}/{EPOCHS}]  "
                  f"Train Loss: {train_loss:.4f}  "
                  f"Train Acc: {train_acc*100:.2f}%  │  "
                  f"Val Loss: {val_loss:.4f}  "
                  f"Val Acc: {val_acc*100:.2f}%")

        # ── Early stopping check ───────────────────────
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_val_acc   = val_acc
            best_epoch     = epoch
            patience_count = 0

            # Save best model weights
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(),
                       'models/mlp_best.pth')
            # state_dict() = all the learned weights
            # We save the BEST weights not the final ones

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
# PLOTTING FUNCTIONS
# ══════════════════════════════════════════════════════

def plot_training_history(history):
    """
    Plots training and validation loss/accuracy curves.
    These curves tell us if the model is learning well
    or overfitting.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('MLP Training History',
                 fontsize=13, fontweight='bold')

    epochs_range = range(1, len(history['train_loss']) + 1)

    # ── Loss curves ────────────────────────────────────
    axes[0].plot(epochs_range, history['train_loss'],
                 label='Train Loss',
                 color='steelblue', linewidth=1.5)
    axes[0].plot(epochs_range, history['val_loss'],
                 label='Val Loss',
                 color='coral', linewidth=1.5)
    axes[0].set_title('Loss Over Epochs\n'
                      '(both should decrease together)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # ── Accuracy curves ────────────────────────────────
    axes[1].plot(epochs_range,
                 [a*100 for a in history['train_acc']],
                 label='Train Accuracy',
                 color='steelblue', linewidth=1.5)
    axes[1].plot(epochs_range,
                 [a*100 for a in history['val_acc']],
                 label='Val Accuracy',
                 color='coral', linewidth=1.5)
    axes[1].set_title('Accuracy Over Epochs\n'
                      '(both should increase together)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('models/mlp_training_history.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Training history plot saved!")


def plot_confusion_matrix(y_true, y_pred, title):
    """
    Plots a confusion matrix heatmap.

    The confusion matrix shows:
    - Diagonal = correct predictions
    - Off-diagonal = mistakes (what got confused with what)
    """
    cm = confusion_matrix(y_true, y_pred)

    # Normalize to percentages
    cm_percent = cm.astype('float') / \
                 cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent,
                annot=True,
                fmt='.1f',
                cmap='Blues',
                xticklabels=EMOTION_NAMES,
                yticklabels=EMOTION_NAMES,
                vmin=0, vmax=100)
    plt.title(f'{title}\n'
              f'(values are % — diagonal = correct)')
    plt.ylabel('True Emotion')
    plt.xlabel('Predicted Emotion')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('models/mlp_confusion_matrix.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Confusion matrix saved!")


# ══════════════════════════════════════════════════════
# MAIN — RUNS EVERYTHING
# ══════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("   Speech Emotion Recognition — MLP Model")
    print("=" * 55)

    # ── Step 1: Load data ──────────────────────────────
    (train_loader, val_loader, test_loader,
     y_test, class_weights) = load_data()

    # ── Step 2: Build model ────────────────────────────
    print("\nBuilding MLP model...")
    model = EmotionMLP().to(device)
    # .to(device) moves model to GPU if available

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    # numel() = number of elements in a tensor
    # parameters() = all learnable weights in model

    # ── Step 3: Define loss and optimizer ──────────────
    # CrossEntropyLoss for multi-class classification
    # weight=class_weights handles class imbalance
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Adam optimizer with our learning rate
    optimizer = optim.Adam(model.parameters(),
                           lr=LEARNING_RATE,
                           weight_decay=1e-4)
    # weight_decay adds L2 regularization
    # another technique to prevent overfitting

    # ── Step 4: Train model ────────────────────────────
    history = train_model(model, train_loader,
                          val_loader, criterion, optimizer)

    # ── Step 5: Plot training history ─────────────────
    plot_training_history(history)

    # ── Step 6: Load best weights and evaluate ─────────
    print("\nLoading best model weights...")
    model.load_state_dict(
        torch.load('models/mlp_best.pth',
                   map_location=device)
    )

    # Final evaluation on TEST set
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
                          'MLP Model — Test Set')

    # ── Step 9: Save results ───────────────────────────
    results = {
        'model'        : 'MLP',
        'test_accuracy': round(test_acc * 100, 2),
        'test_loss'    : round(test_loss, 4),
        'epochs_trained': len(history['train_loss']),
        'parameters'   : total_params
    }

    with open('models/mlp_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print(f"\n{'='*55}")
    print(f"   MLP Training Complete! 🎉")
    print(f"{'='*55}")
    print(f"\n  Test Accuracy : {test_acc*100:.2f}%")
    print(f"  Model saved   : models/mlp_best.pth")
    print(f"  Results saved : models/mlp_results.json")
    print(f"\n  Baseline established!")
    print(f"  CNN and LSTM models should beat this.")


if __name__ == '__main__':
    main()