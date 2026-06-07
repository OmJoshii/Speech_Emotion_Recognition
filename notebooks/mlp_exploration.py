# Exploring MLP concepts before building the real model

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score,
                              classification_report,
                              confusion_matrix)
import seaborn as sns

print("=" * 50)
print("      MLP Exploration")
print("=" * 50)

# ── Load preprocessed data ─────────────────────────────
print("\nLoading preprocessed data...")

X_train = np.load('data/processed/X_fixed_train.npy')
X_val   = np.load('data/processed/X_fixed_val.npy')
X_test  = np.load('data/processed/X_fixed_test.npy')
y_train = np.load('data/processed/y_train.npy')
y_val   = np.load('data/processed/y_val.npy')
y_test  = np.load('data/processed/y_test.npy')

print(f"Training   : {X_train.shape}")
print(f"Validation : {X_val.shape}")
print(f"Test       : {X_test.shape}")

# ── Emotion names for display ──────────────────────────
emotion_names = ['neutral', 'calm', 'happy', 'sad',
                 'angry', 'fearful', 'disgust', 'surprised']

# ── Quick Sklearn MLP ──────────────────────────────────
# sklearn has a simple MLP we can use for quick testing
# before building our PyTorch version

print("\nTraining quick sklearn MLP...")
print("(This is just for exploration — real model uses PyTorch)")

clf = MLPClassifier(
    hidden_layer_sizes=(256, 128),  # 2 hidden layers
    activation='relu',               # ReLU activation
    max_iter=50,                     # 50 epochs
    random_state=42,
    verbose=False
)

clf.fit(X_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, clf.predict(X_train))
val_acc   = accuracy_score(y_val,   clf.predict(X_val))

print(f"\nTraining accuracy   : {train_acc*100:.2f}%")
print(f"Validation accuracy : {val_acc*100:.2f}%")

# ── Confusion Matrix ───────────────────────────────────
y_pred = clf.predict(X_val)

cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm,
            annot=True,        # show numbers in cells
            fmt='d',           # integer format
            cmap='Blues',      # color scheme
            xticklabels=emotion_names,
            yticklabels=emotion_names)
plt.title('Confusion Matrix — sklearn MLP (Exploration)\n'
          'Diagonal = correct predictions')
plt.ylabel('True Emotion')
plt.xlabel('Predicted Emotion')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('notebooks/confusion_matrix.png',
            dpi=150, bbox_inches='tight')
plt.show()

# ── Classification Report ──────────────────────────────
print("\n── Classification Report ──")
print(classification_report(y_val, y_pred,
                             target_names=emotion_names))

print("Exploration complete!")
print("Now building the real PyTorch MLP...")