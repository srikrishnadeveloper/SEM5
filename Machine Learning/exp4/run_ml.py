import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer

# ------------------------------------------------------------------
# 1. Load the handwritten character recognition dataset
# ------------------------------------------------------------------
# The CSV has image filenames in first column, labels in second
# Each image is 16x16 grayscale -> 256 flat pixel features
csv_path = r'C:/Users/srik2/Desktop/College/Machine Learning/exp4/english_dataset/english.csv'
df = pd.read_csv(csv_path, header=None)
data_rows = df.iloc[1:]  # skip the header row

n_samples = len(data_rows)
X = np.zeros((n_samples, 256))  # 256 features = 16x16 pixels
y = []

for i, (_, row) in enumerate(data_rows.iterrows()):
    img_name = row[0]
    img_path = os.path.join(
        r'C:/Users/srik2/Desktop/College/Machine Learning/exp4/english_dataset/Img',
        img_name
    )
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        # Convert to grayscale if needed
        arr = img[:, :, 0] if img.ndim == 3 else img
        arr = arr / 255.0  # normalize to 0-1 range

        # Resize to exactly 16x16 if different
        if arr.shape != (16, 16):
            h, w = arr.shape
            start_h, start_w = max(0, (h-16)//2), max(0, (w-16)//2)
            arr = arr[start_h:start_h+16, start_w:start_w+16]

        X[i] = arr.flatten()
    else:
        X[i] = 0  # zero vector for missing images

    # Parse label from the filename string (e.g. 'img001-026.png' -> '026' -> 26)
    label_str = row[0].split('-')[1] if '-' in row[0] else row[0]
    if label_str.isdigit():
        y.append(int(label_str))
    else:
        c = label_str.upper()
        if 'A' <= c <= 'Z':
            y.append(ord(c) - ord('A') + 10)
        elif 'a' <= c <= 'z':
            y.append(ord(c) - ord('a') + 36)
        else:
            y.append(0)

# Binarize labels for OvR classification
lb = LabelBinarizer()
lb.fit(np.unique(y))
Y = lb.transform(y)

print(f"Dataset: {len(X)} samples, {len(np.unique(y))} classes")


# ------------------------------------------------------------------
# 2. Perceptron Learning Algorithm (implemented from scratch)
# ------------------------------------------------------------------
class PerceptronFromScratch:
    """A simple single-layer perceptron classifier."""

    def __init__(self, learning_rate=0.01, max_iter=500):
        self.lr = learning_rate
        self.max_iter = max_iter
        self.weights = None
        self.bias = None
        self.losses = []

    def _step(self, x):
        """Step activation: outputs 1 if dot product >= 0, else 0."""
        return 1 if np.dot(x, self.weights) + self.bias >= 0 else 0

    def fit(self, X, y_bin):
        """
        Train the perceptron on binary labels (0 or 1).
        
        For each iteration, we loop through every training sample.
        If the prediction is wrong, we update the weights using the
        classic perceptron rule: w += lr * (y_true - y_pred) * x
        
        Training stops early if no errors in an iteration.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for it in range(self.max_iter):
            errors = 0
            for idx in range(n_samples):
                y_pred = self._step(X[idx])
                if y_pred != y_bin[idx]:
                    error = y_bin[idx] - y_pred
                    self.weights += self.lr * error * X[idx]
                    self.bias += self.lr * error
                    errors += 1
            self.losses.append(errors)
            if errors == 0:
                break  # converged
        return self

    def predict(self, X):
        """Classify all samples in X."""
        return np.array([self._step(x) for x in X])


# ------------------------------------------------------------------
# 3. Train PLA with One-vs-Rest for all 62 classes
# ------------------------------------------------------------------
print("Training PLA (One-vs-Rest)...")
pca_models = {}
pLA_scores = {}

for c in range(62):
    # Binary label: 1 for samples of class c, 0 for all others
    y_binary = (np.array(y) == c).astype(int)
    perceptron = PerceptronFromScratch(learning_rate=0.1, max_iter=500)
    perceptron.fit(X, y_binary)
    pca_models[c] = perceptron

    y_pred = perceptron.predict(X)
    pLA_scores[c] = (y_pred == y_binary).sum() / len(y_binary)

print(f"PLA per-class accuracies (first 10): {sorted(pLA_scores.items())[:10]}")
print(f"PLA avg accuracy: {np.mean(list(pLA_scores.values())[:10]):.4f}")


# ------------------------------------------------------------------
# 4. Train MLP with different hyperparameter configs
# ------------------------------------------------------------------
print("\nTraining MLP with 3 configs...")
mlp_params = [
    {"hidden_layer_sizes": (128,), "activation": "relu", "solver": "sgd",
     "learning_rate_init": 0.01, "max_iter": 100},
    {"hidden_layer_sizes": (256, 128), "activation": "relu", "solver": "adam",
     "learning_rate_init": 0.001, "max_iter": 100},
    {"hidden_layer_sizes": (512, 256, 128), "activation": "tanh", "solver": "adam",
     "learning_rate_init": 0.0005, "max_iter": 100},
]

mlp_results = []
for i, params in enumerate(mlp_params, 1):
    mlp = MLPClassifier(**params, random_state=42)
    mlp.fit(X, y)
    acc = accuracy_score(y, mlp.predict(X))
    try:
        auc = roc_auc_score(Y, mlp.predict_proba(X), multi_class='ovr', average='macro')
    except Exception:
        auc = 0.0
    mlp_results.append({"params": params, "accuracy": acc, "auc": auc,
                         "loss_curve": mlp.loss_curve_ if hasattr(mlp, "loss_curve_") else []})
    print(f"MLP {i}: acc={acc:.4f}, auc={auc:.4f}")


# ------------------------------------------------------------------
# 5. Generate plots and save results
# ------------------------------------------------------------------
os.makedirs("results", exist_ok=True)

plt.figure(figsize=(10, 6))
for i, res in enumerate(mlp_results):
    plt.plot(res["loss_curve"], label=f"MLP {i}: {res['params']}")
plt.title("MLP Training Loss Curves")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("results/mlp_loss_curves.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 4))
plt.plot(list(pca_models[0].losses))
plt.title("PLA Training Loss (Class 0)")
plt.xlabel("Iteration")
plt.ylabel("Misclassifications")
plt.savefig("results/pla_loss.png", dpi=150)
plt.close()

print("\nDone. Results saved in results/")