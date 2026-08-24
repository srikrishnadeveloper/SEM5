import os, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer

csv_abs = r'C:\Users\srik2\Desktop\College\Machine Learning\exp4\english_dataset\english.csv'
df = pd.read_csv(csv_abs, header=None)
data_rows = df.iloc[1:]  # skip header row
n_samples = len(data_rows)
print(f"Samples: {n_samples}")

X = np.zeros((n_samples, 256))
y = []
for i, (_, row) in enumerate(data_rows.iterrows()):
    img_name = row[0]
    img_path = os.path.join(r'C:\Users\srik2\Desktop\College\Machine Learning\exp4\english_dataset\Img', img_name)
    if os.path.exists(img_path):
        img = plt.imread(img_path)
        if img.ndim == 3:
            arr = img[:,:,0]
        else:
            arr = img
        arr = arr / 255.0
        if arr.shape == (16,16):
            X[i] = arr.flatten()
        else:
            h, w = arr.shape
            start_h = max(0, (h-16)//2)
            start_w = max(0, (w-16)//2)
            patch = arr[start_h:start_h+16, start_w:start_w+16]
            X[i, :patch.shape[0]*patch.shape[1]] = patch.flatten()
    else:
        X[i] = 0
    # parse label from csv first column (after header)
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

from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
lb.fit(np.unique(y))
Y = lb.transform(y)
print(f"Dataset: {len(X)} samples, {len(np.unique(y))} classes")

# PLA OvR
class PerceptronFromScratch:
    def __init__(self, lr=0.1, max_iter=500):
        self.lr = lr; self.max_iter = max_iter; self.weights = None; self.bias = None; self.losses = []
    def _step(self, x):
        return 1 if np.dot(x, self.weights) + self.bias >= 0 else 0
    def fit(self, X, y_bin):
        n, d = X.shape
        self.weights = np.zeros(d); self.bias = 0
        for it in range(self.max_iter):
            loss = 0
            for idx in range(n):
                x_i = X[idx]; yt = y_bin[idx]
                yp = self._step(x_i)
                if yp != yt:
                    error = yt - yp
                    self.weights += self.lr * error * x_i
                    self.bias += self.lr * error
                    loss += 1
            self.losses.append(loss)
            if loss == 0: break
        return self
    def predict(self, X):
        return np.array([self._step(x) for x in X])

print("Training PLA OvR...")
pca = {}
sc = {}
for c in range(62):
    yb = (np.arange(62) == c).astype(int)
    p = PerceptronFromScratch(lr=0.1, max_iter=500)
    p.fit(X, yb)
    pca[c] = p
    yp = p.predict(X)
    sc[c] = (yp == yb).sum() / len(yb)

print(f"PLA first 10 accs: {sorted(sc.items())[:10]}")
print(f"PLA avg acc (10 classes): {np.mean(list(sc.values())[:10]):.4f}")

# MLP
print("Training MLP...")
mlp_params = [
    {"hidden_layer_sizes": (128,), "activation": "relu", "solver": "sgd", "learning_rate": "0.01", "max_iter": 100},
    {"hidden_layer_sizes": (256, 128), "activation": "relu", "solver": "adam", "learning_rate": "0.001", "max_iter": 100},
    {"hidden_layer_sizes": (512, 256, 128), "activation": "tanh", "solver": "adam", "learning_rate": "0.0005", "max_iter": 100},
]
mlp_res = []
for i, pr in enumerate(mlp_params, 1):
    mlp = MLPClassifier(**pr, random_state=42)
    mlp.fit(X, y)
    acc = accuracy_score(y, mlp.predict(X))
    try:
        auc = roc_auc_score(Y, mlp.predict_proba(X), multi_class='ovr', average='macro')
    except: auc = 0.0
    mlp_res.append({"params": pr, "accuracy": acc, "auc": auc, "loss": mlp.loss_curve_ if hasattr(mlp, "loss_curve_") else []})
    print(f"MLP {i}: acc={acc:.4f}, auc={auc:.4f}")

os.makedirs("results", exist_ok=True)
plt.figure(); plt.plot([r["loss"] for r in mlp_res]); plt.title("MLP Loss"); plt.savefig("results/mlp_loss.png", dpi=150); plt.close()
plt.figure(); plt.plot(list(pca[0].losses)); plt.title("PLA Loss"); plt.savefig("results/pla_loss.png", dpi=150); plt.close()
print("\nDone. Results in /results/")