import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings

warnings.filterwarnings('ignore')

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment
from confusion_matrix import confusion_matrix

RANDOM_STATE = 42

# Load dataset
DATA_DIR = r'C:\Users\srik2\Desktop\College\Machine Learning\exp6\UCI HAR Dataset'

X_train = pd.read_csv(f'{DATA_DIR}/train/X_train.txt', header=None, sep='\\s+')
y_train = pd.read_csv(f'{DATA_DIR}/train/y_train.txt', header=None)[0]
subject_train = pd.read_csv(f'{DATA_DIR}/train/subject_train.txt', header=None)[0]

X_test = pd.read_csv(f'{DATA_DIR}/test/X_test.txt', header=None, sep='\\s+')
y_test = pd.read_csv(f'{DATA_DIR}/test/y_test.txt', header=None)[0]

features = pd.read_csv(f'{DATA_DIR}/features.txt', header=None, sep='\\s+')[1]
X_train.columns = features
X_test.columns = features

activity_labels = pd.read_csv(f'{DATA_DIR}/activity_labels.txt', header=None, sep='\\s+')
activity_map = dict(zip(activity_labels[0], activity_labels[1]))

# Preprocessing
X_train = X_train.loc[:, ~X_train.columns.duplicated()]
X_test = X_test.loc[:, ~X_test.columns.duplicated()]
X_train = X_train.dropna(axis=1)
X_test = X_test.dropna(axis=1)
common = list(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common]
X_test = X_test[common]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA for visualization
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# K-Means clustering
kmeans = KMeans(n_clusters=6, random_state=RANDOM_STATE, n_init=10)
start = time.time()
cluster_labels = kmeans.fit_predict(X_train_scaled)
train_time = time.time() - start

test_clusters = kmeans.predict(X_test_scaled)
inertia = round(kmeans.inertia_, 2)
sil_train = round(silhouette_score(X_train_scaled, cluster_labels), 4)
sil_test = round(silhouette_score(X_test_scaled, test_clusters), 4)

# Elbow method
inertias = []
silhouettes = []
K_range = range(2, 11)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_train_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_train_scaled, km.labels_))

# Plot elbow and silhouette (just save figures; data used for PDF table later)
fig, ax1 = plt.subplots(figsize=(8, 4))
ax1.plot(K_range, inertias, 'b-o', label='Inertia')
ax1.set_xlabel('Number of clusters (k)')
ax1.set_ylabel('Inertia', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax2 = ax1.twinx()
ax2.plot(K_range, silhouettes, 'r-s', label='Silhouette')
ax2.set_ylabel('Silhouette score', color='r')
ax2.tick_params(axis='y', labelcolor='r')
plt.title('Elbow method and Silhouette scores')
plt.tight_layout()
plt.savefig('elbow_silhouette.png')
plt.close()

# PCA plot: cluster vs actual
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# ground truth
for i, ax in enumerate(axes):
    if i == 0:
        labels = y_train.values
        title = 'Ground truth activities'
    else:
        labels = cluster_labels
        title = 'K-Means clusters'
    for cls in range(6):
        idx = labels == cls
        ax.scatter(X_train_pca[idx, 0], X_train_pca[idx, 1], s=8, alpha=0.7, label=f'Activity {cls+1}' if i == 0 else "")
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
plt.tight_layout()
plt.savefig('pca_clusters.png')
plt.close()

# Cluster vs activity alignment
train_activities = y_train.values
cm = confusion_matrix(train_activities, cluster_labels)
row_ind, col_ind = linear_sum_assignment(-cm)
label_map = {col: row+1 for col, row in zip(col_ind, row_ind)}
mapped_clusters = np.array([label_map.get(c, 0) for c in cluster_labels])

ct = pd.crosstab(mapped_clusters, train_activities, rownames=['Mapped cluster'], colnames=['True activity'])
ct.columns = [activity_map[c] for c in ct.columns]

# Accuracy
from sklearn.metrics import accuracy_score
acc = round(accuracy_score(train_activities, mapped_clusters), 4)

# Final metrics print
print(f'Clusters used: 6')
print(f'Training time: {train_time:.4f}s')
print(f'Inertia: {inertia}')
print(f'Silhouette (train): {sil_train}')
print(f'Silhouette (test): {sil_test}')
print(f'Alignment accuracy: {acc}')