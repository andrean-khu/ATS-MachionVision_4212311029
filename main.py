# Source code for EMNIST Letters classification using SVM with HOG features
# 4212311029 - Andrean Bagas Khudori

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib

from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, LeaveOneOut, cross_val_predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



# DATASET

print("=== Memuat dataset EMNIST Letters ===")
train_df = pd.read_csv('emnist-letters-train.csv')
print("Shape data:", train_df.shape)

# Pisahkan label dan fitur
y_all = train_df.iloc[:, 0].values
X_all = train_df.iloc[:, 1:].values
X_all = X_all.reshape(-1, 28, 28)


# ORIENTASI GAMBAR

def fix_emnist_orientation(images):
    fixed = []
    for img in images:
        img = np.rot90(img, 3)  
        img = np.flipud(img)     
        fixed.append(img)
    return np.array(fixed)

print("Memperbaiki orientasi gambar...")
X_all = fix_emnist_orientation(X_all)


# SAMPLING 50 PER KELAS

def balanced_sample(X, y, n_per_class=500, random_state=42):
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)
    Xs, ys = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        sel = rng.choice(idx, size=n_per_class, replace=False)
        Xs.append(X[sel])
        ys.append(y[sel])
    return np.vstack(Xs), np.concatenate(ys)

print("Sampling 500 data per kelas...")
X_sample, y_sample = balanced_sample(X_all, y_all)
print("Total data:", X_sample.shape[0])


# FITUR HOG

def extract_hog_features(images, orientations=9, pixels_per_cell=(8,8), cells_per_block=(2,2)):
    feats = []
    for img in images:
        feat = hog(img, orientations=orientations,
                   pixels_per_cell=pixels_per_cell,
                   cells_per_block=cells_per_block,
                   block_norm='L2-Hys', feature_vector=True)
        feats.append(feat)
    return np.array(feats)

print("Mengekstraksi fitur HOG...")
start = time.time()
X_hog = extract_hog_features(X_sample)
print("Selesai! Waktu:", round(time.time() - start, 2), "detik")
print("Dimensi fitur HOG:", X_hog.shape)


# STANDARISASI DATA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_hog)
print("Data berhasil distandarisasi.")


# TUNING PARAMETER SVM (GRID SEARCH)

print("\n=== Proses tuning parameter SVM ===")
param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}

grid = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1, verbose=2)
grid.fit(X_scaled, y_sample)

best_svm = grid.best_estimator_
print("Parameter terbaik:", grid.best_params_)


# LOOCV (SUBSET)

print("\n=== Evaluasi model dengan LOOCV (subset 13000 data) ===")
X_small = X_scaled[:13000]
y_small = y_sample[:13000]

loo = LeaveOneOut()
start = time.time()
y_pred = cross_val_predict(best_svm, X_small, y_small, cv=loo, n_jobs=-1)
end = time.time()

print("Selesai! Waktu total:", round((end - start) / 60, 2), "menit")
print("\n=== Hasil Evaluasi ===")
print(classification_report(y_small, y_pred, digits=4))

acc = accuracy_score(y_small, y_pred)
print("Akurasi total:", round(acc * 100, 2), "%")


# VISUALISASI CONFUSION MATRIX

cm = confusion_matrix(y_small, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, cmap='Blues', cbar=True)
plt.title("Confusion Matrix (LOOCV Subset 13000 Data)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
 

# SIMPAN MODEL DAN HASIL

print("\nMenyimpan model dan hasil...")
joblib.dump(best_svm, 'svm_emnist_model.joblib')
joblib.dump(scaler, 'hog_scaler.joblib')
np.save('confusion_matrix.npy', cm)
print("Semua hasil tersimpan!")

print("\n=== Selesai! ===")
