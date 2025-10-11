import os
import cv2
import pickle
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# --- CONFIGURACIÓN ---
DATASET_DIR = os.path.join(os.path.dirname(__file__), "images")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# --- EXTRACCIÓN DE DESCRIPTORES ---
sift = cv2.SIFT_create()
descriptors_list = []
labels = []
label_names = []

print("🔍 Extrayendo características...")

for label, folder in enumerate(os.listdir(DATASET_DIR)):
    path = os.path.join(DATASET_DIR, folder)
    if not os.path.isdir(path):
        continue
    label_names.append(folder)

    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, desc = sift.detectAndCompute(gray, None)
        if desc is not None:
            descriptors_list.append(desc)
            labels.append(label)

# --- UNIR DESCRIPTORES ---
all_descriptors = np.vstack(descriptors_list)
print(f"Total de descriptores: {all_descriptors.shape[0]}")

# --- CREAR DICCIONARIO VISUAL ---
k = 60
print("🧱 Creando diccionario visual (KMeans)...")
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(all_descriptors)

# --- HISTOGRAMAS ---
features = []
for desc in descriptors_list:
    words = kmeans.predict(desc)
    hist, _ = np.histogram(words, bins=np.arange(k + 1))
    features.append(hist)

features = np.array(features)
scaler = StandardScaler().fit(features)
features_scaled = scaler.transform(features)

# --- ENTRENAR SVM ---
print("🤖 Entrenando SVM...")
svm = SVC(kernel='linear', probability=True)
svm.fit(features_scaled, labels)

# --- GUARDAR MODELOS ---
with open(os.path.join(MODEL_DIR, "codebook.pkl"), "wb") as f:
    pickle.dump((kmeans, scaler), f)

with open(os.path.join(MODEL_DIR, "svm.pkl"), "wb") as f:
    pickle.dump((svm, label_names), f)

print("✅ Modelos guardados correctamente en /models/")
