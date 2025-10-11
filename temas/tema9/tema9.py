import os
import cv2
import pickle
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

def run():
    st.title("👗👟 Reconocimiento de prendas (Dress vs Footwear)")
    st.write("""
    Este programa usa **SIFT**, un **diccionario visual (KMeans)** y un **SVM** para reconocer si una imagen
    pertenece a la clase *dress* o *footwear*.
    """)
    
    # --- Ruta base y modelos ---
    BASE_DIR = os.path.dirname(__file__)
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    svm_path = os.path.join(MODELS_DIR, "svm.pkl")
    codebook_path = os.path.join(MODELS_DIR, "codebook.pkl")
    
    # Mostrar ruta buscada
    st.write("📁 Ruta de modelos:")
    st.code(MODELS_DIR)
    
    # Verificar existencia
    if not all(map(os.path.exists, [svm_path, codebook_path])):
        st.warning("⚠️ No se encontraron los archivos del modelo entrenado. Ejecuta `train.py` primero.")
        st.stop()
    
    # --- Cargar modelos ---
    with open(codebook_path, "rb") as f:
        kmeans, scaler = pickle.load(f)
    with open(svm_path, "rb") as f:
        svm, label_names = pickle.load(f)
    
    st.success("✅ Modelos cargados correctamente.")
    
    # --- Subir imagen ---
    uploaded_file = st.file_uploader("📤 Sube una imagen (JPG o PNG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Leer imagen
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Mostrar imagen usando matplotlib
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(image_rgb)
        ax.axis('off')
        ax.set_title("Imagen cargada", fontsize=14)
        st.pyplot(fig)
        plt.close()
        
        # --- Extracción de características ---
        sift = cv2.SIFT_create()
        _, desc = sift.detectAndCompute(gray, None)
        
        if desc is None:
            st.error("⚠️ No se detectaron descriptores en la imagen. Intenta con otra.")
            return
        
        # --- Histograma de palabras visuales ---
        words = kmeans.predict(desc)
        hist, _ = np.histogram(words, bins=np.arange(kmeans.n_clusters + 1))
        hist_scaled = scaler.transform(hist.reshape(1, -1))
        
        # --- Clasificación ---
        prediction = svm.predict(hist_scaled)[0]
        prob = svm.predict_proba(hist_scaled)[0][prediction]
        
        # --- Resultado ---
        class_name = label_names[prediction]
        st.markdown(f"### 🏷️ Predicción: **{class_name.upper()}**")
        st.progress(float(prob))
        st.write(f"Confianza: **{prob*100:.2f}%**")
