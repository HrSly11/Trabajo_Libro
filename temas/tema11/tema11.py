# ============================================================
# 🤖 PROGRAMA: Clasificador de Imágenes con ANN (Capítulo 11)
# ------------------------------------------------------------
# Permite subir una imagen y clasificarla en categorías
# entrenadas (mochila, vestido, calzado) usando una red neuronal.
# ============================================================

import streamlit as st
import cv2
import numpy as np
import pickle
from PIL import Image
import tempfile
import os
import sys
import matplotlib.pyplot as plt

# --- Estilos visuales personalizados ---
def run():
    st.markdown("""
        <style>
        .main {
            background-color: #0E1117;
            color: white;
        }

        </style>
    """, unsafe_allow_html=True)

    # ✅ Para importar create_features correctamente
    sys.path.append(os.path.join(os.path.dirname(__file__)))
    import create_features as cf

    # ---- Rutas absolutas de los modelos ----
    BASE_DIR = os.path.dirname(__file__)       # tema11/
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    CODEBOOK_FILE = os.path.join(MODEL_DIR, "codebook.pkl")
    ANN_FILE = os.path.join(MODEL_DIR, "ann.yaml")
    LE_FILE = os.path.join(MODEL_DIR, "le.pkl")

    # ---- Verificar que existan los archivos ----
    if not os.path.exists(ANN_FILE):
        st.error(f"❌ No se encontró el modelo ANN: {ANN_FILE}")
        return
    if not os.path.exists(LE_FILE):
        st.error(f"❌ No se encontró el archivo de etiquetas: {LE_FILE}")
        return
    if not os.path.exists(CODEBOOK_FILE):
        st.error(f"❌ No se encontró el archivo codebook: {CODEBOOK_FILE}")
        return

    # ---- Inicializar el clasificador ----
    class ClasificadorImagen:
        def __init__(self, ann_file, le_file, codebook_file):
            self.ann = cv2.ml.ANN_MLP_load(ann_file)
            with open(le_file, "rb") as f:
                self.le = pickle.load(f)
            with open(codebook_file, "rb") as f:
                self.kmeans, self.centroids = pickle.load(f)

        def clasificar(self, img):
            img = cf.resize_to_size(img, 150)
            feature_vector = cf.FeatureExtractor().get_feature_vector(img, self.kmeans, self.centroids)
            _, prediction = self.ann.predict(feature_vector)
            label = self.le.inverse_transform(np.asarray(prediction))
            return label[0]

    # ---- Interfaz de Streamlit ----
    st.title("🤖 Clasificador Visual - Capítulo 11")
    st.markdown("""
    Este modelo utiliza una **Red Neuronal Artificial (ANN)** para **reconocer objetos** en imágenes.
    Sube una imagen y el sistema predecirá a qué clase pertenece según su entrenamiento previo.
    """)
    
    # Mostrar las categorías disponibles
    st.info("📋 **Categorías disponibles:** Dress (Vestido 👗) • Backpack (Mochila 🎒) • Footwear (Calzado 👞)")

    uploaded_file = st.file_uploader("📸 Sube una imagen para clasificar", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Abrir imagen con PIL
        image = Image.open(uploaded_file)
        
        # Mostrar imagen usando matplotlib en lugar de st.image()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(image)
        ax.axis('off')
        ax.set_title("🖼️ Imagen seleccionada", fontsize=14)
        st.pyplot(fig)
        plt.close()

        # --- Guardar la imagen temporalmente con extensión ---
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image.save(temp_file.name, format="PNG")
            temp_path = temp_file.name

        # Leer la imagen con OpenCV
        img_cv = cv2.imread(temp_path)

        with st.spinner("🧠 Analizando con la red neuronal..."):
            clasificador = ClasificadorImagen(ANN_FILE, LE_FILE, CODEBOOK_FILE)
            resultado = clasificador.clasificar(img_cv)

        # ---- Traducir el resultado al español ----
        traducciones = {
            "backpack": "Mochila 🎒",
            "dress": "Vestido 👗",
            "footwear": "Calzado 👞"
        }
        resultado_es = traducciones.get(resultado.lower(), resultado)
        resultado_en = resultado.capitalize()

        # Mostrar resultado con ambos idiomas
        st.success(f"🎯 **Resultado:** {resultado_en} ({resultado_es})")
        st.info("✅ Clasificación completada mediante un modelo ANN entrenado.")
