import cv2
import numpy as np
import math
import streamlit as st


def run():
    st.title("🌊 Efectos de Ondas en Imágenes")

    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Leer imagen
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        rows, cols = img.shape

        # Mostrar original
        st.subheader("Imagen Original")
        st.image(img, channels="GRAY")

        # Onda Vertical
        st.subheader("Efecto: Onda Vertical")
        img_output = np.zeros(img.shape, dtype=img.dtype)
        for i in range(rows):
            for j in range(cols):
                offset_x = int(25.0 * math.sin(2 * 3.14 * i / 180))
                if j + offset_x < cols:
                    img_output[i, j] = img[i, (j + offset_x) % cols]
        st.image(img_output, channels="GRAY")

        # Onda Horizontal
        st.subheader("Efecto: Onda Horizontal")
        img_output = np.zeros(img.shape, dtype=img.dtype)
        for i in range(rows):
            for j in range(cols):
                offset_y = int(16.0 * math.sin(2 * 3.14 * j / 150))
                if i + offset_y < rows:
                    img_output[i, j] = img[(i + offset_y) % rows, j]
        st.image(img_output, channels="GRAY")

        # Onda Multidireccional
        st.subheader("Efecto: Onda Multidireccional")
        img_output = np.zeros(img.shape, dtype=img.dtype)
        for i in range(rows):
            for j in range(cols):
                offset_x = int(20.0 * math.sin(2 * 3.14 * i / 150))
                offset_y = int(20.0 * math.cos(2 * 3.14 * j / 150))
                if i + offset_y < rows and j + offset_x < cols:
                    img_output[i, j] = img[(i + offset_y) % rows, (j + offset_x) % cols]
        st.image(img_output, channels="GRAY")

        # Efecto Cóncavo
        st.subheader("Efecto: Cóncavo")
        img_output = np.zeros(img.shape, dtype=img.dtype)
        for i in range(rows):
            for j in range(cols):
                offset_x = int(128.0 * math.sin(2 * 3.14 * i / (2 * cols)))
                if j + offset_x < cols:
                    img_output[i, j] = img[i, (j + offset_x) % cols]
        st.image(img_output, channels="GRAY")