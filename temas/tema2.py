import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


def run():
    st.title("✨ Enhancing the Contrast in an Image")
    st.write("Aplicamos un filtro de **Motion Blur** para crear un efecto de movimiento.")
    
    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Cargar imagen
        image = Image.open(uploaded_file).convert("RGB")
        img = np.array(image)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original")
            st.image(img)
        
        # Crear kernel de motion blur
        kernel_size = 15
        motion_kernel = np.zeros((kernel_size, kernel_size))
        motion_kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        motion_kernel = motion_kernel / kernel_size
        
        # Aplicar filtro
        blurred = cv2.filter2D(img, -1, motion_kernel)
        
        with col2:
            st.subheader("Con Motion Blur")
            st.image(blurred)
        
        # Descarga
        output_img = Image.fromarray(blurred)
        buffer = BytesIO()
        output_img.save(buffer, format="JPEG")
        
        st.download_button(
            label="Descargar imagen procesada",
            data=buffer.getvalue(),
            file_name="motion_blur.jpg",
            mime="image/jpeg"
        )


if __name__ == "__main__":
    run()