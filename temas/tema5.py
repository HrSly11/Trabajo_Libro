import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def run():
    st.title("⚡ Detección de Características FAST")
    
    uploaded = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])
    
    if uploaded:
        # Cargar imagen
        img_pil = Image.open(uploaded).convert("RGB")
        img_color = np.array(img_pil)
        img_gray = cv2.cvtColor(img_color, cv2.COLOR_RGB2GRAY)
        
        st.subheader("Configuración del detector")
        
        col1, col2 = st.columns(2)
        with col1:
            umbral = st.slider("Umbral de detección", 5, 100, 20)
        with col2:
            suprimir = st.checkbox("Supresión no-máxima", value=True)
        
        # Crear detector FAST
        detector_fast = cv2.FastFeatureDetector_create(
            threshold=umbral,
            nonmaxSuppression=suprimir
        )
        
        # Detectar características
        puntos_clave = detector_fast.detect(img_gray, None)
        
        # Dibujar puntos en la imagen
        img_resultado = img_color.copy()
        for punto in puntos_clave:
            coord_x, coord_y = int(punto.pt[0]), int(punto.pt[1])
            cv2.circle(img_resultado, (coord_x, coord_y), 4, (255, 0, 0), -1)
        
        # Visualizar
        st.subheader("Resultado")
        figura, eje = plt.subplots(figsize=(12, 8))
        eje.imshow(img_resultado)
        eje.axis('off')
        eje.set_title(f"Características detectadas: {len(puntos_clave)}", fontsize=16)
        st.pyplot(figura)
        plt.close()
        
        # Estadísticas
        st.divider()
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            st.metric("Total de puntos", len(puntos_clave))
        
        with col_b:
            if len(puntos_clave) > 0:
                tam_promedio = np.mean([p.size for p in puntos_clave])
                st.metric("Tamaño promedio", f"{tam_promedio:.2f}")
        
        with col_c:
            if len(puntos_clave) > 0:
                resp_promedio = np.mean([p.response for p in puntos_clave])
                st.metric("Respuesta promedio", f"{resp_promedio:.2f}")


if __name__ == "__main__":
    run()