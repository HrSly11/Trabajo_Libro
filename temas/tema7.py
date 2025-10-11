import streamlit as st
import numpy as np
import cv2
from PIL import Image


def run():
    st.title("💧 Segmentación con Algoritmo Watershed")
    
    archivo = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])
    
    if archivo:
        # Cargar imagen
        pil_img = Image.open(archivo)
        imagen = np.array(pil_img)
        
        st.subheader("Original")
        st.image(imagen)
        
        # Procesar
        gris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
        
        # Binarización
        _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Limpieza morfológica
        elemento = np.ones((3, 3), np.uint8)
        apertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, elemento, iterations=3)
        
        # Identificar fondo
        fondo_seguro = cv2.dilate(apertura, elemento, iterations=2)
        
        # Identificar objetos
        transform_dist = cv2.distanceTransform(apertura, cv2.DIST_L2, 5)
        _, objetos_seguros = cv2.threshold(transform_dist, 0.6 * transform_dist.max(), 255, 0)
        
        # Zona desconocida
        objetos_seguros = np.uint8(objetos_seguros)
        zona_desconocida = cv2.subtract(fondo_seguro, objetos_seguros)
        
        # Marcadores
        _, etiquetas = cv2.connectedComponents(objetos_seguros)
        etiquetas = etiquetas + 1
        etiquetas[zona_desconocida == 255] = 0
        
        # Watershed
        etiquetas = cv2.watershed(imagen, etiquetas.copy())
        imagen_resultado = imagen.copy()
        imagen_resultado[etiquetas == -1] = [0, 255, 0]  # Bordes en verde
        
        # Mostrar
        st.subheader("Proceso de segmentación")
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(binaria, caption="Binarización")
            st.image(objetos_seguros, caption="Objetos detectados")
        
        with col_b:
            st.image(fondo_seguro, caption="Fondo identificado")
            st.image(imagen_resultado, caption="Resultado Watershed")
        
        # Estadísticas
        objetos_totales = len(np.unique(etiquetas)) - 2
        st.success(f"Objetos segmentados: {objetos_totales}")


if __name__ == "__main__":
    run()