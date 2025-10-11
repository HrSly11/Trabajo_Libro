import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


def run():
    # Funciones auxiliares
    def calcular_energia(imagen):
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        grad_x = cv2.Sobel(gris, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gris, cv2.CV_64F, 0, 1, ksize=3)
        abs_x = cv2.convertScaleAbs(grad_x)
        abs_y = cv2.convertScaleAbs(grad_y)
        return cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)

    def energia_modificada(imagen, area):
        matriz_energia = calcular_energia(imagen)
        x, y, w, h = area
        matriz_energia[y:y+h, x:x+w] = 0
        return matriz_energia

    def buscar_seam_vertical(imagen, energia):
        filas, columnas = imagen.shape[:2]
        seam = np.zeros(filas)
        distancia = np.zeros((filas, columnas)) + float("inf")
        distancia[0, :] = np.zeros(columnas)
        camino = np.zeros((filas, columnas))

        for f in range(filas - 1):
            for c in range(columnas):
                if c != 0 and distancia[f+1, c-1] > distancia[f, c] + energia[f+1, c-1]:
                    distancia[f+1, c-1] = distancia[f, c] + energia[f+1, c-1]
                    camino[f+1, c-1] = 1
                if distancia[f+1, c] > distancia[f, c] + energia[f+1, c]:
                    distancia[f+1, c] = distancia[f, c] + energia[f+1, c]
                    camino[f+1, c] = 0
                if c != columnas-1 and distancia[f+1, c+1] > distancia[f, c] + energia[f+1, c+1]:
                    distancia[f+1, c+1] = distancia[f, c] + energia[f+1, c+1]
                    camino[f+1, c+1] = -1

        seam[filas-1] = np.argmin(distancia[filas-1, :])
        for i in range(filas-1, 0, -1):
            seam[i-1] = seam[i] + camino[i, int(seam[i])]
        return seam

    def eliminar_seam_vertical(imagen, seam):
        filas, columnas = imagen.shape[:2]
        for f in range(filas):
            for c in range(int(seam[f]), columnas-1):
                imagen[f, c] = imagen[f, c+1]
        return imagen[:, 0:columnas-1]

    def procesar_eliminacion(imagen, area):
        cantidad_seams = area[2] + 10
        energia = energia_modificada(imagen, area)
        barra = st.progress(0)
        
        for idx in range(cantidad_seams):
            seam = buscar_seam_vertical(imagen, energia)
            imagen = eliminar_seam_vertical(imagen, seam)
            x, y, w, h = area
            energia = energia_modificada(imagen, (x, y, max(1, w-idx), h))
            barra.progress((idx + 1) / cantidad_seams)
        
        barra.empty()
        return imagen

    # Interfaz
    st.title("✂️ Eliminación de Objetos con Seam Carving")
    
    archivo = st.file_uploader("Sube una imagen", type=["jpg", "png", "jpeg"])

    if archivo:
        imagen = Image.open(archivo)
        img_array = np.array(imagen)
        
        # Convertir a BGR
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
        else:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        alto, ancho = img_array.shape[:2]
        
        st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), caption="Imagen original")
        
        st.divider()
        st.subheader("Define el área a eliminar")
        
        c1, c2 = st.columns(2)
        
        with c1:
            pos_x = st.slider("Posición X", 0, max(0, ancho-20), ancho//4)
            pos_y = st.slider("Posición Y", 0, max(0, alto-20), alto//4)
        
        with c2:
            tam_w = st.slider("Ancho", 10, ancho - pos_x, min(100, ancho - pos_x))
            tam_h = st.slider("Alto", 10, alto - pos_y, min(100, alto - pos_y))

        # Vista previa
        area_seleccionada = (int(pos_x), int(pos_y), int(tam_w), int(tam_h))
        img_preview = img_array.copy()
        cv2.rectangle(img_preview, (pos_x, pos_y), (pos_x+tam_w, pos_y+tam_h), (0, 0, 255), 2)
        
        st.image(cv2.cvtColor(img_preview, cv2.COLOR_BGR2RGB), caption="Vista previa del área")
        
        if st.button("Eliminar objeto", type="primary"):
            with st.spinner("Procesando..."):
                img_copia = img_array.copy()
                resultado = procesar_eliminacion(img_copia, area_seleccionada)
                resultado_rgb = cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB)
                
                st.divider()
                st.subheader("Resultado")
                
                col_antes, col_despues = st.columns(2)
                with col_antes:
                    st.image(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB), caption="Antes")
                with col_despues:
                    st.image(resultado_rgb, caption="Después")
                
                # Descarga
                img_pil = Image.fromarray(resultado_rgb)
                buffer = BytesIO()
                img_pil.save(buffer, format="PNG")
                
                st.download_button(
                    "Descargar resultado",
                    data=buffer.getvalue(),
                    file_name="objeto_eliminado.png",
                    mime="image/png"
                )


if __name__ == "__main__":
    run()