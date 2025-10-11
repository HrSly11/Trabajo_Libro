import streamlit as st
import cv2
import tempfile
import numpy as np
import matplotlib.pyplot as plt


def run():
    st.title("🎯 Detección Automática de Movimiento")
    
    archivo_video = st.file_uploader("Sube un video", type=["mp4", "avi", "mov"])
    
    if archivo_video:
        # Guardar temporalmente
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_file.write(archivo_video.read())
        temp_file.close()
        
        # Configuración
        col1, col2 = st.columns(2)
        with col1:
            umbral_mov = st.slider("Umbral de detección", 10, 100, 25)
            area_min = st.slider("Área mínima", 100, 2000, 700)
        with col2:
            intervalo = st.slider("Mostrar cada n frames", 1, 10, 3)
            limite = st.slider("Frames máximos", 50, 500, 200)
        
        if st.button("Procesar", type="primary"):
            with st.spinner("Analizando movimiento..."):
                detectar_movimiento(temp_file.name, umbral_mov, area_min, intervalo, limite)


def detectar_movimiento(ruta, umbral, area_minima, mostrar_cada, max_frames):
    video = cv2.VideoCapture(ruta)
    
    # Primer frame
    exito, cuadro_inicial = video.read()
    if not exito:
        st.error("Error al leer video")
        return
    
    cuadro_inicial = cv2.resize(cuadro_inicial, (640, 480))
    gris_previo = cv2.cvtColor(cuadro_inicial, cv2.COLOR_BGR2GRAY)
    
    contenedor = st.empty()
    contador = 0
    
    while contador < max_frames:
        exito, cuadro_actual = video.read()
        if not exito:
            break
        
        contador += 1
        cuadro_actual = cv2.resize(cuadro_actual, (640, 480))
        gris_actual = cv2.cvtColor(cuadro_actual, cv2.COLOR_BGR2GRAY)
        
        # Detectar diferencias
        diferencia = cv2.absdiff(gris_previo, gris_actual)
        _, binario = cv2.threshold(diferencia, umbral, 255, cv2.THRESH_BINARY)
        
        # Morfología
        elemento = np.ones((7, 7), np.uint8)
        binario = cv2.dilate(binario, elemento, iterations=1)
        
        # Contornos
        contornos, _ = cv2.findContours(binario, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Marcar objetos
        hay_movimiento = False
        for cnt in contornos:
            if cv2.contourArea(cnt) > area_minima:
                hay_movimiento = True
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(cuadro_actual, (x, y), (x + w, y + h), (255, 0, 0), 3)
        
        # Texto de estado
        texto = "MOVIMIENTO" if hay_movimiento else "QUIETO"
        color = (0, 255, 0) if hay_movimiento else (200, 200, 0)
        cv2.putText(cuadro_actual, texto, (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
        
        gris_previo = gris_actual.copy()
        
        # Visualizar
        if contador % mostrar_cada == 0:
            rgb = cv2.cvtColor(cuadro_actual, cv2.COLOR_BGR2RGB)
            
            figura, eje = plt.subplots(figsize=(10, 7))
            eje.imshow(rgb)
            eje.axis('off')
            eje.set_title(f"Frame {contador}", fontsize=14)
            
            contenedor.pyplot(figura)
            plt.close(figura)
    
    video.release()
    st.success(f"Análisis completado: {contador} frames")


if __name__ == "__main__":
    run()