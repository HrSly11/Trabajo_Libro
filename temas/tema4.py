import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile


def run():
    st.title("👃 Detector de Narices con Haar Cascade")
    
    # Cargar clasificador
    detector = cv2.CascadeClassifier('./cascade_files/haarcascade_mcs_nose.xml')
    if detector.empty():
        st.error("No se encontró el archivo de clasificación")
        st.stop()

    # Estado
    if 'imagen_actual' not in st.session_state:
        st.session_state.imagen_actual = None

    # Selector de entrada
    tipo_entrada = st.tabs(["📷 Cámara", "🖼️ Imagen", "🎬 Video"])
    
    source_img = None

    # === TAB CÁMARA ===
    with tipo_entrada[0]:
        captura = st.camera_input("Captura tu foto")
        if captura:
            img_pil = Image.open(captura)
            source_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            st.session_state.imagen_actual = source_img

    # === TAB IMAGEN ===
    with tipo_entrada[1]:
        archivo_img = st.file_uploader("Sube una imagen", type=['jpg', 'png', 'jpeg', 'bmp'])
        if archivo_img:
            img_pil = Image.open(archivo_img)
            source_img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            st.session_state.imagen_actual = source_img

    # === TAB VIDEO ===
    with tipo_entrada[2]:
        archivo_video = st.file_uploader("Sube un video", type=['mp4', 'avi', 'mov', 'mkv'])
        if archivo_video:
            temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp.write(archivo_video.read())
            
            video = cv2.VideoCapture(temp.name)
            num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_seleccionado = st.slider("Frame", 0, num_frames - 1, 0)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_seleccionado)
            exito, frame = video.read()
            video.release()
            
            if exito:
                source_img = frame
                st.session_state.imagen_actual = source_img

    # === PROCESAR IMAGEN ===
    if st.session_state.imagen_actual is not None:
        st.divider()
        st.subheader("Ajustes de detección")
        
        col_a, col_b = st.columns(2)
        with col_a:
            factor_escala = st.slider("Factor de escala", 1.05, 2.0, 1.25, 0.05)
            vecinos_min = st.slider("Vecinos mínimos", 1, 12, 4)
        with col_b:
            color_rect = st.color_picker("Color", "#FF6B6B")
            grosor = st.slider("Grosor", 1, 8, 2)
        
        # Convertir color
        hex_code = color_rect.lstrip('#')
        r, g, b = tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))
        color_bgr = (b, g, r)
        
        # Detectar
        imagen_copia = st.session_state.imagen_actual.copy()
        gris = cv2.cvtColor(imagen_copia, cv2.COLOR_BGR2GRAY)
        
        detecciones = detector.detectMultiScale(
            gris,
            scaleFactor=factor_escala,
            minNeighbors=vecinos_min
        )
        
        # Dibujar
        cantidad = len(detecciones)
        for (x, y, w, h) in detecciones:
            cv2.rectangle(imagen_copia, (x, y), (x + w, y + h), color_bgr, grosor)
        
        # Mostrar
        st.subheader("Resultado")
        if cantidad > 0:
            st.success(f"Detectadas: {cantidad} nariz/narices")
        else:
            st.warning("No se encontraron narices")
        
        img_rgb = cv2.cvtColor(imagen_copia, cv2.COLOR_BGR2RGB)
        st.image(img_rgb)
        
        # Acciones
        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("Reiniciar"):
                st.session_state.imagen_actual = None
                st.rerun()
        with btn2:
            _, encoded = cv2.imencode('.png', imagen_copia)
            st.download_button(
                "Descargar resultado",
                data=encoded.tobytes(),
                file_name="narices_detectadas.png",
                mime="image/png"
            )
    else:
        st.info("Selecciona una fuente de entrada en las pestañas superiores")


if __name__ == "__main__":
    run()