import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile


def run():
    st.title("📐 Dibujar Rectángulos en Imágenes y Videos")
    
    # Estado
    if 'current_image' not in st.session_state:
        st.session_state.current_image = None

    # Selector de entrada
    opcion = st.selectbox(
        "Selecciona la fuente",
        ["Capturar foto", "Cargar imagen", "Extraer frame de video"]
    )

    img_source = None

    # === CAPTURA DE CÁMARA ===
    if opcion == "Capturar foto":
        foto = st.camera_input("Usa tu cámara")
        if foto:
            pil_img = Image.open(foto)
            img_source = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            st.session_state.current_image = img_source

    # === SUBIR IMAGEN ===
    elif opcion == "Cargar imagen":
        archivo = st.file_uploader("Selecciona una imagen", type=['jpg', 'png', 'jpeg'])
        if archivo:
            pil_img = Image.open(archivo)
            img_source = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            st.session_state.current_image = img_source

    # === PROCESAR VIDEO ===
    elif opcion == "Extraer frame de video":
        video_file = st.file_uploader("Selecciona un video", type=['mp4', 'avi', 'mov'])
        if video_file:
            # Guardar temporalmente
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            temp_file.write(video_file.read())
            
            # Abrir video
            video = cv2.VideoCapture(temp_file.name)
            frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Seleccionar frame
            frame_idx = st.slider("Frame a extraer", 0, frame_count - 1, 0)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video.read()
            video.release()
            
            if success:
                img_source = frame
                st.session_state.current_image = img_source

    # === DIBUJAR RECTÁNGULO ===
    if st.session_state.current_image is not None:
        work_img = st.session_state.current_image.copy()
        h, w = work_img.shape[:2]
        
        st.subheader("Configuración del rectángulo")
        
        # Coordenadas
        c1, c2 = st.columns(2)
        with c1:
            inicio_x = st.slider("Inicio X", 0, w, w // 4)
            inicio_y = st.slider("Inicio Y", 0, h, h // 4)
        with c2:
            fin_x = st.slider("Fin X", 0, w, w * 3 // 4)
            fin_y = st.slider("Fin Y", 0, h, h * 3 // 4)
        
        # Estilo
        c3, c4 = st.columns(2)
        with c3:
            rect_color = st.color_picker("Color", "#FF0000")
        with c4:
            border_width = st.slider("Grosor", 1, 15, 2)
        
        # Convertir color
        hex_val = rect_color.lstrip('#')
        r, g, b = tuple(int(hex_val[i:i+2], 16) for i in (0, 2, 4))
        bgr_color = (b, g, r)
        
        # Dibujar
        cv2.rectangle(work_img, (inicio_x, inicio_y), (fin_x, fin_y), bgr_color, border_width)
        
        # Mostrar
        st.subheader("Resultado")
        rgb_img = cv2.cvtColor(work_img, cv2.COLOR_BGR2RGB)
        st.image(rgb_img)
        
        # Acciones
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Reiniciar"):
                st.session_state.current_image = None
                st.rerun()
        with col_b:
            _, encoded = cv2.imencode('.png', work_img)
            st.download_button(
                "Descargar",
                data=encoded.tobytes(),
                file_name="rectangulo.png",
                mime="image/png"
            )
    else:
        st.info("Selecciona una fuente para comenzar")


if __name__ == "__main__":
    run()