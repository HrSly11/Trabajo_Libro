import streamlit as st
import cv2
import numpy as np
import tempfile
from PIL import Image
import io


def run():
    st.title("🎯 Seguimiento por Flujo Óptico")
    
    video_file = st.file_uploader("Sube un video", type=['mp4', 'avi', 'mov', 'mkv'])

    if video_file:
        # Guardar temporalmente
        temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp.write(video_file.read())
        ruta_video = temp.name
        
        # Info del video
        video = cv2.VideoCapture(ruta_video)
        total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_video = int(video.get(cv2.CAP_PROP_FPS))
        video.release()
        
        st.info(f"Video cargado: {total} frames, {fps_video} FPS")
        
        # Configuración
        st.subheader("Configuración")
        
        c1, c2 = st.columns(2)
        with c1:
            escala = st.slider("Escala", 0.2, 1.0, 0.4, 0.1)
            seguimiento = st.slider("Frames por trayectoria", 3, 15, 7)
        with c2:
            salto = st.slider("Intervalo de detección", 1, 8, 3)
            limite_frames = st.slider("Frames a procesar", 20, min(total, 200), min(80, total))
        
        # Parámetros LK
        params_lk = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.02)
        )
        
        if st.button("Procesar video", type="primary"):
            with st.spinner("Procesando..."):
                video = cv2.VideoCapture(ruta_video)
                
                trayectorias = []
                idx_frame = 0
                gris_anterior = None
                resultado_frames = []
                
                barra = st.progress(0)
                
                while idx_frame < limite_frames:
                    ok, cuadro = video.read()
                    if not ok:
                        break
                    
                    # Escalar
                    cuadro = cv2.resize(cuadro, None, fx=escala, fy=escala, 
                                       interpolation=cv2.INTER_AREA)
                    cuadro_gris = cv2.cvtColor(cuadro, cv2.COLOR_BGR2GRAY)
                    salida = cuadro.copy()
                    
                    # Seguimiento
                    if len(trayectorias) > 0 and gris_anterior is not None:
                        pts_actuales = [t[-1] for t in trayectorias]
                        pts0 = np.float32(pts_actuales).reshape(-1, 1, 2)
                        
                        pts1, estado, _ = cv2.calcOpticalFlowPyrLK(
                            gris_anterior, cuadro_gris, pts0, None, **params_lk
                        )
                        
                        pts0_rev, _, _ = cv2.calcOpticalFlowPyrLK(
                            cuadro_gris, gris_anterior, pts1, None, **params_lk
                        )
                        
                        diferencia = abs(pts0 - pts0_rev).reshape(-1, 2).max(-1)
                        puntos_buenos = diferencia < 1.5
                        
                        nuevas_trayectorias = []
                        for traj, (x, y), bueno in zip(trayectorias, 
                                                       pts1.reshape(-1, 2), 
                                                       puntos_buenos):
                            if not bueno:
                                continue
                            traj.append((x, y))
                            if len(traj) > seguimiento:
                                del traj[0]
                            nuevas_trayectorias.append(traj)
                            cv2.circle(salida, (int(x), int(y)), 4, (255, 0, 0), -1)
                        
                        trayectorias = nuevas_trayectorias
                        rutas = [np.int32(t) for t in trayectorias]
                        cv2.polylines(salida, rutas, False, (0, 255, 255), 2)
                    
                    # Detectar nuevos puntos
                    if idx_frame % salto == 0:
                        mascara = np.zeros_like(cuadro_gris)
                        mascara[:] = 255
                        for x, y in [np.int32(t[-1]) for t in trayectorias]:
                            cv2.circle(mascara, (x, y), 8, 0, -1)
                        
                        caracteristicas = cv2.goodFeaturesToTrack(
                            cuadro_gris, mask=mascara, maxCorners=400, 
                            qualityLevel=0.25, minDistance=10, blockSize=5
                        )
                        
                        if caracteristicas is not None:
                            for x, y in np.float32(caracteristicas).reshape(-1, 2):
                                trayectorias.append([(x, y)])
                    
                    # Info en frame
                    cv2.putText(salida, f"F: {idx_frame + 1}", (15, 35),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(salida, f"T: {len(trayectorias)}", (15, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    resultado_frames.append(salida)
                    gris_anterior = cuadro_gris
                    idx_frame += 1
                    
                    barra.progress(idx_frame / limite_frames)
                
                video.release()
                barra.empty()
                st.success(f"Procesados {len(resultado_frames)} frames")
                
                # Visualizar
                st.subheader("Resultado")
                
                selector = st.slider("Frame", 0, len(resultado_frames) - 1, 0)
                frame_rgb = cv2.cvtColor(resultado_frames[selector], cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption=f"Frame {selector + 1}")
                
                # Descarga
                _, buffer = cv2.imencode('.png', resultado_frames[selector])
                st.download_button(
                    "Descargar frame",
                    data=buffer.tobytes(),
                    file_name=f"flujo_optico_{selector}.png",
                    mime="image/png"
                )


if __name__ == "__main__":
    run()