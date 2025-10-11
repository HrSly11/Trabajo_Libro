import streamlit as st
from temas import tema1, tema2, tema3, tema4, tema5, tema6, tema7, tema8, tema10
from temas.tema11 import tema11
from temas.tema9 import tema9

# Configuración
st.set_page_config(page_title="Ejercicios de Programación", layout="wide")

# Datos de los temas
temas = [
    {"titulo": "Applying Geometric Transformations", "desc": "Aplicar transformaciones geométricas"},
    {"titulo": "Detecting Edges and Applying Image Filters", "desc": "Detectar bordes y aplicar filtros"},
    {"titulo": "Cartoonizing an Image", "desc": "Transformar una fotografía en caricatura"},
    {"titulo": "Detecting Body Parts", "desc": "Detectar y rastrear partes del cuerpo"},
    {"titulo": "Extracting Features", "desc": "Extraer características de imágenes"},
    {"titulo": "Seam Carving of an Image", "desc": "Redimensionar imágenes sin distorsión"},
    {"titulo": "Detecting Shapes and Segmenting an Image", "desc": "Detectar formas y segmentar regiones"},
    {"titulo": "Tracking an Object in a Live Video", "desc": "Rastrear objetos en movimiento en video"},
    {"titulo": "Object Recognition", "desc": "Reconocer objetos mediante visión artificial"},
    {"titulo": "Augmented Reality", "desc": "Crear experiencias de realidad aumentada"},
    {"titulo": "Machine Learning by Neural Network", "desc": "Clasificar imágenes con redes neuronales"},
]

modulos = [tema1, tema2, tema3, tema4, tema5, tema6, tema7, tema8, tema9, tema10, tema11]

# Estado inicial
if 'tema_seleccionado' not in st.session_state:
    st.session_state.tema_seleccionado = None

# Sidebar
with st.sidebar:
    st.header("📚 Temas")
    for i, tema in enumerate(temas):
        if st.button(f"{i+1}. {tema['titulo']}", key=f"tema_{i}", use_container_width=True):
            st.session_state.tema_seleccionado = i
            st.rerun()
    
    st.divider()
    
    if st.button("🏠 Volver al Inicio", use_container_width=True):
        st.session_state.tema_seleccionado = None
        st.rerun()

# Contenido principal
if st.session_state.tema_seleccionado is None:
    st.title("📘 Ejercicios de Visión por Computadora e IA")
    st.write("Explora los **11 programas** desarrollados. Selecciona un tema desde el menú lateral.")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    for i, tema in enumerate(temas):
        with col1 if i % 2 == 0 else col2:
            with st.container(border=True):
                st.subheader(f"{i+1}. {tema['titulo']}")
                st.write(tema['desc'])
                if st.button(f"Ver Tema {i+1}", key=f"btn{i}", use_container_width=True):
                    st.session_state.tema_seleccionado = i
                    st.rerun()

else:
    idx = st.session_state.tema_seleccionado
    tema_actual = temas[idx]
    st.title(f"{idx+1}. {tema_actual['titulo']}")
    st.write(tema_actual['desc'])
    st.divider()
    modulos[idx].run()