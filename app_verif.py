import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # FORCE LE CPU ET IGNORE LES ERREURS GPU
import streamlit as st
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, TimeDistributed, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from ultralytics import YOLO

# --- 1. CONFIGURATION & DESIGN (CSS) ---
st.set_page_config(
    page_title="VetScan AI - Détection Rage",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS pour améliorer le look (Centrage, Cartes, Ombres)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        font-weight: bold;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 30px;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-top: 20px;
    }
    .safe { background-color: #D1FAE5; color: #065F46; border: 1px solid #34D399; }
    .danger { background-color: #FEE2E2; color: #991B1B; border: 1px solid #F87171; }
    .warning { background-color: #FEF3C7; color: #92400E; border: 1px solid #FBBF24; }
    
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. CONSTANTES & CLASSES ---
MODEL_PATH = "models/best_model_checkpoint 2022.keras" 
CLASSES_PATH = "models/class_names2022.npy"
IMG_HEIGHT, IMG_WIDTH = 160, 160
FRAMES_PER_CLIP = 20

RABIES_SYMPTOMS = [
    'Incoordination', 'Paralysis', 'Restlesness', 'Sezure', 
    'Sudden aggression', 'bone in throat syndrome', 
    'dropped jaw, toungh', 'hyper salivation'
]
NORMAL_BEHAVIORS = [
    'barking', 'digging', 'playing', 
    'running dogs', 'tail wagging'
]

# --- 3. CHARGEMENT DES MODÈLES (CACHÉ) ---
@st.cache_resource
def load_yolo():
    return YOLO('yolov8n.pt')

@st.cache_resource
def load_ai_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        st.error("❌ Fichiers modèles manquants.")
        return None, None
    try:
        class_names = np.load(CLASSES_PATH)
        base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights=None)
        base_model.trainable = False 
        video_input = Input(shape=(FRAMES_PER_CLIP, IMG_HEIGHT, IMG_WIDTH, 3))
        cnn = TimeDistributed(base_model)(video_input)
        pool = TimeDistributed(GlobalAveragePooling2D())(cnn)
        gru = GRU(64, return_sequences=False, dropout=0.4)(pool)
        x = Dropout(0.5)(gru)
        output = Dense(len(class_names), activation='softmax')(x)
        model = Model(inputs=video_input, outputs=output)
        model.load_weights(MODEL_PATH)
        return model, class_names
    except Exception as e:
        st.error(f"Erreur chargement modèle : {e}")
        return None, None

# --- 4. FONCTIONS DE TRAITEMENT ---
def check_dog_presence(video_path):
    model_yolo = load_yolo()
    cap = cv2.VideoCapture(video_path)
    found, frame_count = False, 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame_count > 80: break
            if frame_count % 8 == 0:
                res = model_yolo.predict(frame, classes=[16], conf=0.4, verbose=False)
                if len(res[0].boxes) > 0:
                    found = True
                    break
            frame_count += 1
    finally:
        cap.release()
    return found

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0
            frames.append(frame)
    finally:
        cap.release()
    if len(frames) < 20: return None
    indices = np.linspace(0, len(frames)-1, FRAMES_PER_CLIP, dtype=int)
    return np.expand_dims(np.array(frames)[indices], axis=0)

# --- 5. INTERFACE UTILISATEUR ---

# A. Header
st.markdown('<p class="main-header">🐾 VetScan AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Assistant intelligent de détection précoce de la rage canine</p>', unsafe_allow_html=True)

# B. Sidebar (Mode d'emploi)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3048/3048122.png", width=100)
    st.title("Mode d'emploi")
    st.markdown("""
    1. **Filmez** le chien pendant 5 à 10 secondes.
    2. **Importez** la vidéo ici.
    3. **L'IA vérifie** d'abord si c'est un chien.
    4. **L'analyse** comportementale démarre.
    """)
    st.divider()
    st.warning("⚠️ **AVERTISSEMENT LÉGAL**\n\nCet outil est une aide au diagnostic. Il ne remplace PAS un vétérinaire. En cas de doute, isolez l'animal et consultez.")

# C. Zone Principale
col1, col2 = st.columns([1, 1])

with col1:
    st.info("📹 **Étape 1 : Importation**")
    uploaded_file = st.file_uploader("Glissez votre vidéo ici", type=["mp4", "avi", "mov"], label_visibility="collapsed")
    
    if uploaded_file:
        temp_path = "temp_video.mp4"
        with open(temp_path, "wb") as f: f.write(uploaded_file.read())
        st.video(temp_path)

with col2:
    st.info("🧠 **Étape 2 : Analyse**")
    
    if uploaded_file:
        if st.button("Lancer le Diagnostic", type="primary"):
            model_rage, class_names = load_ai_model()
            
            if model_rage:
                # Utilisation de st.status pour une UX fluide
                with st.status("Analyse en cours...", expanded=True) as status:
                    
                    st.write("🔍 Recherche de chien dans l'image (YOLOv8)...")
                    is_dog = check_dog_presence(temp_path)
                    
                    if not is_dog:
                        status.update(label="Échec", state="error", expanded=True)
                        st.error("🚫 Aucun chien détecté.")
                        st.caption("Assurez-vous que l'animal est bien visible.")
                    else:
                        st.write("✅ Chien identifié.")
                        st.write("🧬 Analyse des mouvements neurologiques (MobileNetV2)...")
                        
                        tensor = process_video(temp_path)
                        if tensor is not None:
                            preds = model_rage.predict(tensor)
                            idx = np.argmax(preds[0])
                            label = class_names[idx]
                            conf = preds[0][idx] * 100
                            status.update(label="Diagnostic terminé", state="complete", expanded=False)
                            
                            # --- AFFICHAGE RÉSULTAT ---
                            st.divider()
                            
                            if conf < 60:
                                st.markdown(f"""
                                <div class="result-card warning">
                                    <h3>⚠️ RÉSULTAT INCERTAIN</h3>
                                    <p>Confiance : {conf:.1f}%</p>
                                    <p>Le comportement est ambigu.</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            elif label in RABIES_SYMPTOMS:
                                st.markdown(f"""
                                <div class="result-card danger">
                                    <h3>🔴 ALERTE RAGE</h3>
                                    <h1>{label}</h1>
                                    <p>Probabilité : <strong>{conf:.1f}%</strong></p>
                                    <hr>
                                    <p>🚨 <strong>ACTION REQUISE :</strong> ISOLEMENT IMMÉDIAT</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            else:
                                st.markdown(f"""
                                <div class="result-card safe">
                                    <h3>🟢 COMPORTEMENT NORMAL</h3>
                                    <h1>{label}</h1>
                                    <p>Probabilité : <strong>{conf:.1f}%</strong></p>
                                    <p>Aucun signe neurologique grave.</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                        else:
                            status.update(label="Erreur", state="error")
                            st.error("Vidéo trop courte.")
    else:
        st.write("👈 *En attente d'une vidéo...*")
        # Placeholder visuel
        st.markdown("""
        <div style="text-align: center; color: #ccc; padding: 50px; border: 2px dashed #ccc; border-radius: 10px;">
            En attente du fichier...
        </div>
        """, unsafe_allow_html=True)