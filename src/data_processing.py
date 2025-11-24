import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# --- CONSTANTES ---
CLIP_DURATION = 3
FRAMES_PER_CLIP = 20
IMG_HEIGHT = 160
IMG_WIDTH = 160

def augment_frames(frames):
    """
    Applique des transformations aléatoires pour la Data Augmentation.
    Rend le modèle plus robuste aux variations de lumière et d'orientation.
    """
    # 1. Flip Horizontal (Miroir) - 50% de chance
    if tf.random.uniform(()) > 0.5:
        frames = tf.image.flip_left_right(frames)
    
    # 2. Luminosité (Simulation jour/nuit/intérieur)
    frames = tf.image.random_brightness(frames, max_delta=0.2)
    
    # 3. Contraste (Qualité caméra variable)
    frames = tf.image.random_contrast(frames, lower=0.8, upper=1.2)
    
    return frames

def create_clip_manifest(base_paths):
    """
    Scanne les dossiers et crée une liste de tous les clips possibles
    sans charger les vidéos en mémoire.
    """
    manifest = []
    class_names = []
    all_folders = []

    # 1. Identification des dossiers de classes
    for base_path in base_paths:
        if not os.path.exists(base_path):
            print(f"⚠️ Chemin introuvable : {base_path}")
            continue
        for class_folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, class_folder)
            if os.path.isdir(folder_path):
                all_folders.append((folder_path, class_folder))
                if class_folder not in class_names:
                    class_names.append(class_folder)

    # 2. Scan des fichiers vidéo
    for folder_path, class_name in all_folders:
        video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            
            # Vérification rapide de la vidéo
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): continue
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if fps <= 0 or fps > 200: fps = 25
            if total_frames < 20: continue
            
            # Découpage logique en clips
            frames_in_clip = int(CLIP_DURATION * fps)
            step = frames_in_clip // 2 # Chevauchement 50%
            
            for start in range(0, total_frames - frames_in_clip, step):
                manifest.append((video_path, class_name, start))
                
    return manifest, sorted(class_names)

def load_and_process_clip(video_path, start_frame):
    """Fonction Pythn pure pour charger un clip spécifique"""
    video_path = video_path.numpy().decode('utf-8')
    start_frame = int(start_frame.numpy())
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 25
    
    end_frame = start_frame + int(CLIP_DURATION * fps)
    
    # Sélection uniforme des frames
    frame_indices = np.linspace(start_frame, end_frame-1, FRAMES_PER_CLIP, dtype=int)
    frames = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0 # Normalisation 0-1
            frames.append(frame)
    cap.release()
    
    # Padding si échec de lecture
    if len(frames) == FRAMES_PER_CLIP:
        return np.array(frames, dtype=np.float32)
    return np.zeros((FRAMES_PER_CLIP, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)

@tf.function
def tf_load_clip(path, label, start):
    """Wrapper TensorFlow pour le dataset"""
    [clip,] = tf.py_function(load_and_process_clip, [path, start], [tf.float32])
    clip.set_shape((FRAMES_PER_CLIP, IMG_HEIGHT, IMG_WIDTH, 3))
    
    # Application de l'augmentation
    clip = augment_frames(clip)
    
    return clip, label

def create_dataset(manifest, class_names, batch_size):
    """Crée un générateur tf.data optimisé"""
    le = LabelEncoder()
    le.fit(class_names)
    
    paths, labels, starts = [], [], []
    for (p, c, s) in manifest:
        paths.append(p)
        labels.append(le.transform([c])[0])
        starts.append(float(s))

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, starts))
    ds = ds.map(tf_load_clip, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return ds, le