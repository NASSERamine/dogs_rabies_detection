import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# --- 1. PARAMÈTRES GLOBAUX ---
CLIP_DURATION = 3
FRAMES_PER_CLIP = 20
IMG_HEIGHT = 160
IMG_WIDTH = 160

def create_clip_manifest(base_paths):
    """
    Scanne les dossiers, ne charge PAS les vidéos.
    Crée une "liste de clips" (manifest) avec (video_path, class_name, start_frame).
    """
    manifest = [] # (video_path, class_name, start_frame)
    class_names = []
    
    all_folders = []
    for base_path in base_paths:
        if not os.path.exists(base_path):
            print(f"AVERTISSEMENT : Le chemin {base_path} n'existe pas. Ignoré.")
            continue
        for class_folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, class_folder)
            if os.path.isdir(folder_path):
                all_folders.append((folder_path, class_folder))
                if class_folder not in class_names:
                    class_names.append(class_folder)

    print(f"Classes trouvées : {sorted(class_names)}")
    if not class_names:
        raise ValueError("Aucun dossier de classe trouvé. Vérifiez vos chemins.")

    for folder_path, class_name in all_folders:
        print(f"...Scan de : {class_name}")
        video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            
            # Tenter d'ouvrir la vidéo juste pour obtenir les métadonnées
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"  Avertissement : Impossible d'ouvrir {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            if fps == 0 or fps > 200: fps = 25
            if total_frames < 10: continue # Ignorer les vidéos trop courtes

            frames_in_clip_duration = int(CLIP_DURATION * fps)
            step = frames_in_clip_duration // 2  # Chevauchement de 50%

            if frames_in_clip_duration <= 0:
                print(f"  Avertissement : Durée de clip invalide pour {video_path}, fps={fps}")
                continue

            for start_frame in range(0, total_frames - frames_in_clip_duration, step):
                manifest.append((video_path, class_name, start_frame))
                
    if not manifest:
        raise ValueError("Aucun clip n'a pu être créé. Vérifiez vos fichiers vidéo.")
        
    return manifest, sorted(class_names)

def load_and_process_clip(video_path, start_frame):
    """
    Charge et traite UN SEUL clip à partir d'un chemin et d'un start_frame.
    """
    video_path = video_path.decode('utf-8') # Nécessaire pour tf.py_function
    start_frame = int(start_frame)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erreur (load_clip): Impossible d'ouvrir {video_path}")
        return np.zeros((FRAMES_PER_CLIP, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps > 200: fps = 25
    
    frames_in_clip_duration = int(CLIP_DURATION * fps)
    end_frame = start_frame + frames_in_clip_duration
    
    frame_indices = np.linspace(start_frame, end_frame - 1, FRAMES_PER_CLIP, dtype=int)
    
    clip_frames = []
    frames_read_success = 0
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (IMG_HEIGHT, IMG_WIDTH))
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0  # Normalisation
            clip_frames.append(frame)
            frames_read_success += 1
    
    cap.release()

    if frames_read_success == FRAMES_PER_CLIP:
        return np.array(clip_frames, dtype=np.float32)
    else:
        return np.zeros((FRAMES_PER_CLIP, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)

@tf.function
def tf_load_and_process_clip(video_path, label, start_frame):
    """
    Wrapper TensorFlow pour appeler notre fonction Python (OpenCV).
    """
    [clip_array,] = tf.py_function(
        load_and_process_clip, # Notre fonction OpenCV
        [video_path, start_frame], # Les arguments
        [tf.float32] # Le type de retour
    )
    
    clip_array.set_shape((FRAMES_PER_CLIP, IMG_HEIGHT, IMG_WIDTH, 3))
    label.set_shape(()) # Le label est un scalaire
    
    return clip_array, label

def create_dataset(manifest, class_names, batch_size):
    """
    Crée le pipeline tf.data à partir du "manifest" de clips.
    """
    
    # 1. Encodage des labels
    le = LabelEncoder()
    le.fit(class_names)
    
    # 2. Préparer les listes pour TensorFlow
    video_paths = []
    labels = []
    start_frames = []
    
    for (path, class_name, start) in manifest:
        video_paths.append(path)
        labels.append(le.transform([class_name])[0])
        start_frames.append(float(start))

    # 3. Créer le dataset de base à partir des listes (très léger en RAM)
    dataset = tf.data.Dataset.from_tensor_slices(
        (video_paths, labels, start_frames)
    )
    
    # 4. Appliquer le chargement et le traitement "à la volée"
    dataset = dataset.map(
        tf_load_and_process_clip, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    # 5. Configurer le pipeline pour la performance
    dataset = dataset.shuffle(buffer_size=500) # Mélanger
    dataset = dataset.batch(batch_size) # Mettre en lots
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # Pré-charger
    
    return dataset, le