import os
import cv2
import numpy as np

def load_and_process_videos(base_paths, img_height, img_width, frames_per_clip, clip_duration):
    """
    Charge toutes les vidéos, les découpe en "clips" avec chevauchement,
    et extrait 'frames_per_clip' de chaque clip.
    
    VERSION MISE À JOUR : Gère robustement les vidéos N&B et les convertit en RGB.
    """
    clips = []  # Va contenir les (N, 20, 160, 160, 3)
    labels = []  # Va contenir les (N,) labels (ex: 'seizure', 'playing')
    class_names = []

    # Collecter tous les dossiers de classe
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
        print(f"ERREUR: Aucun dossier de classe trouvé. Vérifiez vos chemins.")
        return None, None, None

    for folder_path, class_name in all_folders:
        print(f"--- Traitement de la classe : {class_name} ---")
        video_files = [f for f in os.listdir(folder_path) if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        for video_file in video_files:
            video_path = os.path.join(folder_path, video_file)
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"  Avertissement : Impossible d'ouvrir {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # <-- La variable correcte
            
            if fps == 0 or fps > 200: 
                print(f"  Avertissement : FPS invalide ({fps}) pour {video_path}. On utilise 25 par défaut.")
                fps = 25

            frames_in_clip_duration = int(clip_duration * fps)
            step = frames_in_clip_duration // 2  # Chevauchement de 50%

            # --- CORRECTION DE LA TYPO ICI ---
            # J'utilise "total_frames" et non "total_tables"
            for start_frame in range(0, total_frames - frames_in_clip_duration, step):
                end_frame = start_frame + frames_in_clip_duration
                frame_indices = np.linspace(start_frame, end_frame - 1, frames_per_clip, dtype=int)
                
                clip_frames = []
                frames_read_success = 0
                
                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    
                    if ret:
                        frame = cv2.resize(frame, (img_height, img_width))

                        if len(frame.shape) == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                        elif frame.shape[2] == 1:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                        else:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        frame = frame / 255.0  
                        
                        clip_frames.append(frame)
                        frames_read_success += 1
                
                if frames_read_success == frames_per_clip:
                    clips.append(np.array(clip_frames))
                    labels.append(class_name)
            
            cap.release()
            
    if not clips:
        print("ERREUR : Aucun clip n'a été chargé. Vérifiez vos fichiers vidéo.")
        return None, None, None
            
    return np.array(clips), np.array(labels), class_names