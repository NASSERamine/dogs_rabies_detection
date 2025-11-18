import os
import argparse  # Pour lire les arguments de Colab
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# --- NOS PROPRES MODULES (C'est ça, le MLOps !) ---
from data_processing import load_and_process_videos
from model import build_model

# --- 1. DÉFINIR LES PARAMÈTRES D'ENTRAÎNEMENT ---
CLIP_DURATION = 3
FRAMES_PER_CLIP = 20
IMG_HEIGHT = 160
IMG_WIDTH = 160
BATCH_SIZE = 8

def main(args):
    """
    Fonction principale d'entraînement.
    """
    
    # --- 2. CHARGEMENT DES DONNÉES (via data_processing.py) ---
    print("Démarrage du chargement et du prétraitement des données...")
    all_data_paths = [args.data_path, args.normal_path]
    X, y, class_names = load_and_process_videos(
        all_data_paths, IMG_HEIGHT, IMG_WIDTH, FRAMES_PER_CLIP, CLIP_DURATION
    )
    
    if X is None or X.size == 0:
        print("ERREUR FATALE : Aucune donnée n'a été chargée.")
        return

    print(f"Données chargées. Forme de X : {X.shape}, Forme de y : {y.shape}")

    # --- 3. ENCODAGE DES LABELS ---
    print("Encodage des labels...")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    y_categorical = to_categorical(y_encoded, num_classes=len(class_names))
    
    # Créer le dossier de sauvegarde s'il n'existe pas
    os.makedirs(args.save_path, exist_ok=True)
    
    # Sauvegarder les classes
    np.save(os.path.join(args.save_path, "class_names.npy"), le.classes_)
    print(f"Classes sauvegardées dans {args.save_path}")

    # --- 4. SPLIT TRAIN/TEST ---
    print("Division des données (Train/Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical,
        test_size=0.20,  # 20% pour le test
        random_state=42,
        stratify=y_categorical
    )

    # --- 5. CONSTRUCTION DU MODÈLE (via model.py) ---
    print("Construction du modèle...")
    NUM_CLASSES = len(class_names)
    model = build_model(FRAMES_PER_CLIP, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES)
    
    # Compilation
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # --- 6. ENTRAÎNEMENT ---
    print("\n--- Démarrage de l'entraînement ---")
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=args.epochs,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping]
    )
    
    print("\n--- Entraînement terminé ---")

    # --- 7. SAUVEGARDE DU MODÈLE FINAL ---
    model_save_path = os.path.join(args.save_path, "rabies_behavior_model.keras")
    model.save(model_save_path)
    print(f"Modèle sauvegardé avec succès dans : {model_save_path}")

if __name__ == "__main__":
    # --- ARGUMENTS DE LIGNE DE COMMANDE ---
    # C'est ce qui permet à Colab de parler à notre script
    parser = argparse.ArgumentParser(description="Script d'entraînement du modèle de détection de rage.")
    
    parser.add_argument('--data_path', type=str, required=True, help="Chemin vers le dossier 'Dataset' (symptômes)")
    parser.add_argument('--normal_path', type=str, required=True, help="Chemin vers le dossier 'Normal dog'")
    parser.add_argument('--save_path', type=str, required=True, help="Dossier où sauvegarder le modèle et les classes")
    parser.add_argument('--epochs', type=int, default=30, help="Nombre d'époques pour l'entraînement")
    
    args = parser.parse_args()
    
    # Lancer la fonction principale
    main(args)