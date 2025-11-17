import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# --- NOS PROPRES MODULES ---
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
    
    # --- 2. CHARGEMENT DES DONNÉES ---
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
    
    os.makedirs(args.save_path, exist_ok=True)
    np.save(os.path.join(args.save_path, "class_names.npy"), le.classes_)
    print(f"Classes sauvegardées dans {args.save_path}")

    # --- 4. SPLIT TRAIN/TEST ---
    print("Division des données (Train/Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.20, random_state=42, stratify=y_categorical
    )

    # --- 5. CONSTRUCTION OU CHARGEMENT DU MODÈLE (MODIFIÉ V3) ---
    NUM_CLASSES = len(class_names)
    
    # Définir les DEUX noms de fichiers possibles
    checkpoint_save_path = os.path.join(args.save_path, "best_model_checkpoint.keras")
    legacy_model_path = os.path.join(args.save_path, "rabies_behavior_model.keras") # L'ancien nom

    if os.path.exists(checkpoint_save_path):
        print(f"--- Chargement du checkpoint (reprise V2+) : {checkpoint_save_path} ---")
        model = tf.keras.models.load_model(checkpoint_save_path)
        print("Modèle chargé.")
    elif os.path.exists(legacy_model_path):
        print(f"--- Chargement du modèle V1 (reprise V1) : {legacy_model_path} ---")
        model = tf.keras.models.load_model(legacy_model_path)
        print("Modèle V1 chargé. L'entraînement va continuer.")
    else:
        print("--- Construction d'un nouveau modèle (entraînement V1) ---")
        model = build_model(FRAMES_PER_CLIP, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES)
        print("Compilation du nouveau modèle...")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    model.summary()

    # --- 6. ENTRAÎNEMENT (AVEC CALLBACKS) ---
    print("\n--- Démarrage de l'entraînement ---")
    
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    
    # Le ModelCheckpoint est toujours nécessaire pour sauvegarder le *nouveau* meilleur modèle
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_save_path, # Il sauvegardera sous le NOUVEAU nom
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    
    callbacks_list = [early_stopping, model_checkpoint]

    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=args.epochs, # Il s'entraînera jusqu'au nouveau nombre (ex: 20)
        validation_data=(X_test, y_test),
        callbacks=callbacks_list
    )
    
    print("\n--- Entraînement terminé ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'entraînement du modèle de détection de rage.")
    parser.add_argument('--data_path', type=str, required=True, help="Chemin vers le dossier 'Dataset' (symptômes)")
    parser.add_argument('--normal_path', type=str, required=True, help="Chemin vers le dossier 'Normal dog'")
    parser.add_argument('--save_path', type=str, required=True, help="Dossier où sauvegarder le modèle et les classes")
    parser.add_argument('--epochs', type=int, default=30, help="Nombre d'époques pour l'entraînement")
    args = parser.parse_args()
    main(args)