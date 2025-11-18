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

# --- 1. DÉFINIR LES PARAMÈTRES ---
CLIP_DURATION = 3
FRAMES_PER_CLIP = 20
IMG_HEIGHT = 160
IMG_WIDTH = 160
BATCH_SIZE = 8 # Si Colab plante (RAM), changez ceci en 4

def main(args):
    """
    Fonction principale d'entraînement avec REPRISE AUTOMATIQUE.
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

    # --- 4. SPLIT TRAIN/TEST ---
    print("Division des données (Train/Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.20, random_state=42, stratify=y_categorical
    )

    # --- 5. LOGIQUE DE REPRISE (C'EST ICI LA CLÉ) ---
    NUM_CLASSES = len(class_names)
    
    # Chemin exact de votre ancien modèle
    model_path = os.path.join(args.save_path, "rabies_behavior_model.keras")
    
    if os.path.exists(model_path):
        print(f"\n>>> 🔄 MODÈLE TROUVÉ : {model_path}")
        print(">>> Chargement du modèle existant pour continuer l'entraînement (Fine-tuning)...")
        # On charge le modèle, ses poids ET l'état de l'optimiseur (le cerveau se souvient)
        model = tf.keras.models.load_model(model_path)
    else:
        print("\n>>> ✨ AUCUN MODÈLE TROUVÉ.")
        print(">>> Construction d'un nouveau modèle (départ de zéro)...")
        model = build_model(FRAMES_PER_CLIP, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    model.summary()

    # --- 6. ENTRAÎNEMENT (AJOUT D'ÉPOQUES) ---
    print(f"\n--- Démarrage de l'entraînement pour {args.epochs} époques supplémentaires ---")
    
    checkpoint_path = os.path.join(args.save_path, "best_model_checkpoint.keras")
    
    callbacks_list = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
    ]

    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=args.epochs, # Cela rajoute 10 tours de plus
        validation_data=(X_test, y_test),
        callbacks=callbacks_list
    )
    
    print("\n--- Entraînement terminé ---")

    # --- 7. SAUVEGARDE DU RÉSULTAT ---
    # On met à jour le fichier pour la prochaine fois
    model.save(model_path)
    print(f"Modèle mis à jour sauvegardé dans : {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--normal_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()
    main(args)