import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split

# --- NOS PROPRES MODULES (Version Générateur) ---
from data_processing import create_clip_manifest, create_dataset
from model import build_model

# --- 1. DÉFINIR LES PARAMÈTRES D'ENTRAÎNEMENT ---
CLIP_DURATION = 3
FRAMES_PER_CLIP = 20
IMG_HEIGHT = 160
IMG_WIDTH = 160

def main(args):
    """
    Fonction principale d'entraînement (utilisant tf.data.Dataset).
    """
    
    # --- 2. CRÉATION DU MANIFEST (Léger en RAM) ---
    print("Démarrage du scan des vidéos (création du manifest)...")
    all_data_paths = [args.data_path, args.normal_path]
    
    # manifest est la liste de (path, class_name, start_frame)
    manifest, class_names = create_clip_manifest(all_data_paths)
    
    if not manifest:
        print("ERREUR FATALE : Le manifest est vide.")
        return

    print(f"Scan terminé. {len(manifest)} clips trouvés.")

    # --- 3. SAUVEGARDE DES CLASSES (avant le split) ---
    os.makedirs(args.save_path, exist_ok=True)
    np.save(os.path.join(args.save_path, "class_names.npy"), np.array(class_names))
    print(f"Classes sauvegardées dans {args.save_path}")

    # --- 4. SPLIT TRAIN/TEST (sur le manifest, pas sur les données) ---
    print("Division du manifest (Train/Test)...")
    
    # Extraire les labels pour un split stratifié
    manifest_labels = [item[1] for item in manifest]
    
    train_manifest, test_manifest = train_test_split(
        manifest,
        test_size=0.20,
        random_state=42,
        stratify=manifest_labels
    )
    
    print(f"Taille du set d'entraînement : {len(train_manifest)} clips")
    print(f"Taille du set de test : {len(test_manifest)} clips")

    # --- 5. CRÉATION DES PIPELINES tf.data ---
    print("Création des générateurs de données (tf.data)...")
    
    train_dataset, le = create_dataset(train_manifest, class_names, args.batch_size)
    test_dataset, _ = create_dataset(test_manifest, class_names, args.batch_size)
    
    # --- 6. CONSTRUCTION OU CHARGEMENT DU MODÈLE ---
    NUM_CLASSES = len(class_names)
    checkpoint_save_path = os.path.join(args.save_path, "best_model_checkpoint.keras")
    legacy_model_path = os.path.join(args.save_path, "rabies_behavior_model.keras")

    if os.path.exists(checkpoint_save_path):
        print(f"--- Chargement du checkpoint (reprise V2+) : {checkpoint_save_path} ---")
        model = tf.keras.models.load_model(checkpoint_save_path)
    elif os.path.exists(legacy_model_path):
        print(f"--- Chargement du modèle V1 (reprise V1) : {legacy_model_path} ---")
        model = tf.keras.models.load_model(legacy_model_path)
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

    # --- 7. ENTRAÎNEMENT (AVEC GÉNÉRATEURS) ---
    print("\n--- Démarrage de l'entraînement ---")
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(
        filepath=checkpoint_save_path,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1
    )
    callbacks_list = [early_stopping, model_checkpoint]

    # Calculer les "steps" (nombre de lots par époque)
    steps_per_epoch = len(train_manifest) // args.batch_size
    validation_steps = len(test_manifest) // args.batch_size
    
    if steps_per_epoch == 0 or validation_steps == 0:
         print(f"ERREUR: Pas assez de données pour un batch complet (Batch size: {args.batch_size}). Réduisez le batch_size.")
         return

    model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=test_dataset,
        callbacks=callbacks_list,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    
    print("\n--- Entraînement terminé ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'entraînement du modèle de détection de rage.")
    parser.add_argument('--data_path', type=str, required=True, help="Chemin vers le dossier 'Dataset' (symptômes)")
    parser.add_argument('--normal_path', type=str, required=True, help="Chemin vers le dossier 'Normal dog'")
    parser.add_argument('--save_path', type=str, required=True, help="Dossier où sauvegarder le modèle et les classes")
    parser.add_argument('--epochs', type=int, default=30, help="Nombre d'époques pour l'entraînement")
    parser.add_argument('--batch_size', type=int, default=4, help="Taille des lots (batch size)")
    
    args = parser.parse_args()
    main(args)