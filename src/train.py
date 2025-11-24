import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
# Imports locaux
from data_processing import create_clip_manifest, create_dataset
from model import build_model

# Paramètres globaux (doivent matcher data_processing)
CLIP_DURATION = 3
FRAMES_PER_CLIP = 20
IMG_HEIGHT = 160
IMG_WIDTH = 160

def main(args):
    print(f"--- CONFIGURATION: Epochs={args.epochs}, Batch_Size={args.batch_size} ---")
    
    # 1. Préparation des données
    print(">>> Scan des fichiers vidéo...")
    all_data_paths = [args.data_path, args.normal_path]
    manifest, class_names = create_clip_manifest(all_data_paths)
    
    if not manifest:
        print("❌ ERREUR : Aucune vidéo trouvée. Vérifiez les chemins.")
        return

    print(f"✅ Clips trouvés : {len(manifest)}")
    print(f"✅ Classes : {class_names}")

    # Sauvegarde des noms de classes pour l'inférence (app.py)
    os.makedirs(args.save_path, exist_ok=True)
    np.save(os.path.join(args.save_path, "class_names.npy"), np.array(class_names))

    # 2. Séparation Train / Test
    # On split le manifeste, pas les données chargées (gain de RAM)
    labels = [m[1] for m in manifest]
    train_manifest, test_manifest = train_test_split(
        manifest, 
        test_size=0.20, 
        random_state=42, 
        stratify=labels
    )
    
    # 3. Création des Générateurs tf.data
    train_dataset, le = create_dataset(train_manifest, class_names, args.batch_size)
    test_dataset, _ = create_dataset(test_manifest, class_names, args.batch_size)

    # 4. Construction du Modèle
    NUM_CLASSES = len(class_names)
    print(">>> Construction du modèle V2 (Fine-Tuning)...")
    model = build_model(FRAMES_PER_CLIP, IMG_HEIGHT, IMG_WIDTH, NUM_CLASSES)
    
    # Compilation
    # Note : Learning rate bas (0.0001) car on fait du Fine-Tuning
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    model.summary()

    # 5. Callbacks (Sauvegarde automatique)
    checkpoint_path = os.path.join(args.save_path, "best_model_checkpoint.keras")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
    ]

    # 6. Lancement de l'entraînement
    print(">>> Démarrage de l'entraînement...")
    model.fit(
        train_dataset,
        epochs=args.epochs,
        validation_data=test_dataset,
        callbacks=callbacks
    )
    
    # Sauvegarde finale
    final_path = os.path.join(args.save_path, "rabies_behavior_model_final.keras")
    model.save(final_path)
    print(f"✅ Entraînement terminé. Modèle sauvegardé sous : {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement Détection Rage Canine")
    parser.add_argument('--data_path', type=str, required=True, help="Dossier des symptômes")
    parser.add_argument('--normal_path', type=str, required=True, help="Dossier des chiens normaux")
    parser.add_argument('--save_path', type=str, required=True, help="Dossier de sortie")
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=4)
    
    args = parser.parse_args()
    main(args)