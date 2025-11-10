import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, TimeDistributed, Flatten, Dropout
from tensorflow.keras.applications import MobileNetV2

def build_model(num_frames, img_height, img_width, num_classes):
    """
    Construit l'architecture du modèle CNN (figé) + GRU.
    """
    
    # --- 1. L'extracteur de caractéristiques (Les "Yeux") ---
    base_model = MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,  # On retire la tête de classification ImageNet
        weights='imagenet'  # On charge les poids pré-entraînés
    )
    # On gèle le CNN ! On ne le ré-entraîne pas.
    base_model.trainable = False

    # --- 2. Définition de l'architecture complète ---
    
    # L'entrée de notre modèle
    video_input = Input(shape=(num_frames, img_height, img_width, 3))
    
    # On applique le CNN à chaque frame de la vidéo
    cnn_features = TimeDistributed(base_model)(video_input)
    
    # On aplatit la sortie de chaque frame
    flattened_features = TimeDistributed(Flatten())(cnn_features)
    
    # --- 3. L'analyseur de séquence (La "Mémoire") ---
    # Le GRU analyse la séquence de caractéristiques
    temporal_features = GRU(128)(flattened_features)
    
    # --- 4. La tête de classification (Le "Décideur") ---
    x = Dropout(0.5)(temporal_features) # Régularisation pour éviter l'overfitting
    output = Dense(num_classes, activation='softmax')(x)
    
    # On crée le modèle final
    model = Model(inputs=video_input, outputs=output)
    
    return model