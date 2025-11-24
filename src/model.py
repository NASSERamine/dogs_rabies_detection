from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, TimeDistributed, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2

def build_model(num_frames, img_height, img_width, num_classes):
    """
    Construit l'architecture CNN + RNN (MobileNetV2 + GRU).
    Version V2 : Fine-Tuning + GlobalAveragePooling
    """
    
    # 1. Base MobileNetV2 (CNN)
    base_model = MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # --- FINE TUNING ---
    # On rend le modèle trainable
    base_model.trainable = True
    # Mais on re-gèle les premières couches pour ne pas tout casser
    # On laisse seulement les 30 dernières couches apprendre les textures "Rage"
    for layer in base_model.layers[:-30]:
        layer.trainable = False
        
    # 2. Entrée Vidéo
    video_input = Input(shape=(num_frames, img_height, img_width, 3))
    
    # 3. Extraction de caractéristiques sur chaque frame
    cnn_features = TimeDistributed(base_model)(video_input)
    
    # 4. Pooling Spatial (Réduit la dimensionnalité efficacement)
    pooled_features = TimeDistributed(GlobalAveragePooling2D())(cnn_features)
    
    # 5. Analyse Temporelle (GRU)
    # 64 neurones suffisent avec le GlobalPooling
    temporal_features = GRU(64, return_sequences=False, dropout=0.4)(pooled_features)
    
    # 6. Classification
    x = Dropout(0.5)(temporal_features)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=video_input, outputs=output)
    
    return model