🐾 VetScan AI : Détection Précoce de la Rage Canine

<div align="center">

Un système de Computer Vision hybride (CNN + RNN) pour l'analyse comportementale des chiens et le diagnostic vétérinaire assisté.

[Démo Vidéo] | [Lire le Rapport] | [Télécharger le Modèle]

</div>

📖 À Propos du Projet

La rage est une maladie virale mortelle qui tue encore environ 59 000 personnes par an dans le monde, principalement transmise par les chiens. Le diagnostic précoce est crucial mais difficile, car les signes cliniques (comportementaux) peuvent être subtils avant l'apparition des symptômes physiques évidents.

VetScan AI est une solution d'intelligence artificielle conçue pour analyser des séquences vidéo de chiens et détecter les signes neurologiques précurseurs de la rage.

🎯 Objectifs

Triage Rapide : Permettre une évaluation préliminaire en moins de 10 secondes.

Non-invasif : Analyse à distance via une simple vidéo smartphone.

Accessibilité : Déploiement facile sur des appareils grand public via une interface web.

⚙️ Architecture Technique

Ce projet utilise une approche Deep Learning Hybride pour combiner l'analyse visuelle et temporelle :

Détection d'Objet (YOLOv8) : * Agit comme un "gardien" pour vérifier la présence d'un chien dans la vidéo avant l'analyse.

Élimine les faux positifs (chats, humains, objets).

Extraction de Caractéristiques (MobileNetV2 - CNN) :

Analyse chaque image (frame) de la vidéo pour extraire des caractéristiques visuelles (textures, formes).

Utilise le Transfer Learning (pré-entraîné sur ImageNet) avec Fine-Tuning des 30 dernières couches.

Analyse Séquentielle (GRU - RNN) :

Traite la séquence temporelle des caractéristiques extraites.

Détecte les anomalies de mouvement (incoordination, tremblements, agressivité soudaine).

Classification :

Couche dense finale avec activation Softmax.

13 Classes : 8 Symptômes de rage vs 5 Comportements normaux.

📊 Performance et Résultats

Le modèle a été entraîné sur un dataset propriétaire de 1200+ clips vidéo.

Métrique

Score

Description

Précision Globale (Accuracy)

83%

Sur le jeu de test (données inconnues)

Détection Agressivité

97%

Précision sur la classe critique "Sudden Aggression"

Vitesse d'Inférence

< 200ms

Temps de traitement par vidéo (sur CPU standard)

Note : Le modèle a été optimisé pour minimiser les Faux Négatifs sur les classes dangereuses.

🚀 Installation et Utilisation

Prérequis

Python 3.10+

Un environnement virtuel (recommandé)

1. Cloner le dépôt

git clone [https://github.com/NASSERamine/dogs_rabies_detection.git](https://github.com/NASSERamine/dogs_rabies_detection.git)
cd dogs_rabies_detection


2. Installer les dépendances

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Sur Windows : .venv\Scripts\activate

# Installer les librairies
pip install -r requirements.txt


3. Télécharger le Modèle Entraîné

En raison de la taille des fichiers, le modèle .keras n'est pas inclus dans le dépôt Git.

Téléchargez le fichier best_model_checkpoint.keras [Lien vers votre Google Drive/Release].

Placez-le dans le dossier models/ à la racine du projet.

4. Lancer l'Application (Démo)

Nous fournissons une interface web interactive basée sur Streamlit.

streamlit run app_local.py


L'application s'ouvrira automatiquement dans votre navigateur à l'adresse http://localhost:8501.

📂 Structure du Projet

dogs_rabies_detection/
├── data/                  # (Ignoré par Git) Données brutes
├── models/                # Fichiers modèles (.keras, .npy)
│   ├── best_model_checkpoint 2022.keras
│   └── class_names2022.npy
├── src/                   # Code source du pipeline MLOps
│   ├── data_processing.py # Chargement, Augmentation, Générateurs tf.data
│   ├── model.py           # Architecture CNN-RNN (MobileNetV2 + GRU)
│   └── train.py           # Script d'entraînement avec Callbacks
├── app_local.py           # Application de démo (Streamlit + YOLO + Keras)
├── requirements.txt       # Liste des dépendances
└── README.md              # Documentation


🛠️ Pipeline d'Entraînement (Pour les développeurs)

Si vous souhaitez ré-entraîner le modèle avec vos propres données :

Organisez vos vidéos dans data/Dataset (Malades) et data/Normal dog (Sains).

Lancez le script d'entraînement :

python src/train.py --data_path "data/Dataset" --normal_path "data/Normal dog" --save_path "models" --epochs 20 --batch_size 4


Le script gère automatiquement la reprise d'entraînement (Resume) si un modèle existe déjà.

⚠️ Avertissement Légal et Éthique

Ce projet est un outil de recherche et d'aide à la décision. Il ne remplace en aucun cas l'avis d'un vétérinaire professionnel.

Un résultat "Positif" doit entraîner l'isolement immédiat de l'animal et un contact avec les autorités sanitaires.

Un résultat "Négatif" ne garantit pas l'absence de pathologie.

👤 Auteur

Nasser Amine

LinkedIn

GitHub

<div align="center">
<sub>Projet réalisé dans le cadre de [Nom de votre Formation/Diplôme] - 2025</sub>
</div>
