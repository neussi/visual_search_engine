# 👕 FashionSearch - Moteur de Recherche Visuel de Vêtements

**FashionSearch** est un moteur de recherche visuel basé sur le contenu (**CBIR - Content-Based Image Retrieval**) développé avec Django. Il permet de retrouver instantanément des vêtements similaires dans un catalogue en téléchargeant une photo ou en sélectionnant un échantillon. Le système extrait les descripteurs caractéristiques des images et effectue une recherche de plus proches voisins par **similarité cosinus**.

---

## 🚀 Fonctionnalités Clés

1. **Upload Drag-and-Drop** : Zone de glisser-déposer moderne avec prévisualisation immédiate de l'image de requête.
2. **Recherche Instantanée sur Échantillons** : Galerie de 12 images aléatoires du catalogue permettant de tester les performances du moteur de recherche en un seul clic.
3. **Extraction de Caractéristiques Avancée (MobileNetV2)** : Extraction de vecteurs descripteurs de dimension $1280$ issus de l'avant-dernière couche de MobileNetV2 (pré-entraîné sur ImageNet).
4. **Descripteur de Recours (Color/Texture Histogram)** : Si PyTorch n'est pas disponible, l'application bascule automatiquement sur un descripteur hybride Couleur (Histogramme R, G, B de 24 bins) et Texture Spatiale (pixels flous de 256 dimensions), offrant une recherche visuelle basée sur les teintes et motifs avec une latence quasi nulle ($<5$ ms).
5. **Résultats Triés par Score** : Affichage dynamique sous forme de grille responsive des 8 vêtements les plus similaires avec affichage du pourcentage de ressemblance et de la catégorie.

---

## 🛠️ Stack Technique

- **Framework Web** : Django 6.0+
- **Deep Learning (Feature Extraction)** : PyTorch & Torchvision (MobileNetV2)
- **Computer Vision Classique** : Pillow (PIL), NumPy (RGB Histograms)
- **Calcul Linéaire / Similarité** : Scikit-Learn (Cosine Similarity)
- **Style frontend** : Tailwind CSS, Outfit Typography

---

## 📦 Installation et Lancement Local

### Prerequisites
- Python 3.10+
- Pip & Virtualenv
- Utilitaire `curl` (pour le téléchargement rapide du dataset)

### Étapes d'installation

1. **Activer l'environnement virtuel et installer les dépendances** :
   ```bash
   source ../venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Télécharger le jeu de données d'images de vêtements** :
   Utilisez le script de téléchargement parallèle haute performance pour récupérer un sous-ensemble de 150 images du dataset vestimentaire :
   ```bash
   python ../fast_clothing_download.py
   ```

3. **Générer l'index de caractéristiques (Indexation)** :
   Exécutez le script d'indexation pour extraire les vecteurs caractéristiques de toutes les images du catalogue et générer l'artéfact `visual_search_assets.pkl` :
   ```bash
   python train_visual.py
   ```

4. **Appliquer les migrations de base de données** :
   ```bash
   python manage.py migrate
   ```

5. **Lancer le serveur de développement** :
   ```bash
   python manage.py runserver 0.0.0.0:8003
   ```

Accédez à l'application sur [http://localhost:8003/](http://localhost:8003/).

---

## 📂 Structure du Projet

```
├── visual_project/          # Configuration Django et Vues de recherche vectorielle
│   ├── settings.py          # Inclut le mappage static vers le dossier des images
│   ├── urls.py
│   └── views.py             # Logique d'indexation, d'extraction de vecteur et de calcul cosinus
├── dataset/                 # Contient le dataset d'images (clothing-dataset/images/)
├── models/                  # Index de recherche vectorielle sérialisé (visual_search_assets.pkl)
├── static/                  # Fichiers statiques
├── templates/
│   └── index.html           # Interface utilisateur interactive (Upload zone et Grille de résultats)
├── train_visual.py          # Script d'indexation (MobileNetV2 / Color-Spatial histogram fallback)
├── visual_search.ipynb      # Notebook Jupyter d'EDA et de tracé d'histogrammes couleur
└── manage.py
```
