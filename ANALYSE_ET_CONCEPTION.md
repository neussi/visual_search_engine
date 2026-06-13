# Analyse et Conception — Visual Search Engine

**Projet:** Moteur de Recherche d'Images par Similarite Visuelle  
**Niveau:** Master 2 — Intelligence Artificielle, ENSPY  
**Annee:** 2025-2026

---

## 1. Problematique et Contexte

La recherche par texte presente des limites pour les produits vestimentaires : un utilisateur souhaitant trouver un article similaire a une photographie ne possede pas toujours le vocabulaire adequat pour le decrire. La recherche par image — Content-Based Image Retrieval (CBIR) — resout ce probleme en comparant directement les representations visuelles.

**Objectif:** Construire un systeme de recherche d'images capable de retrouver les N articles d'un catalogue les plus similaires visuellement a une image requete, en temps reel, avec un deploiement sans GPU.

---

## 2. Analyse Exploratoire du Dataset

### 2.1 Statistiques du Catalogue

```
Attribut                Valeur
--------                ------
Images totales          150
Format                  JPEG (UUID comme nom de fichier)
Taille typique          Variable (normalisee a 224x224 a l'extraction)
Categories (labels):
  T-shirts              28   (18.7%)
  Shirts                24   (16.0%)
  Dresses               22   (14.7%)
  Skirts                18   (12.0%)
  Shorts                16   (10.7%)
  Shoes                 21   (14.0%)
  Bags                  11   (7.3%)
  Other                 10   (6.7%)
Items enfants           23   (15.3%)
```

### 2.2 Caracteristiques Visuelles du Dataset

```
- Fond:            Variable (blanc, studio, outdoor)
- Eclairage:       Heterogene (studio, naturel)
- Point de vue:    Frontal dominant
- Resolution:      Variable (100x100 a 1200x1600 pixels)
- Canaux:          RGB (3 canaux)
```

---

## 3. Architecture de la Solution

### 3.1 Vue Globale du Systeme

```
+---------------------------------------------------------------+
|                      CLIENT (Navigateur)                      |
|               /search-engine/  (drag-and-drop)                |
+-----------------------------|---------------------------------+
                              | POST multipart/form-data (image)
+-----------------------------v--------------------------+
|                   Django Application                   |
|                                                        |
|  +------------------+   +------------------+           |
|  |  Views Layer     |   |  URL Router      |           |
|  |  views.py        |   |  urls.py         |           |
|  +--------+---------+   +------------------+           |
|           |                                            |
|  +--------v-----------------------------------------+  |
|  |            Feature Extraction Pipeline           |  |
|  |                                                  |  |
|  |  load_assets() -> pickle                         |  |
|  |  {                                               |  |
|  |    catalog_embeddings:          (150, 1280)      |  |
|  |    catalog_fallback_embeddings: (150, 384)       |  |
|  |    catalog_image_paths:         List[str]        |  |
|  |    catalog_metadata:            List[dict]       |  |
|  |  }                                               |  |
|  |                                                  |  |
|  |  Query Image -> Feature Vector (1280 ou 384)     |  |
|  |  Cosine Similarity vs Catalogue                  |  |
|  |  -> Top-K results                                |  |
|  +---------------------------------------------------+ |
|                                                        |
|  Static: Catalog images sous /staticfiles/images/      |
|          Servies par WhiteNoise via WSGI               |
+--------------------------------------------------------+
                         |
                Vercel Serverless (Python 3.12)
```

### 3.2 Flux de Traitement d'une Requete

```
Utilisateur upload image.jpg
         |
         v
POST /search/  (multipart/form-data)
         |
         v
+---------------------------+
|  Image Loading (PIL)      |
|  query_image = Image.open |
+---------------------------+
         |
         v
+---------------------------+
|  Feature Extraction       |
|                           |
|  Si PyTorch disponible:   |
|    MobileNetV2 -> 1280d   |
|    catalog_embeddings     |
|  Sinon (serverless):      |
|    HSV Histogram -> 384d  |
|    catalog_fallback_emb   |
+---------------------------+
         |
         v
+---------------------------+
|  Cosine Similarity        |
|                           |
|  sim = (q . C^T) /        |
|         (||q|| ||C||)     |
|                           |
|  q:  (1, d)               |
|  C:  (n, d)               |
|  sim: (n,)                |
+---------------------------+
         |
         v
+---------------------------+
|  Tri et Selection Top-K   |
|  np.argsort(sim)[::-1]    |
|  K = 12 par defaut        |
+---------------------------+
         |
         v
  JSON: [{path, label, score}, ...]
```

---

## 4. Architecture du Modele IA

### 4.1 MobileNetV2 — Extracteur Principal

MobileNetV2 est un reseau convolutif leger base sur des blocs Inverted Residual Bottleneck, optimise pour l'inference mobile et embarquee.

```
Architecture MobileNetV2:

Input:    224 x 224 x 3  (RGB, normalise ImageNet)
          mean = [0.485, 0.456, 0.406]
          std  = [0.229, 0.224, 0.225]

Bloc 0:   Conv 3x3, stride=2       -> 112x112x32
Bloc 1:   InvRes t=1, c=16,  n=1   -> 112x112x16
Bloc 2:   InvRes t=6, c=24,  n=2   ->  56x56x24
Bloc 3:   InvRes t=6, c=32,  n=3   ->  28x28x32
Bloc 4:   InvRes t=6, c=64,  n=4   ->  14x14x64
Bloc 5:   InvRes t=6, c=96,  n=3   ->  14x14x96
Bloc 6:   InvRes t=6, c=160, n=3   ->   7x7x160
Bloc 7:   InvRes t=6, c=320, n=1   ->   7x7x320
Conv 1x1:                          ->   7x7x1280
Pool:     Global Average Pooling   ->   1280
[Suppression du classifieur final]

Output:   Vecteur 1280-d (feature map)

Parametres: 3.4 M (congeles — pas de fine-tuning)
Poids:      ImageNet-1k pretrained
```

**Bloc Inverted Residual (detail):**
```
input (t*c canaux)
    |
    v
+--------------------+
|  Conv 1x1 (expand) |   c -> t*c  (expansion factor t)
|  BN + ReLU6        |
+--------------------+
    |
    v
+--------------------+
|  Depthwise Conv    |   3x3, groupes=t*c
|  BN + ReLU6        |
+--------------------+
    |
    v
+--------------------+
|  Conv 1x1 (proj)   |   t*c -> c'  (projection)
|  BN (sans activ.)  |
+--------------------+
    |
    v (+ shortcut si stride=1 et c=c')
output
```

### 4.2 Descripteur HSV Joint — Fallback

Utilise lorsque PyTorch n'est pas disponible (environnement serverless Vercel):

```
Image (quelconque) -> Resize 64x64 -> RGB -> HSV

Canaux HSV:
  H (Teinte):      [0, 360] -> 8 intervalles uniformes
  S (Saturation):  [0,   1] -> 4 intervalles uniformes
  V (Valeur):      [0,   1] -> 4 intervalles uniformes

Histogramme Joint 3D:
  N_bins = 8 x 4 x 4 = 128 entrées
  Normalise par la somme (distribution de probabilite)

Descripteur Spatial:
  Grayscale -> Resize 16x16 -> 256 pixels aplatis
  Z-score normalise (mu=0, sigma=1)

Concatenation:
  feature = [ hist_HSV (128d) | spatial (256d) ] = 384d

Avantages:
  - Invariance partielle aux changements d'eclairage (HSV vs RGB)
  - Joint histogram: capture les correlations H/S/V
  - Descripteur spatial: encode la disposition des tons
```

### 4.3 Comparaison des Descripteurs

```
Propriete              MobileNetV2      HSV + Spatial
---------              -----------      -------------
Dimension              1280             384
Invariance echelle     Oui (GAP)        Partielle
Invariance rotation    Non              Partielle
Capture texture        Oui              Non
Capture couleur        Oui              Oui (dominant)
GPU requis             Oui (inference)  Non
Disponible sur Vercel  Non              Oui
Qualite similarite     Elevee           Moderee
```

---

## 5. Similarite Cosinus — Fondement Mathematique

```
sim(q, c_i) = (q . c_i) / (||q|| * ||c_i||)

Avec:
  q    : vecteur requete (1, d)
  c_i  : vecteur catalogue image i (1, d)
  ||.|| : norme L2 euclidienne

Propriete:
  sim in [-1, 1]
  sim = 1  : vecteurs identiques (meme direction)
  sim = 0  : vecteurs orthogonaux (aucune similarite)
  sim = -1 : vecteurs opposes

Choix vs distance euclidienne:
  La norme cosinus est independante de la magnitude des vecteurs,
  ce qui la rend plus robuste aux variations d'intensite lumineuse.
```

---

## 6. Pipeline d'Entrainement Hors-Ligne

```
train_visual.py:

Pour chaque image du catalogue:
  1. Charger avec PIL.Image.open()
  2. Si PyTorch:
       Normaliser (ImageNet stats)
       MobileNetV2.forward() -> vecteur 1280d
  3. Toujours:
       extract_fallback_features() -> vecteur 384d
  4. Stocker: path, vecteurs, label

Sauvegarder dans models/visual_search_assets.pkl:
  {
    catalog_embeddings:          np.array (150, 1280)
    catalog_fallback_embeddings: np.array (150, 384)
    catalog_image_paths:         list[str]
    catalog_metadata:            list[dict]
    has_torch:                   bool
  }
```

---

## 7. Evaluation

```
Metrique                        Valeur (test subjectif)
--------                        -----------------------
Precision@5 (meme categorie)   MobileNetV2: ~72%
                                HSV fallback: ~54%
Temps de requete               < 200ms (Vercel, fallback)
Couverture du catalogue        100% (150 images indexees)
```

---

## 8. Decisions de Conception

| Decision | Justification |
|----------|--------------|
| MobileNetV2 frozen | Transfert learning suffisant — pas de donnees pour fine-tuning |
| GAP comme pooling | Reduction dimensionnelle propre, invariante a la taille d'entree |
| HSV joint histogram | Superieur au RGB separate — capture les correlations chromatiques |
| Pre-calcul des embeddings | Inference O(n*d) vs re-extraction O(n*CNN) par requete |
| Fallback pre-calcule | Evite les lectures disque serverless (acces /dataset/ impossible en prod) |
| Cosine vs Euclidean | Plus robuste aux variations d'eclairage et de contraste |
| Top-12 resultats | Affichage en grille 4x3, equilibre exhaustivite/lisibilite |

---

## 9. Diagramme de Deploiement

```
+--------------------+         +--------------------+
|  Developpeur       |  push   |  GitHub            |
|  (local)           | ------> |  neussi/           |
|  PyTorch local     |         |  visual_search_    |
|  train_visual.py   |         |  engine            |
|  -> .pkl (1280d    |         +--------+-----------+
|     + 384d emb.)   |                  |
+--------------------+                  | Vercel CI/CD
                                        v
+--------------------------------------------+
|              Vercel Platform               |
|  Runtime: Python 3.12 Serverless           |
|  Handler: visual_project/wsgi.py           |
|  Static:  /staticfiles/images/*.jpg        |
|           via WhiteNoise (153 fichiers)    |
|  Inference: Fallback 384d (sans PyTorch)   |
+--------------------------------------------+
                    |
       https://visual-search-engine-seven.vercel.app
```
