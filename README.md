# Visual Search Engine — Content-Based Image Retrieval

A production-ready visual search platform using deep feature extraction (MobileNetV2) and cosine similarity to retrieve clothing items visually similar to a query image, deployed on Vercel with a premium Django interface.

**Production URL:** https://visual-search-engine-seven.vercel.app  
**Repository:** https://github.com/neussi/visual_search_engine

---

## Platform Overview

The engine encodes catalog images into compact feature vectors using MobileNetV2 (a lightweight CNN pretrained on ImageNet). At query time, the uploaded image is projected into the same feature space and cosine similarity is computed against all catalog embeddings to retrieve the Top-K most visually similar garments.

A high-precision fallback descriptor based on joint HSV color histograms and spatial layout is automatically activated when the PyTorch runtime is unavailable (serverless environments).

| Route | Section | Description |
|-------|---------|-------------|
| `/` | Home | Platform overview and feature explanation |
| `/search-engine/` | Moteur de Recherche | Drag-and-drop image upload + similarity results |
| `/analytics/` | Analytique | Category distribution, feature space analysis |
| `/contact/` | Contact | SMTP contact form (Gmail backend) |
| `/search/` | API | JSON endpoint for image similarity search |

---

## Project Structure

```
visual_search_engine/
|
+-- visual_project/             Django project configuration
|   +-- settings.py             Application settings (WhiteNoise, SMTP, CORS)
|   +-- urls.py                 URL routing
|   +-- views.py                View functions, feature extraction, similarity search
|   +-- wsgi.py                 WSGI entry point (Vercel serverless)
|   +-- asgi.py                 ASGI entry point
|
+-- templates/
|   +-- base.html               Master layout (Tailwind CSS, Outfit font, MathJax)
|   +-- home.html               Landing page with feature highlights
|   +-- search.html             Search interface with drag-and-drop upload
|   +-- analytics.html          Dataset analysis and model diagnostics
|   +-- contact.html            Contact form with AJAX submission
|
+-- static/
|   +-- images/                 Pre-generated analysis plots (PNG)
|
+-- staticfiles/                Collected static assets (WhiteNoise, Vercel)
|   +-- images/                 153 clothing catalog images served as static assets
|
+-- dataset/
|   +-- clothing-dataset/
|       +-- images/             150 clothing catalog images (JPEG, UUID filenames)
|       +-- images.csv          Image metadata (label, kids flag)
|
+-- models/
|   +-- visual_search_assets.pkl    Catalog embeddings, fallback embeddings, metadata
|
+-- docs/
|   +-- images/                 High-resolution plots for documentation
|
+-- train_visual.py             Offline feature extraction and embedding pipeline
+-- visual_search.ipynb         Full Jupyter analysis and training notebook
+-- requirements.txt
+-- vercel.json
+-- manage.py
+-- .gitignore
```

---

## AI Model Architecture

### Primary Pipeline: MobileNetV2 + Cosine Similarity

```
Query Image (uploaded by user)
          |
          v
+------------------------------+
|  MobileNetV2 Backbone        |
|  Input: 224 x 224 x 3        |
|  Pretrained: ImageNet-1k     |
|                              |
|  Inverted Residuals:         |
|   Block 1 :  32 filters      |
|   Block 2 :  16 filters      |
|   Block 3 :  24 filters  x2  |
|   Block 4 :  32 filters  x3  |
|   Block 5 :  64 filters  x4  |
|   Block 6 :  96 filters  x3  |
|   Block 7 : 160 filters  x3  |
|   Block 8 : 320 filters  x1  |
|  Conv 1x1 : 1280 filters     |
|  Global Average Pooling      |
|  Output: 1280-dim vector     |
+------------------------------+
          |
          v
  Query Feature Vector q in R^1280
          |
          v
+------------------------------+
|  Cosine Similarity           |
|                              |
|        q . c_i               |
|  s_i = ---------             |
|        ||q|| ||c_i||         |
|                              |
|  for each catalog vector c_i |
+------------------------------+
          |
          v
  Ranked Results (Top-K by s_i)
```

### Fallback Pipeline: Joint HSV Histogram + Spatial Descriptor

Activated automatically when PyTorch is unavailable (serverless runtime):

```
Input Image
    |
    v
+--------------------------------------+
|  Resize to 64 x 64                   |
|  Convert to HSV color space          |
|                                      |
|  Hue bins:        8  channels        |
|  Saturation bins: 4  channels        |
|  Value bins:      4  channels        |
|  Joint 3D histogram: 8x4x4 = 128 d   |
+--------------------------------------+
    |
    v
+--------------------------------------+
|  Grayscale Spatial Layout            |
|  Resize to 16 x 16 = 256 pixels      |
|  Z-score normalization               |
+--------------------------------------+
    |
    v
  Concatenate -> 384-dim descriptor
    |
    v
  Cosine Similarity against catalog
```

### Training Complexity

| Parameter | Value |
|-----------|-------|
| Catalog size | 150 images |
| Embedding dimension (MobileNetV2) | 1280 |
| Embedding dimension (Fallback) | 384 |
| Similarity metric | Cosine similarity |
| Query time complexity | O(n * d) |
| Preprocessing: MobileNetV2 | ImageNet mean/std normalization |
| Model parameters (frozen backbone) | 3.4M |

---

## Dataset

**Source:** Clothing Dataset (Kaggle)  
**Images:** 150 labeled clothing photographs (JPEG)  
**Categories:** T-shirts, shirts, dresses, skirts, shorts, shoes, bags  
**Image size:** Variable (normalized to 224x224 during feature extraction)  
**Metadata:** `images.csv` with UUID filename, category label, and kids flag

---

## Feature Extraction Pipeline

Embeddings are pre-computed offline and stored in `models/visual_search_assets.pkl`:

```python
# Asset structure
{
    "catalog_embeddings":          np.ndarray (n, 1280)  # MobileNetV2 vectors
    "catalog_fallback_embeddings": np.ndarray (n, 384)   # HSV + spatial vectors
    "catalog_image_paths":         List[str]             # relative image paths
    "catalog_metadata":            List[dict]            # label, kids flag
    "has_torch":                   bool                  # whether PyTorch was used
}
```

---

## Local Development

```bash
git clone https://github.com/neussi/visual_search_engine.git
cd visual_search_engine

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run feature extraction (requires PyTorch locally for MobileNetV2)
python train_visual.py

python manage.py runserver
```

---

## Deployment

Deployed on **Vercel** via `@vercel/python`. WhiteNoise serves all static and catalog images through the WSGI handler. Pre-computed fallback embeddings ensure zero-dependency inference in the serverless runtime.

| Variable | Description |
|----------|-------------|
| `EMAIL_HOST_PASSWORD` | Gmail App Password for SMTP |

---

## Dependencies

| Package | Role |
|---------|------|
| `django>=5.0` | Web framework |
| `numpy` | Vector arithmetic and cosine similarity |
| `pandas` | Dataset metadata loading |
| `pillow` | Image loading and preprocessing |
| `scikit-learn` | Cosine similarity computation |
| `whitenoise` | Static file serving |
| `django-cors-headers` | Cross-origin request handling |

---

## Contact

**Institution:** Ecole Nationale Superieure Polytechnique de Yaounde (ENSPY)  
**Level:** AIA4 - Intelligence Artificielle  
**Contact:** npe.techs@gmail.com | +237 650 970 526
