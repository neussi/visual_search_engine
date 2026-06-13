import os
import pickle
import numpy as np
from PIL import Image
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from sklearn.metrics.pairwise import cosine_similarity

try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Load index and feature extractor
ASSETS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'visual_search_assets.pkl')
visual_assets = None
feature_extractor = None
transform = None

if HAS_TORCH:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def extract_fallback_features(img):
    img_rgb = img.convert('RGB')
    img_small = img_rgb.resize((64, 64))
    pixels = np.array(img_small)
    
    r_hist, _ = np.histogram(pixels[:, :, 0], bins=8, range=(0, 256))
    g_hist, _ = np.histogram(pixels[:, :, 1], bins=8, range=(0, 256))
    b_hist, _ = np.histogram(pixels[:, :, 2], bins=8, range=(0, 256))
    
    r_hist = r_hist / (np.sum(r_hist) + 1e-8)
    g_hist = g_hist / (np.sum(g_hist) + 1e-8)
    b_hist = b_hist / (np.sum(b_hist) + 1e-8)
    
    color_hist = np.concatenate([r_hist, g_hist, b_hist])
    
    img_gray = img_rgb.convert('L')
    img_gray_small = img_gray.resize((16, 16))
    gray_pixels = np.array(img_gray_small).flatten() / 255.0
    gray_pixels = gray_pixels - np.mean(gray_pixels)
    std = np.std(gray_pixels)
    if std > 0:
        gray_pixels = gray_pixels / std
        
    return np.concatenate([color_hist, gray_pixels])

def load_assets():
    global visual_assets, feature_extractor
    if visual_assets is None and os.path.exists(ASSETS_PATH):
        with open(ASSETS_PATH, 'rb') as f:
            visual_assets = pickle.load(f)
            
    if HAS_TORCH and feature_extractor is None:
        try:
            mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            mobilenet.classifier = torch.nn.Identity()
            mobilenet.eval()
            feature_extractor = mobilenet
        except Exception as e:
            print(f"Error loading PyTorch MobileNet: {e}")
            
    return visual_assets

def index(request):
    assets = load_assets()
    samples = []
    if assets:
        np.random.seed(42)
        indices = np.random.choice(len(assets['catalog_image_paths']), min(12, len(assets['catalog_image_paths'])), replace=False)
        for idx in indices:
            samples.append({
                'path': assets['catalog_image_paths'][idx],
                'label': assets['catalog_metadata'][idx]['label']
            })
    return render(request, 'index.html', {'samples': samples})

@csrf_exempt
def search(request):
    assets = load_assets()
    if not assets:
        return JsonResponse({'error': 'Index non chargé ou modèle indisponible. Exécutez le script d\'entraînement.'}, status=500)
        
    query_image = None
    if request.method == 'POST' and request.FILES.get('query_file'):
        query_image = Image.open(request.FILES['query_file'])
    elif request.GET.get('sample_path'):
        sample_rel = request.GET.get('sample_path')
        full_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', 'clothing-dataset', sample_rel)
        if os.path.exists(full_path):
            query_image = Image.open(full_path)
            
    if query_image is None:
        return JsonResponse({'error': 'Image de requête introuvable'}, status=400)
        
    try:
        # Check if assets were trained with PyTorch or fallback
        assets_have_torch = assets.get('has_torch', False)
        
        if HAS_TORCH and assets_have_torch and feature_extractor is not None:
            img_rgb = query_image.convert('RGB')
            img_tensor = transform(img_rgb).unsqueeze(0)
            with torch.no_grad():
                query_vector = feature_extractor(img_tensor).squeeze().numpy()
        else:
            query_vector = extract_fallback_features(query_image)
            
        catalog_embeddings = assets['catalog_embeddings']
        catalog_image_paths = assets['catalog_image_paths']
        catalog_metadata = assets['catalog_metadata']
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector.reshape(1, -1), catalog_embeddings).flatten()
        
        # Get top-8 matches
        top_k = min(8, len(similarities))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'image_path': '/static/' + catalog_image_paths[idx],
                'similarity': float(similarities[idx]),
                'label': catalog_metadata[idx]['label']
            })
            
        return JsonResponse({'results': results})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
