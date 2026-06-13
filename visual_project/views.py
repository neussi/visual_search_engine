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
    # PIL Image to RGB
    img_rgb = img.convert('RGB')
    
    # 1. Joint HSV Color Histogram
    img_small = img_rgb.resize((64, 64))
    pixels = np.array(img_small) / 255.0
    r, g, b = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
    
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    df = mx - mn
    
    # Hue
    h = np.zeros_like(r)
    idx = (mx == r) & (df != 0)
    h[idx] = (60 * ((g[idx] - b[idx]) / df[idx]) + 360) % 360
    idx = (mx == g) & (df != 0)
    h[idx] = (60 * ((b[idx] - r[idx]) / df[idx]) + 120) % 360
    idx = (mx == b) & (df != 0)
    h[idx] = (60 * ((r[idx] - g[idx]) / df[idx]) + 240) % 360
    
    # Saturation
    s = np.zeros_like(r)
    idx = mx != 0
    s[idx] = df[idx] / mx[idx]
    
    # Value
    v = mx
    
    # Joint Bins: 8 Hue, 4 Saturation, 4 Value
    h_bins = np.digitize(h, np.linspace(0, 360, 9)) - 1
    s_bins = np.digitize(s, np.linspace(0, 1.0, 5)) - 1
    v_bins = np.digitize(v, np.linspace(0, 1.0, 5)) - 1
    
    joint_indices = h_bins * 16 + s_bins * 4 + v_bins
    joint_indices = np.clip(joint_indices, 0, 127)
    
    h_joint, _ = np.histogram(joint_indices, bins=128, range=(0, 128))
    h_joint = h_joint / (np.sum(h_joint) + 1e-8)
    
    # 2. Spatial Layout: Grayscale 16x16
    img_gray = img_rgb.convert('L')
    img_gray_small = img_gray.resize((16, 16))
    gray_pixels = np.array(img_gray_small).flatten() / 255.0
    gray_pixels = gray_pixels - np.mean(gray_pixels)
    std = np.std(gray_pixels)
    if std > 0:
        gray_pixels = gray_pixels / std
        
    return np.concatenate([h_joint, gray_pixels])

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

def home(request):
    return render(request, 'home.html')

def search_view(request):
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
    return render(request, 'search.html', {'samples': samples})

def analytics_view(request):
    return render(request, 'analytics.html')

def contact_view(request):
    return render(request, 'contact.html')

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
            catalog_embeddings = assets['catalog_embeddings']
        else:
            query_vector = extract_fallback_features(query_image)
            catalog_embeddings = assets.get('catalog_fallback_embeddings', assets['catalog_embeddings'])
            
        catalog_image_paths = assets['catalog_image_paths']
        catalog_metadata = assets['catalog_metadata']

        # Fallback if dimensions still mismatch
        if catalog_embeddings.shape[1] != len(query_vector):
            fallback_embeddings = []
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for path in catalog_image_paths:
                full_p = os.path.join(base_dir, "dataset", "clothing-dataset", path)
                try:
                    with Image.open(full_p) as img:
                        fallback_embeddings.append(extract_fallback_features(img))
                except Exception:
                    fallback_embeddings.append(np.zeros(len(query_vector)))
            catalog_embeddings = np.array(fallback_embeddings)
            assets['catalog_fallback_embeddings'] = catalog_embeddings  # cache it
            
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

from django.core.mail import send_mail
from django.conf import settings

@csrf_exempt
def contact(request):
    if request.method == 'POST':
        name = request.POST.get('name', '')
        email = request.POST.get('email', '')
        subject = request.POST.get('subject', '')
        message = request.POST.get('message', '')
        
        if not name or not email or not message:
            return JsonResponse({'error': 'Veuillez remplir tous les champs obligatoires.'}, status=400)
            
        full_message = f"Message de {name} ({email}) :\n\n{message}"
        try:
            send_mail(
                subject=f"[Contact Plateforme] {subject or 'Nouveau Message'}",
                message=full_message,
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=['npe.techs@gmail.com'],
                fail_silently=False,
            )
            return JsonResponse({'success': True})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)
            
    return JsonResponse({'error': 'Méthode non autorisée'}, status=405)

