import os
import pickle
import numpy as np
import pandas as pd
from PIL import Image

try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def extract_fallback_features(img):
    # PIL Image to RGB
    img_rgb = img.convert('RGB')
    img_small = img_rgb.resize((64, 64))
    pixels = np.array(img_small)
    
    # 8-bin histograms for R, G, B
    r_hist, _ = np.histogram(pixels[:, :, 0], bins=8, range=(0, 256))
    g_hist, _ = np.histogram(pixels[:, :, 1], bins=8, range=(0, 256))
    b_hist, _ = np.histogram(pixels[:, :, 2], bins=8, range=(0, 256))
    
    # Normalize
    r_hist = r_hist / (np.sum(r_hist) + 1e-8)
    g_hist = g_hist / (np.sum(g_hist) + 1e-8)
    b_hist = b_hist / (np.sum(b_hist) + 1e-8)
    
    color_hist = np.concatenate([r_hist, g_hist, b_hist])
    
    # Spatial layout (grayscale downsampled)
    img_gray = img_rgb.convert('L')
    img_gray_small = img_gray.resize((16, 16))
    gray_pixels = np.array(img_gray_small).flatten() / 255.0
    gray_pixels = gray_pixels - np.mean(gray_pixels)
    std = np.std(gray_pixels)
    if std > 0:
        gray_pixels = gray_pixels / std
        
    return np.concatenate([color_hist, gray_pixels])

def main():
    device = None
    mobilenet = None
    transform = None
    
    if HAS_TORCH:
        print("Initializing MobileNetV2 for feature extraction...")
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        mobilenet.classifier = torch.nn.Identity()
        mobilenet.eval()
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mobilenet.to(device)
    else:
        print("PyTorch not installed. Using Color Histogram + Spatial Layout features as fallback...")
        
    dataset_dir = "dataset/clothing-dataset"
    csv_path = os.path.join(dataset_dir, "images.csv")
    
    if not os.path.exists(csv_path):
        print(f"Error: dataset CSV not found at {csv_path}. Please download the clothing dataset first.")
        return
        
    df = pd.read_csv(csv_path)
    
    catalog_embeddings = []
    catalog_image_paths = []
    catalog_metadata = []
    
    print(f"Extracting embeddings for {len(df)} images...")
    for idx, row in df.iterrows():
        img_filename = row['image'] + ".jpg" if not row['image'].endswith('.jpg') else row['image']
        img_path = os.path.join(dataset_dir, "images", img_filename)
        
        if os.path.exists(img_path):
            try:
                img = Image.open(img_path)
                if HAS_TORCH:
                    img_rgb = img.convert('RGB')
                    img_tensor = transform(img_rgb).unsqueeze(0).to(device)
                    with torch.no_grad():
                        vector = mobilenet(img_tensor).squeeze().cpu().numpy()
                else:
                    vector = extract_fallback_features(img)
                    
                catalog_embeddings.append(vector)
                catalog_image_paths.append(os.path.join("images", img_filename))
                catalog_metadata.append({
                    'label': row['label'],
                    'sender_id': row.get('sender_id', 'unknown'),
                    'kids': row.get('kids', False)
                })
            except Exception as e:
                print(f"Error processing {img_filename}: {e}")
                
    catalog_embeddings = np.array(catalog_embeddings)
    print(f"Successfully processed {len(catalog_embeddings)} images.")
    
    # Save assets
    os.makedirs("models", exist_ok=True)
    assets = {
        "catalog_embeddings": catalog_embeddings,
        "catalog_image_paths": catalog_image_paths,
        "catalog_metadata": catalog_metadata,
        "has_torch": HAS_TORCH
    }
    
    with open("models/visual_search_assets.pkl", "wb") as f:
        pickle.dump(assets, f)
    print("Visual Search assets saved successfully to models/visual_search_assets.pkl!")

if __name__ == "__main__":
    main()
