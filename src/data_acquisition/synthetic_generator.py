import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import random

# Configurare Căi
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw' / 'Train'
GENERATED_DIR = BASE_DIR / 'data' / 'generated'

# Creăm folderul pentru date generate
GENERATED_DIR.mkdir(parents=True, exist_ok=True)

def augment_image(image_path):
    """
    Funcție care simulează condiții de mediu:
    - Întuneric (Noapte)
    - Blur (Mișcare)
    - Zgomot (Senzor slab)
    """
    try:
        img = Image.open(image_path)
        
        # 1. Simulare variație luminozitate (Zi/Noapte)
        enhancer = ImageEnhance.Brightness(img)
        factor = random.uniform(0.3, 1.2) # Întunecat sau normal
        img = enhancer.enhance(factor)
        
        # 2. Simulare blur (Mișcare sau ploaie)
        if random.random() > 0.5:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5)))
            
        # 3. Simulare rotație ușoară (Unghi cameră)
        angle = random.uniform(-10, 10)
        img = img.rotate(angle)
        
        return img
    except Exception as e:
        print(f"Eroare procesare {image_path}: {e}")
        return None

def generate_synthetic_dataset(samples_needed=100):
    """
    Generează un set de date sintetice pentru a atinge cerința de 40% originalitate.
    """
    print(f">>> Generare {samples_needed} imagini sintetice în {GENERATED_DIR}...")
    
    # Luăm clase la întâmplare
    classes = os.listdir(RAW_DATA_DIR)
    
    count = 0
    while count < samples_needed:
        # Alegem o clasă random și o imagine random din ea
        cls = random.choice(classes)
        cls_path = RAW_DATA_DIR / cls
        
        if not cls_path.exists(): continue
        
        images = os.listdir(cls_path)
        if not images: continue
        
        img_name = random.choice(images)
        original_img_path = cls_path / img_name
        
        # Augmentăm
        new_img = augment_image(original_img_path)
        
        if new_img:
            # Salvăm în folderul generated, păstrând structura claselor
            save_folder = GENERATED_DIR / cls
            save_folder.mkdir(exist_ok=True)
            
            save_path = save_folder / f"synth_{count}_{img_name}"
            new_img.save(save_path)
            count += 1
            
            if count % 50 == 0:
                print(f"    Generat: {count}/{samples_needed}")

    print(">>> Generare completă! Datele originale sunt în data/generated/")

if __name__ == "__main__":
    # Pentru testare rapidă generăm doar 100. Pentru proiect real am genera mii.
    generate_synthetic_dataset(samples_needed=100)