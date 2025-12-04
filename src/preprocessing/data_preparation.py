import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from pathlib import Path

# --- CONFIGURARE ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = BASE_DIR / 'data' / 'raw'
PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
IMG_HEIGHT = 30
IMG_WIDTH = 30
NUM_CLASSES = 43

# Creăm folderul processed dacă nu există
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
(BASE_DIR / 'docs').mkdir(parents=True, exist_ok=True)

def load_and_preprocess():
    print(">>> [1/4] Incepem incarcarea datelor brute...")
    data = []
    labels = []
    
    # Calea catre folderul Train
    train_path = RAW_DATA_DIR / 'Train'
    
    # Verificam daca folderul exista
    if not train_path.exists():
        print(f"EROARE: Nu gasesc folderul {train_path}")
        print("Verifica daca ai mutat folderul 'Train' (cel cu 0, 1, 2...) in data/raw/")
        return

    # Contor pentru grafic
    class_counts = {}

    for i in range(NUM_CLASSES):
        path = train_path / str(i)
        if not path.exists():
            continue
            
        images = os.listdir(path)
        class_counts[i] = len(images)
        
        for img_name in images:
            try:
                # Incarcare si redimensionare
                image = Image.open(path / img_name)
                image = image.resize((IMG_WIDTH, IMG_HEIGHT))
                image = np.array(image)
                
                data.append(image)
                labels.append(i)
            except Exception as e:
                print(f"Eroare la imaginea: {img_name}")

    data = np.array(data)
    labels = np.array(labels)
    
    print(f"    Total imagini incarcate: {data.shape[0]}")

    # --- Salvare Grafic Distributie ---
    print(">>> [2/4] Generare grafic (EDA)...")
    plt.figure(figsize=(10, 5))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title("Distributia imaginilor per clasa")
    plt.xlabel("ID Clasa")
    plt.ylabel("Numar Imagini")
    plt.savefig(BASE_DIR / 'docs' / 'distributie_clase.png')
    print("    Grafic salvat in docs/")

    # --- Preprocesare ---
    print(">>> [3/4] Normalizare si Split...")
    # Normalizare (0-255 -> 0-1)
    X = data.astype('float32') / 255.0
    y = labels

    # Split: 80% Train, 20% Validare
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # --- Salvare ---
    print(">>> [4/4] Salvare date procesate...")
    np.save(PROCESSED_DATA_DIR / 'X_train.npy', X_train)
    np.save(PROCESSED_DATA_DIR / 'y_train.npy', y_train)
    np.save(PROCESSED_DATA_DIR / 'X_val.npy', X_val)
    np.save(PROCESSED_DATA_DIR / 'y_val.npy', y_val)
    
    print("SUCCES! Datele sunt gata.")

if __name__ == "__main__":
    load_and_preprocess()