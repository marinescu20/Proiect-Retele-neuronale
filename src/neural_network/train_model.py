import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from pathlib import Path
import os

# --- CĂI ȘI CONFIGURĂRI ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'models'
DOCS_DIR = BASE_DIR / 'docs'

# Creăm folderele necesare dacă nu există
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)

def build_and_train():
    print(">>> [1/3] Incarcare date procesate...")
    try:
        X_train = np.load(PROCESSED_DIR / 'X_train.npy')
        y_train = np.load(PROCESSED_DIR / 'y_train.npy')
        X_val = np.load(PROCESSED_DIR / 'X_val.npy')
        y_val = np.load(PROCESSED_DIR / 'y_val.npy')
    except FileNotFoundError:
        print("EROARE: Nu gasesc fisierele .npy! Ruleaza intai src/preprocessing/data_preparation.py")
        return

    # One-hot encoding
    y_train = to_categorical(y_train, 43)
    y_val = to_categorical(y_val, 43)

    print(f"    Date incarcate. Train: {X_train.shape}, Val: {X_val.shape}")

    # --- DEFINIREA MODELULUI (CNN) ---
    print(">>> [2/3] Construire si antrenare model CNN...")
    model = Sequential()
    
    # Strat 1
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    # Strat 2
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    # Strat 3
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))

    # Compilare
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Antrenare (15 epoci)
    epochs = 15
    history = model.fit(X_train, y_train, batch_size=64, epochs=epochs, validation_data=(X_val, y_val))

    # --- SALVARE ---
    print(">>> [3/3] Salvare model si grafice...")
    
    model.save(MODELS_DIR / 'traffic_classifier.h5')
    print(f"    Model salvat in: models/traffic_classifier.h5")

    # Grafice
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Antrenare')
    plt.plot(history.history['val_accuracy'], label='Validare')
    plt.title('Acuratete')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Antrenare')
    plt.plot(history.history['val_loss'], label='Validare')
    plt.title('Eroare (Loss)')
    plt.legend()

    plt.savefig(DOCS_DIR / 'training_performance.png')
    print("    Grafice salvate in docs/")

if __name__ == "__main__":
    build_and_train()