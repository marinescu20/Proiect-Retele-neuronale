# 📘 README – Etapa 3: Analiza și Pregătirea Setului de Date pentru Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Marinescu Robert-Antonio  
**Grupa:** 633AB  
**Data:** [Pune Data Curentă]  

---

## Introducere

Acest document descrie activitățile realizate în **Etapa 3**, în care s-a analizat setul de date GTSRB (German Traffic Sign Recognition Benchmark) și s-au realizat procedurile de preprocesare necesare pentru antrenarea rețelei neuronale convoluționale (CNN).

---

##  1. Structura Repository-ului Github (versiunea Etapei 3)

```text
Proiect_Semne_RN/
├── README.md
├── docs/
│   └── distributie_clase.png  # Graficul generat în urma analizei
├── data/
│   ├── raw/               # Conține folderele 'Train', 'Test' și fișierele CSV originale
│   ├── processed/         # Conține fișierele .npy (X_train.npy, y_train.npy etc.)
│   ├── train/             # (Virtual) Gestionat intern prin split în memorie
│   ├── validation/        # (Virtual) Gestionat intern prin split în memorie
│   └── test/              # Imaginile de test brute
├── src/
│   ├── preprocessing/     
│   │   └── data_preparation.py  # Scriptul care face EDA și Preprocesarea
│   └── neural_network/    # (Urmează în etapa 4)
└── requirements.txt       # tensorflow, pandas, numpy, matplotlib, pillow

---

##  2. Descrierea Setului de Date

### 2.1 Sursa datelor

* **Origine:** GTSRB - German Traffic Sign Recognition Benchmark (Kaggle).
* **Modul de achiziție:**  Fișier extern (Descărcare publică arhiva .zip).
* **Perioada / condițiile colectării:** Imagini reale capturate în Germania în diverse condiții de iluminare și vreme.

### 2.2 Caracteristicile dataset-ului

* **Număr total de observații:** ~39,209 imagini.
* **Număr de caracteristici (features):** 43 de clase distincte (semne de circulație).
* **Tipuri de date:** Imagini Color (RGB).
* **Format fișiere:**  .PNG (imagini) și .CSV (metadate).

### 2.3 Descrierea fiecărei caracteristici

Caracteristică,Tip,Unitate,Descriere,Domeniu valori
Pixel Height,numeric,px,Înălțimea imaginii,Variabil (15px - 250px)
Pixel Width,numeric,px,Lățimea imaginii,Variabil (15px - 250px)
Canale culoare,numeric,-,"RGB (Red, Green, Blue)",3 canale
Class ID,categorial,-,ID-ul semnului rutier,0 - 42

---

##  3. Analiza Exploratorie a Datelor (EDA) – Sintetic

### 3.1 Statistici descriptive aplicate

Distribuția pe clase: S-a generat un grafic de tip Bar Chart pentru a vizualiza câte imagini există pentru fiecare semn.

Rezultat: S-a observat un dezechilibru (Class Imbalance). Unele semne (ex: Limită de viteză) au ~2000 imagini, iar altele (ex: Pericol) au sub 200.

### 3.2 Analiza calității datelor

Valori lipsă: Nu există pixeli lipsă, dar s-au verificat fișiere corupte la încărcarea cu biblioteca Pillow.

Dimensiuni variabile: Imaginile brute au dimensiuni diferite, ceea ce necesită redimensionare obligatorie.

### 3.3 Probleme identificate

Variabilitate: Dimensiunile imaginilor nu sunt uniforme.

Soluție: Toate imaginile vor fi redimensionate la 30x30 pixeli.

---

##  4. Preprocesarea Datelor

### 4.1 Curățarea datelor

Tratarea erorilor: Scriptul data_preparation.py folosește un bloc try-except pentru a sări peste fișierele care nu pot fi deschise (imagini corupte).

### 4.2 Transformarea caracteristicilor

Redimensionare: Toate imaginile au fost aduse la rezoluția 30x30 pixeli.

Normalizare: Valorile pixelilor (0-255) au fost împărțite la 255.0 pentru a obține valori în intervalul [0, 1]. Acest pas este crucial pentru convergența rapidă a Rețelei Neuronale.

Encoding: Etichetele (Labels 0-42) au fost păstrate numeric pentru antrenare, urmând a fi transformate în One-Hot Encoding în etapa de antrenare.

### 4.3 Structurarea seturilor de date

**Împărțire recomandată:**
* 70–80% – train
* 10–15% – validation
* 10–15% – test

Împărțire realizată:

80% Antrenare (Train)

20% Validare (Validation)

S-a folosit funcția train_test_split cu parametru de stratificare (stratify=y) pentru a păstra proporțiile claselor în ambele seturi.

**Principii respectate:**
* Stratificare pentru clasificare
* Fără scurgere de informație (data leakage)
* Statistici calculate DOAR pe train și aplicate pe celelalte seturi

### 4.4 Salvarea rezultatelor preprocesării

Datele au fost serializate și salvate în format binar NumPy (.npy) pentru încărcare rapidă în etapa următoare:

data/processed/X_train.npy

data/processed/y_train.npy

data/processed/X_val.npy

data/processed/y_val.npy

---

##  5. Fișiere Generate în Această Etapă

src/preprocessing/data_preparation.py – Codul sursă Python.

data/processed/*.npy – Matricele numerice gata de intrare în rețea.

docs/distributie_clase.png – Graficul distribuției datelor.

---

##  6. Stare Etapă (de completat de student)

[x] Structură repository configurată

[x] Dataset analizat (EDA realizată - grafic generat)

[x] Date preprocesate (Resize, Normalize)

[x] Seturi train/val generate și salvate

[x] Documentație actualizată

---
