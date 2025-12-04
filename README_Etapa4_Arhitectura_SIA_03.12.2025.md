# 📘 README – Etapa 4: Arhitectura Completă a Aplicației SIA bazată pe Rețele Neuronale

**Disciplina:** Rețele Neuronale  
**Instituție:** POLITEHNICA București – FIIR  
**Student:** Marinescu Robert-Antonio  
**Grupa:** 633AB  
**Data:** 04.12.2025  

---

## Scopul Etapei 4
Livrarea unui SCHELET COMPLET și FUNCȚIONAL al întregului Sistem cu Inteligență Artificială (SIA).

---

## 1. Tabelul Nevoie Reală → Soluție SIA → Modul Software

| **Nevoie reală concretă** | **Cum o rezolvă SIA-ul vostru** | **Modul software responsabil** |
|---------------------------|--------------------------------|--------------------------------|
| Asistență la conducere pentru evitarea accidentelor prin nerespectarea semnelor | Clasificare imagine cameră bord → alertă vizuală șofer în < 100ms | Modul Data Acquisition + RN + UI |
| Adaptarea automată a vitezei autovehiculului la limita legală | Identificare semn limită de viteză → transmitere comandă (simulată) reducere viteză | Modul RN Inference + Control Logic |
| Detectarea semnelor în condiții meteo adverse (ploaie/ceață) | Antrenare pe date sintetice augmentate → Acuratețe > 90% în condiții de zgomot | Modul Data Augmentation (Acquisition) |

---

## 2. Contribuția Originală la Setul de Date

### Contribuția originală la setul de date:

**Total observații finale:** ~65,000 (după Etapa 3 + Etapa 4)
**Observații originale:** ~25,000 (~40%)

**Tipul contribuției:**
[X] Date sintetice prin metode avansate (Simulare condiții meteo și de mediu)
[ ] Date achiziționate cu senzori proprii
[ ] Etichetare/adnotare manuală

**Descriere detaliată:**
Am preluat setul de date GTSRB (German Traffic Sign Recognition Benchmark) și am dezvoltat un modul de generare de date sintetice. Deoarece un vehicul autonom întâlnește semne în condiții variabile, am implementat algoritmi de procesare de imagine pentru a simula:
1. **Condiții de iluminare scăzută** (simulare condus noaptea).
2. **Zgomot de senzor și blur de mișcare** (simulare viteză mare sau cameră ieftină).
3. **Obstrucții parțiale și rotații** (simulare unghiuri diferite de vizualizare).

Aceste date noi sunt generate procedural și adăugate la setul de antrenare pentru a crește robustețea modelului.

**Locația codului:** `src/data_acquisition/synthetic_generator.py`
**Locația datelor:** `data/generated/`

---

## 3. Diagrama State Machine a Întregului Sistem

Diagrama se găsește în: `docs/state_machine.png`

### Justificarea State Machine-ului ales:

Am ales o arhitectură de tip **Monitorizare Continuă cu Feedback Vizual**, specifică sistemelor de asistență pentru șoferi (ADAS).

Stările principale sunt:
1. **IDLE / WAIT_CAMERA:** Așteptarea inițializării camerei video.
2. **ACQUIRE_FRAME:** Captura unui cadru video în timp real (30 FPS).
3. **PREPROCESS_ROI:** Decuparea zonei de interes și redimensionarea la 30x30 pixeli.
4. **INFERENCE:** Rularea modelului CNN pentru a obține clasa și probabilitatea.
5. **VALIDATE:** Verificarea dacă probabilitatea depășește pragul de siguranță (ex: 75%).

Starea **ERROR** este critică deoarece conexiunea cu camera se poate pierde sau modelul poate fi neîncărcat corect, caz în care aplicația trebuie să notifice utilizatorul fără a se bloca.

---

## 4. Structura Modulelor

| **Modul** | **Implementare** | **Status** |
|-----------|------------------|------------|
| **1. Data Logging / Acquisition** | `src/data_acquisition/synthetic_generator.py` | Funcțional. Generează imagini augmentate (blur, noise). |
| **2. Neural Network Module** | `src/neural_network/architecture.py` | Funcțional. Modelul CNN este definit și compilat. |
| **3. Web Service / UI** | `src/app/gui_app.py` | Funcțional. Interfață grafică Tkinter pentru încărcare și predicție. |