import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from pathlib import Path

# --- CONFIGURARE ---
# Încărcăm modelul antrenat
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'traffic_classifier.h5'

print("Se incarca modelul... (poate dura cateva secunde)")
model = load_model(MODEL_PATH)
print("Model incarcat!")

# Dicționarul claselor (trebuie să fie identic cu cel din gui_app)
classes = { 
    0:'Limita de viteza (20km/h)',
    1:'Limita de viteza (30km/h)', 
    2:'Limita de viteza (50km/h)', 
    3:'Limita de viteza (60km/h)', 
    4:'Limita de viteza (70km/h)', 
    5:'Limita de viteza (80km/h)', 
    6:'Sfarsit limita (80km/h)', 
    7:'Limita de viteza (100km/h)', 
    8:'Limita de viteza (120km/h)', 
    9:'Depasirea interzisa', 
    10:'Depasirea interzisa pt camioane', 
    11:'Prioritate la urmatoarea intersectie', 
    12:'Drum cu prioritate', 
    13:'Cedeaza trecerea', 
    14:'Stop', 
    15:'Accesul interzis', 
    16:'Accesul interzis camioanelor', 
    17:'Accesul interzis', 
    18:'Atentie! Pericol', 
    19:'Curba periculoasa la stanga', 
    20:'Curba periculoasa la dreapta', 
    21:'Curbe duble', 
    22:'Drum cu denivelari', 
    23:'Drum alunecos', 
    24:'Drum ingustat pe dreapta', 
    25:'Lucrari', 
    26:'Semafor', 
    27:'Pietoni', 
    28:'Copii / Scoala', 
    29:'Biciclisti', 
    30:'Gheata / Zapada',
    31:'Animale salbatice', 
    32:'Sfarsit restrictii', 
    33:'Obligatoriu la dreapta', 
    34:'Obligatoriu la stanga', 
    35:'Obligatoriu inainte', 
    36:'Obligatoriu inainte sau la dreapta', 
    37:'Obligatoriu inainte sau la stanga', 
    38:'Obligatoriu la dreapta', 
    39:'Obligatoriu la stanga', 
    40:'Sens giratoriu', 
    41:'Sfarsit interzicere depasire', 
    42:'Sfarsit interzicere depasire camioane' 
}

def preprocessing(img):
    # Transformăm imaginea exact cum am făcut la antrenare
    img = cv2.resize(img, (30, 30))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Modelul a invatat pe RGB, OpenCV citeste BGR
    img = img / 255.0  # Normalizare
    img = np.expand_dims(img, axis=0)
    return img

# Pornim camera (0 este de obicei camera laptopului)
cap = cv2.VideoCapture(0)
cap.set(3, 640) # Latime fereastra
cap.set(4, 480) # Inaltime fereastra

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    success, frame = cap.read()
    if not success:
        break

    # --- ZONA DE INTERES (ROI) ---
    # Desenăm un pătrat unde utilizatorul trebuie să pună semnul
    # Coordonate: (x1, y1) si (x2, y2)
    cv2.rectangle(frame, (200, 100), (450, 350), (0, 255, 0), 2)
    cv2.putText(frame, "PUNE SEMNUL IN PATRAT", (180, 90), font, 0.7, (0, 255, 0), 2)

    # Decupăm doar ce este în pătrat pentru analiză
    crop_img = frame[100:350, 200:450]

    try:
        # Procesăm imaginea decupată
        img_processed = preprocessing(crop_img)
        
        # Facem predicția
        prediction = model.predict(img_processed, verbose=0)
        class_index = np.argmax(prediction)
        probability_value = np.amax(prediction)

        # Afișăm rezultatul DOAR dacă e sigur peste 75%
        # (Altfel arată erori când e peretele gol)
        if probability_value > 0.75:
            text_rezultat = f"{classes[class_index]} ({round(probability_value*100, 2)}%)"
            cv2.putText(frame, text_rezultat, (50, 420), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, "DETECTAT:", (50, 390), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(frame, "...", (50, 420), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

    except Exception as e:
        pass

    # Afișăm fereastra
    cv2.imshow("Detectie Live Semne Circulatie", frame)

    # Dacă apeși tasta 'q', se închide
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()