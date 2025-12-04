import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy as np
from tensorflow.keras.models import load_model
from pathlib import Path

# Configurăm calea către model

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / 'models' / 'traffic_classifier.h5'

# Încărcăm modelul antrenat
print("Se încarcă modelul... Așteptați.")
try:
    model = load_model(MODEL_PATH)
    print("Model încărcat cu succes!")
except Exception as e:
    print(f"EROARE CRITICĂ: Nu pot încărca modelul din {MODEL_PATH}")
    print("Asigură-te că ai rulat 'train_model.py' înainte!")
    exit()

# Dicționarul cu numele semnelor (în engleză sau română, cum preferi)
classes = { 
    1:'Limită de viteză (20km/h)',
    2:'Limită de viteză (30km/h)', 
    3:'Limită de viteză (50km/h)', 
    4:'Limită de viteză (60km/h)', 
    5:'Limită de viteză (70km/h)', 
    6:'Limită de viteză (80km/h)', 
    7:'Sfârșit limită (80km/h)', 
    8:'Limită de viteză (100km/h)', 
    9:'Limită de viteză (120km/h)', 
    10:'Depășirea interzisă', 
    11:'Depășirea interzisă pt camioane', 
    12:'Prioritate la următoarea intersecție', 
    13:'Drum cu prioritate', 
    14:'Cedează trecerea', 
    15:'Stop', 
    16:'Accesul interzis', 
    17:'Accesul interzis camioanelor', 
    18:'Accesul interzis', 
    19:'Atenție! Pericol', 
    20:'Curbă periculoasă la stânga', 
    21:'Curbă periculoasă la dreapta', 
    22:'Curbe duble', 
    23:'Drum cu denivelări', 
    24:'Drum alunecos', 
    25:'Drum îngustat pe dreapta', 
    26:'Lucrări', 
    27:'Semafor', 
    28:'Pietoni', 
    29:'Copii / Școală', 
    30:'Bicicliști', 
    31:'Gheață / Zăpadă',
    32:'Animale sălbatice', 
    33:'Sfârșit restricții', 
    34:'Obligatoriu la dreapta', 
    35:'Obligatoriu la stânga', 
    36:'Obligatoriu înainte', 
    37:'Obligatoriu înainte sau la dreapta', 
    38:'Obligatoriu înainte sau la stânga', 
    39:'Obligatoriu la dreapta', 
    40:'Obligatoriu la stânga', 
    41:'Sens giratoriu', 
    42:'Sfârșit interzicere depășire', 
    43:'Sfârșit interzicere depășire camioane' 
}

# Inițializare GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Recunoașterea Semnelor de Circulație')
top.configure(background='#CDCDCD')

label = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((30,30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    
    # Predicție
    pred_probs = model.predict([image])[0]
    pred_class = np.argmax(pred_probs)
    sign = classes[pred_class + 1]
    
    print(f"Detectat: {sign}")
    label.configure(foreground='#011638', text=sign) 

def show_classify_button(file_path):
    classify_b = Button(top, text="Verifică Imaginea", command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)
        
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload = Button(top, text="Încarcă o imagine", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Verifică Semnul Rutier", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()

top.mainloop()