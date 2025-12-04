from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

def build_model(input_shape=(30, 30, 3), num_classes=43):
    """
    Definește arhitectura CNN pentru Traffic Sign Recognition.
    Această funcție returnează un model compilat, gata de antrenare.
    """
    model = Sequential()
    
    # Bloc 1: Extragere trăsături
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    # Bloc 2: Extragere trăsături complexe
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))

    # Bloc 3: Clasificare
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compilare
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    # Test simplu că arhitectura se compilează
    model = build_model()
    model.summary()
    print(">>> Model definit și compilat cu succes (SCHELET ETAPA 4).")