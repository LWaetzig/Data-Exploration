import os
import cv2
import numpy as np
from tensorflow import keras







# Funktionen zum Laden aller Daten definieren

# Bildgröße für Bildvorverarbeitung festlegen
image_size = (32, 32)

# Funktion zum Laden und Vorverarbeiten der Bilder
def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Einlesen des Bildes in Graustufen (alternativ das ", cv2.IMREAD_GRAYSCALE" entfernen)
    img = cv2.resize(img, image_size)
    img = img / 255.0  # Normalisierung der Pixelwerte auf den Bereich [0, 1]
    return img


# Funktion zum Einlesen der Bilder und deren Labels
def load_data(df):
    images = []
    labels = []
    for i, row in df.iterrows():
        path = row['Path']
        label = row['ClassId']
        image_path = os.path.join(path)
        image = load_image(image_path)
        images.append(image)
        labels.append(label)
        
    return np.array(images), np.array(labels)








# Funktion zur Modellerstellung (Die verschiedenen Layer festlegen und anschließend Modell testen um während des Trainings zu überwachen)
def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 1)), # Bei input_shape die letzte Zahl von 1 auf 3 ändern, sollten die Bilder NICHT in Graustufen eingelesen werden
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(43, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model
