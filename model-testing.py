import os
import cv2
import numpy as np
import pandas as pd
from tensorflow import keras



# Ordnerstruktur bereitstellen
data_dir = 'data'
test_dir = os.path.join(data_dir, 'Test')
train_dir = os.path.join(data_dir, 'Train')


# CSV-Dateien einlesen
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))


# Bildgröße für Bildvorverarbeitung festlegen
image_size = (32, 32)


# Pfad zur gespeicherten Modelldatei
model_path = os.path.join('saved-models', 'final_model.h5')


# Funktionen zum Laden aller Daten definieren
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


# Trainingsdaten laden
train_images, train_labels = load_data(train_df)


# Testdaten laden
test_images, test_labels = load_data(test_df)


# Modell laden
model = keras.models.load_model(model_path)


# Modell auf Testdaten evaluieren
train_loss, train_acc = model.evaluate(train_images, train_labels, verbose=2)
print(f'Trainings accuracy: {train_acc}')


# Modell auf Testdaten evaluieren
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')