import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tqdm import tqdm

# Pfad zu den Datenordnern
data_folder = os.path.join("data")
train_folder = os.path.join(data_folder, "Train")
meta_folder = os.path.join(data_folder, "Meta")
train_csv_file = os.path.join(data_folder, "train.csv")

# Lade die Trainingsdaten aus den Bildern und CSV-Datei
def load_train_data():
    train_images = []
    train_labels = []

    # Lese die CSV-Datei mit den Trainingsinformationen
    train_df = pd.read_csv(train_csv_file)

    # Iteriere über jeden Ordner im Trainingsordner
    for folder_name in tqdm(os.listdir(train_folder)):
        folder_path = os.path.join(train_folder, folder_name)
        if not os.path.isdir(folder_path):
            continue

        class_id = int(folder_name)
        # Iteriere über jedes Bild im Ordner
        for image_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image_name)
            # Lade das Bild mit OpenCV
            image = cv2.imread(image_path)
            # Füge das Bild und die Label zur Liste hinzu
            train_images.append(image)
            train_labels.append(class_id)

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    return train_images, train_labels

# Lade die Trainingsdaten
train_images, train_labels = load_train_data()

# Aufteilung der Daten in Trainings- und Validierungssets
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.2)

# Normalisiere die Bilder
train_images = train_images / 255.0
val_images = val_images / 255.0

# Definiere das Modell
model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(43, activation='softmax')
])

# Kompilieren des Modells
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Trainiere das Modell
model.fit(train_images, train_labels, epochs=10, batch_size=32, validation_data=(val_images, val_labels))
