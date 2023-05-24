import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

# Einlesen und Vorbereiten der Trainingsdaten

train_dir = os.path.join("data", "Train")
meta_dir = os.path.join("data", "Meta")
csv_dir = os.path.join("data")

# Bilddaten einlesen
images = []
labels = []

for label in tqdm(range(43)):
    label_dir = os.path.join(train_dir, str(label))
    for img_path in os.listdir(label_dir):
        img = cv2.imread(os.path.join(label_dir, img_path))
        img = cv2.resize(img, (32, 32))  # Größe anpassen, falls erforderlich
        images.append(img)
        labels.append(label)

# Labels einlesen
label_images = []
for label in tqdm(range(43)):
    img = cv2.imread(os.path.join(meta_dir, str(label) + '.png'))
    img = cv2.resize(img, (32, 32))  # Größe anpassen, falls erforderlich
    label_images.append(img)

# Daten in NumPy-Arrays konvertieren
images = np.array(images)
labels = np.array(labels)
label_images = np.array(label_images)

# Modellarchitektur definieren

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(43, activation='softmax')
])

# Modell kompilieren und trainieren

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(images, labels, epochs=20, batch_size=32, validation_split=0.2)

# Modell auf Testdaten testen und Bildnamen mit vorhergesagtem Label ausgeben lassen

test_dir = os.path.join("Data-Exploration", "data", "Test")
test_csv = os.path.join(csv_dir, "test.csv")

# Testdaten einlesen
test_images = []
test_labels = []

test_df = pd.read_csv(test_csv)
test_paths = test_df['Path'].values

for img_path in tqdm(test_paths):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (32, 32))  # Größe anpassen, falls erforderlich
    test_images.append(img)

# Daten in NumPy-Array konvertieren
test_images = np.array(test_images)

# Vorhersagen mit dem Modell machen
predictions = model.predict(test_images)

# Labels aus den Vorhersagen extrahieren
predicted_labels = np.argmax(predictions, axis=1)

# Ergebnisse ausgeben
for i, img_path in enumerate(test_paths):
    true_label = test_df.loc[i, 'ClassId']
    print("Bild:", img_path)
    print("Vorhergesagtes Label:", predicted_labels[i])
    print("Wahres Label:", true_label)
    print("")

# Genauigkeit auf den Testdaten berechnen
test_labels = test_df['ClassId'].values
accuracy = np.mean(predicted_labels == test_labels)
print("Genauigkeit auf Testdaten:", accuracy)
