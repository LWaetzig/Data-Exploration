import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# Pfad zum Datenordner
data_dir = 'data'
train_dir = os.path.join(data_dir, 'Train')
test_dir = os.path.join(data_dir, 'Test')
meta_dir = os.path.join(data_dir, 'Meta')

# CSV-Dateien einlesen
train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))

# Bildgröße
image_size = (32, 32)

# Funktion zum Laden der Bilder
def load_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, image_size)
    img = img / 255.0  # Normalisierung der Pixelwerte auf den Bereich [0, 1]
    return img

# Funktion zum Laden der Trainingsdaten
def load_train_data(df):
    images = []
    labels = []
    for i, row in df.iterrows():
        image_path = os.path.join(row['Path'])
        image = load_image(image_path)
        label = row['ClassId']
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)

# Funktion zum Laden der Testdaten
def load_test_data(df):
    images = []
    labels = []
    for i, row in df.iterrows():
        image_path = os.path.join(row['Path'])
        image = load_image(image_path)
        label = row['ClassId']
        images.append(image)
        labels.append(label)
    return np.array(images), np.array(labels)

# Trainingsdaten laden
train_images, train_labels = load_train_data(train_df)

# Testdaten laden
test_images, test_labels = load_test_data(test_df)

# Anzahl der Klassen
num_classes = len(np.unique(train_labels))

# Modell definieren
model = tf.keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_classes, activation='softmax')
])

# Modell kompilieren
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Modell trainieren
model.fit(train_images, train_labels, epochs=10, batch_size=32)

# Modell auf Testdaten evaluieren
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')

# Vorhersagen für Testdaten
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Vergleich der Vorhersagen mit den wahren Labels
correct_predictions = np.sum(predicted_labels == test_labels)
total_predictions = len(test_labels)
accuracy = correct_predictions / total_predictions
print(f'Accuracy on test data: {accuracy}')
