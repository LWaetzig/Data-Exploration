import pandas as pd 
import os
import numpy as np
import cv2 #function for image editing
from sklearn.model_selection import train_test_split
import random

#import the data
path = 'data/Train'
classes = os.listdir(path)
data = []
labels = []

#label the data
for i, c in enumerate(classes):
    class_path = os.path.join(path, c)
    for img_file in os.listdir(class_path):
        try:
            img = cv2.imread(os.path.join(class_path, img_file))
            img = cv2.resize(img, (32, 32,))
            img_arr = np.array(img)
            data.append([img_arr, i])
#            labels.append(i)
        except:
            print(f"Error {img_file}")

random.shuffle(data)
print(len(data))

f = []
l = []

for features,label in data:
    f.append(features)
    l.append(label)

f = np.array(f)
l = np.array(l)

train_F, val_F, train_L, val_L = train_test_split(f, l, test_size=0.2, random_state=43)

print('Trainingsdaten:', train_F.shape)
print('Trainingslabels:', val_F.shape)
print('validationdata:', train_L.shape)
print('validationlabels:', val_L.shape)

train_F = train_F/255.0
val_F = val_F/255.0

print("Shape of train_images is:", train_F.shape)
print("Shape of labels is:", train_L.shape)

import tensorflow as tf
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
# Annahme: train_F sind deine Trainingsdaten (RGB-Bilder)
# Annahme: train_L sind deine Trainingslabels (One-Hot-kodierte Klassen)

# Datenvorbereitung
train_F_gray = np.dot(train_F[..., :3], [0.2989, 0.5870, 0.1140])
train_F_gray = train_F_gray / 255.0
train_F_gray = train_F_gray.reshape(-1, 32, 32, 1)

# Anzahl der Klassen (Verkehrsschilder) in deinem Datensatz
num_classes = 43


# Definition des CNN-Modells
model = tf.keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))

# Kompilieren des Modells
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Anzahl der Epochen für das Training
epochs = 20

# Training des Modells
history = model.fit(train_F_gray, train_L, batch_size=32, epochs=epochs, validation_split=0.2)

# Anzeige der Genauigkeit und des Verlusts
plt.figure(0)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.figure(1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()