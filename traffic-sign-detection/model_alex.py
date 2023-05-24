import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import random
import tensorflow as tf
from keras import layers

# read out the data
path = "data\Train"  # path to the train-data
classes = os.listdir(path)
data = []
labels = []

# reading the input data
for i, c in enumerate(classes):
    class_path = os.path.join(path, c)
    for img_file in os.listdir(class_path):
        try:
            img = cv2.imread(os.path.join(class_path, img_file))
            img = cv2.resize(img, (32, 32))
            img_arr = np.array(img)
            data.append(img_arr)
            labels.append(i)
        except:
            print(f"Error {img_file}")

# convert to numpy arrays
data = np.array(data)
labels = np.array(labels)

# use one-hot encoding for the labels
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()
labels = label_binarizer.fit_transform(labels)

# split the data into training and validation data
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.2, random_state=43)

# normalize the data
train_data = train_data/255.0
val_data = val_data/255.0

# print the shapes of the data
print('Train-data:', train_data.shape)
print('Validation-data:', val_data.shape)
print('Train-labels:', train_labels.shape)
print('Validation-labels:', val_labels.shape)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

# create the model
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(43, activation='softmax'))

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train the model
model.fit(train_data, train_labels, batch_size=32, epochs=10, validation_data=(val_data, val_labels))


# prediction with the test data
from sklearn.metrics import accuracy_score
import pandas as pd 
from PIL import Image
import numpy as np  

y_test = pd.read_csv('data/test.csv')

labels = y_test["ClassId"].values
imgs = y_test["Path"].values

data=[]

for img in imgs:
    image = Image.open(img)
    image = image.resize((32,32))
    data.append(np.array(image))

X_test=np.array(data)

pred = np.argmax(model.predict(X_test), axis=-1)

#Accuracy with the test data
from sklearn.metrics import accuracy_score
print(accuracy_score(labels, pred))
