import os
import cv2 as cv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt


train_path = os.path.join("data" , "Train")
train_data = list()
train_labels = list()

for dir in os.listdir(train_path):
    for file in os.listdir(os.path.join(train_path, dir)):
        img = cv.imread(os.path.join(train_path, dir, file))
        img = cv.resize(img, (32, 32))
        img = img.astype(np.float32)
        img = img / 255
        train_data.append(img)
        train_labels.append(int(dir))


train_data = np.array(train_data)
train_data = train_data.reshape(-1, 32*32*3)

kmeans = KMeans(n_clusters=43, random_state=0).fit(train_data)
pred = kmeans.predict(train_data)

print(accuracy_score(train_labels, pred))