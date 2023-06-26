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

#save model
model.save('model.test5')

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

path = r'C:\AP\Studium\DataExploration\Data-Exploration\data\randomPic\stop.jpg'
#path = r'C:\AP\Studium\DataExploration\Data-Exploration\data\Test\00002.png'

img = Image.open(path)
img = img.resize((32, 32))
plt.imshow(img)
x = np.expand_dims(np.array(img), axis=0)

from keras.models import load_model
model = load_model('model.test5')

value = classes = np.argmax(model.predict(x, batch_size=32), axis=-1) #predict the label for the image

if classes[0]==0:
    print('Speed limit (20km/h)') #print the content
elif classes[0]==1:
      print('Speed limit (30km/h)') #print the content
elif classes[0]==2:
      print('Speed limit (50km/h)') #print the content
elif classes[0]==3:
      print(' Speed limit (60km/h)') #print the content
elif classes[0]==4:
      print('Speed limit (70km/h)') #print the content
elif classes[0]==5:
      print('Speed limit (80km/h)') #print the content
elif classes[0]==6:
      print('End of speed limit (80km/h)') #print the content
elif classes[0]==7:
      print('Speed limit (100km/h)') #print the content
elif classes[0]==8:
      print('Speed limit (120km/h)') #print the conten
elif classes[0]==9:
      print('No passing') #print the content     
elif classes[0]==10:
      print('No passing veh over 3.5 tons') #print the content       
elif classes[0]==11:
      print(', Right-of-way at intersection') #print the content     
elif classes[0]==12:
      print('Priority road') #print the content     
elif classes[0]==13:
      print('Yield') #print the content             
elif classes[0]==14:
      print('Stop') #print the content             
elif classes[0]==15:
      print('No vehicles') #print the content                
elif classes[0]==16:
      print('Veh > 3.5 tons prohibited') #print the content              
elif classes[0]==17:
      print('No entry') #print the content                             
elif classes[0]==18:
      print('General caution') #print the content                     
elif classes[0]==19:
      print('Dangerous curve left') #print the content              
elif classes[0]==20:
      print('Dangerous curve right') #print the content              
elif classes[0]==21:
      print('Double curve') #print the content             
elif classes[0]==22:
      print('Bumpy road') #print the content                    
elif classes[0]==23:
      print('Slippery road') #print the content             
elif classes[0]==24:
      print('Road narrows on the right') #print the content                     
elif classes[0]==25:
      print('Road work') #print the content              
elif classes[0]==26:
      print('Traffic signals') #print the content       
elif classes[0]==27:
      print('Pedestrians') #print the content                      
elif classes[0]==28:
      print('Children crossing') #print the content        
elif classes[0]==29:
      print( 'Bicycles crossing') #print the content               
elif classes[0]==30:
      print('Beware of ice/snow') #print the content              
elif classes[0]==31:
      print('Wild animals crossing') #print the content                      
elif classes[0]==32:
      print('End speed + passing limits') #print the content               
elif classes[0]==33:
      print('Turn right ahead') #print the content       
elif classes[0]==34:
      print('Turn left ahead') #print the content                
elif classes[0]==35:
      print('Ahead only') #print the content            
elif classes[0]==36:
      print('Go straight or right') #print the content  
elif classes[0]==37:
      print('Go straight or left') #print the content             
elif classes[0]==38:
      print('Keep right') #print the content       
elif classes[0]==39:
      print('Keep left') #print the content              
elif classes[0]==40:
      print('Roundabout mandatory') #print the content
elif classes[0]==41:
      print('End of no passing') #print the content               
else:
      print('End no passing veh > 3.5 tons') #print the content