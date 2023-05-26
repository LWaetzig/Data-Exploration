import os
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical


# create model
def create_model(input_shape, num_classes):
    # CNN für die Merkmalsextraktion
    inputs = tf.keras.Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation="relu")(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation="relu")(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)

    # Klassifikationsausgabe
    classification_output = Dense(64, activation="relu")(x)
    classification_output = Dense(
        num_classes, activation="softmax", name="classification_output"
    )(classification_output)

    # ROI-Regression
    roi_output = Dense(4, name="roi_output")(x)

    # Gesamtmodell erstellen
    model = tf.keras.Model(inputs=inputs, outputs=[classification_output, roi_output])

    return model


input_shape = (32, 32, 3)
num_classes = 43
model = create_model(input_shape, num_classes)

# model.summary()
# plot_model(model, to_file='model.png', show_shapes=True)


def preprocess_images(image):
    image = cv.resize(image, (32, 32))
    image = image / 255.0
    return image


train_meta = pd.read_csv(os.path.join("data", "train.csv"), index_col=0)
val_meta = pd.read_csv(os.path.join("data", "validation.csv"), index_col=0)

X_train = list()
y_train = list()
y_train_roi = list()

for i in range(len(train_meta)):
    img = cv.imread(train_meta.iloc[i]["Path"])
    img = preprocess_images(img)

    X_train.append(img)
    y_train.append(train_meta.iloc[i]["ClassId"])
    y_train_roi.append(
        np.column_stack(
            (
                train_meta.iloc[i]["Roi.X1"],
                train_meta.iloc[i]["Roi.Y1"],
                train_meta.iloc[i]["Roi.X2"],
                train_meta.iloc[i]["Roi.Y2"],
            )
        )
    )


X_train = np.array(X_train)
y_train = np.array(y_train)
y_train_one_hot = to_categorical(y_train, num_classes=num_classes)


X_val = list()
y_val = list()
y_val_roi = list()

for i in range(len(val_meta)):
    img = cv.imread(val_meta.iloc[i]["Path"])
    img = preprocess_images(img)

    X_val.append(img)
    y_val.append(val_meta.iloc[i]["ClassId"])
    y_val_roi.append(
        np.column_stack(
            (
                val_meta.iloc[i]["Roi.X1"],
                val_meta.iloc[i]["Roi.Y1"],
                val_meta.iloc[i]["Roi.X2"],
                val_meta.iloc[i]["Roi.Y2"],
            )
        )
    )

X_val = np.array(X_val)
y_val = np.array(y_val)
y_val_one_hot = to_categorical(y_val, num_classes=num_classes)


model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss={"classification_output": "categorical_crossentropy", "roi_output": "mse"},
    metrics={"classification_output": "accuracy", "roi_output": "mse"},
)

epochs = 20
batch_size = 32

y_train_roi = np.array(y_train_roi)
y_train_roi = np.repeat(y_train_roi, len(X_train), axis=0)
# y_val_roi = np.array(y_val_roi)
# y_val_roi = np.repeat(y_val_roi, len(X_val), axis=0)


model.fit(
    X_train,
    {"classification_output": y_train_one_hot, "roi_output": y_train_roi},
    validation_data=(X_val, {"classification_output": y_val_one_hot, "roi_output": y_val_roi}),
    epochs=epochs,
    batch_size=batch_size,
)

print(model)