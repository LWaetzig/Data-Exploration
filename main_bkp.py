import warnings

warnings.filterwarnings("ignore")

import math
import os
import random
import shutil

import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.initializers import he_normal, zeros, glorot_normal, RandomNormal
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.regularizers import l1, l2, l1_l2
from tqdm import tqdm

from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score
from keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import f1_score, precision_score, recall_score, auc
import datetime
import time

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

test = pd.read_csv("data/test.csv", index_col=0)
train = pd.read_csv("data/train.csv", index_col=0)
validation = pd.read_csv("data/validation.csv", index_col=0)

train_path = os.path.join("data", "Train")
val_path = os.path.join("data", "Validation")

# Set up variables

target_img_width = 30
target_img_height = 30
n_channels = 3
batch_size = 32
val_batch_size = 32
n_epochs = 200
class_names = list(range(43))
n_classes = len(class_names)


train_image_paths = train["Path"].values
val_iamge_paths = validation["Path"].values

train_labels = train["ClassId"].values
val_labels = validation["ClassId"].values

for i, row in train.iterrows():
    width = row["Width"]
    height = row["Height"]
    if width > target_img_width:
        widht_diff = width - target_img_width
        train.loc[i, "Roi.X2"] = train.loc[i, "Roi.X2"] - widht_diff
    else:
        widht_diff = target_img_width - width
        train.loc[i, "Roi.X2"] = train.loc[i, "Roi.X2"] + widht_diff

    if height > target_img_height:
        height_diff = height - target_img_height
        train.loc[i, "Roi.Y2"] = train.loc[i, "Roi.Y2"] - height_diff
    else:
        height_diff = target_img_height - height
        train.loc[i, "Roi.Y2"] = train.loc[i, "Roi.Y2"] + height_diff


for i, row in validation.iterrows():
    width = row["Width"]
    height = row["Height"]
    if width > target_img_width:
        widht_diff = width - target_img_width
        validation.loc[i, "Roi.X2"] = validation.loc[i, "Roi.X2"] - widht_diff
    else:
        widht_diff = target_img_width - width
        validation.loc[i, "Roi.X2"] = validation.loc[i, "Roi.X2"] + widht_diff

    if height > target_img_height:
        height_diff = height - target_img_height
        validation.loc[i, "Roi.Y2"] = validation.loc[i, "Roi.Y2"] - height_diff
    else:
        height_diff = target_img_height - height
        validation.loc[i, "Roi.Y2"] = validation.loc[i, "Roi.Y2"] + height_diff


def parse_function(filename, labels, df):
        """Function to preprocess the images"""
        # reading path
        image_string = tf.io.read_file(filename)
        # decoding image
        image = tf.image.decode_png(image_string, channels=n_channels)
        # This will convert to float values in [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Adjusting contrast and brightness of the image
        if tf.math.reduce_mean(image) < 0.3:
            image = tf.image.adjust_contrast(image, 5)
            image = tf.image.adjust_brightness(image, 0.2)
        # resize the image
        image = tf.image.resize(
            image,
            [target_img_height, target_img_width],
            method="nearest",
            preserve_aspect_ratio=False,
        )
        image = image / 255.0
        # one hot coding for label
        # y = tf.one_hot(tf.cast(label, tf.uint8), N_CLASSES)
        return image, {"classification": labels, "regression": df}


def tfdata_generator(images, labels, df, is_training, batch_size=32):
    """Construct a data generator using tf.Dataset"""

    ## creating a dataset from tensorslices
    dataset = tf.data.Dataset.from_tensor_slices((images, labels, df))
    if is_training:
        dataset = dataset.shuffle(30000)  # depends on sample size
    # Transform and batch data at the same time
    dataset = dataset.map(
        parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    # prefetch the data into CPU/GPU
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset



train = train.drop(['Height', 'Width', 'ClassId', 'Path'], axis=1)
validation = validation.drop(['Height', 'Width', 'ClassId', 'Path'], axis=1)

# Train and Validation data generators :
tf_image_generator_train = tfdata_generator(
    train_image_paths,
    train_labels,
    train,
    is_training=True,
    batch_size=32,
)
tf_image_generator_val = tfdata_generator(
    val_iamge_paths,
    val_labels,
    validation,
    is_training=False,
    batch_size=32,
)


class Sharpen(tf.keras.layers.Layer):
    """
    Sharpen layer sharpens the edges of the image.
    """
    def __init__(self, num_outputs) :
        super(Sharpen, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape) :
        self.kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        self.kernel = tf.expand_dims(self.kernel, 0)
        self.kernel = tf.expand_dims(self.kernel, 0)
        self.kernel = tf.cast(self.kernel, tf.float32)

    def call(self, input_) :
        return tf.nn.conv2d(input_, self.kernel, strides=[1, 1, 1, 1], padding='SAME')


def get_model() :
    #Input layer
    input_layer = Input(shape=(target_img_width, target_img_height, n_channels, ), name="input_layer", dtype='float32')
    #Sharpen Layer to sharpen the edges of the image.
    sharp = Sharpen(num_outputs=(target_img_width, target_img_height, n_channels, ))(input_layer)
    #Convolution, maxpool and dropout layers
    conv_1 = Conv2D(filters=32, kernel_size=(5,5), activation=relu,
                    kernel_initializer=he_normal(seed=54), bias_initializer=zeros(),
                    name="first_convolutional_layer") (sharp)
    conv_2 = Conv2D(filters=64, kernel_size=(3,3), activation=relu,
                    kernel_initializer=he_normal(seed=55), bias_initializer=zeros(),
                    name="second_convolutional_layer") (conv_1)                  
    maxpool_1 = MaxPool2D(pool_size=(2,2), name = "first_maxpool_layer")(conv_2)
    dr1 = Dropout(0.25)(maxpool_1)
    conv_3 = Conv2D(filters=64, kernel_size=(3,3), activation=relu,
                    kernel_initializer=he_normal(seed=56), bias_initializer=zeros(),
                    name="third_convolutional_layer") (dr1)
    maxpool_2 = MaxPool2D(pool_size=(2,2), name = "second_maxpool_layer")(conv_3)
    dr2 = Dropout(0.25)(maxpool_2) 
    flat = Flatten(name="flatten_layer")(dr2)

    #Fully connected layers
    d1 = Dense(units=256, activation=relu, kernel_initializer=he_normal(seed=45),
                bias_initializer=zeros(), name="first_dense_layer_classification", kernel_regularizer = l2(0.001))(flat)
    dr3 = Dropout(0.5)(d1)
    
    classification = Dense(units = 43, activation=None, name="classification",  kernel_regularizer = l2(0.0001))(dr3)
    
    regression = Dense(units = 4, activation = 'linear', name = "regression", 
                        kernel_initializer=RandomNormal(seed=43), kernel_regularizer = l2(0.1))(dr3)
    #Model
    model = Model(inputs = input_layer, outputs = [classification, regression])
    model.summary()
    return model



class Metrics(Callback) :
    """
    Custom callback to print the weighted F1-Score
    at the end of each training epoch.
    """
    def __init__(self, validation_data_generator) :
        self.validation_data_generator = validation_data_generator

    def on_train_begin(self, logs={}) :
        '''
        This function initializes lists to store AUC and Micro F1 scores
        '''
        self.val_f1s = []
        self.val_precisions = []
        self.val_recalls = []
        self.batches = self.validation_data_generator.as_numpy_iterator()

    def on_epoch_end(self, epoch, logs = {}) :
        '''
        This function calculates the micro f1 and auc scores
        at the end of each epochs
        '''
        current_batch = self.batches.next()
        images = current_batch[0]
        labels = current_batch[1]
        labels = labels["classification"]
        labels = np.array(labels)
        pred = self.model.predict(images)
        pred = pred[0]
        val_predict = (np.asarray(pred)).round()
        idx = np.argmax(val_predict, axis=-1)
        a = np.zeros( val_predict.shape )
        a[ np.arange(a.shape[0]), idx] = 1
        val_predict = [np.where(r==1)[0][0] for r in a]
        val_predict = np.array(val_predict)
        val_targ = labels
        _val_f1 = f1_score(val_targ, val_predict, average = 'weighted')
        _val_precision = precision_score(val_targ, val_predict, average='weighted')
        _val_recall = recall_score(val_targ, val_predict, average='weighted')
        print("\nEpoch : {0} -  Precision_Score : {1:.2f} - Recall_Score : {2:.2f} - F1_Score : {3:.2f}\n".format(epoch, _val_precision, _val_recall, _val_f1))
        self.val_f1s.append(_val_f1)
        self.val_precisions.append(_val_precision)
        self.val_recalls.append(_val_recall)
        return

#Defining loss functions for classification and regression
#Loss function for bounding box regression
def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#loss function for classification
loss = SparseCategoricalCrossentropy(from_logits=True)

#Compiling the model
model = get_model()
model.compile(optimizer="adam", loss = {"classification" : loss, "regression" : "mse"}, metrics={"classification" : "acc", "regression" : r2_keras}, loss_weights = {"classification" : 5, "regression" : 1})

#Callbacks
#Tensorboard callback
%load_ext tensorboard
log_dir="/content/drive/My Drive/CaseStudy2/logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images = True)
#ModelCheckpoint
NAME = "TrafficSignRecog-first-cut-{0}".format(int(time.time()))
save_best_model = ModelCheckpoint(filepath='/content/drive/My Drive/CaseStudy2/best_models/{0}'.format(NAME), monitor='val_loss', save_best_only = True, mode = 'min', save_freq = 'epoch')

#Early stopping to avoide model overfitting
early_stop = EarlyStopping(monitor='loss', min_delta=0.0001, patience=5)
metrics = Metrics(tf_image_generator_val)

#Training
history = model.fit_generator(
    generator = tf_image_generator_train, steps_per_epoch = 32, #train batch size
    epochs = n_epochs,
    validation_data = tf_image_generator_val, validation_steps = 32, #val batch size
    callbacks = [save_best_model, tensorboard_callback, metrics, early_stop]
)
