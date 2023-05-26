import math
import os
import random
import shutil

import pandas as pd


def create_validation_set():
    train = pd.read_csv("data/train.csv", index_col=0)
    validation = pd.DataFrame(columns=train.columns)

    validation_dir = os.path.join("data", "Validation")
    train_dir = os.path.join("data", "Train")

    if not os.path.exists(validation_dir):
        os.mkdir(validation_dir)

    n_classes = len(os.listdir(os.path.join("data", "Train")))

    val_classes = random.sample(range(n_classes), math.ceil(n_classes / 4))

    for val_class in val_classes:
        path = os.path.join(validation_dir, str(val_class))
        os.makedirs(path) if not os.path.exists(path) else None

        src = os.path.join(train_dir, str(val_class))
        for file in os.listdir(src):
            shutil.move(os.path.join(src, file), os.path.join(path, file))
        

    for i , row in train.iterro():
        if not os.path.exists(row["Path"]):
            validation.loc[i, "Width"] = row["Width"]
            validation.loc[i, "Height"] = row["Height"]
            validation.loc[i, "Roi.X1"] = row["Roi.X1"]
            validation.loc[i, "Roi.Y1"] = row["Roi.Y1"]
            validation.loc[i, "Roi.X2"] = row["Roi.X2"]
            validation.loc[i, "Roi.Y2"] = row["Roi.Y2"]
            validation.loc[i, "ClassId"] = row["ClassId"]
            validation.loc[i, "Path"] = row["Path"].replace("Train", "Validation")
            train.drop(i, inplace=True)

    train.to_csv("data/train.csv")
    validation.to_csv("data/validation.csv")