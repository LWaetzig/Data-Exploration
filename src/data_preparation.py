import os
import random
import math
import pandas as pd
import shutil

def create_validation_set(
        train_path: str,
        train_csv_path: str,
        validation_path: str,
        validation_csv_path: str,
    ) -> None:
        """Create Validation Set from Test Set, move files and create new csv files

        Args:
            train_path (str): path to train directory
            train_csv_path (str): path to train csv file
            validation_path (str): save path to validation directory
            validation_csv_path (str): save path to validation csv file

        Returns:
            None
        """
        train = pd.read_csv(train_csv_path, index_col=0)
        validation = pd.DataFrame(columns=train.columns)

        if not os.path.exists(validation_path):
            os.mkdir(validation_path)

        n_classes = len(os.listdir(os.path.join("data", "Train")))

        val_classes = random.sample(range(n_classes), math.ceil(n_classes / 4))

        for val_class in val_classes:
            path = os.path.join(validation_path, str(val_class))
            os.makedirs(path) if not os.path.exists(path) else None

            src = os.path.join(train_path, str(val_class))
            for file in os.listdir(src):
                shutil.move(os.path.join(src, file), os.path.join(path, file))

        for i, row in train.iterro():
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

        train.to_csv(train_csv_path)
        validation.to_csv(validation_csv_path)

        return None



meta = pd.read_csv(os.path.join("data", "meta.csv"), index_col=0)
train = pd.read_csv(os.path.join("data", "train.csv"), index_col=0)

for i , row in train.iterrows():
    train.loc[i, "shape"] = int(meta.loc[meta["class"] == row["ClassId"]]["shape"].values[0])
    train.loc[i, "color"] = int(meta.loc[meta["class"] == row["ClassId"]]["color"].values[0])

