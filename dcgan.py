import os
import time
from PIL import Image
import pathlib
import tensorflow as tf
from random import shuffle
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.layers import (
    Input,
    Dense,
    Reshape,
    Flatten,
    BatchNormalization,
    LeakyReLU,
    Conv2D,
    Conv2DTranspose,
    MaxPool2D,
    Dropout,
)
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.compat.v1 import ConfigProto, InteractiveSession
import scipy.special as sc
import scipy.stats as stats
from scipy import signal
import math
import tensorflow.experimental.numpy as tnp
import matplotlib.pyplot as plt
import numpy as np
import pydot
import random
import pandas as pd
from tqdm import tqdm
import cv2 as cv
import multiprocessing
import opendatasets as od

# Load Habana
from habana_frameworks.tensorflow import load_habana_module

load_habana_module()


# Enable numpy behavior for TF
tnp.experimental_enable_numpy_behavior()


class DCGAN:
    def __init__(self):
        self.img_size = 256
        self.batch_size = 12
        self.latent_dim = 256
        self.img_shape = (self.img_size, self.img_size, 3)

    def download_data(self):
        """
        Download Required Data
        """
        od.download("https://www.kaggle.com/andrewmvd/ocular-disease-recognition-odir5k")
        data_dir = pathlib.Path("/content/ocular-disease-recognition-odir5k/ODIR-5K/ODIR-5K/Training Images/")

    def process_data(self, data_image_list: list, data_folder: str) -> list:
        """
        Process Data
        """
        data_df = []
        for img in tqdm(data_image_list):
            path = os.path.join(data_folder, img)
            img = cv.imread(path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            img = cv.resize(img, (self.img_size, self.img_size), interpolation=cv.INTER_NEAREST)
            data_df.append([np.array(img)])
        shuffle(data_df)
        return data_df

    def has_cataract(self, text) -> bool:
        if "cataract" in text:
            return 1
        else:
            return 0

    def show_images(self, data: list):
        print(f"Cataract Image Set")

        f, ax = plt.subplots(5, 5, figsize=(15, 15))

        for i, data in enumerate(data[:25]):
            img_data = data[0]
            ax[i // 5, i % 5].imshow(img_data)
            ax[i // 5, i % 5].axis("off")

        plt.show()

    def prepare_dataset(self) -> list:
        """
        Prepare dataset
        """
        training_data_folder = os.path.join("./ocular-disease-recognition-odir5k/ODIR-5K/ODIR-5K/Training Images")
        df = pd.read_csv(os.path.join("./ocular-disease-recognition-odir5k/", "full_df.csv"))

        df["left_cataract"] = df["Left-Diagnostic Keywords"].apply(lambda x: self.has_cataract(x))
        df["right_cataract"] = df["Right-Diagnostic Keywords"].apply(lambda x: self.has_cataract(x))

        left_cataract = df.loc[(df.C == 1) & (df.left_cataract == 1)]["Left-Fundus"].values
        right_cataract = df.loc[(df.C == 1) & (df.right_cataract == 1)]["Right-Fundus"].values

        print("Number of images in left cataract: {}".format(len(left_cataract)))
        print("Number of images in right cataract: {}".format(len(right_cataract)))

        cataract_list = np.concatenate((left_cataract, right_cataract), axis=0)

        print('Total "cataract" images : ', len(cataract_list))

        cat_df = self.process_data(cataract_list, training_data_folder)

        return cat_df

    def rescale_data(self, data_image_list: np.array) -> np.array:
        min_max = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 127.5, offset=-1)
        processed_data = min_max(data_image_list)

        return processed_data

    def create_dataset(self) -> np.array:
        # Download Data
        self.download_data()

        # Create Training Set
        train = self.prepare_dataset()
        shuffle(train)

        # Visualize Training Set
        self.show_images(train)

        # Split Training Set into X, Y
        X = np.array([i[0] for i in train]).astype("float32").reshape(-1, self.img_size, self.img_size, 3)

        # Normalize Training Data (MinMax Scaling)
        X = self.rescale_data(X)

        print(f"X Shape : {X.shape}")

        return X

    def run(self):
        """
        Run DCGAN
        """
        X = self.create_dataset()


if __name__ == "__main__":
    DCGAN().run()
