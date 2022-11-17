import glob
import numpy as np
import pandas as pd
import os
import PIL.Image as Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Input
from tensorflow.keras.models import Model, Sequential
import time

root_dir = "../data/all_images"
aug_dir = "../data/augmented"
df = pd.read_csv("../data/csvs/HAM10000_metadata.csv")
img_resize = (100, 100)

def get_images(n):
    X = []
    start = time.time()
    for i, row in enumerate(np.array(df)):
        if i >= n:
            break
        img_path = os.path.join(root_dir, row[1] + ".jpg")
        X.append(np.array(Image.open(img_path).resize(img_resize, Image.Resampling.BICUBIC)))
    print(time.time() - start)
    return X

