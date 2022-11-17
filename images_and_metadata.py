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

root_dir = "data/all_images"
df = pd.read_csv("data/csvs/HAM10000_metadata.csv")
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

total_images = 50

encoder = LabelEncoder()
binarizer = LabelBinarizer()

dx_type = encoder.fit_transform(df.dx_type).reshape(-1, 1)[:total_images]
age = encoder.fit_transform(df.age).reshape(-1, 1)[:total_images]
sex = encoder.fit_transform(df.sex).reshape(-1, 1)[:total_images]
localization = encoder.fit_transform(df.localization).reshape(-1, 1)[:total_images]

metadata = np.concatenate((dx_type, age, sex, localization), axis=1)
y = encoder.fit_transform(df.dx)[:total_images]

images_X = get_images(total_images)
zipped = list(zip(metadata, images_X, y))
np.random.shuffle(zipped)
n = len(zipped)
train_split_index = int(0.8 * n)
train, test = zipped[:train_split_index], zipped[train_split_index:]

meta_train, image_train, y_train = zip(*train)
meta_test, image_test, y_test = zip(*test)

meta_train = np.array(meta_train)
image_train = np.array(image_train)
y_train = np.array(y_train)

img_input = Input(shape=(*img_resize, 3))
x = layers.Conv2D(64, (5, 5), activation="relu")(img_input)
x = layers.MaxPool2D((2, 2))(x)
x = layers.Conv2D(64, (5, 5), activation="relu")(x)
x = layers.MaxPool2D((2, 2))(x)
x = layers.Conv2D(32, (5, 5), activation="relu")(x)
img_output = layers.GlobalMaxPooling2D()(x)

meta_input = Input(shape=(4,))
x = layers.Concatenate()([meta_input, img_output])
x = layers.Dense(100, activation="relu")(x)
outputs = layers.Dense(7, activation="softmax")(x)

model = Model([img_input, meta_input], outputs)
print(model.summary())

model.compile(optimizer="adam", metrics=["accuracy"], loss="categorical_crossentropy")

model.fit([image_train, meta_train], y_train)

#print(confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1)))
