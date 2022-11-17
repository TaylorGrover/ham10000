import json
import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

csv_root = "../data/csvs"
dims = (28, 28)
image_dir = "../data/all_images"

def get_class_counts():
    counts = dict()
    csv_path = os.path.join(csv_root, "HAM10000_metadata.csv")
    df = pd.read_csv(csv_path)
    for dx in list(set(df.dx)):
        counts[dx] = int(np.sum(df.dx == dx))
    return counts
    

def get_generators(dims=(100, 100), batch_size=32):
    data_generator = ImageDataGenerator(
        rescale=1 / 255.0,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        height_shift_range=0.1,
        rotation_range=180
    )
    val_datagen = ImageDataGenerator(rescale=1/255.0)
    train_generator = data_generator.flow_from_directory(
        image_dir,
        target_size=dims,
        batch_size=batch_size,
        subset="training",
        class_mode="binary",
    )
    validation_generator = val_datagen.flow_from_directory(
        image_dir,
        target_size=dims,
        batch_size=batch_size,
        class_mode="binary",
    )

    '''dataset = tf.keras.utils.image_dataset_from_directory(
        image_dir,
        labels="inferred",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=dims,
        shuffle=True,
        validation_split=0.3,
        subset="both",
        seed=23,
        interpolation="bicubic",
    )'''
    return train_generator, validation_generator


def get_X_y_csv():
    csv_path = os.path.join(csv_root, "hmnist_28_28_RGB.csv")
    df = pd.read_csv(csv_path)
    data = np.array(df)
    np.random.shuffle(data)
    X = data[:, :-1]
    X = X.reshape(X.shape[0], 28, 28, 3)
    y = data[:, -1]
    y = y.reshape(y.shape[0], 1)
    onehot = OneHotEncoder()
    y = onehot.fit_transform(y).toarray()
    return X, y


def build_model(dims):
    model = Sequential()
    model.add(layers.Conv2D(80, (5, 5), activation="relu", input_shape=(*dims, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization(axis=-1))
    model.add(layers.Conv2D(64, (5, 5), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation="relu", use_bias=True))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(7, activation="softmax", use_bias=True))
    return model


if __name__ == "__main__":
    X, y = get_X_y_csv()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    tensor_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    tensor_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

    model = build_model(dims)
    model.summary()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    tf.keras.utils.plot_model(
        model,
        to_file="model.png",
        show_shapes=True,
        show_dtype=False,
        show_layer_names=False,
        show_layer_activations=False
    )

    # Saves the model with highest validation accuracy in all epochs
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=".",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )
    counts = get_class_counts()
    print(json.dumps(counts, indent=4))
    #train_gen, val_gen = get_generators(dims)
    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        callbacks=[checkpoint_callback],
        validation_split=0.2,
    )
    print(model.evaluate(X_test, y_test))
