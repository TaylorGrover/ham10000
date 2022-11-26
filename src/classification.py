import json
import numpy as np
import os
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.metrics import categorical_accuracy, Precision
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

csv_root = "../data/csvs"
dims = (28, 28)
epochs = 20
image_dir = "../data/all_images"

def get_class_dist():
    counts = dict()
    df = get_metadata()
    for dx in list(set(df.dx)):
        counts[dx] = np.mean(df.dx == dx)
    return counts


def get_metadata(is_numeric=False):
    """
    :return: pandas dataframe of all metadata if not is_numeric, otherwise 
    returns [dx_type, age, sex, localization]
    """
    csv_path = os.path.join(csv_root, "HAM10000_metadata.csv")
    df = pd.read_csv(csv_path)
    le = LabelEncoder()
    if is_numeric:
        df.dx_type = le.fit_transform(df.dx_type)
        df.sex = le.fit_transform(df.sex)
        df.localization = le.fit_transform(df.localization)
        df = np.array(df.iloc["dx_type", "age", "sex", "localization"])
    return df
    

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
    """
    X is the pixel values of the 28x28x3 skin lesion images. y is a binary 
    one-hot vector (i.e. [0, 1] or [1, 0])
    """
    csv_path = os.path.join(csv_root, "hmnist_28_28_RGB.csv")
    df = pd.read_csv(csv_path)
    data = np.array(df)
    np.random.shuffle(data)
    X = data[:, :-1]
    X = X.reshape(X.shape[0], 28, 28, 3) / 255
    y = data[:, -1]
    y_dist = [np.mean(y == i) for i in range(max(y))]
    y = (y.reshape(y.shape[0], 1) == np.argmax(y_dist)) * 1
    onehot = OneHotEncoder()
    y = onehot.fit_transform(y).toarray()
    return X, y


def build_conv_model(dims, out_dim):
    model = Sequential()
    model.add(layers.Conv2D(80, (5, 5), activation="relu", input_shape=(*dims, 3)))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.BatchNormalization(-1))
    model.add(layers.Dropout(.2))
    model.add(layers.Conv2D(64, (5, 5), activation="relu"))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization(-1))
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.AveragePooling2D((2, 2)))
    model.add(layers.BatchNormalization(-1))
    model.add(layers.Dropout(.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(out_dim, activation="softmax", use_bias=True))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def experimental_model(img_dim, meta_len, out_dim):
    """
    Use metadata in addition to pixel data to perform classification
    """
    conv_input = tf.keras.Input(shape=img_dim)
    conv_x = layers.Conv2D(80, (5, 5), activation="relu", input_shape=(*dims, 3))(conv_input)
    conv_x = layers.MaxPool2D((2, 2))(conv_x)
    conv_x = layers.BatchNormalization(-1)(conv_x)
    conv_x = layers.Conv2D(64, (5, 5), activation="relu")(conv_x)
    conv_x = layers.MaxPool2D((2, 2))(conv_x)
    conv_x = layers.BatchNormalization(-1)(conv_x)
    conv_x = layers.Conv2D(32, (3, 3), activation="relu")(conv_x)
    conv_x = layers.AveragePooling2D((2, 2))(conv_x)
    conv_x = layers.BatchNormalization(-1)(conv_x)
    conv_x = layers.Dropout(.2)(conv_x)
    conv_x = layers.Flatten()(conv_x)

    meta_input = tf.keras.Input(shape=(meta_len,))
    x = layers.Dense(100, activation="relu")([meta_input, conv_x])
    output = layers.Dense(meta_len, activation="softmax")(x)

    model = Model(inputs=[conv_input, meta_input], outputs=output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def method_conv():
    X, y = get_X_y_csv()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    tensor_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    tensor_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

    model = build_conv_model(dims, y_train.shape[1])
    model.summary()

    tf.keras.utils.plot_model(
        model,
        to_file="model.png",
        show_shapes=True,
        show_dtype=True,
        show_layer_names=False,
        show_layer_activations=True
    )

    # Saves the model with highest validation accuracy in all epochs
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=".",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )
    # To load the model, just do `model = tf.keras.models.load(".")`
    dist = get_class_dist()
    print(json.dumps(dist, indent=4))
    #train_gen, val_gen = get_generators(dims)
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        callbacks=[checkpoint_callback],
        validation_split=0.2,
    )
    print(model.evaluate(X_test, y_test))


def method_experimental():
    X, y = get_X_y_csv()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    meta_df = get_metadata(is_numeric=True)
    


if __name__ == "__main__":
    pass
    #method_conv()
