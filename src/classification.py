import json
import numpy as np
import os
import pandas as pd
import pdb
import tensorflow as tf

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import layers, regularizers
from tensorflow.keras.metrics import categorical_accuracy, Precision
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

csv_root = "../data/csvs"
dims = (28, 28)
epochs = 20
image_dir = "../data/images_binary"

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
        df = np.array(df.iloc[:,3:7])
    return df


def get_generators(dims=(100, 100), batch_size=32):
    data_generator = ImageDataGenerator(
        rescale=1 / 255.0,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        height_shift_range=0.1,
        rotation_range=180,
        image_size=dims,
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


def get_checkpoint(name):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=name,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )


def plot_model(model, name):
    tf.keras.utils.plot_model(
        model,
        to_file=name,
        show_shapes=True,
        show_dtype=True,
        show_layer_names=False,
        show_layer_activations=True
    )


def get_X_y_csv(is_binary=True):
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
    if is_binary:
        y_dist = [np.mean(y == i) for i in range(max(y))]
        y = (y.reshape(y.shape[0], 1) == np.argmax(y_dist)) * 1
    onehot = OneHotEncoder()
    y = onehot.fit_transform(y).toarray()
    return X, y


def build_conv_model(dims, out_dim):
    model = Sequential()
    model.add(layers.Conv2D(80, (5, 5), activation="relu", input_shape=(*dims, 3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.BatchNormalization(-1))
    model.add(layers.Dropout(.2))
    model.add(layers.Conv2D(64, (5, 5), activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization(-1))
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.BatchNormalization(-1))
    model.add(layers.Dropout(.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(out_dim, activation="softmax", use_bias=True))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def experimental_model(img_dim, meta_len, out_len):
    """
    Use metadata in addition to pixel data to perform classification
    """
    conv_input = tf.keras.Input(shape=img_dim)
    conv_x = layers.Conv2D(80, (5, 5), activation="relu", padding="same", input_shape=(*dims, 3))(conv_input)
    conv_x = layers.MaxPool2D((2, 2), padding="same")(conv_x)
    conv_x = layers.BatchNormalization(-1)(conv_x)
    conv_x = layers.Conv2D(64, (5, 5), activation="relu")(conv_x)
    conv_x = layers.AveragePooling2D((2, 2))(conv_x)
    conv_x = layers.BatchNormalization(-1)(conv_x)
    conv_x = layers.Dropout(.2)(conv_x)
    conv_x = layers.Flatten()(conv_x)

    meta_input = tf.keras.Input(shape=(meta_len,))
    merged = layers.Concatenate(axis=1)([conv_x, meta_input])
    x = layers.Dense(500, activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-4))(merged)
    output = layers.Dense(out_len, activation="softmax")(x)

    model = Model(inputs=[conv_input, meta_input], outputs=output)
    model.compile(optimizer="ftrl", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def method_conv(is_binary=True):
    X, y = get_X_y_csv(is_binary=is_binary)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = build_conv_model(dims, y_train.shape[1])
    model.summary()

    plot_model(model, "conv.png")
    # Saves the model with highest validation accuracy in all epochs
    checkpoint_callback = get_checkpoint("conv")
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
    return model, X_train, X_test, y_train, y_test


def method_cascade(is_binary=True):
    X, y = get_X_y_csv(is_binary=is_binary)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    

def method_SVM(is_binary=True):
    X, y = get_X_y_csv(is_binary=is_binary)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    flat_train = X_train.reshape(X_train.shape[0], np.prod(X_train.shape[1:]))
    flat_test = X_test.reshape(X_test.shape[0], np.prod(X_test.shape[1:]))
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    svm_full = LinearSVC(verbose=1)
    svm_full.fit(flat_train, y_train)
    print("Full Linear SVM test accuracy: %f" % (svm_full.score(flat_test, y_test)))
    svm_pca = LinearSVC(verbose=2)
    pca = PCA(n_components=200)
    pca.fit(flat_train.T)
    cmp_train = pca.components_.T
    pca.fit(flat_test.T)
    cmp_test = pca.components_.T
    svm_pca.fit(cmp_train, y_train)
    print("Linear SVM using PCA: %f" % (svm_pca.score(cmp_test, y_test)))
    return svm_full, svm_pca, X_train, X_test, y_train, y_test


def method_conv_and_metadata():
    """
    Combine the pixels and metadata
    """
    X, y = get_X_y_csv()
    meta_x = get_metadata(is_numeric=True)
    concat = list(zip(X, meta_x))
    X_train, X_test, y_train, y_test = train_test_split(concat, y, test_size=0.2)
    X_train, meta_train = zip(*X_train)
    X_test, meta_test = zip(*X_test)
    X_train = np.array(X_train)
    meta_train = np.array(meta_train)
    X_test = np.array(X_test)
    meta_test = np.array(meta_test)

    model = experimental_model(X.shape[1:], meta_x.shape[1], y.shape[1])
    print(model.summary())
    
    plot_model(model, "conv_and_metadata.png")
    checkpoint_callback = get_checkpoint("conv_and_metadata")
    history = model.fit(
        [X_train, meta_train], 
        y_train, 
        epochs=20, 
        batch_size=48, 
        callbacks=[checkpoint_callback],
        validation_split=0.2
    )
    print(model.evaluate([X_test, meta_test], y_test))
    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    pass
    #svm_full, svm_pca, X_train, X_test, y_train, y_test = method_SVM()
    model, X_train, X_test, y_train, y_test = method_conv(is_binary=False)
    #model, X_train, X_test, y_train, y_test = method_conv_and_metadata()
