import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pdb
import tensorflow as tf
import time

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier, SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras import layers, regularizers
from tensorflow.keras.metrics import categorical_accuracy, Precision
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

EPOCHS = 20
CSV_ROOT = "../data/csvs"
image_dir_bin = "../data/images_binary"
image_dir_multi = "../data/images_multiclass"
model_dir = "models"
accuracy_dir = "accuracies"
chart_dir = "training_charts"
model_plots_dir = "model_plots"

def get_class_dist():
    """
    Get class distribution: weight of each class in dataset
    """
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
    csv_path = os.path.join(CSV_ROOT, "HAM10000_metadata.csv")
    df = pd.read_csv(csv_path)
    le = LabelEncoder()
    if is_numeric:
        df.dx_type = le.fit_transform(df.dx_type)
        df.sex = le.fit_transform(df.sex)
        df.localization = le.fit_transform(df.localization)
        df = np.array(df.iloc[:,3:7])
    return df


def get_datagen(dims=(100, 100), batch_size=32):

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        featurewise_center=False,
        samplewise_center=False,
        zca_whitening=False,
    )
    '''train_dataset = tf.keras.utils.image_dataset_from_directory(
        image_dir_multi,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=dims,
        shuffle=True,
        subset="training",
        seed=1,
        validation_split=0.2,
        interpolation="bilinear"
    )
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        image_dir_multi,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=dims,
        shuffle=True,
        subset="validation",
        seed=1,
        validation_split=0.2,
        interpolation="bilinear"
    )'''

    return datagen


def get_checkpoint(name):
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, name),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )


def plot_model(model, name):
    tf.keras.utils.plot_model(
        model,
        to_file=os.path.join(model_plots_dir, name),
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
    csv_path = os.path.join(CSV_ROOT, "hmnist_28_28_RGB.csv")
    df = pd.read_csv(csv_path)
    data = np.array(df)
    np.random.shuffle(data)
    X = data[:, :-1]
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std
    X = X.reshape(X.shape[0], 28, 28, 3)
    y = data[:, -1]
    y = y.reshape(y.shape[0], 1)
    if is_binary:
        y_dist = np.array([np.mean(y == i) for i in range(np.max(y) + 1)])
        #y = (y == np.argmax(y_dist)) * 1 # For nv 
        melanoma_index = np.where(np.abs(y_dist - .1111333) < .0001)[0][0]
        y = (y == melanoma_index) * 1 # Get melanoma encoding
    onehot = OneHotEncoder()
    y = onehot.fit_transform(y).toarray()
    return X, y


def build_conv_model(dims, out_dim):
    model = Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation="relu", input_shape=dims))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=dims))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.BatchNormalization(-1))
    model.add(layers.Dropout(.2))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization(-1))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.BatchNormalization(-1))
    model.add(layers.Dropout(.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(200, activation="relu", use_bias=True))
    model.add(layers.Dense(100, activation="relu", use_bias=True))
    model.add(layers.Dense(out_dim, activation="softmax", use_bias=True))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_history(name, history, metric):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_ylim(.55, 1)
    ax.plot(history.history[metric])
    ax.plot(history.history["val_{}".format(metric)])
    ax.set_title(name)
    ax.set_ylabel(metric)
    ax.set_xlabel("epoch")
    ax.legend(["train", "val"], loc="upper left")
    fig.savefig(os.path.join(chart_dir, name + ".png"))
    return fig, ax


def experimental_model(img_dim, meta_len, out_len):
    """
    Use metadata in addition to pixel data to perform classification
    """
    conv_input = tf.keras.Input(shape=img_dim)
    conv_x = layers.Conv2D(80, (5, 5), activation="relu", padding="same", input_shape=img_dim)(conv_input)
    conv_x = layers.MaxPool2D((2, 2), padding="same")(conv_x)
    conv_x = layers.BatchNormalization(-1)(conv_x)
    conv_x = layers.Conv2D(64, (5, 5), activation="relu")(conv_x)
    conv_x = layers.MaxPool2D((2, 2))(conv_x)
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


def performance(model, X_train, y_train, X_test, y_test, checkpoint_callback, hist_name, plot_name, epochs):
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        callbacks=[checkpoint_callback],
        validation_split=0.2,
    )
    plot_history(hist_name, history, "accuracy")
    model = tf.keras.models.load_model(os.path.join(model_dir, plot_name))
    test_acc = model.evaluate(X_test, y_test)
    conf = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(model.predict(X_test), axis=1))
    data = {
        "name": plot_name,
        "conf": conf.tolist(),
        "test": test_acc,
        "epochs": epochs,
    }
    print(json.dumps(data, indent=4))
    with open(os.path.join(accuracy_dir, plot_name + ".json"), "w") as f:
        json.dump(data, f, indent=4)
    return model


def method_conv(is_binary=True):
    X, y = get_X_y_csv(is_binary=is_binary)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = build_conv_model(X.shape[1:], y_train.shape[1])
    model.summary()

    hist_name = "Deep Convolutional Model Accuracy"
    plot_name = "conv"
    if is_binary: 
        hist_name += " (binary)"
        plot_name += "_bin"
    plot_model(model, plot_name + ".png")
    # Saves the model with highest validation accuracy in all epochs
    checkpoint_callback = get_checkpoint(plot_name)
    # To load the model, just do `model = tf.keras.models.load(".")`
    dist = get_class_dist()
    print(json.dumps(dist, indent=4))
    #train_gen, val_gen = get_generators(dims)
    model = performance(model, X_train, y_train, X_test, y_test, checkpoint_callback, hist_name, plot_name, EPOCHS)
    return model, X_train, X_test, y_train, y_test


def build_weak_model(dims, out_dim):
    model = Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation="relu", input_shape=dims))
    model.add(layers.Conv2D(16, (3, 3), activation="relu", input_shape=dims))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.BatchNormalization(-1))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", kernel_regularizer=regularizers.L1L2(l1=1e-5, l2=1e-4)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.BatchNormalization(-1))
    model.add(layers.Dropout(0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation="sigmoid"))
    model.add(layers.Dense(out_dim, activation="softmax", use_bias=True))
    model.compile(optimizer="adagrad", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def build_ensemble(num_models, input_shape, output_shape):
    weak_models = [build_weak_model(input_shape, output_shape) for i in range(num_models)]
    model_input = tf.keras.Input(shape=input_shape)
    outputs = [model(model_input) for model in weak_models]
    ensemble_output = layers.Average()(outputs)
    model = Model(inputs=model_input, outputs=ensemble_output)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def method_ensemble(is_binary=True, num_models=3):
    X, y = get_X_y_csv(is_binary=is_binary)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = build_ensemble(num_models, X.shape[1:], y.shape[1])

    hist_name = f"Weak CNN Ensemble (Model count: {num_models})"
    plot_name = f"ensemble_{num_models}"
    if is_binary:
        hist_name += " (binary)"
        plot_name += "_bin"

    plot_model(model, plot_name + ".png")
    checkpoint_callback = get_checkpoint(plot_name)
    model = performance(model, X_train, y_train, X_test, y_test, checkpoint_callback, hist_name, plot_name, EPOCHS)
    return model, X_train, X_test, y_train, y_test
    

def method_SVM(is_binary=True):
    X, y = get_X_y_csv(is_binary=is_binary)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    flat_train = X_train.reshape(X_train.shape[0], np.prod(X_train.shape[1:]))
    flat_test = X_test.reshape(X_test.shape[0], np.prod(X_test.shape[1:]))
    y_train = np.argmax(y_train, axis=1)
    y_test = np.argmax(y_test, axis=1)

    svm_full = LinearSVC(verbose=1)
    svm_full.fit(flat_train, y_train)
    pred = svm_full.predict(flat_test)
    print("Full Linear SVM test accuracy: %f" % (svm_full.score(flat_test, y_test)))
    print("Full Linear Test Confusion Matrix: {}".format(confusion_matrix(y_test, pred)))
    svm_pca = LinearSVC(verbose=2)
    pca = PCA(n_components=200)
    pca.fit(flat_train.T)
    cmp_train = pca.components_.T
    pca.fit(flat_test.T)
    cmp_test = pca.components_.T
    svm_pca.fit(cmp_train, y_train)
    pred = svm_pca.predict(cmp_test)
    print("Linear SVM using PCA: %f" % (svm_pca.score(cmp_test, y_test)))
    print("Linear SVM Test Confusion Matrix: {}".format(confusion_matrix(y_test, pred)))
    return svm_full, svm_pca, X_train, X_test, y_train, y_test


def method_conv_and_metadata(is_binary=True):
    """
    Combine the pixels and metadata
    """
    X, y = get_X_y_csv(is_binary=is_binary)
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
    hist_name = "Convolution with Metadata Inputs"
    plot_name = "conv_and_metadata"
    if is_binary:
        hist_name += " (binary)"
        plot_name += "_bin"
    plot_model(model, plot_name + ".png")
    checkpoint_callback = get_checkpoint(plot_name)
    model = performance(model, [X_train, meta_train], y_train, [X_test, meta_test], y_test, checkpoint_callback, hist_name, plot_name, 3)
    return model, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    start = time.time()
    #svm_full, svm_pca, X_train, X_test, y_train, y_test = method_SVM()
    print("Conv (Multiclass)")
    model, X_train, X_test, y_train, y_test = method_conv(is_binary=False)
    #model, X_train, X_test, y_train, y_test = method_conv_and_metadata(is_binary=False)
    print("Ensemble (Multi)")
    model, X_train, X_test, y_train, y_test = method_ensemble(is_binary=False, num_models=3)
    print("Conv (Binary)")
    model, X_train, X_test, y_train, y_test = method_conv(is_binary=True)
    #model, X_train, X_test, y_train, y_test = method_conv_and_metadata(is_binary=True)
    print("Ensemble (Binary)")
    model, X_train, X_test, y_train, y_test = method_ensemble(is_binary=True, num_models=3)
    elapsed = time.time() - start
    print("Training time for all models: {}".format(elapsed))
