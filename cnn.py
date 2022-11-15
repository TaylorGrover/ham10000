import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

df = pd.read_csv("data/csvs/hmnist_28_28_RGB.csv")
data = np.array(df)
np.random.shuffle(data)
X = data[:, :-1]
X = X.reshape(X.shape[0], 28, 28, 3)
y = data[:, -1]
y = y.reshape(y.shape[0], 1)
onehot = OneHotEncoder()
y = onehot.fit_transform(y).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
tensor_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
tensor_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
model = Sequential()
#conv2d = tf.keras.layers.Conv2D(10, 3, activation="relu", input_shape=(10, 28, 28, 3))
model.add(layers.Conv2D(80, (5, 5), activation="relu", input_shape=(28, 28, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.BatchNormalization(axis=-1))
model.add(layers.Conv2D(64, (5, 5), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(100, activation="relu", use_bias=True))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(7, activation="softmax", use_bias=True))
model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=".",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True
)
history = model.fit(
    x=tensor_train,
    y=y_train,
    epochs=100,
    batch_size=50,
    validation_split=0.1,
    callbacks=[checkpoint_callback],
)
print(model.evaluate(X_test, y_test))
