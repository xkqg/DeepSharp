# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the models Keras saved that the Keras importer's tests read, and what Keras answered with each. Its output, when
# it was run, is keras-fixtures.txt beside it.
#
# Run with Keras 3.15.1 on PyTorch 2.10.0 for the CPU, on the rows make-titanic-rows.cs writes:
#
#     dotnet run make-titanic-rows.cs -- rows
#     KERAS_BACKEND=torch python keras-fixtures.py rows
#
#   keras-titanic.keras, .h5   the networks sample's Titanic network as Keras has it — sixteen units through a relu, one
#                              through a sigmoid, a binary cross-entropy named by its string — trained on the pipeline's
#                              training rows, judged by its validation rows, stopped early and kept at its best epoch;
#                              saved in Keras 3's own format and in the legacy HDF5 one.
#   keras-images.keras, .h5    a network over small images that walks every kind the importer reads: a reshape, a window
#                              of three rows by two columns at a stride of two padded as 'same', a batch normalisation of
#                              momentum 0.9 and epsilon 0.001, a window of two padded as 'valid', a relu layer, a flatten,
#                              a dropout, a dense layer, a tanh activation layer, a layer normalisation of epsilon 0.001,
#                              and a softmax over three classes that the categorical cross-entropy, named as an object, owns;
#                              trained on random images so every number, the running statistics too, has moved.
#   keras-refused.keras        a network holding what nothing here walks: a stride of its own for each axis, a dilated
#                              window, channels in groups and a pooling layer.
#   keras-fixtures.json        the test rows and the two served passengers as they were handed to Keras, the images, and
#                              what Keras answered for each: the chances and the shares, as single-precision numbers.

import json
import os
import platform
import sys

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import torch
from keras import layers

torch.set_num_threads(1)
here = os.path.dirname(os.path.abspath(__file__))
rows = sys.argv[1] if len(sys.argv) > 1 else "rows"


def read(name, answers=True):
    table = np.loadtxt(os.path.join(rows, name), delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
    return (table[:, :14], table[:, 14:15]) if answers else table


def values(array):
    return [float(value) for value in np.asarray(keras.ops.convert_to_numpy(array), dtype="float32").reshape(-1)]


def save(model, name):
    model.save(os.path.join(here, f"{name}.keras"))
    model.save(os.path.join(here, f"{name}.h5"))


made = f"keras {keras.__version__} on torch {torch.__version__}, numpy {np.__version__}, Python {platform.python_version()}"
print(made)

# The Titanic network, as the networks sample trains it: Adam at 0.01, batches of 32, at most a hundred epochs, early
# stopping of patience ten that restores the best.
train, train_answers = read("train.csv")
validation, validation_answers = read("validation.csv")
test, _ = read("test.csv")
served = read("served.csv", answers=False)

keras.utils.set_random_seed(20260929)
titanic = keras.Sequential([keras.Input((14,)), layers.Dense(16, activation="relu"), layers.Dense(1, activation="sigmoid")])
titanic.compile(optimizer=keras.optimizers.Adam(0.01), loss="binary_crossentropy")
history = titanic.fit(
    train, train_answers, validation_data=(validation, validation_answers), epochs=100, batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)], verbose=0)
save(titanic, "keras-titanic")

titanic_answers = {
    "epochs": len(history.history["loss"]),
    "test": {"rows": [values(row) for row in test], "chances": values(titanic(test, training=False))},
    "served": {"rows": [values(row) for row in served], "chances": values(titanic(served, training=False))},
}
print(f"titanic: {titanic_answers['epochs']} epochs; a third-class man of 22 {titanic_answers['served']['chances'][0]!r}, "
      f"a first-class woman of 38 {titanic_answers['served']['chances'][1]!r}")

# The images: every kind the importer reads, trained a few epochs on random images and classes so that every number moves.
keras.utils.set_random_seed(20260930)
rng = np.random.default_rng(20260930)
images = keras.Sequential(
    [
        keras.Input((84,)),
        layers.Reshape((6, 7, 2)),
        layers.Conv2D(4, (3, 2), strides=2, padding="same", activation="relu"),
        layers.BatchNormalization(momentum=0.9, epsilon=1e-3),
        layers.Conv2D(3, 2, padding="valid"),
        layers.ReLU(),
        layers.Flatten(),
        layers.Dropout(0.25),
        layers.Dense(6),
        layers.Activation("tanh"),
        layers.LayerNormalization(epsilon=1e-3),
        layers.Dense(3, activation="softmax"),
    ]
)
images.compile(optimizer=keras.optimizers.Adam(0.01), loss=keras.losses.CategoricalCrossentropy())
pixels = rng.normal(size=(64, 84)).astype("float32")
classes = keras.utils.to_categorical(rng.integers(0, 3, size=64), 3).astype("float32")
images.fit(pixels, classes, epochs=5, batch_size=16, verbose=0)
save(images, "keras-images")

seen = rng.normal(size=(8, 84)).astype("float32")
images_answers = {"rows": [values(row) for row in seen], "shares": values(images(seen, training=False))}
norm = images.layers[2]
print("images: running mean", " ".join(f"{value:.6f}" for value in values(keras.ops.convert_to_numpy(norm.moving_mean))),
      "running var", " ".join(f"{value:.6f}" for value in values(keras.ops.convert_to_numpy(norm.moving_variance))))
print("images: shares of the first", " ".join(f"{value:.8f}" for value in images_answers["shares"][:3]))

# What nothing here walks, one of each, in one network: the importer names all four at once, each at its layer.
refused = keras.Sequential(
    [
        keras.Input((16, 16, 4)),
        layers.Conv2D(4, 3, strides=(2, 1), padding="same"),
        layers.Conv2D(4, 3, dilation_rate=2, padding="same"),
        layers.Conv2D(4, 3, groups=2, padding="same"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),
    ]
)
refused.compile(optimizer="adam", loss="binary_crossentropy")
refused.save(os.path.join(here, "keras-refused.keras"))

with open(os.path.join(here, "keras-fixtures.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump({"made": made, "titanic": titanic_answers, "images": images_answers}, file, indent=1)
    file.write("\n")

for name in ("keras-titanic.keras", "keras-titanic.h5", "keras-images.keras", "keras-images.h5", "keras-refused.keras"):
    print(f"{name}: {os.path.getsize(os.path.join(here, name))} bytes")
