# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes keras-refused.keras, the model the Keras importer's tests read for what nothing here walks: a stride of its own for
# each axis, a dilated window, channels in groups, and a layer of a kind that is not read. It was once written by
# keras-fixtures.py, with a pooling layer for the last; pooling is read now, so the layer of a kind not read is an
# up-sampling layer, and this script writes the file alone, leaving the models keras-fixtures.py made as they were.
#
# Run with Keras 3.15.1 on PyTorch for the CPU:
#
#     KERAS_BACKEND=torch python keras-refused-fixture.py
#
# The three convolutions carry the names conv2d_2, conv2d_3 and conv2d_4 that the tests name, as they did when keras-fixtures.py
# wrote the file after two convolutions of its own.

import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers

here = os.path.dirname(os.path.abspath(__file__))

refused = keras.Sequential(
    [
        keras.Input((16, 16, 4)),
        layers.Conv2D(4, 3, strides=(2, 1), padding="same", name="conv2d_2"),
        layers.Conv2D(4, 3, dilation_rate=2, padding="same", name="conv2d_3"),
        layers.Conv2D(4, 3, groups=2, padding="same", name="conv2d_4"),
        layers.UpSampling2D(2),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),
    ]
)
refused.compile(optimizer="adam", loss="binary_crossentropy")
refused.save(os.path.join(here, "keras-refused.keras"))

print(f"keras-refused.keras: {os.path.getsize(os.path.join(here, 'keras-refused.keras'))} bytes")
