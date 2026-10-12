# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the Keras models whose layers this release reads for the first time, and keras-new.json: the examples each was
# handed and what Keras answers through it. Its output, when it was run, is keras-new.txt beside it.
#
# Each is compiled with a mean squared error, since a network here answers through the loss its file names, and trained a few
# epochs on random answers, so that every number has moved: a bias stays nought otherwise.
#
# Run from this folder with Keras 3.15.1 on its torch backend:
#
#     python keras-new-fixtures.py > keras-new.txt
#
#   keras-new-grouped.keras      Conv2D(6, 3, padding="same", groups=2), Conv2D(4, 3, strides=(2, 1), dilation_rate=1)
#   keras-new-depthwise.keras    DepthwiseConv2D(3, depth_multiplier=2, padding="same"), Conv2D(3, 1)
#   keras-new-separable.keras    SeparableConv2D(5, 3, depth_multiplier=2, padding="same")
#   keras-new-transposed.keras   Conv2DTranspose(4, 3, strides=2, padding="valid"), Conv2DTranspose(3, 2)
#   keras-new-padded.keras       ZeroPadding2D((1, 2)), Conv2D(3, 3)
#   keras-new-dilated.keras      Conv1D(4, 3, dilation_rate=2, padding="causal")
#   keras-new-series.keras       ZeroPadding1D(2), SeparableConv1D(4, 3, depth_multiplier=2, padding="same"),
#                                DepthwiseConv1D(3, depth_multiplier=2), Conv1DTranspose(3, 3, strides=2, padding="same"),
#                                Conv1DTranspose(2, 3, strides=2, output_padding=1)
#   keras-new-volume.keras       ZeroPadding3D(((1, 0), (0, 1), (1, 1))), Conv3DTranspose(3, 2, strides=(2, 1, 2), padding="same")

import json
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
from keras import layers

here = os.path.dirname(os.path.abspath(__file__))
keras.utils.set_random_seed(20261012)
rng = np.random.default_rng(20261012)

nets = {
    "keras-new-grouped": ([layers.Conv2D(6, 3, padding="same", groups=2), layers.Conv2D(4, 3, strides=(2, 1))], (6, 7, 6, 4)),
    "keras-new-depthwise": ([layers.DepthwiseConv2D(3, depth_multiplier=2, padding="same"), layers.Conv2D(3, 1)], (6, 6, 5, 3)),
    "keras-new-separable": ([layers.SeparableConv2D(5, 3, depth_multiplier=2, padding="same")], (6, 6, 5, 3)),
    "keras-new-transposed": ([layers.Conv2DTranspose(4, 3, strides=2), layers.Conv2DTranspose(3, 2)], (6, 4, 4, 2)),
    "keras-new-padded": ([layers.ZeroPadding2D((1, 2)), layers.Conv2D(3, 3)], (6, 5, 5, 2)),
    "keras-new-dilated": ([layers.Conv1D(4, 3, dilation_rate=2, padding="causal")], (6, 11, 2)),
    "keras-new-series": (
        [
            layers.ZeroPadding1D(2),
            layers.SeparableConv1D(4, 3, depth_multiplier=2, padding="same"),
            layers.DepthwiseConv1D(3, depth_multiplier=2),
            layers.Conv1DTranspose(3, 3, strides=2, padding="same"),
            layers.Conv1DTranspose(2, 3, strides=2, output_padding=1),
        ],
        (6, 11, 2)),
    "keras-new-volume": (
        [layers.ZeroPadding3D(((1, 0), (0, 1), (1, 1))), layers.Conv3DTranspose(3, 2, strides=(2, 1, 2), padding="same")],
        (6, 4, 4, 4, 2)),
}

fixture = {"made": f"keras {keras.__version__}, backend {keras.backend.backend()}"}
print(fixture["made"])

for name, (built, shape) in nets.items():
    model = keras.Sequential([layers.Input(shape=shape[1:])] + built + [layers.Flatten(), layers.Dense(2)])
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")

    # Trained a few epochs on random examples and random answers, so that every number has moved: a bias stays nought otherwise.
    model.fit(
        rng.normal(size=(48, *shape[1:])).astype(np.float32),
        rng.normal(size=(48, 2)).astype(np.float32),
        epochs=6,
        batch_size=16,
        verbose=0)

    examples = rng.normal(size=shape).astype(np.float32)
    outputs = np.asarray(model.predict(examples, verbose=0), dtype=np.float32)

    model.save(os.path.join(here, f"{name}.keras"))
    fixture[name] = {
        "examples": {"shape": list(examples.shape), "values": [float(value) for value in examples.reshape(-1)]},
        "outputs": [float(value) for value in outputs.reshape(-1)],
    }
    print(f"{name}: {[layer.__class__.__name__ for layer in model.layers]} -> {list(outputs.shape)}")
    print("  weights:", [(weight.path.split('/')[-1], tuple(weight.shape)) for weight in model.weights])
    print("  outputs:", " ".join(f"{value:.8f}" for value in fixture[name]["outputs"][:6]), "...")

with open(os.path.join(here, "keras-new.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
