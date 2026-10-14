# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the Keras models whose layers keep no bias, no scale or no shift, and keras-free.json: the examples each was handed
# and what Keras answers through it. Its output, when it was run, is keras-free.txt beside it.
#
# Each is compiled with a mean squared error, since a network here answers through the loss its file names, and trained a few
# epochs on random answers, so that every number has moved.
#
# Run from this folder with Keras 3.15.1 on its torch backend:
#
#     python keras-free-fixtures.py > keras-free.txt
#
#   keras-free-dense.keras    Dense(6, use_bias=False, relu), BatchNormalization(center=False), Dense(4, use_bias=False, tanh),
#                             LayerNormalization(scale=False), Dense(3, use_bias=False), BatchNormalization(scale=False),
#                             BatchNormalization(center=False, scale=False), Dense(2)
#   keras-free-images.keras   Conv2D(4, 3, use_bias=False, padding="same"), Conv2DTranspose(3, 2, use_bias=False),
#                             DepthwiseConv2D(2, use_bias=False), SeparableConv2D(4, 2, use_bias=False), SeparableConv2D(3, 1)
#   keras-free-series.keras   Conv1D(4, 3, use_bias=False, padding="causal"), SeparableConv1D(3, 3, padding="same"),
#                             SeparableConv1D(2, 2, use_bias=False), Conv1DTranspose(2, 3, strides=2, use_bias=False)
#   keras-free-volume.keras   Conv3D(3, 2, use_bias=False), Conv3DTranspose(2, 2, use_bias=False)

import json
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
from keras import layers

here = os.path.dirname(os.path.abspath(__file__))
keras.utils.set_random_seed(20261014)
rng = np.random.default_rng(20261014)

nets = {
    "keras-free-dense": (
        [
            layers.Dense(6, use_bias=False, activation="relu"),
            layers.BatchNormalization(center=False),
            layers.Dense(4, use_bias=False, activation="tanh"),
            layers.LayerNormalization(scale=False),
            layers.Dense(3, use_bias=False),
            layers.BatchNormalization(scale=False),
            layers.BatchNormalization(center=False, scale=False),
        ],
        (8, 5)),
    "keras-free-images": (
        [
            layers.Conv2D(4, 3, use_bias=False, padding="same"),
            layers.Conv2DTranspose(3, 2, use_bias=False),
            layers.DepthwiseConv2D(2, use_bias=False),
            layers.SeparableConv2D(4, 2, use_bias=False),
            layers.SeparableConv2D(3, 1),
        ],
        (6, 6, 6, 2)),
    "keras-free-series": (
        [
            layers.Conv1D(4, 3, use_bias=False, padding="causal"),
            layers.SeparableConv1D(3, 3, padding="same"),
            layers.SeparableConv1D(2, 2, use_bias=False),
            layers.Conv1DTranspose(2, 3, strides=2, use_bias=False),
        ],
        (6, 11, 2)),
    "keras-free-volume": (
        [layers.Conv3D(3, 2, use_bias=False), layers.Conv3DTranspose(2, 2, use_bias=False)],
        (6, 4, 4, 4, 2)),
}

fixture = {"made": f"keras {keras.__version__}, backend {keras.backend.backend()}"}
print(fixture["made"])

for name, (built, shape) in nets.items():
    model = keras.Sequential([layers.Input(shape=shape[1:])] + built + [layers.Flatten(), layers.Dense(2)])
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")

    # Trained a few epochs on random examples and random answers, so that every number has moved.
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

with open(os.path.join(here, "keras-free.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
