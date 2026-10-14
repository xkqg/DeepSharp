# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the Keras models whose layers carry rules for their numbers - regularizers, MaxNorm constraints and layers that do not
# train - and keras-rules.json: the examples each was handed and what Keras answers through it. Its output, when it was run, is
# keras-rules.txt beside it. A regularizer, a constraint and trainable change nothing of what a model answers, so the answers hold
# the models to being read as they were; the rules themselves are held to what the files say of them.
#
# Run from this folder with Keras 3.15.1 on its torch backend:
#
#     python keras-rules-fixtures.py > keras-rules.txt
#
#   keras-rules-dense.keras    Dense(6, relu, L2(0.01) kernel, L1(0.02) bias, MaxNorm(2) kernel), BatchNormalization (not trainable),
#                              Dense(4, L1L2(0.001, 0.002) kernel, MaxNorm(1.5) bias), Dense(3, tanh, not trainable), Dense(2)
#   keras-rules-images.keras   Conv2D(4, 3, L2(0.01) kernel, MaxNorm(2, axis=[0, 1, 2]) kernel, MaxNorm(1) bias),
#                              DepthwiseConv2D(2, L1(0.01) depthwise, MaxNorm(1.5, axis=[0, 1]) depthwise),
#                              SeparableConv2D(3, 1, L1(0.03) depthwise, L2(0.02) pointwise, MaxNorm(2, axis=[0, 1, 2]) pointwise,
#                              L2(0.04) bias), Flatten, Dense(2)

import json
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
from keras import constraints, layers, regularizers

here = os.path.dirname(os.path.abspath(__file__))
keras.utils.set_random_seed(20261014)
rng = np.random.default_rng(20261014)

fixture = {"made": f"keras {keras.__version__}, backend {keras.backend.backend()}"}
print(fixture["made"])


def build(name, built, shape, frozen=()):
    model = keras.Sequential([layers.Input(shape=shape[1:])] + built + [layers.Flatten(), layers.Dense(2)] if name == "keras-rules-dense" else [layers.Input(shape=shape[1:])] + built)
    for place in frozen:
        model.layers[place].trainable = False
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
    model.fit(
        rng.normal(size=(48, *shape[1:])).astype(np.float32),
        rng.normal(size=(48, 2)).astype(np.float32),
        epochs=4,
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
    print("  trainable:", [layer.trainable for layer in model.layers])
    print("  weights:", [(weight.path.split('/')[-1], tuple(weight.shape)) for weight in model.weights])


build(
    "keras-rules-dense",
    [
        layers.Dense(6, activation="relu", kernel_regularizer=regularizers.L2(0.01), bias_regularizer=regularizers.L1(0.02), kernel_constraint=constraints.MaxNorm(2.0)),
        layers.BatchNormalization(),
        layers.Dense(4, kernel_regularizer=regularizers.L1L2(l1=0.001, l2=0.002), bias_constraint=constraints.MaxNorm(1.5)),
        layers.Dense(3, activation="tanh"),
    ],
    (8, 5),
    frozen=(1, 3))

build(
    "keras-rules-images",
    [
        layers.Conv2D(4, 3, kernel_regularizer=regularizers.L2(0.01), kernel_constraint=constraints.MaxNorm(2.0, axis=[0, 1, 2]), bias_constraint=constraints.MaxNorm(1.0)),
        layers.DepthwiseConv2D(2, depthwise_regularizer=regularizers.L1(0.01), depthwise_constraint=constraints.MaxNorm(1.5, axis=[0, 1])),
        layers.SeparableConv2D(
            3, 1, depthwise_regularizer=regularizers.L1(0.03), pointwise_regularizer=regularizers.L2(0.02),
            pointwise_constraint=constraints.MaxNorm(2.0, axis=[0, 1, 2]), bias_regularizer=regularizers.L2(0.04)),
        layers.Flatten(),
        layers.Dense(2),
    ],
    (6, 6, 6, 2))

with open(os.path.join(here, "keras-rules.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
