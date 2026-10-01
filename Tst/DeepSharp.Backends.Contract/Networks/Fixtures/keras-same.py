# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes keras-same.json: a network Keras describes and runs, with every number it holds and what it answers, for the
# contract to hold every engine's network in Keras's words to. Its output, when it was run, is keras-same.txt beside it.
#
# Run with Keras 3.15.1 on PyTorch 2.10.0 for the CPU (the backend Keras runs on here is PyTorch's; TensorFlow's 'same'
# and Keras's are one rule, which Keras's own backend for PyTorch works out as TensorFlow does):
#
#     KERAS_BACKEND=torch python keras-same.py
#
# The network walks every setting the Keras words translate: a window of four rows by three columns with a stride of two
# over 7 by 8 images, padded as 'same' — one row of nothing before and two after, none before and one column after, where
# no border stated for every side gives the same numbers — a batch normalisation with Keras's momentum and epsilon, whose
# running variance is small enough for the epsilon to matter, an even window at a stride of one padded as 'same' — none
# before, one after — a layer normalisation with Keras's epsilon, and a last sigmoid, which Keras applies in the network
# and DeepSharp in the loss.

import json
import os
import platform

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import torch
from keras import layers

keras.utils.set_random_seed(20260930)
rng = np.random.default_rng(20260930)

model = keras.Sequential(
    [
        keras.Input((7, 8, 2)),
        layers.Conv2D(4, (4, 3), strides=2, padding="same"),
        layers.BatchNormalization(momentum=0.9, epsilon=1e-3),
        layers.ReLU(),
        layers.Conv2D(3, 2, padding="same"),
        layers.Flatten(),
        layers.Dense(5),
        layers.LayerNormalization(epsilon=1e-3),
        layers.Activation("tanh"),
        layers.Dense(1),
        layers.Activation("sigmoid"),
    ]
)

for place in (0, 3, 5, 8):
    bias = model.layers[place].bias
    bias.assign(rng.uniform(-0.1, 0.1, bias.shape).astype("float32"))

norm = model.layers[1]
norm.gamma.assign(rng.uniform(0.5, 1.5, 4).astype("float32"))
norm.beta.assign(rng.uniform(-0.2, 0.2, 4).astype("float32"))
norm.moving_mean.assign(rng.uniform(-0.3, 0.3, 4).astype("float32"))
norm.moving_variance.assign(rng.uniform(0.0005, 0.004, 4).astype("float32"))
layer_norm = model.layers[6]
layer_norm.gamma.assign(rng.uniform(0.5, 1.5, 5).astype("float32"))
layer_norm.beta.assign(rng.uniform(-0.2, 0.2, 5).astype("float32"))

images = rng.normal(size=(6, 7, 8, 2)).astype("float32")

# Each Keras variable under the path of the DeepSharp slot it goes into: the layer's place, and the slot's name there.
names = {"kernel": "weight", "bias": "bias", "gamma": "weight", "beta": "bias", "moving_mean": "running_mean", "moving_variance": "running_var"}


def values(array):
    return [float(value) for value in np.asarray(array, dtype="float32").reshape(-1)]


slots = {}
for place, layer in enumerate(model.layers):
    for weight in layer.weights:
        name = weight.path.split("/")[-1]
        array = keras.ops.convert_to_numpy(weight)
        slots[f"{place}.{names[name]}"] = {"keras": weight.path, "shape": list(array.shape), "values": values(array)}

logits = keras.ops.convert_to_numpy(keras.Model(model.inputs[0], model.layers[-2].output)(images, training=False))
predictions = keras.ops.convert_to_numpy(model(images, training=False))

# One training pass: the batch normalisation measures the batch and moves its running statistics, Keras's way.
model(images, training=True)
moved = {
    "1.running_mean": values(keras.ops.convert_to_numpy(norm.moving_mean)),
    "1.running_var": values(keras.ops.convert_to_numpy(norm.moving_variance)),
}

fixture = {
    "made": f"keras {keras.__version__} on torch {torch.__version__}, numpy {np.__version__}, Python {platform.python_version()}",
    "images": {"shape": list(images.shape), "values": values(images)},
    "slots": slots,
    "logits": {"shape": list(logits.shape), "values": values(logits)},
    "predictions": {"shape": list(predictions.shape), "values": values(predictions)},
    "trained": moved,
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "keras-same.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")

print(fixture["made"])
for path, slot in slots.items():
    print(f"{path:16} <- {slot['keras']:40} {slot['shape']}")
print("logits      ", " ".join(f"{value:.8f}" for value in fixture["logits"]["values"]))
print("predictions ", " ".join(f"{value:.8f}" for value in fixture["predictions"]["values"]))
print("moved mean  ", " ".join(f"{value:.8f}" for value in moved["1.running_mean"]))
print("moved var   ", " ".join(f"{value:.8f}" for value in moved["1.running_var"]))
