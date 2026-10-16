# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the Keras models made of the words 0.11.0 reads, and keras-vocabulary.json: the examples each was handed and what Keras
# answers through it. Its output, when it was run, is keras-vocabulary.txt beside it.
#
# Each is compiled with a mean squared error, since a network here answers through the loss its file names, and trained a few
# epochs on random answers, so that every number has moved.
#
# Run from this folder with Keras 3.15.1 on its torch backend:
#
#     python keras-vocabulary-fixtures.py > keras-vocabulary.txt
#
#   keras-vocabulary-rows.keras       Dense(6, gelu), Dense(6, silu), Dense(6, mish), Dense(6, leaky_relu), LeakyReLU(0.1), PReLU,
#                                     ELU(0.7), Dense(6, elu), Activation(selu), GroupNormalization(3), RMSNormalization,
#                                     Dense(4, softmax) - the last softmax kept, a mean squared error applying none
#   keras-vocabulary-images.keras     Conv2D(4, 3), GroupNormalization(2), PReLU(shared_axes=[1, 2]), GroupNormalization(-1),
#                                     Softmax, Flatten, Dense(2)
#   keras-vocabulary-embedding.keras  Embedding(5, 3), Flatten, Dense(2), over one category of five: Keras is handed its place,
#                                     and DeepSharp the row of noughts and a one its encoder writes it down as
#   keras-vocabulary-rms-scaling.keras  LayerNormalization(rms_scaling=True), which is refused

import json
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
from keras import layers

here = os.path.dirname(os.path.abspath(__file__))
keras.utils.set_random_seed(20261016)
rng = np.random.default_rng(20261016)

nets = {
    "keras-vocabulary-rows": (
        [
            layers.Dense(6, activation="gelu"),
            layers.Dense(6, activation="silu"),
            layers.Dense(6, activation="mish"),
            layers.Dense(6, activation="leaky_relu"),
            layers.LeakyReLU(negative_slope=0.1),
            layers.PReLU(),
            layers.ELU(alpha=0.7),
            layers.Dense(6, activation="elu"),
            layers.Activation("selu"),
            layers.GroupNormalization(groups=3),
            layers.RMSNormalization(),
            layers.Dense(4, activation="softmax"),
        ],
        (8, 5)),
    "keras-vocabulary-images": (
        [
            layers.Conv2D(4, 3),
            layers.GroupNormalization(groups=2),
            layers.PReLU(shared_axes=[1, 2]),
            layers.GroupNormalization(groups=-1),
            layers.Softmax(),
            layers.Flatten(),
            layers.Dense(2),
        ],
        (6, 6, 6, 2)),
}

fixture = {"made": f"keras {keras.__version__}, backend {keras.backend.backend()}"}
print(fixture["made"])


def trained(model, examples, answers):
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
    model.fit(examples, answers, epochs=6, batch_size=16, verbose=0)


def keep(name, model, examples, handed):
    outputs = np.asarray(model.predict(handed, verbose=0), dtype=np.float32)
    model.save(os.path.join(here, f"{name}.keras"))
    fixture[name] = {
        "examples": {"shape": list(examples.shape), "values": [float(value) for value in examples.reshape(-1)]},
        "outputs": [float(value) for value in outputs.reshape(-1)],
    }
    print(f"{name}: {[layer.__class__.__name__ for layer in model.layers]} -> {list(outputs.shape)}")
    print("  weights:", [(weight.path.split('/')[-1], tuple(weight.shape)) for weight in model.weights])
    print("  outputs:", " ".join(f"{value:.8f}" for value in fixture[name]["outputs"][:6]), "...")


for name, (built, shape) in nets.items():
    model = keras.Sequential([layers.Input(shape=shape[1:])] + built)
    width = model.output_shape[-1]
    trained(model, rng.normal(size=(48, *shape[1:])).astype(np.float32), rng.normal(size=(48, width)).astype(np.float32))
    examples = rng.normal(size=shape).astype(np.float32)
    keep(name, model, examples, examples)

# The embedding: Keras looks a category up by its place; DeepSharp multiplies the row of noughts and a one by the same weight.
embedding = keras.Sequential([layers.Input(shape=(1,), dtype="int32"), layers.Embedding(5, 3), layers.Flatten(), layers.Dense(2)])
trained(embedding, rng.integers(0, 5, size=(48, 1)).astype(np.int32), rng.normal(size=(48, 2)).astype(np.float32))
places = np.array([[0], [3], [1], [4], [2], [3]], dtype=np.int32)
keep("keras-vocabulary-embedding", embedding, np.eye(5, dtype=np.float32)[places[:, 0]], places)

# A layer normalisation that scales by rms_scaling, which Keras works out from the variance and which is refused.
scaled = keras.Sequential([layers.Input(shape=(4,)), layers.Dense(4), layers.LayerNormalization(rms_scaling=True), layers.Dense(1)])
scaled.compile(optimizer="adam", loss="mse")
scaled.save(os.path.join(here, "keras-vocabulary-rms-scaling.keras"))
print("keras-vocabulary-rms-scaling:", [layer.__class__.__name__ for layer in scaled.layers])

with open(os.path.join(here, "keras-vocabulary.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
