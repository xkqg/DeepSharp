# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes Keras models that look across the steps of a series, each in both files Keras 3 saves a model to, and keras-attention.json: the
# series each was handed and what Keras answers through it. Its output, when it was run, is keras-attention.txt beside it.
#
#   keras-attention          MultiHeadAttention(2 heads, key_dim 4) over the series itself, averaged over the steps, Dense(2)
#   keras-attention-causal   the same, each step looking only at the steps up to its own (use_causal_mask)
#   keras-attention-block    the attention added back to the series and normalised, a feed-forward part added back and normalised, averaged, Dense(2)
#   keras-attention-cross    MultiHeadAttention over two series (query and value apart): a model a network here does not read
#
# Each model is compiled with a mean squared error and trained a few epochs on random answers, so that every number has moved.
#
# Run from this folder with Keras 3.15.1 on its torch backend:
#
#     python keras-attention-fixtures.py > keras-attention.txt

import json
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
from keras import layers

here = os.path.dirname(os.path.abspath(__file__))
keras.utils.set_random_seed(20261023)
rng = np.random.default_rng(20261023)

STEPS, CHANNELS = 5, 8


def pooled(inputs, looked):
    return layers.Dense(2)(layers.GlobalAveragePooling1D()(looked))


def make_attention():
    inputs = keras.Input(shape=(STEPS, CHANNELS))
    return keras.Model(inputs, pooled(inputs, layers.MultiHeadAttention(num_heads=2, key_dim=4)(inputs, inputs)))


def make_causal():
    inputs = keras.Input(shape=(STEPS, CHANNELS))
    return keras.Model(inputs, pooled(inputs, layers.MultiHeadAttention(num_heads=2, key_dim=4)(inputs, inputs, use_causal_mask=True)))


def make_block():
    inputs = keras.Input(shape=(STEPS, CHANNELS))
    looked = layers.MultiHeadAttention(num_heads=2, key_dim=4)(inputs, inputs)
    first = layers.LayerNormalization(epsilon=1e-5)(layers.Add()([inputs, looked]))
    wide = layers.Dense(16, activation="relu")(first)
    second = layers.LayerNormalization(epsilon=1e-5)(layers.Add()([first, layers.Dense(CHANNELS)(wide)]))
    return keras.Model(inputs, pooled(inputs, second))


def make_cross():
    inputs = keras.Input(shape=(STEPS, CHANNELS))
    other = keras.Input(shape=(STEPS, CHANNELS))
    return keras.Model([inputs, other], pooled(inputs, layers.MultiHeadAttention(num_heads=2, key_dim=4)(inputs, other)))


models = {"attention": make_attention(), "attention-causal": make_causal(), "attention-block": make_block(), "attention-cross": make_cross()}


def values(array):
    return [float(value) for value in np.asarray(array).reshape(-1)]


fixture = {"made": f"keras {keras.__version__}, backend {keras.backend.backend()}"}
examples = rng.normal(size=(6, STEPS, CHANNELS)).astype(np.float32)

for name, model in models.items():
    model.compile(optimizer=keras.optimizers.Adam(0.05), loss="mse")
    several = len(model.inputs) > 1
    train = [rng.normal(size=(48, STEPS, CHANNELS)).astype(np.float32) for _ in model.inputs] if several else rng.normal(size=(48, STEPS, CHANNELS)).astype(np.float32)
    model.fit(train, rng.normal(size=(48, 2)).astype(np.float32), epochs=6, batch_size=16, verbose=0)
    model.save(os.path.join(here, f"keras-{name}.keras"))
    model.save(os.path.join(here, f"keras-{name}.h5"))
    asked = [examples, examples] if several else examples
    fixture[name] = {
        "examples": {"shape": list(examples.shape), "values": values(examples)},
        "outputs": values(model.predict(asked, verbose=0)),
    }
    print(name, "first outputs", fixture[name]["outputs"][:2])

with open(os.path.join(here, "keras-attention.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
