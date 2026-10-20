# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes two Keras functional models, each in both files Keras 3 saves a model to, and keras-functional.json: the examples each
# was handed, what Keras answers through it, and the numbers of its layers. Its output, when it was run, is
# keras-functional.txt beside it.
#
#   keras-residual   one input of six values: a dense layer, a second one, the input added back (Add), the input laid beside that
#                    (Concatenate) and a head - a residual step and a dense block in one model, every layer called by a name of its own.
#   keras-two-inputs two inputs, of three and of four values: each through a layer of its own, then added - the model Keras is
#                    handed a list of two arrays for, which a network here is handed as the columns of one row, in the order of the
#                    model's inputs.
#
# Each model is compiled with a mean squared error and trained a few epochs on random answers, so that every number has moved.
#
# Run from this folder with Keras 3.15.1 on its torch backend:
#
#     python keras-functional-fixtures.py > keras-functional.txt

import json
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
from keras import layers

here = os.path.dirname(os.path.abspath(__file__))
keras.utils.set_random_seed(20261020)
rng = np.random.default_rng(20261020)


def values(array):
    return [float(value) for value in np.asarray(array).reshape(-1)]


def weights(model):
    return {layer.name: [values(weight) for weight in layer.get_weights()] for layer in model.layers if layer.get_weights()}


rows = keras.Input(shape=(6,), name="rows")
first = layers.Dense(6, activation="relu", name="first")(rows)
second = layers.Dense(6, name="second")(first)
added = layers.Add(name="residual")([rows, second])
joined = layers.Concatenate(name="joined")([rows, added])
head = layers.Dense(2, name="head")(joined)
residual = keras.Model(rows, head, name="residual_model")
residual.compile(optimizer=keras.optimizers.Adam(0.05), loss="mse")
residual.fit(rng.normal(size=(48, 6)).astype(np.float32), rng.normal(size=(48, 2)).astype(np.float32), epochs=6, batch_size=16, verbose=0)

left = keras.Input(shape=(3,), name="left")
right = keras.Input(shape=(4,), name="right")
seen = layers.Dense(5, activation="relu", name="seen")(left)
heard = layers.Dense(5, name="heard")(right)
meeting = layers.Add(name="meeting")([seen, heard])
answer = layers.Dense(1, name="answer")(meeting)
two = keras.Model([left, right], answer, name="two_inputs_model")
two.compile(optimizer=keras.optimizers.Adam(0.05), loss="mse")
two.fit([rng.normal(size=(48, 3)).astype(np.float32), rng.normal(size=(48, 4)).astype(np.float32)], rng.normal(size=(48, 1)).astype(np.float32), epochs=6, batch_size=16, verbose=0)

examples = rng.normal(size=(5, 6)).astype(np.float32)
lefts = rng.normal(size=(5, 3)).astype(np.float32)
rights = rng.normal(size=(5, 4)).astype(np.float32)

for name, model in (("residual", residual), ("two-inputs", two)):
    model.save(os.path.join(here, f"keras-{name}.keras"))
    model.save(os.path.join(here, f"keras-{name}.h5"))

fixture = {
    "made": f"keras {keras.__version__}, backend {keras.backend.backend()}",
    "residual": {
        "examples": {"shape": list(examples.shape), "values": values(examples)},
        "outputs": values(residual.predict(examples, verbose=0)),
        "weights": weights(residual),
    },
    "twoInputs": {
        "lefts": {"shape": list(lefts.shape), "values": values(lefts)},
        "rights": {"shape": list(rights.shape), "values": values(rights)},
        "outputs": values(two.predict([lefts, rights], verbose=0)),
        "weights": weights(two),
    },
}

print(fixture["made"])
print("residual outputs of the first example", fixture["residual"]["outputs"][:2])
print("two inputs outputs of the first example", fixture["twoInputs"]["outputs"][:1])

with open(os.path.join(here, "keras-functional.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
