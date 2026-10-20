# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes Keras models that read a series a step at a time, each in both files Keras 3 saves a model to, and keras-recurrent.json: the
# series each was handed and what Keras answers through it. Its output, when it was run, is keras-recurrent.txt beside it.
#
#   keras-lstm      LSTM(4) read to its last state, Dense(2)
#   keras-gru       GRU(4) read at every step, Flatten, Dense(2)
#   keras-stacked   LSTM(4) at every step, GRU(3) to its last state, Dense(1)
#   keras-both      Bidirectional(LSTM(4)) to its last states, Dense(2)
#   keras-both-seq  Bidirectional(GRU(3), return_sequences) every step, Flatten, Dense(1)
#   keras-gru-old   GRU(4, reset_after=False): the form Keras 2 made the default, which a network here does not read
#
# Each model is compiled with a mean squared error and trained a few epochs on random answers, so that every number has moved.
#
# Run from this folder with Keras 3.15.1 on its torch backend:
#
#     python keras-recurrent-fixtures.py > keras-recurrent.txt

import json
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
from keras import layers

here = os.path.dirname(os.path.abspath(__file__))
keras.utils.set_random_seed(20261022)
rng = np.random.default_rng(20261022)

STEPS, CHANNELS = 5, 3

models = {
    "lstm": keras.Sequential([layers.Input(shape=(STEPS, CHANNELS)), layers.LSTM(4), layers.Dense(2)]),
    "gru": keras.Sequential([layers.Input(shape=(STEPS, CHANNELS)), layers.GRU(4, return_sequences=True), layers.Flatten(), layers.Dense(2)]),
    "stacked": keras.Sequential([layers.Input(shape=(STEPS, CHANNELS)), layers.LSTM(4, return_sequences=True), layers.GRU(3), layers.Dense(1)]),
    "both": keras.Sequential([layers.Input(shape=(STEPS, CHANNELS)), layers.Bidirectional(layers.LSTM(4)), layers.Dense(2)]),
    "both-seq": keras.Sequential(
        [layers.Input(shape=(STEPS, CHANNELS)), layers.Bidirectional(layers.GRU(3, return_sequences=True)), layers.Flatten(), layers.Dense(1)]
    ),
    "gru-old": keras.Sequential([layers.Input(shape=(STEPS, CHANNELS)), layers.GRU(4, reset_after=False), layers.Dense(1)]),
}


def values(array):
    return [float(value) for value in np.asarray(array).reshape(-1)]


fixture = {"made": f"keras {keras.__version__}, backend {keras.backend.backend()}"}
examples = rng.normal(size=(6, STEPS, CHANNELS)).astype(np.float32)

for name, model in models.items():
    outputs = model.output_shape[-1]
    model.compile(optimizer=keras.optimizers.Adam(0.05), loss="mse")
    model.fit(rng.normal(size=(48, STEPS, CHANNELS)).astype(np.float32), rng.normal(size=(48, outputs)).astype(np.float32), epochs=6, batch_size=16, verbose=0)
    model.save(os.path.join(here, f"keras-{name}.keras"))
    model.save(os.path.join(here, f"keras-{name}.h5"))
    fixture[name] = {
        "examples": {"shape": list(examples.shape), "values": values(examples)},
        "outputs": values(model.predict(examples, verbose=0)),
    }
    print(name, "first outputs", fixture[name]["outputs"][:2])

with open(os.path.join(here, "keras-recurrent.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
