# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes a Keras model whose first Dense layer has a low-rank pair beside its held kernel (enable_lora), in both files Keras saves
# a model to, and keras-lora.json: the examples it was handed, what Keras answers through it, and the numbers of the pair. Its
# output, when it was run, is keras-lora.txt beside it.
#
# The two files hold the pair differently, which is what the reader is held to:
#
#   keras-lora.keras   the archive Keras 3 saves to: the kernel is saved MERGED - the held kernel plus the pair's product scaled by
#                      alpha over the rank - and neither matrix of the pair, with lora_rank and lora_alpha in config.json.
#   keras-lora.h5      the HDF5 file Keras saved to before: the held kernel, the bias and the two matrices, each as it is.
#
# The model is compiled with a mean squared error, since a network here answers through the loss its file names, and trained a
# few epochs on random answers so that the pair and the biases have moved: the kernel does not.
#
# Run from this folder with Keras 3.15.1 on its torch backend:
#
#     python keras-lora-fixtures.py > keras-lora.txt

import json
import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
from keras import layers

here = os.path.dirname(os.path.abspath(__file__))
keras.utils.set_random_seed(20261019)
rng = np.random.default_rng(20261019)

RANK, ALPHA = 2, 4.0

model = keras.Sequential([layers.Input(shape=(3,)), layers.Dense(4, activation="relu", name="wide"), layers.Dense(1, name="narrow")])
wide = model.layers[0]
wide.enable_lora(RANK, lora_alpha=ALPHA)
model.compile(optimizer=keras.optimizers.Adam(0.05), loss="mse")
model.fit(rng.normal(size=(48, 3)).astype(np.float32), rng.normal(size=(48, 1)).astype(np.float32), epochs=8, batch_size=16, verbose=0)

kernel = wide._kernel.numpy()
a = wide.lora_kernel_a.numpy()
b = wide.lora_kernel_b.numpy()
bias = wide.bias.numpy()

assert np.abs(b).max() > 1e-3, "the pair has not moved"
assert not wide._kernel.trainable

examples = rng.normal(size=(6, 3)).astype(np.float32)
outputs = np.asarray(model.predict(examples, verbose=0), dtype=np.float32)

model.save(os.path.join(here, "keras-lora.keras"))
model.save(os.path.join(here, "keras-lora.h5"))


def values(array):
    return [float(value) for value in np.asarray(array).reshape(-1)]


fixture = {
    "made": f"keras {keras.__version__}, backend {keras.backend.backend()}",
    "rank": RANK,
    "alpha": ALPHA,
    "examples": {"shape": list(examples.shape), "values": values(examples)},
    "outputs": values(outputs),
    "merged": values(kernel + (ALPHA / RANK) * (a @ b)),
    "kernel": values(kernel),
    "a": values(a),
    "b": values(b),
    "bias": values(bias),
}

print(fixture["made"])
print(f"rank {RANK}, alpha {ALPHA}; largest number of b {np.abs(b).max()!r}")

with open(os.path.join(here, "keras-lora.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
