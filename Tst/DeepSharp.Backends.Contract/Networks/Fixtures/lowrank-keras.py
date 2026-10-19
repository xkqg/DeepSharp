# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Prints what Keras's own Dense layer does once enable_lora has been called: the answer is the kernel's product plus the product of
# the pair of matrices, scaled by alpha over the rank, and a step of training moves the pair and the bias and leaves the kernel
# where it was. lowrank-keras.json is what LowRankContract holds a dense layer with a pair to, on any engine: the layer's numbers,
# the answers to four rows, and the numbers after one step of plain gradient descent on the mean squared error. Its output, when it
# was run, is lowrank-keras.txt beside it.
#
# Run once with Keras 3.15.1 on the PyTorch backend and PyTorch 2.14.1 for the CPU:
#
#     KERAS_BACKEND=torch python lowrank-keras.py > lowrank-keras.txt
#
# DeepSharp lays a dense layer's kernel out inputs by outputs, as Keras does, so nothing is turned round.

import json
import os
import platform

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))

rows = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0], [0.5, -0.5, 1.0], [2.0, 1.0, -1.0]], dtype="float32")
targets = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [-1.0, 1.0]], dtype="float32")
kernel = np.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]], dtype="float32")
bias = np.array([0.05, -0.05], dtype="float32")
a = np.array([[0.5, -0.25], [0.125, 0.75], [-0.375, 0.25]], dtype="float32")
b = np.array([[0.2, -0.4], [0.6, 0.1]], dtype="float32")
rank, alpha = 2, 6.0


def values(array):
    return [float(value) for value in np.asarray(array).reshape(-1).tolist()]


def walk(rate):
    dense = keras.layers.Dense(2, name="d")
    model = keras.Sequential([keras.layers.Input((3,)), dense])
    dense.kernel.assign(kernel)
    dense.bias.assign(bias)
    dense.enable_lora(rank, lora_alpha=alpha)
    dense.lora_kernel_a.assign(a)
    dense.lora_kernel_b.assign(b)
    answers = model.predict(rows, verbose=0)
    model.compile(optimizer=keras.optimizers.SGD(rate), loss="mse")
    loss = model.train_on_batch(rows, targets)
    return dense, answers, float(loss)


made = f"keras {keras.__version__} on {keras.backend.backend()}, Python {platform.python_version()}"
print(made)

dense, answers, loss = walk(0.1)
expected = rows @ kernel + (rows @ a @ b) * (alpha / rank) + bias
assert np.allclose(answers, expected, atol=1e-5), "the answer is not the kernel's product plus the pair's scaled by alpha over the rank"
assert not dense._kernel.trainable, "the kernel is held still once the pair exists"

print(f"answers {values(answers)}")
print(f"loss before the step {loss!r}")
print(f"after the step: kernel {values(dense._kernel.numpy())}")
print(f"                bias   {values(dense.bias.numpy())}")
print(f"                a      {values(dense.lora_kernel_a.numpy())}")
print(f"                b      {values(dense.lora_kernel_b.numpy())}")

assert np.allclose(dense._kernel.numpy(), kernel), "a step moved the held kernel"
assert not np.allclose(dense.lora_kernel_b.numpy(), b), "a step moved nothing of the pair"

fixture = {
    "made": made,
    "rank": rank,
    "alpha": alpha,
    "rate": 0.1,
    "rows": values(rows),
    "targets": values(targets),
    "kernel": values(kernel),
    "bias": values(bias),
    "a": values(a),
    "b": values(b),
    "answers": values(answers),
    "loss": loss,
    "after": {
        "kernel": values(dense._kernel.numpy()),
        "bias": values(dense.bias.numpy()),
        "a": values(dense.lora_kernel_a.numpy()),
        "b": values(dense.lora_kernel_b.numpy()),
    },
}

with open(os.path.join(here, "lowrank-keras.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
