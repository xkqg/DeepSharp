# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the models Keras saved that the Keras importer's tests read for the layers that walk a series, an image or a volume
# (convolutions, poolings, global poolings and the dropouts of whole channels), and what Keras answered with each. Its
# output, when it was run, is keras-spatial-fixtures.txt beside it. Run with Keras 3.15.1 on PyTorch for the CPU:
#
#     KERAS_BACKEND=torch python keras-spatial-fixtures.py
#
#   keras-spatial-series.keras, .h5       a series of 13 steps and 3 channels: a causal Conv1D of window 3 (relu), a
#                                         MaxPooling1D of 2, a SpatialDropout1D, an AveragePooling1D of 3 at a stride of 2
#                                         ('valid': the last step is left over), a GlobalAveragePooling1D and a Dense layer
#                                         of two numbers under a mean squared error.
#   keras-spatial-image.keras, .h5        an image of 9 rows, 8 columns and 2 channels: a Conv2D of 3 by 2 (relu), a
#                                         MaxPooling2D of 3 by 2 at a stride of 2 padded as 'same' (one row before and one
#                                         after, none columns before and one after), an AveragePooling2D of 2, a
#                                         SpatialDropout2D, a GlobalMaxPooling2D and a softmax over three classes.
#   keras-spatial-volume.keras, .h5       a volume of 6 planes, 5 rows, 7 columns and 2 channels: a Conv3D of 2 by 3 by 2
#                                         (relu), a MaxPooling3D of 2, an AveragePooling3D of 1 by 1 by 2, a
#                                         GlobalAveragePooling3D that keeps its axes, a Flatten and a sigmoid.
#   keras-spatial-series-same.keras       a series of 10 steps: a Conv1D 'valid' of window 3 (tanh), a Conv1D of window 4 at
#                                         a stride of 2 padded as 'same', an AveragePooling1D of 2 padded as 'same', a
#                                         GlobalMaxPooling1D that keeps its axis, a Flatten and a softmax.
#   keras-spatial-image-same.keras        an image of 7 by 7 and 2 channels: a Conv2D of 3 at a stride of 2 padded as 'same'
#                                         (sigmoid), an AveragePooling2D of 3 at a stride of 2 padded as 'same', a
#                                         GlobalAveragePooling2D and a sigmoid.
#   keras-spatial-volume-same.keras       a volume of 5 by 5 by 5 and 2 channels: a Conv3D of 2 padded as 'same' (tanh), a
#                                         MaxPooling3D of 2 padded as 'same', a SpatialDropout3D, a GlobalMaxPooling3D and
#                                         a Dense layer of two numbers under a mean squared error.
#   keras-spatial.json                    the inputs as they were handed to Keras and what each network answered, flattened.
#
# WHAT KERAS ANSWERED, AND WHAT IS HELD AGAINST IT. Keras 3 on its PyTorch backend does not give TensorFlow's answer for an
# AveragePooling padded as 'same' when a window covers padding and the stride is above one: TensorFlow (and the ONNX
# AveragePool, and this library) leave the padded cells out of the count, the PyTorch backend does not. The padding the
# library holds itself to is TensorFlow's. So every network is also worked out here by hand, in numpy, from its own numbers
# and TensorFlow's documented rule (padding 'same' pads as many places as the stride fits, the odd place after; 'causal'
# pads the window less one before; a maximum never sees padding; an average counts the cells that are real), and the answer
# written to the json is Keras's own whenever the two agree to a few roundings of a single-precision number. For a network
# where the PyTorch backend differs (declared below, and checked: a network that differs without saying so, or says so
# without differing, stops this script) the answer written is the one worked out by hand, and the json says so.

import itertools
import json
import os
import platform

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import torch
from keras import layers

torch.set_num_threads(1)
here = os.path.dirname(os.path.abspath(__file__))


def values(array):
    return [float(value) for value in np.asarray(keras.ops.convert_to_numpy(array), dtype="float32").reshape(-1)]


# ---- TensorFlow's rule, worked out by hand -------------------------------------------------------------------------------

def border(length, size, stride, mode):
    """The places of nothing before and after a side of this length: 'valid' none, 'causal' the window less one before, 'same' as many as the stride fits."""
    if mode == "valid":
        return 0, 0
    if mode == "causal":
        return size - 1, 0
    places = -(-length // stride)
    total = max((places - 1) * stride + size - length, 0)
    return total // 2, total - total // 2


def patches(x, sizes, stride, mode, fill):
    """Every window over x (N, axes..., C): (N, places..., sizes..., C), and how many cells of each window are real."""
    axes = x.shape[1:-1]
    pads = [border(length, size, stride, mode) for length, size in zip(axes, sizes)]
    padded = np.pad(x, [(0, 0)] + pads + [(0, 0)], constant_values=fill)
    real = np.pad(np.ones(axes + (1,)), pads + [(0, 0)], constant_values=0.0)
    places = [(padded.shape[at + 1] - size) // stride + 1 for at, size in enumerate(sizes)]
    found = np.empty((x.shape[0], *places, *sizes, x.shape[-1]))
    counted = np.empty((*places, *sizes, 1))
    for place in itertools.product(*[range(count) for count in places]):
        window = tuple(slice(at * stride, at * stride + size) for at, size in zip(place, sizes))
        found[(slice(None), *place)] = padded[(slice(None), *window)]
        counted[place] = real[window]
    return found, counted


def pooled(x, sizes, stride, mode, how):
    n = len(sizes)
    if how == "max":
        found, _ = patches(x, sizes, stride, mode, -np.inf)
        return found.reshape(found.shape[:1 + n] + (-1, found.shape[-1])).max(axis=-2)
    found, counted = patches(x, sizes, stride, mode, 0.0)
    total = found.reshape(found.shape[:1 + n] + (-1, found.shape[-1])).sum(axis=-2)
    return total / counted.reshape(counted.shape[:n] + (-1,)).sum(axis=-1)[None, ..., None]


def convolved(x, kernel, bias, stride, mode):
    sizes = kernel.shape[:-2]
    found, _ = patches(x, sizes, stride, mode, 0.0)
    rows = found.reshape(found.shape[:1 + len(sizes)] + (-1,))
    return rows @ kernel.reshape(-1, kernel.shape[-1]) + bias


def activated(name, x):
    if name == "linear":
        return x
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "tanh":
        return np.tanh(x)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if name == "softmax":
        shifted = np.exp(x - x.max(axis=-1, keepdims=True))
        return shifted / shifted.sum(axis=-1, keepdims=True)
    raise ValueError(name)


def by_hand(model, x):
    x = np.asarray(x, dtype=np.float64)
    for layer in model.layers:
        kind = type(layer).__name__
        config = layer.get_config()
        if kind.startswith("Conv"):
            kernel, bias = [np.asarray(number, dtype=np.float64) for number in layer.get_weights()]
            x = activated(config["activation"], convolved(x, kernel, bias, config["strides"][0], config["padding"]))
        elif kind.startswith("MaxPooling") or kind.startswith("AveragePooling"):
            how = "max" if kind.startswith("Max") else "average"
            x = pooled(x, tuple(config["pool_size"]), config["strides"][0], config["padding"], how)
        elif kind.startswith("GlobalMaxPooling") or kind.startswith("GlobalAveragePooling"):
            reduce = np.max if kind.startswith("GlobalMax") else np.mean
            x = reduce(x, axis=tuple(range(1, x.ndim - 1)), keepdims=config["keepdims"])
        elif kind == "Flatten":
            x = x.reshape(x.shape[0], -1)
        elif kind == "Dense":
            kernel, bias = [np.asarray(number, dtype=np.float64) for number in layer.get_weights()]
            x = activated(config["activation"], x @ kernel + bias)
        elif kind.startswith("SpatialDropout"):
            pass
        else:
            raise ValueError(kind)
    return x


# ---- The six networks ----------------------------------------------------------------------------------------------------

def make(name, shape, build, loss, targets, torch_differs, seed, save_legacy):
    keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    model = keras.Sequential([keras.Input(shape)] + build())
    model.compile(optimizer=keras.optimizers.Adam(0.01), loss=loss)

    # Trained a few epochs on random inputs and random answers, so that every number has moved: a bias stays nought otherwise.
    pixels = rng.normal(size=(48, *shape)).astype("float32")
    model.fit(pixels, targets(rng, 48), epochs=6, batch_size=16, verbose=0)

    model.save(os.path.join(here, f"{name}.keras"))

    if save_legacy:
        model.save(os.path.join(here, f"{name}.h5"))

    seen = rng.normal(size=(6, *shape)).astype("float32")
    answered = np.asarray(keras.ops.convert_to_numpy(model(seen, training=False)), dtype=np.float64)
    worked = by_hand(model, seen)
    apart = float(np.abs(answered - worked).max())
    differs = apart > 1e-4
    print(f"{name}: {[layer.output.shape[1:] for layer in model.layers]}")
    print(f"{name}: Keras on PyTorch and the rule worked out by hand are {apart:.2e} apart"
          f"{' (the PyTorch backend does not pool as TensorFlow does)' if differs else ''}")

    if differs != torch_differs:
        raise SystemExit(f"{name}: expected the PyTorch backend to {'differ' if torch_differs else 'agree'}, and it does not")

    chosen = worked if differs else answered

    return {
        "file": name,
        "shape": list(shape),
        "rows": [values(row) for row in seen],
        "answers": values(chosen.astype("float32")),
        "answeredBy": "numpy, TensorFlow's rule: the PyTorch backend averages a padded window otherwise" if differs else "keras",
    }


def classes(count):
    return lambda rng, n: keras.utils.to_categorical(rng.integers(0, count, size=n), count).astype("float32")


def amounts(count):
    return lambda rng, n: rng.normal(size=(n, count)).astype("float32")


def yes_or_no(rng, n):
    return rng.integers(0, 2, size=(n, 1)).astype("float32")


made = f"keras {keras.__version__} on torch {torch.__version__}, numpy {np.__version__}, Python {platform.python_version()}"
print(made)

answers = {"made": made}

answers["series"] = make(
    "keras-spatial-series", (13, 3),
    lambda: [
        layers.Conv1D(5, 3, padding="causal", activation="relu"),
        layers.MaxPooling1D(2),
        layers.SpatialDropout1D(0.3),
        layers.AveragePooling1D(3, strides=2, padding="valid"),
        layers.GlobalAveragePooling1D(),
        layers.Dense(2),
    ],
    "mean_squared_error", amounts(2), torch_differs=False, seed=20261009, save_legacy=True)

answers["image"] = make(
    "keras-spatial-image", (9, 8, 2),
    lambda: [
        layers.Conv2D(4, (3, 2), activation="relu"),
        layers.MaxPooling2D((3, 2), strides=2, padding="same"),
        layers.AveragePooling2D(2),
        layers.SpatialDropout2D(0.25),
        layers.GlobalMaxPooling2D(),
        layers.Dense(3, activation="softmax"),
    ],
    keras.losses.CategoricalCrossentropy(), classes(3), torch_differs=False, seed=20261010, save_legacy=True)

answers["volume"] = make(
    "keras-spatial-volume", (6, 5, 7, 2),
    lambda: [
        layers.Conv3D(3, (2, 3, 2), activation="relu"),
        layers.MaxPooling3D(2),
        layers.AveragePooling3D((1, 1, 2), strides=1),
        layers.GlobalAveragePooling3D(keepdims=True),
        layers.Flatten(),
        layers.Dense(1, activation="sigmoid"),
    ],
    "binary_crossentropy", yes_or_no, torch_differs=False, seed=20261011, save_legacy=True)

answers["seriesSame"] = make(
    "keras-spatial-series-same", (10, 3),
    lambda: [
        layers.Conv1D(4, 3, padding="valid", activation="tanh"),
        layers.Conv1D(4, 4, strides=2, padding="same"),
        layers.AveragePooling1D(2, strides=1, padding="same"),
        layers.GlobalMaxPooling1D(keepdims=True),
        layers.Flatten(),
        layers.Dense(3, activation="softmax"),
    ],
    "categorical_crossentropy", classes(3), torch_differs=False, seed=20261012, save_legacy=False)

answers["imageSame"] = make(
    "keras-spatial-image-same", (7, 7, 2),
    lambda: [
        layers.Conv2D(3, 3, strides=2, padding="same", activation="sigmoid"),
        layers.AveragePooling2D(3, strides=2, padding="same"),
        layers.GlobalAveragePooling2D(),
        layers.Dense(1, activation="sigmoid"),
    ],
    "binary_crossentropy", yes_or_no, torch_differs=True, seed=20261013, save_legacy=False)

answers["volumeSame"] = make(
    "keras-spatial-volume-same", (5, 5, 5, 2),
    lambda: [
        layers.Conv3D(3, 2, padding="same", activation="tanh"),
        layers.MaxPooling3D(2, padding="same"),
        layers.SpatialDropout3D(0.2),
        layers.GlobalMaxPooling3D(),
        layers.Dense(2),
    ],
    "mean_squared_error", amounts(2), torch_differs=False, seed=20261014, save_legacy=False)

with open(os.path.join(here, "keras-spatial.json"), "w", encoding="utf-8", newline="\n") as file:
    # One line for each network: the numbers are many, and a line each would only make the file long.
    file.write("{\n" + ",\n".join(f" {json.dumps(key)}: {json.dumps(value)}" for key, value in answers.items()) + "\n}\n")

for name in sorted(name for name in os.listdir(here) if name.startswith("keras-spatial-") and name.endswith((".keras", ".h5"))):
    print(f"{name}: {os.path.getsize(os.path.join(here, name))} bytes")
