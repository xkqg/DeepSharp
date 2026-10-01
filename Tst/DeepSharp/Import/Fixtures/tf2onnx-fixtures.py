# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the ONNX graphs tf2onnx converts TensorFlow SavedModels into, which the ONNX importer's tests read, and what
# TensorFlow answered through each network. Its output, when it was run, is tf2onnx-fixtures.txt beside it.
#
# Run from this folder with TensorFlow 2.21.0, Keras 3.15.1 on TensorFlow, tf2onnx 1.17.0 and onnx, on Python 3.12 (the
# newest Python TensorFlow installs on):
#
#     KERAS_BACKEND=tensorflow python tf2onnx-fixtures.py
#
# Each network is exported as a TensorFlow SavedModel by Keras's model.export(format="tf_saved_model"), and the SavedModel
# converted with `python -m tf2onnx.convert --saved-model <folder> --output <file>`, as a TensorFlow model reaches ONNX.
#
#   tf2onnx-titanic.onnx               keras-titanic.keras — the networks sample's Titanic network Keras trained, which the
#                                      Keras importer's tests read — loaded into Keras on TensorFlow, for batches of any
#                                      length. What Keras answered through it stands in keras-fixtures.json beside it.
#   tf2onnx-convolution.onnx           Conv2D(4, (3, 2), padding="same"), ReLU, Conv2D(3, 2, strides=2), Flatten,
#                                      Dense(5, tanh), Dense(1) — the shapes of the PyTorch network over images the other
#                                      fixtures hold, every bias drawn, over six images of 8 rows by 6 columns and 2
#                                      channels — exported for a batch of six, so the flatten's target is a number.
#   tf2onnx-convolution-batches.onnx   the same network with a batch normalisation after each convolution and after the
#                                      flatten, numbers drawn, exported for batches of any length, as model.export does
#                                      unless told otherwise: what tf2onnx writes then is refused.
#   tf2onnx-fixtures.json              the images as TensorFlow was handed them, channels last, and its outputs for them.

import json
import os
import platform
import subprocess
import sys
import tempfile

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import keras
import numpy as np
import onnx
import tensorflow as tf
import tf2onnx
from keras import layers
from onnx.reference import ReferenceEvaluator

here = os.path.dirname(os.path.abspath(__file__))
work = tempfile.mkdtemp()
answers = json.load(open(os.path.join(here, "keras-fixtures.json"), encoding="utf-8"))["titanic"]

print(f"tensorflow {tf.__version__}, keras {keras.__version__}, tf2onnx {tf2onnx.__version__}, onnx {onnx.__version__}, numpy {np.__version__}, Python {platform.python_version()}")


def converted(model, name, signature=None):
    saved = os.path.join(work, name)
    if signature is None:
        model.export(saved, format="tf_saved_model", verbose=False)
    else:
        model.export(saved, format="tf_saved_model", verbose=False, input_signature=signature)
    path = os.path.join(here, f"{name}.onnx")
    run = subprocess.run([sys.executable, "-m", "tf2onnx.convert", "--saved-model", saved, "--output", path], capture_output=True, text=True)
    if run.returncode != 0:
        print(run.stderr)
        raise SystemExit(run.returncode)
    graph = onnx.load(path)
    print(f"{name}.onnx: {os.path.getsize(path)} bytes; opset {graph.opset_import[0].version}; producer {graph.producer_name} {graph.producer_version}; "
          + " ".join(node.op_type for node in graph.graph.node))
    return ReferenceEvaluator(graph)


def network(normalised):
    body = [keras.Input((8, 6, 2)), layers.Conv2D(4, (3, 2), padding="same", bias_initializer="random_normal")]
    body += [layers.BatchNormalization()] if normalised else []
    body += [layers.ReLU(), layers.Conv2D(3, 2, strides=2, bias_initializer="random_normal")]
    body += [layers.BatchNormalization()] if normalised else []
    body += [layers.Flatten()]
    body += [layers.BatchNormalization()] if normalised else []
    body += [layers.Dense(5, activation="tanh", bias_initializer="random_normal"), layers.Dense(1, bias_initializer="random_normal")]
    model = keras.Sequential(body)
    for norm in (layer for layer in model.layers if isinstance(layer, layers.BatchNormalization)):
        features = norm.moving_mean.shape[0]
        norm.set_weights([rng.uniform(0.5, 1.5, features).astype("float32"), rng.uniform(-0.2, 0.2, features).astype("float32"),
                          rng.uniform(-0.3, 0.3, features).astype("float32"), rng.uniform(0.5, 1.5, features).astype("float32")])
    return model


# The Titanic network Keras trained, read on TensorFlow without its optimizer, which Keras saved as the one of its PyTorch
# backend; only the network is exported.
titanic = keras.saving.load_model(os.path.join(here, "keras-titanic.keras"), compile=False)
evaluator = converted(titanic, "tf2onnx-titanic")
for part in ("test", "served"):
    rows = np.array(answers[part]["rows"], dtype="float32")
    keras_chances = np.array(answers[part]["chances"], dtype="float32")
    chances = evaluator.run(None, {evaluator.input_names[0]: rows})[0].reshape(-1)
    print(f"titanic {part}: {len(rows)} rows; the graph within {np.abs(chances - keras_chances).max():.3g} of Keras's chances on PyTorch,"
          f" TensorFlow within {np.abs(titanic(rows).numpy().reshape(-1) - keras_chances).max():.3g}")

# Convolutions over images with more rows than columns and more than one channel, flattened into a dense layer.
keras.utils.set_random_seed(20260935)
rng = np.random.default_rng(20260935)
images = rng.normal(size=(6, 8, 6, 2)).astype("float32")  # image, row, column, channel: TensorFlow's layout and DeepSharp's
convolution = network(normalised=False)
outputs = convolution(images, training=False).numpy()
evaluator = converted(convolution, "tf2onnx-convolution", [tf.TensorSpec((6, 8, 6, 2), tf.float32)])
print(f"convolution: the graph within {np.abs(evaluator.run(None, {evaluator.input_names[0]: images})[0] - outputs).max():.3g} of TensorFlow's outputs")
print("convolution outputs:", " ".join(f"{value:.8f}" for value in outputs.reshape(-1)))

batches = network(normalised=True)
converted(batches, "tf2onnx-convolution-batches")

with open(os.path.join(here, "tf2onnx-fixtures.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump({"made": f"tensorflow {tf.__version__}, keras {keras.__version__}, tf2onnx {tf2onnx.__version__}",
               "convolution": {"images": {"shape": list(images.shape), "values": [float(value) for value in images.reshape(-1)]},
                               "outputs": [float(value) for value in outputs.reshape(-1)]}}, file, indent=1)
    file.write("\n")
