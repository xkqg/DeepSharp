# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the ONNX graph Keras 3 exports its Titanic network to, which the ONNX importer's tests read, and checks it answers
# as Keras did. Its output, when it was run, is keras-onnx-fixtures.txt beside it.
#
# Run from this folder with Keras 3.15.1 on PyTorch 2.10.0 for the CPU, onnx and onnxscript:
#
#     KERAS_BACKEND=torch python keras-onnx-fixtures.py
#
#   keras-titanic-export.onnx   keras-titanic.keras — the networks sample's Titanic network Keras trained, which the Keras
#                               importer's tests read — exported by the model's own model.export(format="onnx"). What
#                               Keras answered through it stands in keras-fixtures.json beside it.

import json
import os
import platform
import warnings

os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import onnx
import torch
from onnx.reference import ReferenceEvaluator

torch.set_num_threads(1)
here = os.path.dirname(os.path.abspath(__file__))
answers = json.load(open(os.path.join(here, "keras-fixtures.json"), encoding="utf-8"))["titanic"]

print(f"keras {keras.__version__} on torch {torch.__version__}, onnx {onnx.__version__}, numpy {np.__version__}, Python {platform.python_version()}")

model = keras.saving.load_model(os.path.join(here, "keras-titanic.keras"))
served = np.array(answers["served"]["rows"], dtype="float32")
test = np.array(answers["test"]["rows"], dtype="float32")
model(served)  # a model is exported once it has been called

path = os.path.join(here, "keras-titanic-export.onnx")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.export(path, format="onnx", verbose=False)

graph = onnx.load(path)
print(f"keras-titanic-export.onnx: {os.path.getsize(path)} bytes" + (" and a file beside it" if os.path.exists(path + ".data") else "")
      + f"; opset {graph.opset_import[0].version}; " + " ".join(node.op_type for node in graph.graph.node))

# The graph answers as Keras answered: its chances against those Keras gave when it was saved.
evaluator = ReferenceEvaluator(graph)
for part, rows in (("test", test), ("served", served)):
    chances = evaluator.run(None, {evaluator.input_names[0]: rows})[0].reshape(-1)
    keras_chances = np.array(answers[part]["chances"], dtype="float32")
    print(f"{part}: {len(rows)} rows, the graph within {np.abs(chances - keras_chances).max():.3g} of Keras's chances")
