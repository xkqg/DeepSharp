# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the ONNX graphs PyTorch's two exporters make of the words 0.11.0 reads, and onnx-vocabulary.json: the examples each
# graph was handed, as PyTorch lays them out, and what PyTorch answers through it. Its output, when it was run, is
# onnx-vocabulary.txt beside it: each graph's operators, so the spellings the reader matches are on the page.
#
# Run from this folder with PyTorch 2.14.1 for the CPU, onnx 1.23.2 and onnxscript as torch installs them:
#
#     python onnx-vocabulary-fixtures.py > onnx-vocabulary.txt
#
# An operator ONNX gained late is written out of others by an exporter asked for an older operator set: the GELU as Div, Erf,
# Add and two Muls, or as its hyperbolic tangent's chain; the SiLU as a Sigmoid and a Mul; the mish as a Softplus, a Tanh and a
# Mul; a group normalisation as a Reshape, an InstanceNormalization and a Reshape back, then its scale and shift; a root-mean-
# square normalisation as Pow, ReduceMean, Add, Sqrt, Reciprocal and two Muls. Each of those spellings is here as the exporter
# wrote it - the TorchScript one (onnx-vocabulary-ts-*) and the dynamo one (onnx-vocabulary-dynamo-*) - beside the single
# operators of the newer sets. Every network is a Linear(5, 6), the word, and a Linear(6, 2), so the word stands between two
# layers whose numbers the reader also reads; onnx-vocabulary-images is a Conv2d(2, 4, 3), an InstanceNorm2d(4, affine), a
# PReLU(4), a softmax over the channels, a Flatten and a Linear(64, 2).
#
# onnx-vocabulary-groupnorm18.onnx is written by hand with onnx.helper: ONNX's GroupNormalization of operator sets 18 to 20,
# whose scale and shift are one for each group, which the reader refuses.

import io
import json
import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper

here = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(1)
torch.manual_seed(20261016)
rng = np.random.default_rng(20261016)
nn = torch.nn


def rows(word):
    return nn.Sequential(nn.Linear(5, 6), word, nn.Linear(6, 2))


graphs = {
    "ts-gelu": (lambda: rows(nn.GELU()), 17, False),
    "ts-gelu-tanh": (lambda: rows(nn.GELU(approximate="tanh")), 17, False),
    "ts-silu": (lambda: rows(nn.SiLU()), 17, False),
    "ts-mish": (lambda: rows(nn.Mish()), 17, False),
    "ts-groupnorm": (lambda: rows(nn.GroupNorm(3, 6)), 17, False),
    "ts-leaky": (lambda: rows(nn.LeakyReLU(0.1)), 17, False),
    "ts-prelu": (lambda: rows(nn.PReLU(6)), 17, False),
    "ts-elu": (lambda: rows(nn.ELU(0.7)), 17, False),
    "ts-selu": (lambda: rows(nn.SELU()), 17, False),
    "ts-softmax": (lambda: rows(nn.Softmax(dim=1)), 17, False),
    "dynamo-gelu": (lambda: rows(nn.GELU()), 18, True),
    "dynamo-gelu-tanh": (lambda: rows(nn.GELU(approximate="tanh")), 18, True),
    "dynamo-silu": (lambda: rows(nn.SiLU()), 18, True),
    "dynamo-mish": (lambda: rows(nn.Mish()), 18, True),
    "dynamo-groupnorm": (lambda: rows(nn.GroupNorm(3, 6)), 18, True),
    "dynamo-rmsnorm": (lambda: rows(nn.RMSNorm(6)), 18, True),
    "dynamo-gelu-21": (lambda: rows(nn.GELU()), 21, True),
    "dynamo-gelu-tanh-21": (lambda: rows(nn.GELU(approximate="tanh")), 21, True),
    "dynamo-groupnorm-21": (lambda: rows(nn.GroupNorm(3, 6)), 21, True),
    "dynamo-rmsnorm-23": (lambda: rows(nn.RMSNorm(6)), 23, True),
    "images": (lambda: nn.Sequential(nn.Conv2d(2, 4, 3), nn.InstanceNorm2d(4, affine=True), nn.PReLU(4), nn.Softmax(dim=1), nn.Flatten(), nn.Linear(64, 2)), 17, False),
}

fixture = {"made": f"torch {torch.__version__}, onnx {onnx.__version__}"}
print(fixture["made"])

for name, (made, opset, dynamo) in graphs.items():
    model = made()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.4 + (0.6 if parameter.dim() == 1 else 0))

    model.eval()
    shape = (6, 2, 6, 6) if name == "images" else (6, 5)
    examples = torch.from_numpy(rng.normal(size=shape).astype(np.float32))

    with torch.no_grad():
        outputs = model(examples)

    path = os.path.join(here, f"onnx-vocabulary-{name}.onnx")

    if dynamo:
        torch.onnx.export(model, (examples,), dynamo=True, opset_version=opset, verbose=False).save(path)
    else:
        torch.onnx.export(model, (examples,), path, opset_version=opset, dynamo=False, input_names=["input"], output_names=["output"])

    graph = onnx.load(path)
    fixture[name] = {
        "examples": {"shape": list(shape), "values": [float(value) for value in examples.reshape(-1)]},
        "outputs": [float(value) for value in outputs.reshape(-1)],
    }
    print(f"onnx-vocabulary-{name}.onnx: opset {[each.version for each in graph.opset_import if each.domain == '']}, {[node.op_type for node in graph.graph.node]}")

# ONNX's GroupNormalization as operator sets 18 to 20 define it: a scale and a shift for each group.
weights = rng.normal(size=(6, 5)).astype(np.float32)
nodes = [
    helper.make_node("Gemm", ["input", "w", "b"], ["hidden"], transB=1),
    helper.make_node("GroupNormalization", ["hidden", "scale", "shift"], ["output"], num_groups=3, epsilon=1e-5),
]
initializers = [
    helper.make_tensor("w", TensorProto.FLOAT, [6, 5], weights.reshape(-1).tolist()),
    helper.make_tensor("b", TensorProto.FLOAT, [6], [0.0] * 6),
    helper.make_tensor("scale", TensorProto.FLOAT, [3], [1.0, 1.0, 1.0]),
    helper.make_tensor("shift", TensorProto.FLOAT, [3], [0.0, 0.0, 0.0]),
]
handmade = helper.make_model(
    helper.make_graph(nodes, "groupnorm18", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 5])], [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, 6])], initializers),
    opset_imports=[helper.make_opsetid("", 18)])
onnx.save(handmade, os.path.join(here, "onnx-vocabulary-groupnorm18.onnx"))
print("onnx-vocabulary-groupnorm18.onnx: opset [18], ['Gemm', 'GroupNormalization'] written by hand, a scale and a shift for each of 3 groups")

with open(os.path.join(here, "onnx-vocabulary.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
