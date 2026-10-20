# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes ONNX graphs that look across the steps of a series with the Attention operator of the twenty-third operator set, built by hand with
# onnx.helper, and onnx-attention.json: the series each was handed and what ONNX Runtime answers through it. Its output, when it was run, is
# onnx-attention.txt beside it.
#
#   onnx-attention.onnx          the series projected into queries, keys and values (a MatMul and an Add each), Attention over 2 heads, the output
#                                projected, flattened, Gemm(40 -> 2)
#   onnx-attention-causal.onnx   the same with is_causal: each step looks only at the steps up to its own
#   onnx-attention-self.onnx     Attention over the series itself (queries, keys and values the one value), flattened, Gemm
#
# Each takes a batch of series, steps by channels, which a MatMul multiplies along its last axis.
#
# Run from this folder with onnx 1.23.2 and onnxruntime 1.31.0:
#
#     python onnx-attention-fixtures.py > onnx-attention.txt

import json
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

here = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20261024)

STEPS, CHANNELS, HEADS = 5, 8, 2


def tensor(name, array):
    return numpy_helper.from_array(np.asarray(array, dtype=np.float32), name)


def values(array):
    return [float(value) for value in np.asarray(array).reshape(-1)]


def projection(prefix, source, nodes, initializers):
    """A series by a matrix and a bias, as TorchScript writes a linear layer over steps: a MatMul and an Add."""
    weight = rng.normal(scale=0.4, size=(CHANNELS, CHANNELS)).astype(np.float32)
    bias = rng.normal(scale=0.1, size=(CHANNELS,)).astype(np.float32)
    initializers += [tensor(f"{prefix}.weight", weight), tensor(f"{prefix}.bias", bias)]
    nodes += [
        helper.make_node("MatMul", [source, f"{prefix}.weight"], [f"{prefix}.product"], name=f"{prefix}/MatMul"),
        helper.make_node("Add", [f"{prefix}.product", f"{prefix}.bias"], [prefix], name=f"{prefix}/Add"),
    ]
    return prefix


def graph(name, causal, projected):
    nodes, initializers = [], []
    source = "series"
    if projected:
        queries, keys, values_ = (projection(part, source, nodes, initializers) for part in ("query", "key", "value"))
    else:
        queries = keys = values_ = source
    attributes = {"q_num_heads": HEADS, "kv_num_heads": HEADS}
    if causal:
        attributes["is_causal"] = 1
    nodes.append(helper.make_node("Attention", [queries, keys, values_], ["attended"], name="attention", **attributes))
    last = projection("output", "attended", nodes, initializers) if projected else "attended"
    nodes.append(helper.make_node("Flatten", [last], ["flat"], name="flatten", axis=1))
    head = rng.normal(scale=0.3, size=(2, STEPS * CHANNELS)).astype(np.float32)
    bias = rng.normal(scale=0.1, size=(2,)).astype(np.float32)
    initializers += [tensor("head.weight", head), tensor("head.bias", bias)]
    nodes.append(helper.make_node("Gemm", ["flat", "head.weight", "head.bias"], ["out"], name="head", transB=1))
    model = helper.make_model(
        helper.make_graph(
            nodes,
            name,
            [helper.make_tensor_value_info("series", TensorProto.FLOAT, ["N", STEPS, CHANNELS])],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, ["N", 2])],
            initializers,
        ),
        opset_imports=[helper.make_opsetid("", 23)],
        ir_version=11,
    )
    onnx.checker.check_model(model)
    return model


fixture = {"onnx": onnx.__version__, "onnxruntime": ort.__version__}
examples = rng.normal(size=(4, STEPS, CHANNELS)).astype(np.float32)

for name, model in {
    "attention": graph("attention", causal=False, projected=True),
    "attention-causal": graph("attention-causal", causal=True, projected=True),
    "attention-self": graph("attention-self", causal=False, projected=False),
}.items():
    onnx.save(model, os.path.join(here, f"onnx-{name}.onnx"))
    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"series": examples})[0]
    fixture[name] = {"examples": {"shape": list(examples.shape), "values": values(examples)}, "outputs": values(outputs)}
    print(name, "first outputs", values(outputs)[:2])

with open(os.path.join(here, "onnx-attention.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
