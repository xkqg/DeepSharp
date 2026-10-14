# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the ONNX graphs PyTorch exports of layers that keep no bias and no shift, and onnx-free.json: the examples each
# network was handed and what PyTorch answered through it. Its output, when it was run, is onnx-free.txt beside it.
#
# Run with PyTorch 2.14.1 for the CPU, onnx and onnxscript, with UTF-8 output (the default exporter prints a tick):
#
#     python -X utf8 onnx-free-fixtures.py > onnx-free.txt
#
# Every network is exported twice, by the default exporter and by the TorchScript one before it. They write a layer without a
# bias in different words, which is why both are kept:
#
#   the default exporter     a Gemm with no third input, a Conv or a ConvTranspose with no third input, and a
#                            LayerNormalization with no third input
#   the TorchScript exporter a MatMul by weights written inputs by outputs, with no Add after it, a Conv or a ConvTranspose
#                            with no third input, and a LayerNormalization whose shift is a constant of noughts
#
#   onnx-free-rows        Linear(5, 4, bias=False), ReLU, Linear(4, 3, bias=False), LayerNorm(3, bias=False), Tanh, Linear(3, 2)
#   onnx-free-images      Conv2d(2, 4, 3, bias=False), ReLU, ConvTranspose2d(4, 3, 2, bias=False), Flatten, Linear(75, 2)
#   onnx-free-series      Conv1d(2, 4, 3, bias=False), ReLU, ConvTranspose1d(4, 3, 2, bias=False), Flatten, Linear(24, 2)
#   onnx-free-volume      Conv3d(2, 3, 2, bias=False), ReLU, ConvTranspose3d(3, 2, 2, bias=False), Flatten, Linear(128, 2)

import json
import os
import platform
import warnings

import numpy as np
import onnx
import onnxscript
import torch

here = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(1)
torch.manual_seed(20261014)

nets = {
    "onnx-free-rows": (
        torch.nn.Sequential(
            torch.nn.Linear(5, 4, bias=False), torch.nn.ReLU(),
            torch.nn.Linear(4, 3, bias=False), torch.nn.LayerNorm(3, bias=False), torch.nn.Tanh(),
            torch.nn.Linear(3, 2)),
        (6, 5)),
    "onnx-free-images": (
        torch.nn.Sequential(
            torch.nn.Conv2d(2, 4, 3, bias=False), torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(4, 3, 2, bias=False), torch.nn.Flatten(), torch.nn.Linear(75, 2)),
        (6, 2, 6, 6)),
    "onnx-free-series": (
        torch.nn.Sequential(
            torch.nn.Conv1d(2, 4, 3, bias=False), torch.nn.ReLU(),
            torch.nn.ConvTranspose1d(4, 3, 2, bias=False), torch.nn.Flatten(), torch.nn.Linear(24, 2)),
        (6, 2, 9)),
    "onnx-free-volume": (
        torch.nn.Sequential(
            torch.nn.Conv3d(2, 3, 2, bias=False), torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(3, 2, 2, bias=False), torch.nn.Flatten(), torch.nn.Linear(128, 2)),
        (6, 2, 4, 4, 4)),
}

made = f"torch {torch.__version__}, onnx {onnx.__version__}, onnxscript {onnxscript.__version__}, numpy {np.__version__}, Python {platform.python_version()}"
fixture = {"made": made}
print(made)

for name, (model, shape) in nets.items():
    # The numbers drawn, so that a layer normalisation's scale is no row of ones.
    for parameter in model.parameters():
        parameter.data = torch.randn_like(parameter) * 0.5

    model.eval()
    example = torch.randn(*shape)

    with torch.no_grad():
        answers = model(example)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model, (example,), os.path.join(here, f"{name}.onnx"), input_names=["x"], output_names=["y"],
            dynamic_shapes=({0: torch.export.Dim("batch")},), verbose=False)
        torch.onnx.export(model, (example,), os.path.join(here, f"{name}-torchscript.onnx"), dynamo=False, input_names=["x"], output_names=["y"])

    for suffix in ("", "-torchscript"):
        path = os.path.join(here, f"{name}{suffix}.onnx")
        graph = onnx.load(path, load_external_data=False)
        print(f"{name}{suffix}: {os.path.getsize(path)} bytes; opset {graph.opset_import[0].version}; "
              + " ".join(f"{node.op_type}" + ("" if len(node.input) == 0 else f"({len(node.input)})") for node in graph.graph.node))

    fixture[name] = {
        "examples": {"shape": list(example.shape), "values": [float(value) for value in example.reshape(-1)]},
        "outputs": [float(value) for value in answers.reshape(-1)],
    }

with open(os.path.join(here, "onnx-free.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
