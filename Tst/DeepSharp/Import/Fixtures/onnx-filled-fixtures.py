# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the ONNX graphs PyTorch's default exporter makes of networks whose paddings fill the places they add from the example
# - mirrored back (ONNX's Pad of mode reflect), repeated from the edge (edge) and gone round (wrap) - and onnx-filled.json:
# the examples each network was handed and what PyTorch answered through it. Its output, when it was run, is
# onnx-filled.txt beside it.
#
# Run with PyTorch 2.14.1 for the CPU, onnx and onnxscript, with UTF-8 output (the default exporter prints a tick):
#
#     python -X utf8 onnx-filled-fixtures.py > onnx-filled.txt
#
# The TorchScript exporter writes the same layers as a graph that works its pads out from the example's shape, which no
# network here is built of, so only the default exporter's graphs are kept. Wrap is ONNX's since operator set 19.
#
#   onnx-filled-series.onnx   ReflectionPad1d((2, 1)), Conv1d(2, 3, 2), ReplicationPad1d((1, 2)), Conv1d(3, 2, 2), Flatten, Linear
#   onnx-filled-image.onnx    ReflectionPad2d((1, 2, 2, 1)), Conv2d(2, 3, 2), ReplicationPad2d((1, 0, 0, 2)), CircularPad2d((1, 2, 2, 1)),
#                             Conv2d(3, 2, 2), Flatten, Linear
#   onnx-filled-volume.onnx   CircularPad3d((1, 1, 1, 1, 1, 1)), Conv3d(2, 3, 2), ReflectionPad3d((1, 0, 1, 0, 0, 1)), ReplicationPad3d(1),
#                             Flatten, Linear

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
nn = torch.nn

bodies = {
    "onnx-filled-series": (
        [nn.ReflectionPad1d((2, 1)), nn.Conv1d(2, 3, 2), nn.ReplicationPad1d((1, 2)), nn.Conv1d(3, 2, 2), nn.Flatten()],
        (6, 2, 9),
    ),
    "onnx-filled-image": (
        [nn.ReflectionPad2d((1, 2, 2, 1)), nn.Conv2d(2, 3, 2), nn.ReplicationPad2d((1, 0, 0, 2)), nn.CircularPad2d((1, 2, 2, 1)), nn.Conv2d(3, 2, 2), nn.Flatten()],
        (6, 2, 6, 6),
    ),
    "onnx-filled-volume": (
        [nn.CircularPad3d((1, 1, 1, 1, 1, 1)), nn.Conv3d(2, 3, 2), nn.ReflectionPad3d((1, 0, 1, 0, 0, 1)), nn.ReplicationPad3d(1), nn.Flatten()],
        (6, 2, 3, 3, 3),
    ),
}

made = f"torch {torch.__version__}, onnx {onnx.__version__}, onnxscript {onnxscript.__version__}, numpy {np.__version__}, Python {platform.python_version()}"
fixture = {"made": made}
print(made)

for name, (body, shape) in bodies.items():
    example = torch.randn(*shape)

    with torch.no_grad():
        width = nn.Sequential(*body)(example).shape[1]

    model = nn.Sequential(*body, nn.Linear(width, 2))

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.3)

    model.eval()

    with torch.no_grad():
        answers = model(example)

    path = os.path.join(here, f"{name}.onnx")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model, (example,), path, input_names=["x"], output_names=["y"],
            dynamic_shapes=({0: torch.export.Dim("batch")},), verbose=False)

    graph = onnx.load(path, load_external_data=False)
    print(f"{name}: {os.path.getsize(path)} bytes; opset {graph.opset_import[0].version}; "
          + " ".join(node.op_type + "".join(f"[{a.s.decode()}]" for a in node.attribute if a.name == "mode") for node in graph.graph.node))

    # example, then the axes, then the channel: this library's layout; the examples are handed to PyTorch channels first.
    axes = len(shape)
    last_first = example.permute(0, *range(2, axes), 1).contiguous()
    fixture[name] = {
        "examples": {"shape": list(last_first.shape), "values": [float(value) for value in last_first.reshape(-1)]},
        "outputs": [float(value) for value in answers.reshape(-1)],
    }

with open(os.path.join(here, "onnx-filled.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
