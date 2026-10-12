# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the graphs PyTorch's newer exporter makes of a convolution that gives the places back and of a layer that only
# pads, and onnx-transposed.json: the images each was handed and what PyTorch answered through it. The tests read each graph
# into the network it is and hold what it answers to PyTorch's. Its output, when it was run, is onnx-transposed.txt beside it.
#
# Run from this folder with PyTorch 2.14.1 for the CPU:
#
#     python onnx-transposed-fixtures.py > onnx-transposed.txt
#
#   onnx-transposed.onnx          ConvTranspose2d(2, 4, 3, stride=2, padding=1), ReLU, ConvTranspose2d(4, 3, 2), Flatten, Linear
#   onnx-padded.onnx              ZeroPad2d((1, 2, 2, 0)), Conv2d(2, 3, 3), Flatten, Linear: the pads before the window
#   onnx-padded-value.onnx        ConstantPad2d((1, 1, 0, 2), -1.5), Conv2d(2, 3, 3), Flatten, Linear
#   onnx-transposed-series.onnx   ConvTranspose1d(2, 3, 3, stride=2), Flatten, Linear
#   onnx-transposed-volume.onnx   ConvTranspose3d(2, 2, 2, stride=(2, 1, 2)), Flatten, Linear
#   onnx-padded-series.onnx       ConstantPad1d((2, 1), 0.5), Conv1d(2, 3, 2), Flatten, Linear
#   onnx-padded-volume.onnx       ConstantPad3d((1, 1, 0, 1, 1, 0), -0.5), Conv3d(2, 2, 2), Flatten, Linear

import io
import json
import os
import warnings

import numpy as np
import onnx
import torch

warnings.filterwarnings("ignore")
here = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(1)
torch.manual_seed(20261012)
rng = np.random.default_rng(20261012)
nn = torch.nn

nets = {
    "onnx-transposed": (
        nn.Sequential(
            nn.ConvTranspose2d(2, 4, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 3, 2),
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 2),
        ),
        (5, 4, 4, 2),
    ),
    "onnx-padded": (
        nn.Sequential(
            nn.ZeroPad2d((1, 2, 2, 0)),
            nn.Conv2d(2, 3, 3),
            nn.Flatten(),
            nn.Linear(3 * 5 * 6, 2),
        ),
        (5, 5, 5, 2),
    ),
    "onnx-padded-value": (
        nn.Sequential(
            nn.ConstantPad2d((1, 1, 0, 2), -1.5),
            nn.Conv2d(2, 3, 3),
            nn.Flatten(),
            nn.Linear(3 * 5 * 5, 2),
        ),
        (5, 5, 5, 2),
    ),
    "onnx-transposed-series": (
        nn.Sequential(
            nn.ConvTranspose1d(2, 3, 3, stride=2),
            nn.Flatten(),
            nn.Linear(3 * 9, 2),
        ),
        (5, 4, 2),
    ),
    "onnx-transposed-volume": (
        nn.Sequential(
            nn.ConvTranspose3d(2, 2, 2, stride=(2, 1, 2)),
            nn.Flatten(),
            nn.Linear(2 * 6 * 4 * 6, 2),
        ),
        (5, 3, 3, 3, 2),
    ),
    "onnx-padded-series": (
        nn.Sequential(
            nn.ConstantPad1d((2, 1), 0.5),
            nn.Conv1d(2, 3, 2),
            nn.Flatten(),
            nn.Linear(3 * 8, 2),
        ),
        (5, 6, 2),
    ),
    "onnx-padded-volume": (
        nn.Sequential(
            nn.ConstantPad3d((1, 1, 0, 1, 1, 0), -0.5),
            nn.Conv3d(2, 2, 2),
            nn.Flatten(),
            nn.Linear(2 * 3 * 3 * 4, 2),
        ),
        (5, 3, 3, 3, 2),
    ),
}

fixture = {"made": f"torch {torch.__version__}, onnx {onnx.__version__}"}
print(fixture["made"])

for name, (model, shape) in nets.items():
    model.eval()
    examples = rng.normal(size=shape).astype(np.float32)          # example, then the axes walked, then the channel: DeepSharp's layout
    axes = len(shape)
    first = torch.from_numpy(examples).permute(0, axes - 1, *range(1, axes - 1)).contiguous()   # PyTorch's: example, channel, axes

    with torch.no_grad():
        outputs = model(first)

    buffer = io.BytesIO()
    torch.onnx.export(model, (first,), buffer, dynamo=True, opset_version=20)
    graph = onnx.load_from_string(buffer.getvalue())
    onnx.save(graph, os.path.join(here, f"{name}.onnx"))

    fixture[name] = {
        "examples": {"shape": list(examples.shape), "values": [float(value) for value in examples.reshape(-1)]},
        "outputs": [float(value) for value in outputs.reshape(-1).numpy()],
    }
    print(f"{name}: {[node.op_type for node in graph.graph.node]}")
    print(f"  outputs {list(outputs.shape)}:", " ".join(f"{value:.8f}" for value in fixture[name]["outputs"][:8]), "...")

with open(os.path.join(here, "onnx-transposed.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
