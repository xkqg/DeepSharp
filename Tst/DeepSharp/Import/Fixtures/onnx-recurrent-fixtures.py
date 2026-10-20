# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the graphs PyTorch's torch.onnx.export (the TorchScript exporter, operator set 17) makes of networks that read a series a step
# at a time, and onnx-recurrent.json: the series each was handed and what PyTorch answers through it. Its output, when it was run, is
# onnx-recurrent.txt beside it.
#
#   onnx-recurrent-lstm-last.onnx    LSTM(3, 4) to its last state, Linear(4, 2)
#   onnx-recurrent-lstm-every.onnx   LSTM(3, 4) at every step, flattened, Linear(20, 2)
#   onnx-recurrent-gru-every.onnx    GRU(3, 4) at every step, flattened, Linear(20, 2)
#   onnx-recurrent-gru-both.onnx     GRU(3, 4) both ways at every step, flattened, Linear(40, 2)
#   onnx-recurrent-lstm-two.onnx     LSTM(3, 4, two layers, both ways) to its last state, Linear(8, 2): the graph PyTorch spells the last
#                                    states of a stack out of with Gather and Concat, which the reader refuses by name
#
# Each network is handed series of batch-first steps by channels, and holds its state at nought, which the exporter spells as an Expand of a
# constant nought by a shape worked out from the series' own.
#
# Run from this folder with PyTorch 2.14.1 for the CPU and onnx 1.23.2:
#
#     python onnx-recurrent-fixtures.py > onnx-recurrent.txt

import json
import os
import platform

import numpy as np
import onnx
import torch

here = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(1)
torch.manual_seed(20261023)
rng = np.random.default_rng(20261023)

nn = torch.nn


class LstmLast(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(3, 4, batch_first=True)
        self.head = nn.Linear(4, 2)

    def forward(self, x):
        _, (h, _) = self.rnn(x)
        return self.head(h[-1])


class Every(nn.Module):
    def __init__(self, kind, both=False):
        super().__init__()
        self.rnn = kind(3, 4, batch_first=True, bidirectional=both)
        self.head = nn.Linear(40 if both else 20, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out.reshape(out.shape[0], -1))


class Stack(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(3, 4, num_layers=2, bidirectional=True, batch_first=True)
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        _, (h, _) = self.rnn(x)
        return self.head(torch.cat([h[-2], h[-1]], dim=1))


networks = {
    "lstm-last": LstmLast(),
    "lstm-every": Every(nn.LSTM),
    "gru-every": Every(nn.GRU),
    "gru-both": Every(nn.GRU, both=True),
    "lstm-two": Stack(),
}


def values(array):
    return [float(value) for value in np.asarray(array).reshape(-1)]


fixture = {"torch": torch.__version__, "onnx": onnx.__version__, "python": platform.python_version()}

for name, model in networks.items():
    model.eval()
    examples = rng.normal(size=(4, 5, 3)).astype(np.float32)

    with torch.no_grad():
        outputs = model(torch.from_numpy(examples)).numpy()

    torch.onnx.export(model, (torch.from_numpy(examples),), os.path.join(here, f"onnx-recurrent-{name}.onnx"), input_names=["series"], output_names=["out"], dynamo=False, opset_version=17)
    fixture[name] = {"examples": {"shape": list(examples.shape), "values": values(examples)}, "outputs": values(outputs)}
    print(name, "outputs of the first example", values(outputs)[:2])

with open(os.path.join(here, "onnx-recurrent.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
