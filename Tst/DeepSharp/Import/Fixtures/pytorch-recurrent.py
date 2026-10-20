# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the files PyTorch saves the state of four networks that read a series a step at a time - as safetensors and as torch.save
# writes them - and pytorch-recurrent.json: the series each was handed and what PyTorch answers through it. The tests read a file here
# into a network written in DeepSharp, each layer called what the module is called, and hold what it answers to PyTorch's. Its output,
# when it was run, is pytorch-recurrent.txt beside it.
#
# Run from this folder with PyTorch 2.14.1 for the CPU and safetensors 0.8.0:
#
#     python pytorch-recurrent.py > pytorch-recurrent.txt
#
#   recurrent-lstm.*   rnn = LSTM(3, 4, 2 layers, both ways) read to its last state, head = Linear(8, 2)
#   recurrent-gru.*    rnn = GRU(3, 4) read at every step, flattened, head = Linear(20, 2)
#   recurrent-conv.*   conv = Conv1d(2, 3, 2), ReLU, rnn = LSTM(3, 4) read to its last state, head = Linear(4, 1)
#   recurrent-conv-sequence.*  conv = Conv1d(2, 3, 2), ReLU, rnn = LSTM(3, 4) read at every step, flattened, head = Linear(16, 1)
#
# A series goes into PyTorch steps by channels in a batch, and into a convolution channels first, so the convolution is handed the series
# turned round and the recurrent layer what the convolution made turned round again: DeepSharp holds a series channels last throughout.

import json
import os
import platform

import numpy as np
import safetensors
import torch
from safetensors.torch import save_file

here = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(1)
torch.manual_seed(20261021)
rng = np.random.default_rng(20261021)

nn = torch.nn


class LastState(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.LSTM(3, 4, num_layers=2, bidirectional=True, batch_first=True)
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        _, (h, _) = self.rnn(x)
        return self.head(torch.cat([h[-2], h[-1]], dim=1))


class EveryStep(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = nn.GRU(3, 4, batch_first=True)
        self.head = nn.Linear(20, 2)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out.reshape(out.shape[0], -1))


class ConvThenLast(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 3, 2)
        self.rnn = nn.LSTM(3, 4, batch_first=True)
        self.head = nn.Linear(4, 1)

    def forward(self, x):
        seen = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        _, (h, _) = self.rnn(seen)
        return self.head(h[-1])


class ConvThenEvery(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(2, 3, 2)
        self.rnn = nn.LSTM(3, 4, batch_first=True)
        self.head = nn.Linear(16, 1)

    def forward(self, x):
        seen = torch.relu(self.conv(x.transpose(1, 2))).transpose(1, 2)
        out, _ = self.rnn(seen)
        return self.head(out.reshape(out.shape[0], -1))


# Each network, and the shape of the examples as DeepSharp lays them out: steps by channels, and the series' count in front.
networks = {
    "lstm": (LastState(), (4, 5, 3)),
    "gru": (EveryStep(), (4, 5, 3)),
    "conv": (ConvThenLast(), (4, 6, 2)),
    "conv-sequence": (ConvThenEvery(), (4, 5, 2)),
}


def values(array):
    return [float(value) for value in np.asarray(array).reshape(-1)]


fixture = {"torch": torch.__version__, "safetensors": safetensors.__version__, "python": platform.python_version()}

for name, (model, shape) in networks.items():
    model.eval()
    examples = rng.normal(size=shape).astype(np.float32)

    with torch.no_grad():
        outputs = model(torch.from_numpy(examples)).numpy()

    state = {key: tensor.contiguous() for key, tensor in model.state_dict().items()}
    save_file(state, os.path.join(here, f"recurrent-{name}.safetensors"), metadata={"format": "pt"})
    torch.save(model.state_dict(), os.path.join(here, f"recurrent-{name}.pt"))

    fixture[name] = {
        "keys": list(state.keys()),
        "examples": {"shape": list(shape), "values": values(examples)},
        "outputs": values(outputs),
    }
    print(name, "keys", list(state.keys()), "outputs of the first example", values(outputs)[:2])

with open(os.path.join(here, "pytorch-recurrent.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
