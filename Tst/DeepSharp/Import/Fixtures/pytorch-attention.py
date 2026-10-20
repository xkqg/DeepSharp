# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the files PyTorch saves the state of two networks that look across the steps of a series - as safetensors and as torch.save
# writes them - and pytorch-attention.json: the series each was handed and what PyTorch answers through it. The tests read a file here
# into a network written in DeepSharp, each layer called what the module is called, and hold what it answers to PyTorch's. Its output,
# when it was run, is pytorch-attention.txt beside it.
#
# Run from this folder with PyTorch 2.14.1 for the CPU and safetensors 0.8.0:
#
#     python pytorch-attention.py > pytorch-attention.txt
#
#   attention-pooled.*   attn = MultiheadAttention(8, 2), the steps averaged, head = Linear(8, 2)
#   attention-encoder.*  the sine positions added, layer = TransformerEncoderLayer(8, 2, 16), the steps averaged, head = Linear(8, 2)
#
# The sine positions have no state, so they are worked out in the forward pass here by the formula of 'Attention Is All You Need'.

import json
import math
import os
import platform

import numpy as np
import safetensors
import torch
from safetensors.torch import save_file

here = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(1)
torch.manual_seed(20261022)
rng = np.random.default_rng(20261022)

nn = torch.nn


def positions(steps, width):
    return torch.tensor(
        [[(math.sin if c % 2 == 0 else math.cos)(t / 10000 ** ((c - c % 2) / width)) for c in range(width)] for t in range(steps)],
        dtype=torch.float32,
    )


class Pooled(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(8, 2, batch_first=True)
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        looked, _ = self.attn(x, x, x, need_weights=False)
        return self.head(looked.mean(dim=1))


class Encoded(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(8, 2, 16, dropout=0.0, batch_first=True)
        self.head = nn.Linear(8, 2)

    def forward(self, x):
        return self.head(self.layer(x + positions(x.shape[1], x.shape[2])).mean(dim=1))


networks = {
    "pooled": (Pooled(), (4, 5, 8)),
    "encoder": (Encoded(), (4, 5, 8)),
}


def values(array):
    return [float(value) for value in np.asarray(array).reshape(-1)]


fixture = {"torch": torch.__version__, "safetensors": safetensors.__version__, "python": platform.python_version()}

for name, (model, shape) in networks.items():
    model.train()
    examples = rng.normal(size=shape).astype(np.float32)

    with torch.no_grad():
        outputs = model(torch.from_numpy(examples)).numpy()

    state = {key: tensor.contiguous() for key, tensor in model.state_dict().items()}
    save_file(state, os.path.join(here, f"attention-{name}.safetensors"), metadata={"format": "pt"})
    torch.save(model.state_dict(), os.path.join(here, f"attention-{name}.pt"))

    fixture[name] = {
        "keys": list(state.keys()),
        "examples": {"shape": list(shape), "values": values(examples)},
        "outputs": values(outputs),
    }
    print(name, "keys", list(state.keys()), "outputs of the first example", values(outputs)[:2])

with open(os.path.join(here, "pytorch-attention.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
