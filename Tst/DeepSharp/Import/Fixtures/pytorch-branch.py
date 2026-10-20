# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the files PyTorch saves the state of three networks whose paths meet - as safetensors and as torch.save writes them - and
# pytorch-branch.json: the examples each network was handed and what PyTorch answers through it. The tests read a file here into a
# network of steps written in DeepSharp, each step called what the module is called, and hold what it answers to PyTorch's. Its
# output, when it was run, is pytorch-branch.txt beside it.
#
# Run from this folder with PyTorch 2.14.1 for the CPU and safetensors 0.8.0:
#
#     python pytorch-branch.py > pytorch-branch.txt
#
# A module's state says what it holds and not how its forward pass joins what the modules make, so the state's names are the
# modules' attribute names, and a network written in DeepSharp calls its steps those names.
#
#   branch-residual.*  inner = Linear(6, 6), outer = Linear(6, 2): outer(relu(inner(x)) + x)
#   branch-dense.*     a = Linear(5, 4), b = Linear(9, 4), out = Linear(13, 2): a block that reads everything made before it
#   branch-halves.*    left = Linear(3, 4), right = Linear(3, 4): left(x[:, :3]) + right(x[:, 3:])
#
# Each name is written twice: .safetensors by safetensors.torch.save_file, marked "format": "pt" as Hugging Face's transformers
# marks what it saves, and .pt by torch.save(model.state_dict()).

import json
import os
import platform

import numpy as np
import safetensors
import torch
from safetensors.torch import save_file

here = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(1)
torch.manual_seed(20261020)
rng = np.random.default_rng(20261020)

nn = torch.nn


class Residual(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = nn.Linear(6, 6)
        self.outer = nn.Linear(6, 2)

    def forward(self, x):
        return self.outer(torch.relu(self.inner(x)) + x)


class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(5, 4)
        self.b = nn.Linear(9, 4)
        self.out = nn.Linear(13, 2)

    def forward(self, x):
        a = torch.relu(self.a(x))
        b = torch.relu(self.b(torch.cat([x, a], dim=1)))
        return self.out(torch.cat([x, a, b], dim=1))


class Halves(nn.Module):
    def __init__(self):
        super().__init__()
        self.left = nn.Linear(3, 4)
        self.right = nn.Linear(3, 4)

    def forward(self, x):
        return self.left(x[:, :3]) + self.right(x[:, 3:])


networks = {"residual": (Residual(), (6, 6)), "dense": (Dense(), (6, 5)), "halves": (Halves(), (6, 6))}


def values(array):
    return [float(value) for value in np.asarray(array).reshape(-1)]


fixture = {"torch": torch.__version__, "safetensors": safetensors.__version__, "python": platform.python_version()}

for name, (model, shape) in networks.items():
    model.eval()
    examples = rng.normal(size=shape).astype(np.float32)

    with torch.no_grad():
        outputs = model(torch.from_numpy(examples)).numpy()

    state = {key: tensor.contiguous() for key, tensor in model.state_dict().items()}
    save_file(state, os.path.join(here, f"branch-{name}.safetensors"), metadata={"format": "pt"})
    torch.save(model.state_dict(), os.path.join(here, f"branch-{name}.pt"))

    fixture[name] = {
        "keys": list(state.keys()),
        "examples": {"shape": list(shape), "values": values(examples)},
        "outputs": values(outputs),
    }
    print(name, "keys", list(state.keys()), "outputs of the first example", values(outputs)[:2])

with open(os.path.join(here, "pytorch-branch.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
