# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the files PyTorch saves the state of networks whose layers keep no bias and no shift into - as safetensors and as
# torch.save writes them - and pytorch-free.json: the examples each network was handed and what PyTorch answers through it. The
# tests read a file here into a network written in DeepSharp without those biases and hold what it answers to PyTorch's. Its
# output, when it was run, is pytorch-free.txt beside it.
#
# Run from this folder with PyTorch 2.14.1 for the CPU and safetensors 0.8.0:
#
#     python pytorch-free.py > pytorch-free.txt
#
# Every network is put to evaluation, over six examples laid out as DeepSharp lays them out - channels last - and handed to
# PyTorch channels first. A module stands where the layer of the network written in DeepSharp stands, so the state's names are
# the layers' numbers, and a layer without a bias has no `bias` among them.
#
#   free-rows.*      Linear(5, 4, bias=False), ReLU, Linear(4, 3, bias=False), LayerNorm(3, bias=False), Tanh, Linear(3, 2)
#   free-images.*    Conv2d(2, 4, 3, bias=False), ReLU, ConvTranspose2d(4, 3, 2, bias=False), Flatten, Linear(75, 2), over images
#                    of 6 rows, 6 columns and 2 channels
#   free-series.*    Conv1d(2, 4, 3, bias=False), ReLU, ConvTranspose1d(4, 3, 2, bias=False), Flatten, Linear(24, 2), over series
#                    of 9 steps and 2 channels
#   free-volume.*    Conv3d(2, 3, 2, bias=False), ReLU, ConvTranspose3d(3, 2, 2, bias=False), Flatten, Linear(128, 2), over
#                    volumes of 4 planes, 4 rows, 4 columns and 2 channels
#
# Each name is written twice: .safetensors by safetensors.torch.save_file, marked "format": "pt" as Hugging Face's
# transformers marks what it saves, and .pt by torch.save(model.state_dict()), read back here by torch.load(weights_only=True)
# and held to the very bits that were saved.

import json
import os
import platform

import numpy as np
import safetensors
import torch
from safetensors.torch import save_file

here = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(1)
torch.manual_seed(20261014)
rng = np.random.default_rng(20261014)

nn = torch.nn

# Each network, with the shape of the examples as DeepSharp lays them out: channels last, and the examples' count in front.
networks = {
    "rows": (
        nn.Sequential(nn.Linear(5, 4, bias=False), nn.ReLU(), nn.Linear(4, 3, bias=False), nn.LayerNorm(3, bias=False), nn.Tanh(), nn.Linear(3, 2)),
        (6, 5),
    ),
    "images": (
        nn.Sequential(nn.Conv2d(2, 4, 3, bias=False), nn.ReLU(), nn.ConvTranspose2d(4, 3, 2, bias=False), nn.Flatten(), nn.Linear(75, 2)),
        (6, 6, 6, 2),
    ),
    "series": (
        nn.Sequential(nn.Conv1d(2, 4, 3, bias=False), nn.ReLU(), nn.ConvTranspose1d(4, 3, 2, bias=False), nn.Flatten(), nn.Linear(24, 2)),
        (6, 9, 2),
    ),
    "volume": (
        nn.Sequential(nn.Conv3d(2, 3, 2, bias=False), nn.ReLU(), nn.ConvTranspose3d(3, 2, 2, bias=False), nn.Flatten(), nn.Linear(128, 2)),
        (6, 4, 4, 4, 2),
    ),
}


def at(name):
    return os.path.join(here, name)


def values(tensor):
    return [float(value) for value in tensor.detach().to(torch.float32).reshape(-1).numpy()]


def header(path):
    import struct

    data = open(path, "rb").read()
    length = struct.unpack("<Q", data[:8])[0]
    return json.loads(data[8:8 + length])


def same(read, expected):
    """Whether two states hold the same names, and under each the same kind of number, shape and bits."""
    return set(read) == set(expected) and all(
        read[name].dtype == expected[name].dtype
        and read[name].shape == expected[name].shape
        and torch.equal(read[name].reshape(-1).contiguous().view(torch.uint8), expected[name].reshape(-1).contiguous().view(torch.uint8))
        for name in expected
    )


fixture = {"made": f"torch {torch.__version__}, safetensors {safetensors.__version__}, numpy {np.__version__}, Python {platform.python_version()}"}
print(fixture["made"])

for name, (model, shape) in networks.items():
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.2)

    model.eval()
    examples = rng.normal(size=shape).astype(np.float32)
    axes = len(shape)
    # example, then the axes walked, then the channel: DeepSharp's layout; PyTorch's is example, channel, axes.
    handed = torch.from_numpy(examples).permute(0, axes - 1, *range(1, axes - 1)).contiguous() if axes > 2 else torch.from_numpy(examples)

    with torch.no_grad():
        outputs = model(handed)

    state = {key: value.contiguous() for key, value in model.state_dict().items()}
    save_file(state, at(f"free-{name}.safetensors"), metadata={"format": "pt"})
    torch.save(model.state_dict(), at(f"free-{name}.pt"))
    read = torch.load(at(f"free-{name}.pt"), weights_only=True)
    assert same(read, state), name

    fixture[name] = {
        "examples": {"shape": list(examples.shape), "values": [float(value) for value in examples.reshape(-1)]},
        "outputs": values(outputs),
    }
    print(f"{name}: {list(outputs.shape)} outputs", " ".join(f"{value:.8f}" for value in values(outputs)[:12]), "...")
    print(f"free-{name}.safetensors:", json.dumps(header(at(f"free-{name}.safetensors"))))
    print(f"free-{name}.pt: torch.load(weights_only=True) reads {list(read)}; the same bits as saved: {same(read, state)}")

with open(at("pytorch-free.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
