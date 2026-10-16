# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the files PyTorch saves the state of networks made of the words 0.11.0 adds into - as safetensors and as torch.save
# writes them - and pytorch-vocabulary.json: the examples each network was handed and what PyTorch answers through it. The tests
# read a file here into the same network written in DeepSharp and hold what it answers to PyTorch's. Its output, when it was run,
# is pytorch-vocabulary.txt beside it.
#
# Run from this folder with PyTorch 2.14.1 for the CPU and safetensors 0.8.0:
#
#     python pytorch-vocabulary.py > pytorch-vocabulary.txt
#
# Every network is put to evaluation, over six examples laid out as DeepSharp lays them out - channels last - and handed to
# PyTorch channels first. A module stands where the layer of the network written in DeepSharp stands, so the state's names are
# the layers' numbers.
#
#   vocabulary-rows.*       Linear(5, 6), GELU, Linear(6, 6), SiLU, Mish, LeakyReLU(0.1), PReLU(6), ELU(0.7), SELU, GroupNorm(3, 6),
#                           RMSNorm(6), Linear(6, 4), Softmax(dim=1)
#   vocabulary-images.*     Conv2d(2, 4, 3), GroupNorm(2, 4), PReLU(4), InstanceNorm2d(4, affine=True), Flatten, Linear(64, 2),
#                           over images of 6 rows, 6 columns and 2 channels
#   vocabulary-embedding.*  Embedding(5, 3), Flatten, Linear(3, 2), over one category of five: PyTorch is handed its place, and
#                           DeepSharp the row of noughts and a one its encoder writes it down as
#
# Each name is written twice: .safetensors by safetensors.torch.save_file, marked "format": "pt" as Hugging Face's transformers
# marks what it saves, and .pt by torch.save(model.state_dict()), read back here by torch.load(weights_only=True) and held to the
# very bits that were saved.

import json
import os
import platform

import numpy as np
import safetensors
import torch
from safetensors.torch import save_file

here = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(1)
torch.manual_seed(20261016)
rng = np.random.default_rng(20261016)

nn = torch.nn

networks = {
    "rows": (
        nn.Sequential(
            nn.Linear(5, 6), nn.GELU(), nn.Linear(6, 6), nn.SiLU(), nn.Mish(), nn.LeakyReLU(0.1), nn.PReLU(6), nn.ELU(0.7), nn.SELU(),
            nn.GroupNorm(3, 6), nn.RMSNorm(6), nn.Linear(6, 4), nn.Softmax(dim=1)),
        (6, 5),
    ),
    "images": (
        nn.Sequential(nn.Conv2d(2, 4, 3), nn.GroupNorm(2, 4), nn.PReLU(4), nn.InstanceNorm2d(4, affine=True), nn.Flatten(), nn.Linear(64, 2)),
        (6, 6, 6, 2),
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


def save(name, model):
    state = {key: value.contiguous() for key, value in model.state_dict().items()}
    save_file(state, at(f"vocabulary-{name}.safetensors"), metadata={"format": "pt"})
    torch.save(model.state_dict(), at(f"vocabulary-{name}.pt"))
    read = torch.load(at(f"vocabulary-{name}.pt"), weights_only=True)
    assert same(read, state), name
    print(f"vocabulary-{name}.safetensors:", json.dumps(header(at(f"vocabulary-{name}.safetensors"))))
    print(f"vocabulary-{name}.pt: torch.load(weights_only=True) reads {list(read)}; the same bits as saved: {same(read, state)}")


fixture = {"made": f"torch {torch.__version__}, safetensors {safetensors.__version__}, numpy {np.__version__}, Python {platform.python_version()}"}
print(fixture["made"])

for name, (model, shape) in networks.items():
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn_like(parameter) * 0.4 + (0.5 if parameter.dim() == 1 else 0))

    model.eval()
    examples = rng.normal(size=shape).astype(np.float32)
    axes = len(shape)
    handed = torch.from_numpy(examples).permute(0, axes - 1, *range(1, axes - 1)).contiguous() if axes > 2 else torch.from_numpy(examples)

    with torch.no_grad():
        outputs = model(handed)

    save(name, model)
    fixture[name] = {
        "examples": {"shape": list(examples.shape), "values": [float(value) for value in examples.reshape(-1)]},
        "outputs": values(outputs),
    }
    print(f"{name}: {list(outputs.shape)} outputs", " ".join(f"{value:.8f}" for value in values(outputs)[:12]), "...")

# The embedding: PyTorch looks a category up by its place; DeepSharp multiplies the row of noughts and a one by the same weight.
embedding = nn.Sequential(nn.Embedding(5, 3), nn.Flatten(), nn.Linear(3, 2))
with torch.no_grad():
    for parameter in embedding.parameters():
        parameter.copy_(torch.randn_like(parameter) * 0.4)
embedding.eval()
places = np.array([0, 3, 1, 4, 2, 3])
with torch.no_grad():
    outputs = embedding(torch.from_numpy(places).reshape(6, 1))
ones = np.eye(5, dtype=np.float32)[places]
save("embedding", embedding)
fixture["embedding"] = {
    "examples": {"shape": list(ones.shape), "values": [float(value) for value in ones.reshape(-1)]},
    "outputs": values(outputs),
}
print(f"embedding: {list(outputs.shape)} outputs", " ".join(f"{value:.8f}" for value in values(outputs)), "for places", places.tolist())

with open(at("pytorch-vocabulary.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
