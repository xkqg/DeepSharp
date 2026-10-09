# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the files PyTorch saves the state of networks that walk series, images and volumes into - as safetensors and as
# torch.save writes them - and pytorch-spatial.json: the examples each network was handed and what PyTorch answers through
# it. The tests read a file here into a network written in DeepSharp and hold what it answers to PyTorch's. Its output, when
# it was run, is pytorch-spatial.txt beside it. The older fixtures (pytorch.py, torch-save.py) are not written again.
#
# Run from this folder with PyTorch 2.14.1 for the CPU and safetensors 0.8.0:
#
#     python pytorch-spatial.py > pytorch-spatial.txt
#
# Every network is put to evaluation, with every batch normalisation's numbers drawn, over six examples (four volumes) laid
# out as DeepSharp lays them out - channels last - and handed to PyTorch channels first. A module stands where the layer of
# the network written in DeepSharp stands, so the state's names are the layers' numbers.
#
#   spatial-series.*             Conv1d(2, 4, 3, stride=2, padding=1), BatchNorm1d(4), ReLU, MaxPool1d(3, stride=2, padding=1),
#                                Conv1d(4, 5, 2), Dropout1d(0.25), AdaptiveAvgPool1d(1), Flatten, Linear(5, 3), over series of
#                                17 steps and 2 channels
#   spatial-series-flat.*        Conv1d(2, 4, 4, padding='same'), ReLU, AvgPool1d(2, stride=2, padding=1), Conv1d(4, 3, 2),
#                                Flatten, BatchNorm1d(15), Linear(15, 4), Tanh, Linear(4, 1), over series of 10 steps and 2
#                                channels: the flatten makes rows of a series that keeps 5 steps of 3 channels
#   spatial-pooling.*            Conv2d(2, 4, (3, 2), padding=1), BatchNorm2d(4), ReLU, MaxPool2d((3, 2), stride=2, padding=1),
#                                Conv2d(4, 3, 2), AvgPool2d(2, stride=1, padding=1), Dropout2d(0.2), Flatten, Linear(48, 5),
#                                Tanh, Linear(5, 1), over images of 8 rows, 6 columns and 2 channels
#   spatial-pooling-global.*     Conv2d(2, 4, 3, padding=1), ReLU, AdaptiveMaxPool2d(1), Flatten, Linear(4, 3), over images of
#                                6 rows, 5 columns and 2 channels
#   spatial-volume.*             Conv3d(2, 3, (2, 3, 2), padding=1), BatchNorm3d(3), ReLU, MaxPool3d(2, stride=2, padding=1),
#                                Conv3d(3, 2, 2), AvgPool3d(2, stride=1, padding=1), Dropout3d(0.3), Flatten, Linear(96, 3), over
#                                volumes of 5 planes, 6 rows, 4 columns and 2 channels
#   spatial-volume-global.*      Conv3d(2, 3, 2), ReLU, AdaptiveAvgPool3d(1), Flatten, Linear(3, 2), over the same volumes
#
# PyTorch's average pooling counts the cells it pads with (count_include_pad=True) and its max pooling never sees them, so a
# network written in DeepSharp reads the padded average poolings of these with CountsPadding. Each name is written twice: .safetensors
# by safetensors.torch.save_file, marked "format": "pt" as Hugging Face's transformers marks what it saves, and .pt by
# torch.save(model.state_dict()), read back here by torch.load(weights_only=True) and held to the very bits that were saved.

import json
import os
import platform
import warnings

import numpy as np
import safetensors
import torch
from safetensors.torch import save_file

here = os.path.dirname(os.path.abspath(__file__))
# PyTorch warns that padding='same' with an even kernel pads a copy of the input; it pads as TensorFlow's 'same' does.
warnings.filterwarnings("ignore", message="Using padding='same' with even kernel lengths")
torch.set_num_threads(1)
torch.manual_seed(20261009)
rng = np.random.default_rng(20261009)

nn = torch.nn

# Each network, with the shape of one example as DeepSharp lays it out: channels last, and the examples' count in front.
networks = {
    "series": (
        nn.Sequential(
            nn.Conv1d(2, 4, 3, stride=2, padding=1),
            nn.BatchNorm1d(4),
            nn.ReLU(),
            nn.MaxPool1d(3, stride=2, padding=1),
            nn.Conv1d(4, 5, 2),
            nn.Dropout1d(0.25),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(5, 3),
        ),
        (6, 17, 2),
    ),
    "series-flat": (
        nn.Sequential(
            nn.Conv1d(2, 4, 4, padding="same"),
            nn.ReLU(),
            nn.AvgPool1d(2, stride=2, padding=1),
            nn.Conv1d(4, 3, 2),
            nn.Flatten(),
            nn.BatchNorm1d(15),
            nn.Linear(15, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        ),
        (6, 10, 2),
    ),
    "pooling": (
        nn.Sequential(
            nn.Conv2d(2, 4, (3, 2), padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d((3, 2), stride=2, padding=1),
            nn.Conv2d(4, 3, 2),
            nn.AvgPool2d(2, stride=1, padding=1),
            nn.Dropout2d(0.2),
            nn.Flatten(),
            nn.Linear(48, 5),
            nn.Tanh(),
            nn.Linear(5, 1),
        ),
        (6, 8, 6, 2),
    ),
    "pooling-global": (
        nn.Sequential(
            nn.Conv2d(2, 4, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(4, 3),
        ),
        (6, 6, 5, 2),
    ),
    "volume": (
        nn.Sequential(
            nn.Conv3d(2, 3, (2, 3, 2), padding=1),
            nn.BatchNorm3d(3),
            nn.ReLU(),
            nn.MaxPool3d(2, stride=2, padding=1),
            nn.Conv3d(3, 2, 2),
            nn.AvgPool3d(2, stride=1, padding=1),
            nn.Dropout3d(0.3),
            nn.Flatten(),
            nn.Linear(96, 3),
        ),
        (4, 5, 6, 4, 2),
    ),
    "volume-global": (
        nn.Sequential(
            nn.Conv3d(2, 3, 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(3, 2),
        ),
        (4, 5, 6, 4, 2),
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


def drawn(model):
    """Every batch normalisation's numbers drawn, so none is the identity PyTorch starts them as; then to evaluation."""
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                features = module.num_features
                module.weight.copy_(torch.from_numpy(rng.uniform(0.5, 1.5, features).astype(np.float32)))
                module.bias.copy_(torch.from_numpy(rng.uniform(-0.2, 0.2, features).astype(np.float32)))
                module.running_mean.copy_(torch.from_numpy(rng.uniform(-0.3, 0.3, features).astype(np.float32)))
                module.running_var.copy_(torch.from_numpy(rng.uniform(0.5, 1.5, features).astype(np.float32)))
                module.num_batches_tracked.fill_(7)
    return model.eval()


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
    drawn(model)
    examples = rng.normal(size=shape).astype(np.float32)  # example, then the axes walked, then the channel: DeepSharp's layout
    axes = len(shape)
    channels_first = torch.from_numpy(examples).permute(0, axes - 1, *range(1, axes - 1)).contiguous()  # PyTorch's: example, channel, axes
    with torch.no_grad():
        outputs = model(channels_first)

    state = {key: value.contiguous() for key, value in model.state_dict().items()}
    save_file(state, at(f"spatial-{name}.safetensors"), metadata={"format": "pt"})
    torch.save(model.state_dict(), at(f"spatial-{name}.pt"))
    read = torch.load(at(f"spatial-{name}.pt"), weights_only=True)
    assert same(read, state), name

    fixture[name] = {
        "examples": {"shape": list(examples.shape), "values": [float(value) for value in examples.reshape(-1)]},
        "outputs": values(outputs),
    }
    print(f"{name}: {list(outputs.shape)} outputs", " ".join(f"{value:.8f}" for value in values(outputs)[:12]), "...")
    print(f"spatial-{name}.safetensors:", json.dumps(header(at(f"spatial-{name}.safetensors"))))
    print(f"spatial-{name}.pt: torch.load(weights_only=True) reads {list(read)}; the same bits as saved: {same(read, state)}")

with open(at("pytorch-spatial.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
