# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the ONNX graphs PyTorch exports that the ONNX importer's tests read, and what PyTorch answered through each
# network. Its output, when it was run, is onnx-fixtures.txt beside it.
#
# Run with PyTorch 2.10.0 for the CPU, onnx and onnxscript, on the rows make-titanic-rows.cs writes:
#
#     dotnet run make-titanic-rows.cs -- rows
#     python onnx-fixtures.py rows
#
# Every graph is written by torch.onnx.export, by its default exporter unless said; that exporter keeps the larger
# numbers in a file of their own beside the graph, named after it with .data added, and both files are kept here.
#
#   onnx-titanic.onnx (.data)          Linear(14, 16), ReLU, Linear(16, 1), trained on the Titanic training rows with
#                                      BCEWithLogitsLoss and Adam at 0.01, stopped early on the validation rows — the
#                                      network the safetensors reader's tests read, trained the same way on the same rows
#                                      — exported for batches of any length
#   onnx-titanic-torchscript.onnx      the same network, exported by the earlier TorchScript exporter for a batch of two
#   onnx-titanic-sigmoid.onnx          the same network with a Sigmoid after it, as a network exported to be served ends
#   onnx-convolution.onnx (.data)      Conv2d(2, 4, (3, 2), padding=1), BatchNorm2d(4), ReLU, Conv2d(4, 3, 2, stride=2),
#                                      Flatten, BatchNorm1d(36), Linear(36, 5), Tanh, Linear(5, 1): the safetensors
#                                      reader's network over images, every batch normalisation's numbers drawn, over six
#                                      images of 8 rows by 6 columns and 2 channels, exported for batches of any length
#   onnx-convolution-torchscript.onnx  the same network, exported by the TorchScript exporter for a batch of six
#   onnx-kinds.onnx (.data)            Linear(14, 8), LayerNorm(8), Sigmoid, Linear(8, 3), Softmax: a layer
#                                      normalisation whose scale and shift are drawn, and a softmax a cross-entropy owns
#   onnx-refused.onnx (.data)          what nothing here walks, one of each: a stride of its own for each axis, a dilated
#                                      window, channels in groups, a padding of its own for each axis, pooling and a leaky
#                                      relu
#   onnx-branch.onnx (.data)           a network whose forward pass adds its input back after a layer: a branch
#   onnx-fixtures.json                 the rows and the images as they were handed to PyTorch, and what it answered for
#                                      each, as single-precision numbers — and each chance as the double PyTorch's sigmoid
#                                      gave of them

import json
import os
import platform
import sys
import warnings

import numpy as np
import onnx
import onnxscript
import torch

here = os.path.dirname(os.path.abspath(__file__))
rows_folder = sys.argv[1] if len(sys.argv) > 1 else "rows"
torch.set_num_threads(1)
torch.manual_seed(20260930)
rng = np.random.default_rng(20260930)


def rows(name, answers=True):
    array = np.loadtxt(os.path.join(rows_folder, f"{name}.csv"), delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
    return (torch.from_numpy(array[:, :14].copy()), torch.from_numpy(array[:, 14:15].copy())) if answers else torch.from_numpy(array.copy())


train, train_survived = rows("train")
validation, validation_survived = rows("validation")
test, _ = rows("test")
served = rows("served", answers=False)


def values(tensor):
    return [float(value) for value in tensor.detach().to(torch.float32).reshape(-1).numpy()]


def trained(model, seed):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = torch.nn.BCEWithLogitsLoss()
    order = torch.Generator().manual_seed(seed)
    best, kept, waited = float("inf"), None, 0
    for epoch in range(100):
        model.train()
        shuffled = torch.randperm(len(train), generator=order)
        for start in range(0, len(train), 32):
            batch = shuffled[start:start + 32]
            optimizer.zero_grad()
            loss(model(train[batch]), train_survived[batch]).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            measured = loss(model(validation), validation_survived).item()
        if measured < best:
            best, waited, kept = measured, 0, {name: value.clone() for name, value in model.state_dict().items()}
        else:
            waited += 1
            if waited >= 10:
                break
    model.load_state_dict(kept)
    model.eval()
    return epoch + 1


def answers(model, features):
    with torch.no_grad():
        logits = model(features)
    return {"logits": values(logits), "chances": [float(value) for value in torch.sigmoid(logits.double()).reshape(-1).numpy()]}


def exported(model, example, name, names, torchscript=False):
    # For batches of any length, unless the example's batch is kept, as the TorchScript exporter keeps it when told nothing.
    path = os.path.join(here, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if torchscript:
            torch.onnx.export(model, (example,), path, dynamo=False, input_names=[names[0]], output_names=[names[1]])
        else:
            torch.onnx.export(model, (example,), path, input_names=[names[0]], output_names=[names[1]], dynamic_shapes=({0: torch.export.Dim("batch")},), verbose=False)
    graph = onnx.load(path, load_external_data=False)
    print(f"{name}: {os.path.getsize(path)} bytes" + (f", beside it {os.path.getsize(path + '.data')} bytes" if os.path.exists(path + ".data") else "")
          + f"; opset {graph.opset_import[0].version}; " + " ".join(f"{node.op_type}" for node in graph.graph.node))


made = f"torch {torch.__version__}, onnx {onnx.__version__}, onnxscript {onnxscript.__version__}, numpy {np.__version__}, Python {platform.python_version()}"
fixture = {"made": made}
print(made)

# The Titanic network, trained as the safetensors reader's fixtures train it.
titanic = torch.nn.Sequential(torch.nn.Linear(14, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
epochs = trained(titanic, 20260930)
fixture["titanic"] = {
    "epochs": epochs,
    "served": {"features": values(served), **answers(titanic, served)},
    "test": {"features": values(test), **answers(titanic, test)},
}
print(f"titanic: {epochs} epochs; the README's passenger {fixture['titanic']['served']['chances'][0]!r}, the woman of 38 {fixture['titanic']['served']['chances'][1]!r}")
exported(titanic, served, "onnx-titanic.onnx", ["passengers", "logits"])
exported(titanic, served, "onnx-titanic-torchscript.onnx", ["passengers", "logits"], torchscript=True)
exported(torch.nn.Sequential(titanic, torch.nn.Sigmoid()).eval(), served, "onnx-titanic-sigmoid.onnx", ["passengers", "chances"])

# Convolutions over images with more rows than columns and more than one channel, flattened into a linear layer.
torch.manual_seed(20260931)
convolution = torch.nn.Sequential(
    torch.nn.Conv2d(2, 4, (3, 2), padding=1),
    torch.nn.BatchNorm2d(4),
    torch.nn.ReLU(),
    torch.nn.Conv2d(4, 3, 2, stride=2),
    torch.nn.Flatten(),
    torch.nn.BatchNorm1d(36),
    torch.nn.Linear(36, 5),
    torch.nn.Tanh(),
    torch.nn.Linear(5, 1),
)
with torch.no_grad():
    for place, features in ((1, 4), (5, 36)):
        norm = convolution[place]
        norm.weight.copy_(torch.from_numpy(rng.uniform(0.5, 1.5, features).astype(np.float32)))
        norm.bias.copy_(torch.from_numpy(rng.uniform(-0.2, 0.2, features).astype(np.float32)))
        norm.running_mean.copy_(torch.from_numpy(rng.uniform(-0.3, 0.3, features).astype(np.float32)))
        norm.running_var.copy_(torch.from_numpy(rng.uniform(0.5, 1.5, features).astype(np.float32)))
convolution.eval()
images = rng.normal(size=(6, 8, 6, 2)).astype(np.float32)  # image, row, column, channel: DeepSharp's layout
handed = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()  # PyTorch's: image, channel, row, column
with torch.no_grad():
    outputs = convolution(handed)
fixture["convolution"] = {"images": {"shape": list(images.shape), "values": [float(value) for value in images.reshape(-1)]}, "outputs": values(outputs)}
print("convolution outputs:", " ".join(f"{value:.8f}" for value in fixture["convolution"]["outputs"]))
exported(convolution, handed, "onnx-convolution.onnx", ["images", "outputs"])
exported(convolution, handed, "onnx-convolution-torchscript.onnx", ["images", "outputs"], torchscript=True)

# A layer normalisation, a sigmoid inside the network and a softmax at its end.
torch.manual_seed(20260932)
kinds = torch.nn.Sequential(torch.nn.Linear(14, 8), torch.nn.LayerNorm(8), torch.nn.Sigmoid(), torch.nn.Linear(8, 3), torch.nn.Softmax(dim=1))
with torch.no_grad():
    kinds[1].weight.copy_(torch.from_numpy(rng.uniform(0.5, 1.5, 8).astype(np.float32)))
    kinds[1].bias.copy_(torch.from_numpy(rng.uniform(-0.2, 0.2, 8).astype(np.float32)))
kinds.eval()
with torch.no_grad():
    fixture["kinds"] = {"features": values(test), "shares": values(kinds(test))}
print("kinds: shares of the first", " ".join(f"{value:.8f}" for value in fixture["kinds"]["shares"][:3]))
exported(kinds, served, "onnx-kinds.onnx", ["passengers", "shares"])

# What nothing here walks, one of each, in one network: the importer names every one at once, each at its node.
torch.manual_seed(20260933)
refused = torch.nn.Sequential(
    torch.nn.Conv2d(4, 4, 3, stride=(2, 1), padding=1),
    torch.nn.Conv2d(4, 4, 3, dilation=2, padding=2),
    torch.nn.Conv2d(4, 4, 3, groups=2, padding=1),
    torch.nn.Conv2d(4, 4, 3, padding=(1, 0)),
    torch.nn.MaxPool2d(2),
    torch.nn.LeakyReLU(),
    torch.nn.Flatten(),
    torch.nn.Linear(4 * 4 * 3, 1),
).eval()
exported(refused, torch.zeros(2, 4, 16, 8), "onnx-refused.onnx", ["images", "logits"])


# A layer whose input is added back to what it makes: two paths through the network, which a stack does not have.
class Branch(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = torch.nn.Linear(14, 14)
        self.outer = torch.nn.Linear(14, 1)

    def forward(self, passengers):
        return self.outer(torch.relu(self.inner(passengers)) + passengers)


torch.manual_seed(20260934)
exported(Branch().eval(), served, "onnx-branch.onnx", ["passengers", "logits"])

with open(os.path.join(here, "onnx-fixtures.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
