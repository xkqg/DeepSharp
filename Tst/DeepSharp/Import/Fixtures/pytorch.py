# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the safetensors files PyTorch saves its networks' state into, and pytorch.json: what PyTorch answers through each
# network, and what safetensors' own reader makes of files written wrongly on purpose. The tests read a file here into a
# network written in DeepSharp and hold what it answers to PyTorch's. Its output, when it was run, is pytorch.txt beside it.
#
# Run from this folder, once make-rows.cs has written the Titanic rows, with PyTorch 2.10.0 for the CPU and safetensors 0.8.0:
#
#     python pytorch.py > pytorch.txt
#
#   titanic.safetensors          Linear(14, 16), ReLU, Linear(16, 1), trained on the Titanic training rows with
#                                BCEWithLogitsLoss and Adam, stopped early on the validation rows; marked "format": "pt",
#                                as Hugging Face's transformers marks what it saves
#   titanic-f16.safetensors      the same state, held as 16-bit floats
#   titanic-bf16.safetensors     the same state, held as bfloat16
#   titanic-f64.safetensors      the same state, held as 64-bit floats
#   batchnorm.safetensors        Linear(14, 8), BatchNorm1d(8), ReLU, Linear(8, 1), trained the same way; saved by
#                                safetensors.torch.save_model, which marks nothing, so its num_batches_tracked is there too
#   convolution.safetensors      Conv2d(2, 4, (3, 2), padding=1), BatchNorm2d(4), ReLU, Conv2d(4, 3, 2, stride=2), Flatten,
#                                BatchNorm1d(36), Linear(36, 5), Tanh, Linear(5, 1), at PyTorch's start with every batch
#                                normalisation's numbers drawn, over six images of 8 rows by 6 columns and 2 channels

import base64
import json
import os
import platform
import struct

import numpy as np
import safetensors
import torch
from safetensors.torch import save_file, save_model

here = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(1)
torch.manual_seed(20260930)
rng = np.random.default_rng(20260930)


def rows(name):
    array = np.loadtxt(os.path.join(here, f"titanic-{name}.csv"), delimiter=",", skiprows=1, dtype=np.float32, ndmin=2)
    return torch.from_numpy(array[:, :14].copy()), torch.from_numpy(array[:, 14:15].copy())


train, train_survived = rows("train")
validation, validation_survived = rows("validation")
test, _ = rows("test")
served, _ = rows("served")


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


def header(path):
    data = open(path, "rb").read()
    length = struct.unpack("<Q", data[:8])[0]
    return json.loads(data[8:8 + length])


fixture = {"made": f"torch {torch.__version__}, safetensors {safetensors.__version__}, numpy {np.__version__}, Python {platform.python_version()}"}
print(fixture["made"])

# The Titanic network, and its state in each width of float PyTorch saves.
titanic = torch.nn.Sequential(torch.nn.Linear(14, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
epochs = trained(titanic, 20260930)
state = {name: value.contiguous() for name, value in titanic.state_dict().items()}
save_file(state, os.path.join(here, "titanic.safetensors"), metadata={"format": "pt"})
fixture["titanic"] = {
    "epochs": epochs,
    "served": {"features": values(served), **answers(titanic, served)},
    "test": {"features": values(test), **answers(titanic, test)},
}
print(f"titanic: {epochs} epochs; the README's passenger {fixture['titanic']['served']['chances'][0]!r}, the woman of 38 {fixture['titanic']['served']['chances'][1]!r}")
print("titanic.safetensors:", json.dumps(header(os.path.join(here, "titanic.safetensors"))))

widths = {}
for name, dtype in (("f16", torch.float16), ("bf16", torch.bfloat16), ("f64", torch.float64)):
    held = {key: value.to(dtype).contiguous() for key, value in state.items()}
    save_file(held, os.path.join(here, f"titanic-{name}.safetensors"), metadata={"format": "pt"})
    # What PyTorch makes of each as 32-bit floats: the 16-bit widths widened exactly, the 64-bit one rounded to the nearest.
    widths[name] = {key: values(value.to(torch.float32)) for key, value in held.items()}
    print(f"titanic-{name}.safetensors:", {key: header(os.path.join(here, f"titanic-{name}.safetensors"))[key]["dtype"] for key in held})
fixture["widths"] = widths

# A batch normalisation between the two linear layers.
batchnorm = torch.nn.Sequential(torch.nn.Linear(14, 8), torch.nn.BatchNorm1d(8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
epochs = trained(batchnorm, 20260931)
save_model(batchnorm, os.path.join(here, "batchnorm.safetensors"))
fixture["batchnorm"] = {
    "epochs": epochs,
    "served": {"features": values(served), **answers(batchnorm, served)},
    "test": {"features": values(test), **answers(batchnorm, test)},
}
print(f"batchnorm: {epochs} epochs; the README's passenger {fixture['batchnorm']['served']['chances'][0]!r}")
print("batchnorm.safetensors:", json.dumps(header(os.path.join(here, "batchnorm.safetensors"))))

# Convolutions over images with more rows than columns and more than one channel, flattened into a linear layer.
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
        norm.num_batches_tracked.fill_(7)
convolution.eval()
images = rng.normal(size=(6, 8, 6, 2)).astype(np.float32)  # image, row, column, channel: DeepSharp's layout
outputs = convolution(torch.from_numpy(images).permute(0, 3, 1, 2).contiguous())  # PyTorch's: image, channel, row, column
save_file({name: value.contiguous() for name, value in convolution.state_dict().items()}, os.path.join(here, "convolution.safetensors"), metadata={"format": "pt"})
fixture["convolution"] = {
    "images": {"shape": list(images.shape), "values": [float(value) for value in images.reshape(-1)]},
    "outputs": values(outputs),
}
print("convolution outputs:", " ".join(f"{value:.8f}" for value in fixture["convolution"]["outputs"]))
print("convolution.safetensors:", json.dumps(header(os.path.join(here, "convolution.safetensors"))))

# Files written wrongly on purpose, each handed to safetensors' own reader, which says what it makes of it.
tensors = {"0.bias": ("F32", [1], [0, 4]), "0.weight": ("F32", [1, 2], [4, 12])}
data = struct.pack("<3f", 0.5, 1.5, -2.0)


def written(entries=None, metadata='{"format":"pt"}', body=data, text=None, length=None, pad=0):
    if text is None:
        entries = entries if entries is not None else [f'"{name}":{{"dtype":"{dtype}","shape":{json.dumps(shape)},"data_offsets":{json.dumps(offsets)}}}' for name, (dtype, shape, offsets) in tensors.items()]
        text = "{" + ",".join(([f'"__metadata__":{metadata}'] if metadata else []) + entries) + "}"
    raw = text.encode("utf-8") if isinstance(text, str) else text
    raw += b" " * pad
    return struct.pack("<Q", len(raw) if length is None else length) + raw + body


def entry(name, text):
    return f'"{name}":{text}'


bias = entry("0.bias", '{"dtype":"F32","shape":[1],"data_offsets":[0,4]}')
cases = {
    "a file as the writer writes it": written(),
    "a header padded with spaces, as the writer pads one": written(pad=5),
    "no file at all": b"",
    "fewer bytes than the header's length takes": b"\x01\x02\x03",
    "a header said to be longer than a hundred million bytes": struct.pack("<Q", 100_000_001) + b"{}",
    "a header said to be longer than the file": written(length=1000),
    "a header that is no UTF-8": struct.pack("<Q", 3) + b"{\xff}",
    "a header that is no JSON": written(text="{not json}"),
    "a header that is no object": written(text="[]"),
    "a tensor of a type safetensors does not know": written([bias, entry("0.weight", '{"dtype":"F99","shape":[1,2],"data_offsets":[4,12]}')]),
    "a tensor without its shape": written([bias, entry("0.weight", '{"dtype":"F32","data_offsets":[4,12]}')]),
    "a tensor starting where the one before it does not end": written([bias, entry("0.weight", '{"dtype":"F32","shape":[1,2],"data_offsets":[5,13]}')], body=data + b"\x00"),
    "a tensor ending before it starts": written([bias, entry("0.weight", '{"dtype":"F32","shape":[1,2],"data_offsets":[4,3]}')]),
    "a tensor holding fewer bytes than its shape and type take": written([bias, entry("0.weight", '{"dtype":"F32","shape":[1,3],"data_offsets":[4,12]}')]),
    "bytes after the last tensor": written(body=data + b"\x00\x00\x00\x00"),
    "a note that is no text": written(metadata='{"format":1}'),
    "a shape too large to count": written([bias, entry("0.weight", '{"dtype":"F32","shape":[4294967296,4294967296],"data_offsets":[4,12]}')]),
    "a tensor of half bytes ending inside a byte": written([bias, entry("0.weight", '{"dtype":"F4","shape":[3],"data_offsets":[4,6]}')], body=struct.pack("<f", 0.5) + b"\x00\x00"),
    "a tensor named twice": written([bias, bias, entry("0.weight", '{"dtype":"F32","shape":[1,2],"data_offsets":[4,12]}')]),
    "an offset below nothing": written([bias, entry("0.weight", '{"dtype":"F32","shape":[1,2],"data_offsets":[-4,12]}')]),
}
verdicts = []
for name, raw in cases.items():
    try:
        # The reader lists what it read in an order of its own, which changes from one run to the next, so by name here.
        read = sorted(safetensors.deserialize(raw), key=lambda each: each[0])
        verdict = "reads " + ", ".join(f"{tensor} {info['dtype']} {info['shape']}" for tensor, info in read)
    except Exception as refused:  # the reader's own words, whatever it raises
        verdict = f"refuses: {type(refused).__name__}: {refused}"
    verdicts.append({"case": name, "file": base64.b64encode(raw).decode("ascii"), "reference": verdict})
    print(f"{name}: {verdict}")
fixture["reference"] = verdicts

with open(os.path.join(here, "pytorch.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
