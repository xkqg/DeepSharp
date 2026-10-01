# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the files torch.save writes a network's state into — the very state pytorch.py saved as safetensors, loaded
# back into the same PyTorch modules and saved again — and torch-save.json: files written wrongly or with hostile intent
# on purpose, each with what PyTorch's own weights-only reader, torch.load(weights_only=True), makes of it. The tests read
# the files here into networks written in DeepSharp and hold every number to what the safetensors reader put into the
# same slots. Its output, when it was run, is torch-save.txt beside it.
#
# Run from this folder, once pytorch.py has written the safetensors files, with PyTorch 2.10.0 for the CPU and
# safetensors 0.8.0:
#
#     python torch-save.py > torch-save.txt
#
#   titanic.pt                   torch.save(titanic.state_dict()): Linear(14, 16), ReLU, Linear(16, 1)
#   titanic-f16.pt               the same network turned to 16-bit floats, .half(), and its state saved
#   titanic-bf16.pt              turned to bfloat16
#   titanic-f64.pt               turned to 64-bit floats, .double()
#   batchnorm.pt                 Linear(14, 8), BatchNorm1d(8), ReLU, Linear(8, 1), num_batches_tracked and all
#   convolution.pt               Conv2d(2, 4, (3, 2), padding=1), BatchNorm2d(4), ReLU, Conv2d(4, 3, 2, stride=2),
#                                Flatten, BatchNorm1d(36), Linear(36, 5), Tanh, Linear(5, 1)
#   titanic-parameters.pt        torch.save(dict(titanic.named_parameters())): a plain dict of parameters
#   titanic-views.pt             Titanic's state held in views: a weight turned round in its storage, a bias further on in
#                                a longer one, and the second layer's weight and bias sharing one storage
#   titanic-values.pt            Titanic's state carrying, beside it, every kind of value a pickle of numbers holds
#   titanic-big-endian.pt        titanic.pt written again as a big-endian machine writes it: byteorder "big", every number's
#                                bytes the other way round
#   deep.pt                      forty Linear(1, 1) in a row: a pickle long enough to number its memo past 255

import base64
import builtins
import copy
import io
import json
import os
import pickle
import pickletools
import platform
import subprocess
import warnings
import zipfile
from collections import Counter, OrderedDict

import safetensors
import torch
from safetensors.torch import load_file

here = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(1)
warnings.simplefilter("ignore")


def at(name):
    return os.path.join(here, name)


def loaded(model, file):
    model.load_state_dict(load_file(at(file)), strict=True)
    return model.eval()


titanic = loaded(torch.nn.Sequential(torch.nn.Linear(14, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)), "titanic.safetensors")
batchnorm = loaded(
    torch.nn.Sequential(torch.nn.Linear(14, 8), torch.nn.BatchNorm1d(8), torch.nn.ReLU(), torch.nn.Linear(8, 1)), "batchnorm.safetensors")
convolution = loaded(
    torch.nn.Sequential(
        torch.nn.Conv2d(2, 4, (3, 2), padding=1),
        torch.nn.BatchNorm2d(4),
        torch.nn.ReLU(),
        torch.nn.Conv2d(4, 3, 2, stride=2),
        torch.nn.Flatten(),
        torch.nn.BatchNorm1d(36),
        torch.nn.Linear(36, 5),
        torch.nn.Tanh(),
        torch.nn.Linear(5, 1),
    ),
    "convolution.safetensors",
)


def same(read, expected):
    """Whether two states hold the same names, and under each the same kind of number, shape and bits."""
    return set(read) == set(expected) and all(
        read[name].dtype == expected[name].dtype
        and read[name].shape == expected[name].shape
        and torch.equal(read[name].reshape(-1).contiguous().view(torch.uint8), expected[name].reshape(-1).contiguous().view(torch.uint8))
        for name in expected
    )


def saved(obj, **how):
    buffer = io.BytesIO()
    torch.save(obj, buffer, **how)
    return buffer.getvalue()


def verdict(raw):
    """What torch.load(weights_only=True) makes of a file: what it reads, or its refusal's first line."""
    try:
        read = torch.load(io.BytesIO(raw), weights_only=True)
        return "reads " + (", ".join(str(name) for name in read) if isinstance(read, dict) else type(read).__name__)
    except Exception as refused:  # PyTorch's own words, whatever it raises
        text = str(refused)
        # torch.load wraps what its weights-only reader said in advice; what the reader said follows its name.
        text = text.split("WeightsUnpickler error:", 1)[1] if "WeightsUnpickler error:" in text else text
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        said = next((line for line in lines if "GLOBAL" in line), lines[0] if lines else "")
        return f"refuses: {type(refused).__name__}: {said}"


def opcodes(raw):
    names = set()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        pickled = archive.read(next(name for name in archive.namelist() if name.endswith("/data.pkl")))
    for opcode, _, _ in pickletools.genops(pickled):
        names.add(opcode.name)
    return sorted(names)


fixture = {"made": f"torch {torch.__version__}, safetensors {safetensors.__version__}, Python {platform.python_version()}"}
print(fixture["made"])


def write(name, raw, expected):
    with open(at(name), "wb") as file:
        file.write(raw)
    read = torch.load(at(name), weights_only=True)
    print(f"{name}: {len(raw)} bytes; torch.load(weights_only=True) reads {list(read)}; the same bits as expected: {same(read, expected)}")
    print(f"  opcodes: {' '.join(opcodes(raw))}")
    assert same(read, expected), name


# The networks' state, as torch.save writes it, and each width of float the safetensors fixtures hold.
write("titanic.pt", saved(titanic.state_dict()), titanic.state_dict())
for width, turned in (("f16", torch.float16), ("bf16", torch.bfloat16), ("f64", torch.float64)):
    state = copy.deepcopy(titanic).to(turned).state_dict()
    write(f"titanic-{width}.pt", saved(state), load_file(at(f"titanic-{width}.safetensors")))
write("batchnorm.pt", saved(batchnorm.state_dict()), load_file(at("batchnorm.safetensors")))
write("convolution.pt", saved(convolution.state_dict()), load_file(at("convolution.safetensors")))

# The parameters alone, as a plain dict: each a Parameter, rebuilt through torch._utils._rebuild_parameter.
write("titanic-parameters.pt", saved(dict(titanic.named_parameters())), titanic.state_dict())

# The same numbers held in views of other storages: strides that are not a row's, offsets, and a storage two share.
state = titanic.state_dict()
turned = state["0.weight"].t().contiguous()  # 14 by 16 in its storage
longer = torch.cat([torch.full((3,), 99.0), state["0.bias"], torch.full((2,), -99.0)])
shared = torch.cat([state["2.weight"].reshape(-1), state["2.bias"]])
views = OrderedDict(
    [
        ("0.weight", turned.t()),  # 16 by 14, strides (1, 16)
        ("0.bias", longer[3:19]),  # offset 3 in a storage of 21
        ("2.weight", shared[:16].view(1, 16)),
        ("2.bias", shared[16:]),  # offset 16 in the storage the weight is in
    ]
)
assert views["0.weight"].stride() == (1, 16) and views["0.bias"].storage_offset() == 3
assert views["2.weight"].untyped_storage().data_ptr() == views["2.bias"].untyped_storage().data_ptr()
write("titanic-views.pt", saved(views), titanic.state_dict())

# Beside the state, what else a pickle of numbers holds: floats, whole numbers of every width, nothing, truth, tuples,
# lists — kept, as a state dictionary keeps its _metadata, where no slot is.
state = titanic.state_dict()
state._metadata[""] = {"version": 1, "values": [1.5, -7, 300, 70000, 2**40, None, True, (1, 2, 3), (4,), [5], {"six": 6}]}
write("titanic-values.pt", saved(state), titanic.state_dict())

def kept(entry, name=None):
    """A record written again under its own name, or the one given, and the time it was written: the same bytes each run."""
    return zipfile.ZipInfo(name or entry.filename, date_time=entry.date_time)


# Written as a big-endian machine writes it: the note says "big", and every number's bytes are the other way round.
def big_endian(raw):
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as archive, zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as written:
        for entry in archive.infolist():
            data = archive.read(entry.filename)
            if entry.filename.endswith("/byteorder"):
                data = b"big"
            elif "/data/" in entry.filename and not entry.filename.endswith(".pkl"):
                data = b"".join(data[at : at + 4][::-1] for at in range(0, len(data), 4))
            written.writestr(kept(entry), data)
    return out.getvalue()


write("titanic-big-endian.pt", big_endian(saved(titanic.state_dict())), titanic.state_dict())

# Forty layers: the memo numbers past 255, so the pickle puts and gets with four-byte numbers.
torch.manual_seed(20260930)
deep = torch.nn.Sequential(*[torch.nn.Linear(1, 1) for _ in range(40)])
raw = saved(deep.state_dict())
assert {"LONG_BINPUT", "LONG_BINGET"} <= set(opcodes(raw))
write("deep.pt", raw, deep.state_dict())
fixture["deep"] = {name: [float(value) for value in tensor.reshape(-1)] for name, tensor in deep.state_dict().items()}


# Files written wrongly, or to do harm, on purpose; each handed to PyTorch's own weights-only reader.
def with_pickle(pickled, archive=None):
    """titanic.pt, or another archive, with its data.pkl replaced: every storage the state holds is still there."""
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(archive or saved(titanic.state_dict()))) as read, zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as written:
        for entry in read.infolist():
            written.writestr(kept(entry), pickled if entry.filename.endswith("/data.pkl") else read.read(entry.filename))
    return out.getvalue()


def rewritten(raw, change):
    """An archive with each record changed as said: new bytes, or None to leave it out."""
    out = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(raw)) as read, zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as written:
        for entry in read.infolist():
            data = change(entry.filename, read.read(entry.filename))
            if data is not None:
                written.writestr(kept(entry), data)
    return out.getvalue()


class Call:
    """What pickle writes for a call to a function: the function's name, then its arguments."""

    def __init__(self, function, *arguments):
        self.function, self.arguments = function, arguments

    def __reduce__(self):
        return self.function, self.arguments


titanic_pickle = zipfile.ZipFile(io.BytesIO(saved(titanic.state_dict()))).read("archive/data.pkl")
marker = "deepsharp-i6-ran.txt"
harm = {
    "os.system, as pickle writes a call to it": pickle.dumps(Call(os.system, f"echo ran> {marker}"), protocol=2),
    "os.system, named by hand": b"\x80\x02cos\nsystem\nX" + len(f"echo ran> {marker}").to_bytes(4, "little") + f"echo ran> {marker}".encode() + b"\x85R.",
    "builtins.eval, as pickle writes a call to it": pickle.dumps(Call(builtins.eval, f"open('{marker}', 'w')"), protocol=2),
    "builtins.eval, named by hand": b"\x80\x02cbuiltins\neval\nX\x04\x00\x00\x00None\x85R.",
    "subprocess.Popen, as pickle writes a call to it": pickle.dumps(Call(subprocess.Popen, ["cmd", "/c", f"echo ran> {marker}"]), protocol=2),
    "subprocess.Popen, named by hand": b"\x80\x02csubprocess\nPopen\nX\x04\x00\x00\x00echo\x85R.",
    "subprocess.Popen, named on the stack as protocol 4 writes it": pickle.dumps(Call(subprocess.Popen, ["echo"]), protocol=4),
    "subprocess.Popen, named on the stack by hand": b"\x80\x02X\n\x00\x00\x00subprocessX\x05\x00\x00\x00Popen\x93)R.",
    "os.system, named by an old instruction": b"(X\x04\x00\x00\x00echoios\nsystem\n.",
    "a .NET type, named by hand": b"\x80\x02cSystem.Diagnostics\nProcess\n)R.",
    "a type of the tests, named by hand": b"\x80\x02cDeepSharp.Tests.Import\nTripwire\n)R.",
    "an allowed class, built by an old instruction": b"\x80\x02(ccollections\nOrderedDict\no.",
}
shapes = {
    "torch.Size, which PyTorch allows": pickle.dumps(torch.Size([2, 3]), protocol=2),
    "bytes, which PyTorch allows through _codecs.encode": pickle.dumps(b"abc", protocol=2),
    "collections.Counter, which PyTorch allows": pickle.dumps(Counter({"a": 1}), protocol=2),
    "an opcode pickle does not have": b"\x80\x02\xff.",
    "BUILD on a list": b"\x80\x02]}b.",
    "APPEND to a dict": b"\x80\x02}K\x01a.",
    "SETITEM on a list": b"\x80\x02]K\x01K\x02s.",
    "a memo never put": b"\x80\x02h\x05.",
    "a persistent id that is no storage": b"\x80\x02(X\x06\x00\x00\x00moduleK\x01tQ.",
    "a persistent id that is a number": b"\x80\x02K\x01Q.",
    "REDUCE on text": b"\x80\x02X\x01\x00\x00\x00a)R.",
    "a tuple with no mark before it": b"\x80\x02K\x01t.",
    "STOP with nothing made": b"\x80\x02.",
    "no pickle at all": b"",
    "a pickle cut short": titanic_pickle[: len(titanic_pickle) // 2],
    "a storage the archive does not hold": titanic_pickle.replace(b"X\x01\x00\x00\x000q", b"X\x01\x00\x00\x009q", 1),
    "a tensor reaching past its storage": titanic_pickle.replace(b"K\x10K\x0e\x86", b"K\x11K\x0e\x86", 1),
    "the state made by NEWOBJ": titanic_pickle.replace(b")Rq\x01", b")\x81q\x01", 1),
    "a name written as a short byte string": titanic_pickle.replace(b"X\x06\x00\x00\x000.biasq", b"U\x060.biasq", 1),
}
assert titanic_pickle.count(b"X\x01\x00\x00\x000q") >= 1 and titanic_pickle.count(b"K\x10K\x0e\x86") == 1
assert titanic_pickle.count(b")Rq\x01") == 1 and titanic_pickle.count(b"X\x06\x00\x00\x000.biasq") == 1

state = titanic.state_dict()
negated = OrderedDict(state)
negated["0.bias"] = torch._neg_view(-state["0.bias"])
archives = {
    "os.system, among the state's tensors": saved(OrderedDict([*state.items(), ("harm", Call(os.system, f"echo ran> {marker}"))])),
    "a state saved with pickle protocol 3": saved(titanic.state_dict(), pickle_protocol=3),
    "a state saved with pickle protocol 4": saved(titanic.state_dict(), pickle_protocol=4),
    "a state saved in the format before PyTorch 1.6": saved(titanic.state_dict(), _use_new_zipfile_serialization=False),
    "a checkpoint holding the state under a key": saved({"model": titanic.state_dict(), "epoch": 42}),
    "a bias saved as a negated view": saved(negated),
    "a bias saved as whole numbers": saved(OrderedDict((name, tensor.to(torch.int32) if name == "0.bias" else tensor) for name, tensor in state.items())),
    "a TorchScript archive": None,
    "no archive at all": b"not an archive",
    "an archive holding nothing": rewritten(saved(titanic.state_dict()), lambda name, data: None),
    "an archive whose files are in no folder": None,
    "an archive of version 11": rewritten(saved(titanic.state_dict()), lambda name, data: b"11\n" if name.endswith("/version") else data),
    "an archive of version 0": rewritten(saved(titanic.state_dict()), lambda name, data: b"0\n" if name.endswith("/version") else data),
    "an archive with no version": rewritten(saved(titanic.state_dict()), lambda name, data: None if name.endswith("version") else data),
    "an archive with no data.pkl": rewritten(saved(titanic.state_dict()), lambda name, data: None if name.endswith("/data.pkl") else data),
    "an archive of a byte order nobody has": rewritten(saved(titanic.state_dict()), lambda name, data: b"middle" if name.endswith("/byteorder") else data),
    "an archive with no byte order": rewritten(saved(titanic.state_dict()), lambda name, data: None if name.endswith("/byteorder") else data),
    "a storage of fewer bytes than it says": rewritten(saved(titanic.state_dict()), lambda name, data: data[:-4] if name.endswith("/data/0") else data),
}
script = io.BytesIO()
torch.jit.save(torch.jit.script(titanic), script)
archives["a TorchScript archive"] = script.getvalue()
flat = io.BytesIO()
with zipfile.ZipFile(flat, "w", zipfile.ZIP_STORED) as written, zipfile.ZipFile(io.BytesIO(saved(titanic.state_dict()))) as read:
    for entry in read.infolist():
        written.writestr(kept(entry, entry.filename.split("/", 1)[1]), read.read(entry.filename))
archives["an archive whose files are in no folder"] = flat.getvalue()

cases = []
for name, pickled in [*harm.items(), *shapes.items()]:
    cases.append((name, with_pickle(pickled)))
cases.extend(archives.items())

fixture["cases"] = []
for name, raw in cases:
    said = verdict(raw)
    fixture["cases"].append({"case": name, "file": base64.b64encode(raw).decode("ascii"), "torch": said})
    print(f"{name}: {said}")
print("marker file written by anything:", os.path.exists(marker) or os.path.exists(at(marker)))

with open(at("torch-save.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
