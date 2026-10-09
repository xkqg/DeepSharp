# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes the ONNX graphs the ONNX importer's tests read for networks along a series, over an image and through a volume,
# the answers PyTorch gave through each, and the one graph that holds what the importer refuses. Its output, when it was
# run, is onnx-spatial-fixtures.txt beside it.
#
# Run with PyTorch 2.14.1 for the CPU, onnx, onnxscript and onnxruntime:
#
#     python onnx-spatial-fixtures.py
#
# Every graph that stands for a network is written by torch.onnx.export, by its default exporter unless said. The three
# networks share one trunk for each of one, two and three axes — a convolution with a stated padding and a stride, a relu, a
# batch normalisation whose numbers are drawn (not straight after the convolution, which both exporters fold it into), a max
# pooling with padding, a dropout of whole channels, an average pooling that counts its padding, one that does not, and a
# second convolution — and differ in what comes after it:
#
#   flat      a flatten and a linear layer: the default exporter writes onnx-<net>.onnx (a Reshape, the larger numbers in a
#             file beside it named after it with .data added), the TorchScript one onnx-<net>-torchscript.onnx (a Flatten)
#   average   an adaptive average pooling to one place, a flatten and a linear layer, onnx-<net>-global-average.onnx: only
#             the TorchScript exporter writes this as a GlobalAveragePool; the default one writes a ReduceMean over the axes
#             of the places (with an Unsqueeze and a Squeeze round it along a series), onnx-<net>-global-average-default.onnx
#   max       an adaptive max pooling to one place, a flatten and a linear layer, onnx-<net>-global-max.onnx: the TorchScript
#             exporter writes a GlobalMaxPool only where the lengths of the axes are left open to it, and a MaxPool as wide
#             as the axes otherwise, so this one is exported with them open and then given the lengths of the example in
#             the graph's input, as the importer wants every length stated; nothing else in the graph is changed (the default
#             exporter writes a ReduceMax here, onnx-<net>-global-max-default.onnx, and for opset 17 one whose axes are an
#             attribute, onnx-<net>-global-max-opset17.onnx)
#
# <net> is series, image or volume. A dropout of whole channels (Dropout1d, Dropout2d, Dropout3d) is nothing in a network
# exported for inference: neither exporter writes a node for it.
#
# onnx-borders-<net>.onnx are written by hand with onnx.helper, because PyTorch writes only a padding that is the same on
# both sides of an axis: a convolution and poolings padded as SAME_UPPER, and poolings whose pads are written out as
# TensorFlow's 'same' pads the images reaching them — at a stride of one and at a stride of two, where the pads depend on
# the length. PyTorch answers each (an explicit pad of nothing, then the layer), ONNX Runtime is held to the same answers
# here, and the answers are written beside the graph.
#
# onnx-refused.onnx holds what nothing here walks, one of each, in one network: a stride of its own for each axis, a dilated
# window, channels in groups, a padding of its own for each axis, a max pooling that rounds up (ceil_mode) and a leaky relu.
#
# onnx-spatial-fixtures.json  the examples as they were handed to PyTorch — in DeepSharp's layout, channels last — and what
#                             it answered for each, as single-precision numbers

import json
import os
import platform
import warnings

import numpy as np
import onnx
import onnxruntime
import onnxscript
import torch
import torch.nn.functional as functional
from onnx import TensorProto, helper, numpy_helper

here = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(1)
torch.manual_seed(20261009)
rng = np.random.default_rng(20261009)


def values(tensor):
    return [float(value) for value in tensor.detach().to(torch.float32).reshape(-1).numpy()]


def last(tensor):
    # PyTorch's batch, channels, axes... as DeepSharp's batch, axes..., channels.
    return tensor.permute(0, *range(2, tensor.dim()), 1).contiguous()


def exported(model, example, name, mode="default", open_axes=0, names=("x", "y"), opset=None):
    # "default": the default exporter, for batches of any length, for the opset given when one is. "torchscript": the earlier
    # exporter, for the example's batch. "open": the earlier exporter with the lengths of the axes after the channels left open.
    path = os.path.join(here, name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if mode == "default":
            torch.onnx.export(model, (example,), path, input_names=[names[0]], output_names=[names[1]], dynamic_shapes=({0: torch.export.Dim("batch")},), verbose=False, **({} if opset is None else {"opset_version": opset}))
        elif mode == "torchscript":
            torch.onnx.export(model, (example,), path, dynamo=False, input_names=[names[0]], output_names=[names[1]])
        else:
            axes = {axis: f"axis{axis}" for axis in range(2, 2 + open_axes)}
            torch.onnx.export(model, (example,), path, dynamo=False, input_names=[names[0]], output_names=[names[1]], dynamic_axes={names[0]: axes})
            graph = onnx.load(path)
            for axis, length in enumerate(example.shape):
                dim = graph.graph.input[0].type.tensor_type.shape.dim[axis]
                dim.Clear()
                dim.dim_value = length
            for info in list(graph.graph.value_info) + list(graph.graph.output):
                for dim in info.type.tensor_type.shape.dim:
                    if dim.dim_param:
                        dim.Clear()
            onnx.save(graph, path)
    if os.path.exists(path + ".data") and os.path.getsize(path + ".data") == 0:
        os.remove(path + ".data")  # every number small enough to stay in the graph: nothing is kept beside it
    graph = onnx.load(path, load_external_data=False)
    print(f"{name}: {os.path.getsize(path)} bytes" + (f", beside it {os.path.getsize(path + '.data')} bytes" if os.path.exists(path + ".data") else "")
          + f"; opset {graph.opset_import[0].version}; " + " ".join(node.op_type for node in graph.graph.node))


def drawn(norm):
    with torch.no_grad():
        features = norm.num_features
        norm.weight.copy_(torch.from_numpy(rng.uniform(0.5, 1.5, features).astype(np.float32)))
        norm.bias.copy_(torch.from_numpy(rng.uniform(-0.2, 0.2, features).astype(np.float32)))
        norm.running_mean.copy_(torch.from_numpy(rng.uniform(-0.3, 0.3, features).astype(np.float32)))
        norm.running_var.copy_(torch.from_numpy(rng.uniform(0.5, 1.5, features).astype(np.float32)))
    return norm


made = f"torch {torch.__version__}, onnx {onnx.__version__}, onnxscript {onnxscript.__version__}, numpy {np.__version__}, Python {platform.python_version()}"
fixture = {"made": made}
print(made)

# One trunk for each number of axes, and an example of each in DeepSharp's layout: example, axes..., channel.
trunks = {
    "series": (
        (5, 32, 3),
        lambda: torch.nn.Sequential(
            torch.nn.Conv1d(3, 4, 3, stride=2, padding=1),
            torch.nn.ReLU(),
            drawn(torch.nn.BatchNorm1d(4)),
            torch.nn.MaxPool1d(3, stride=2, padding=1),
            torch.nn.Dropout1d(0.25),
            torch.nn.AvgPool1d(3, stride=1, padding=1),
            torch.nn.AvgPool1d(3, stride=2, padding=1, count_include_pad=False),
            torch.nn.Conv1d(4, 5, 2),
        ),
        torch.nn.AdaptiveAvgPool1d,
        torch.nn.AdaptiveMaxPool1d,
    ),
    "image": (
        (6, 12, 10, 3),
        lambda: torch.nn.Sequential(
            torch.nn.Conv2d(3, 4, 3, padding=1),
            torch.nn.ReLU(),
            drawn(torch.nn.BatchNorm2d(4)),
            torch.nn.MaxPool2d(3, stride=2, padding=1),
            torch.nn.Dropout2d(0.25),
            torch.nn.AvgPool2d(3, stride=1, padding=1),
            torch.nn.AvgPool2d(3, stride=2, padding=1, count_include_pad=False),
            torch.nn.Conv2d(4, 5, (2, 1)),
        ),
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.AdaptiveMaxPool2d,
    ),
    "volume": (
        (3, 8, 10, 6, 2),
        lambda: torch.nn.Sequential(
            torch.nn.Conv3d(2, 3, 3, padding=1),
            torch.nn.ReLU(),
            drawn(torch.nn.BatchNorm3d(3)),
            torch.nn.MaxPool3d(2),
            torch.nn.Dropout3d(0.25),
            torch.nn.AvgPool3d(2, stride=1, padding=1),
            torch.nn.AvgPool3d(3, stride=2, padding=1, count_include_pad=False),
            torch.nn.Conv3d(3, 4, (1, 2, 2)),
        ),
        torch.nn.AdaptiveAvgPool3d,
        torch.nn.AdaptiveMaxPool3d,
    ),
}

for net, (shape, trunk_of, average_pool, max_pool) in trunks.items():
    torch.manual_seed(20261010 + len(fixture))
    trunk = trunk_of().eval()
    examples = rng.normal(size=shape).astype(np.float32)
    handed = torch.from_numpy(examples).permute(0, shape.__len__() - 1, *range(1, len(shape) - 1)).contiguous()
    with torch.no_grad():
        made_by_trunk = trunk(handed)
    channels = made_by_trunk.shape[1]
    flat = int(np.prod(made_by_trunk.shape[1:]))
    print(f"{net}: trunk makes {tuple(made_by_trunk.shape)}, a row of {flat}, {channels} channels")

    heads = {
        "flat": torch.nn.Sequential(trunk, torch.nn.Flatten(), torch.nn.Linear(flat, 3)).eval(),
        "average": torch.nn.Sequential(trunk, average_pool(1), torch.nn.Flatten(), torch.nn.Linear(channels, 3)).eval(),
        "max": torch.nn.Sequential(trunk, max_pool(1), torch.nn.Flatten(), torch.nn.Linear(channels, 3)).eval(),
    }
    entry = {"input": {"shape": list(shape), "values": [float(value) for value in examples.reshape(-1)]}}
    for head, model in heads.items():
        with torch.no_grad():
            entry[head] = values(model(handed))
        print(f"  {head}: first outputs", " ".join(f"{value:.8f}" for value in entry[head][:3]))
    fixture[net] = entry

    exported(heads["flat"], handed, f"onnx-{net}.onnx")
    exported(heads["flat"], handed, f"onnx-{net}-torchscript.onnx", mode="torchscript")
    exported(heads["average"], handed, f"onnx-{net}-global-average.onnx", mode="torchscript")
    exported(heads["max"], handed, f"onnx-{net}-global-max.onnx", mode="open", open_axes=len(shape) - 2)

    # What the default exporter writes for an adaptive pooling to one place: a ReduceMean or a ReduceMax over the axes of the places,
    # with an Unsqueeze before and a Squeeze after it along a series. Asked for opset 17, ReduceMax writes its axes as an attribute,
    # as before opset 18; the exporter cannot take ReduceMean down to 17 (it keeps 18), so the attribute form of that is made in a test.
    for head, kind in (("average", "average"), ("max", "max")):
        exported(heads[head], handed, f"onnx-{net}-global-{kind}-default.onnx")
    exported(heads["max"], handed, f"onnx-{net}-global-max-opset17.onnx", opset=17)
    for file in (f"onnx-{net}-global-average-default.onnx", f"onnx-{net}-global-max-default.onnx", f"onnx-{net}-global-max-opset17.onnx"):
        kind = "average" if "average" in file else "max"
        shown = onnxruntime.InferenceSession(os.path.join(here, file)).run(None, {"x": handed.numpy()})[0].reshape(-1)
        assert np.allclose(shown, entry[kind], atol=1e-5), f"{file}: ONNX Runtime disagrees with PyTorch"

    # The graph given the lengths of its axes is still a graph ONNX accepts, with the one global max pooling in it, and ONNX
    # Runtime answers through it as PyTorch did. (ONNX's own reference evaluator, onnx 1.23.2, reduces the wrong axes of a
    # GlobalMaxPool unless the value has four, and pads a MaxPool wrongly, so it cannot answer for either.)
    path = os.path.join(here, f"onnx-{net}-global-max.onnx")
    onnx.checker.check_model(onnx.load(path))
    assert [node.op_type for node in onnx.load(path).graph.node].count("GlobalMaxPool") == 1
    shown = onnxruntime.InferenceSession(path).run(None, {"x": handed.numpy()})[0].reshape(-1)
    assert np.allclose(shown, entry["max"], atol=1e-5), f"{net}: ONNX Runtime disagrees with PyTorch on the global max pooling"


def same_pads(length, size, stride):
    # TensorFlow's 'same': as many places as the stride fits, the odd one after.
    places = -(-length // stride)
    needed = max(0, (places - 1) * stride + size - length)
    return needed // 2, needed - needed // 2


def padded(tensor, begins, ends, fill):
    # ONNX's pads (begin of each axis, then end of each axis) for functional.pad, which goes from the last axis back.
    flat = []
    for axis in reversed(range(len(begins))):
        flat += [begins[axis], ends[axis]]
    return functional.pad(tensor, flat, value=fill)


def pooled(kind, tensor, size, stride, begins, ends, counts):
    walk = getattr(functional, f"{'max' if kind == 'MaxPool' else 'avg'}_pool{len(begins)}d")
    if kind == "MaxPool":
        return walk(padded(tensor, begins, ends, float("-inf")), size, stride)
    if counts:
        return walk(padded(tensor, begins, ends, 0.0), size, stride)
    return walk(padded(tensor, begins, ends, 0.0), size, stride) / walk(padded(torch.ones_like(tensor), begins, ends, 0.0), size, stride)


# What PyTorch cannot write: a border that is not the same on both sides of an axis, written as ONNX's SAME_UPPER or as pads.
for net, spatial in (("series", (23,)), ("image", (14, 12)), ("volume", (9, 8, 6))):
    rank = len(spatial)
    x = torch.from_numpy(rng.normal(size=(2, 2, *spatial)).astype(np.float32))
    kernel = torch.from_numpy(rng.normal(size=(3, 2, *([2] * rank))).astype(np.float32))
    bias = torch.from_numpy(rng.normal(size=(3,)).astype(np.float32))
    nodes, initializers, current, name = [], [numpy_helper.from_array(kernel.numpy(), "kernel"), numpy_helper.from_array(bias.numpy(), "bias")], x, "x"
    lengths = list(spatial)

    def step(kind, size, stride, writes, counts=0):
        global current, name, lengths
        begins, ends = zip(*[same_pads(length, size, stride) for length in lengths])
        attributes = {"kernel_shape": [size] * rank, "strides": [stride] * rank}
        if writes == "auto":
            attributes["auto_pad"] = "SAME_UPPER"
        else:
            attributes["pads"] = [*begins, *ends]
        if kind == "AveragePool":
            attributes["count_include_pad"] = counts
        out = f"{kind.lower()}_{len(nodes)}"
        if kind == "Conv":
            nodes.append(helper.make_node("Conv", [name, "kernel", "bias"], [out], name=out, **attributes))
            current = functional.__dict__[f"conv{rank}d"](padded(current, begins, ends, 0.0), kernel, bias, stride=stride)
        else:
            nodes.append(helper.make_node(kind, [name], [out], name=out, **attributes))
            current = pooled(kind, current, size, stride, begins, ends, counts)
        print(f"  {net} {out}: {size} at {stride} {writes}, pads {list(begins)} {list(ends)}, {list(current.shape[2:])}")
        name, lengths = out, list(current.shape[2:])

    step("Conv", 2, 1, "auto")
    step("AveragePool", 3, 2, "auto", counts=0)
    step("MaxPool", 2, 1, "written")
    step("AveragePool", 3, 2, "written", counts=1)
    step("MaxPool", 3, 2, "written")

    graph = helper.make_graph(
        nodes,
        f"borders_{net}",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, list(current.shape))],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    shown = onnxruntime.InferenceSession(model.SerializeToString()).run(None, {"x": x.numpy()})[0]
    assert shown.shape == tuple(current.shape) and np.allclose(shown, current.numpy(), atol=1e-5), f"{net}: ONNX Runtime disagrees with PyTorch on the borders"
    path = os.path.join(here, f"onnx-borders-{net}.onnx")
    onnx.save(model, path)
    print(f"onnx-borders-{net}.onnx: {os.path.getsize(path)} bytes; opset 20; " + " ".join(node.op_type for node in nodes))
    fixture[f"borders-{net}"] = {
        "input": {"shape": list(last(x).shape), "values": values(last(x))},
        "output": {"shape": list(last(current).shape), "values": values(last(current))},
    }

# What nothing here walks, one of each, in one network: the importer names every one at once, each at its node.
torch.manual_seed(20260933)
refused = torch.nn.Sequential(
    torch.nn.Conv2d(4, 4, 3, stride=(2, 1), padding=1),
    torch.nn.Conv2d(4, 4, 3, dilation=2, padding=2),
    torch.nn.Conv2d(4, 4, 3, groups=2, padding=1),
    torch.nn.Conv2d(4, 4, 3, padding=(1, 0)),
    torch.nn.MaxPool2d(2, ceil_mode=True),
    torch.nn.LeakyReLU(),
    torch.nn.Flatten(),
    torch.nn.Linear(4 * 4 * 3, 1),
).eval()
exported(refused, torch.zeros(2, 4, 16, 8), "onnx-refused.onnx", names=("images", "logits"))

with open(os.path.join(here, "onnx-spatial-fixtures.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
