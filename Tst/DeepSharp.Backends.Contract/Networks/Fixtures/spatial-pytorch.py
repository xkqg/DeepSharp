# Prints PyTorch's values for the one-, two- and three-dimensional convolutions and poolings the contract holds every engine
# to, and the gradients of a loss through them: spatial-pytorch.json, read by SpatialConvolutionContract and PoolingContract.
#
#   python spatial-pytorch.py > spatial-pytorch.json      (PyTorch 2.14.1, CPU, single precision)
#
# Images are laid out with their channels last, as the seam lays them out, so every array printed here is flat in that order,
# and a kernel is printed as this library keeps it: as many rows as the window holds values, channels in included, by as many
# columns as channels out. PyTorch keeps the same numbers channels first. Where PyTorch cannot say what is asked -- a stride
# with padding='same', a causal border, a border no stated padding gives -- the border is worked out here as TensorFlow
# works it out (kernel_shape_util.cc) and padded by hand, so the values are still PyTorch's arithmetic over that border.

import json
import math
import sys

import numpy as np
import torch
import torch.nn.functional as F

random = np.random.RandomState(20261009)
cases = []


def drawn(*shape, low=-1.0, high=1.0):
    return random.uniform(low, high, size=shape).astype(np.float32)


def flat(array):
    return [float(value) for value in np.asarray(array, dtype=np.float32).reshape(-1)]


def tf_same(length, size, stride):
    places = (length + stride - 1) // stride
    needed = max(0, (places - 1) * stride + size - length)
    return needed // 2, needed - needed // 2


def border(mode, length, size, stride):
    # The border along one axis: (before, after).
    if mode == "same":
        return tf_same(length, size, stride)
    if mode == "causal":
        return size - 1, 0
    return mode, mode


def padded(x, borders, value=0.0):
    # x is channels first; borders is one (before, after) per spatial axis, first axis first. F.pad wants the last axis first.
    pads = []
    for before, after in reversed(borders):
        pads += [before, after]
    return F.pad(x, pads, value=value)


def to_first(x_last):
    # (N, s..., C) -> (N, C, s...)
    rank = x_last.dim()
    return x_last.permute(0, rank - 1, *range(1, rank - 1))


def to_last(x_first):
    rank = x_first.dim()
    return x_first.permute(0, *range(2, rank), 1)


def conv_functional(rank):
    return {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}[rank]


def max_functional(rank):
    return {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}[rank]


def avg_functional(rank):
    return {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}[rank]


def convolution(name, rank, n, extents, in_channels, out_channels, sizes, stride, padding):
    x_last = torch.tensor(drawn(n, *extents, in_channels), requires_grad=True)
    weight = torch.tensor(drawn(out_channels, in_channels, *sizes), requires_grad=True)
    bias = torch.tensor(drawn(out_channels), requires_grad=True)

    borders = [border(padding, length, size, stride) for length, size in zip(extents, sizes)]
    y_first = conv_functional(rank)(padded(to_first(x_last), borders), weight, bias, stride=stride)
    y_last = to_last(y_first)

    if isinstance(padding, int):
        check = conv_functional(rank)(to_first(x_last), weight, bias, stride=stride, padding=padding)
        assert torch.allclose(check, y_first, atol=1e-6), name

    g = torch.tensor(drawn(*y_last.shape))
    (y_last * g).sum().backward()

    # Our kernel: rows are (sizes..., channel in), columns channel out.
    ours = weight.detach().permute(*range(2, 2 + rank), 1, 0).reshape(-1, out_channels)
    ours_grad = weight.grad.permute(*range(2, 2 + rank), 1, 0).reshape(-1, out_channels)

    cases.append({
        "name": name, "kind": "convolution", "rank": rank, "n": n, "extents": list(extents), "channels": in_channels,
        "outChannels": out_channels, "sizes": list(sizes), "stride": stride, "padding": padding,
        "input": flat(x_last.detach()), "kernel": flat(ours), "bias": flat(bias.detach()),
        "output": flat(y_last.detach()), "outShape": list(y_last.shape), "gradOutput": flat(g),
        "gradInput": flat(x_last.grad), "gradKernel": flat(ours_grad), "gradBias": flat(bias.grad),
    })


def max_pooling(name, rank, n, extents, channels, sizes, stride, padding):
    x_last = torch.tensor(drawn(n, *extents, channels, low=-3.0, high=1.0), requires_grad=True)
    borders = [border(padding, length, size, stride) for length, size in zip(extents, sizes)]
    y_first = max_functional(rank)(padded(to_first(x_last), borders, value=-math.inf), sizes, stride=stride)
    y_last = to_last(y_first)

    if isinstance(padding, int):
        check = max_functional(rank)(to_first(x_last), sizes, stride=stride, padding=padding)
        assert torch.equal(check, y_first), name

    g = torch.tensor(drawn(*y_last.shape))
    (y_last * g).sum().backward()

    cases.append({
        "name": name, "kind": "max", "rank": rank, "n": n, "extents": list(extents), "channels": channels,
        "sizes": list(sizes), "stride": stride, "padding": padding,
        "input": flat(x_last.detach()), "output": flat(y_last.detach()), "outShape": list(y_last.shape),
        "gradOutput": flat(g), "gradInput": flat(x_last.grad),
    })


def avg_pooling(name, rank, n, extents, channels, sizes, stride, padding, counts_padding):
    x_last = torch.tensor(drawn(n, *extents, channels, low=-3.0, high=1.0), requires_grad=True)
    borders = [border(padding, length, size, stride) for length, size in zip(extents, sizes)]
    volume = int(np.prod(sizes))
    x_first = padded(to_first(x_last), borders)
    sums = avg_functional(rank)(x_first, sizes, stride=stride) * volume

    if counts_padding:
        y_first = sums / volume
    else:
        ones = padded(torch.ones_like(to_first(x_last).detach()), borders)
        y_first = sums / (avg_functional(rank)(ones, sizes, stride=stride) * volume)

    y_last = to_last(y_first)

    if isinstance(padding, int):
        check = avg_functional(rank)(to_first(x_last), sizes, stride=stride, padding=padding, count_include_pad=counts_padding)
        assert torch.allclose(check, y_first, atol=1e-6), name

    g = torch.tensor(drawn(*y_last.shape))
    (y_last * g).sum().backward()

    cases.append({
        "name": name, "kind": "average", "rank": rank, "n": n, "extents": list(extents), "channels": channels,
        "sizes": list(sizes), "stride": stride, "padding": padding, "countsPadding": counts_padding,
        "input": flat(x_last.detach()), "output": flat(y_last.detach()), "outShape": list(y_last.shape),
        "gradOutput": flat(g), "gradInput": flat(x_last.grad),
    })


def global_pooling(name, kind, rank, n, extents, channels):
    x_last = torch.tensor(drawn(n, *extents, channels, low=-3.0, high=1.0), requires_grad=True)
    axes = tuple(range(1, 1 + rank))
    y_last = x_last.mean(dim=axes) if kind == "globalAverage" else x_last.amax(dim=axes)

    g = torch.tensor(drawn(*y_last.shape))
    (y_last * g).sum().backward()

    cases.append({
        "name": name, "kind": kind, "rank": rank, "n": n, "extents": list(extents), "channels": channels,
        "input": flat(x_last.detach()), "output": flat(y_last.detach()), "outShape": list(y_last.shape),
        "gradOutput": flat(g), "gradInput": flat(x_last.grad),
    })


torch.manual_seed(0)

convolution("conv1d stated", 1, 2, (7,), 3, 2, (3,), 2, 1)
convolution("conv1d valid", 1, 1, (9,), 1, 3, (4,), 1, 0)
convolution("conv1d same even", 1, 2, (8,), 2, 2, (4,), 1, "same")
convolution("conv1d same strided", 1, 1, (10,), 2, 2, (3,), 3, "same")
convolution("conv1d causal", 1, 2, (8,), 2, 2, (3,), 1, "causal")
convolution("conv1d causal strided", 1, 1, (9,), 1, 2, (3,), 2, "causal")
convolution("conv2d stated", 2, 2, (5, 6), 2, 2, (3, 2), 2, 1)
convolution("conv3d valid", 3, 1, (4, 5, 6), 2, 2, (2, 3, 2), 1, 0)
convolution("conv3d stated", 3, 2, (5, 5, 5), 1, 2, (3, 3, 3), 2, 1)
convolution("conv3d same strided", 3, 1, (5, 6, 7), 2, 3, (2, 3, 4), 2, "same")
convolution("conv3d causal", 3, 1, (3, 3, 5), 2, 2, (2, 2, 3), 1, "causal")

max_pooling("max1d plain", 1, 2, (9,), 3, (3,), 3, 0)
max_pooling("max1d stated", 1, 2, (8,), 2, (3,), 2, 1)
max_pooling("max1d same", 1, 1, (7,), 2, (3,), 2, "same")
max_pooling("max2d plain", 2, 2, (6, 6), 2, (2, 2), 2, 0)
max_pooling("max2d overlapping", 2, 1, (5, 5), 2, (3, 3), 1, 0)
max_pooling("max2d stated", 2, 2, (5, 5), 2, (3, 3), 2, 1)
max_pooling("max2d same", 2, 1, (5, 6), 2, (3, 3), 2, "same")
max_pooling("max2d same even", 2, 1, (6, 5), 1, (2, 2), 1, "same")
max_pooling("max3d plain", 3, 2, (4, 4, 4), 2, (2, 2, 2), 2, 0)
max_pooling("max3d stated", 3, 1, (5, 5, 5), 2, (3, 3, 3), 2, 1)
max_pooling("max3d same", 3, 1, (5, 4, 3), 2, (2, 2, 2), 2, "same")

avg_pooling("avg1d plain", 1, 2, (9,), 3, (3,), 2, 0, False)
avg_pooling("avg1d stated counting", 1, 2, (8,), 2, (3,), 2, 1, True)
avg_pooling("avg1d stated leaving out", 1, 2, (8,), 2, (3,), 2, 1, False)
avg_pooling("avg1d same", 1, 1, (7,), 2, (3,), 2, "same", False)
avg_pooling("avg2d plain", 2, 2, (6, 6), 2, (2, 2), 2, 0, False)
avg_pooling("avg2d stated counting", 2, 2, (5, 5), 2, (3, 3), 2, 1, True)
avg_pooling("avg2d stated leaving out", 2, 2, (5, 5), 2, (3, 3), 2, 1, False)
avg_pooling("avg2d same", 2, 1, (5, 6), 2, (3, 3), 2, "same", False)
avg_pooling("avg3d plain", 3, 2, (4, 4, 4), 2, (2, 2, 2), 2, 0, False)
avg_pooling("avg3d stated counting", 3, 1, (5, 5, 5), 2, (3, 3, 3), 2, 1, True)
avg_pooling("avg3d same", 3, 1, (5, 4, 3), 2, (2, 2, 2), 2, "same", False)

global_pooling("global average 1d", "globalAverage", 1, 2, (7,), 3)
global_pooling("global average 2d", "globalAverage", 2, 2, (4, 5), 3)
global_pooling("global average 3d", "globalAverage", 3, 2, (3, 4, 2), 2)
global_pooling("global max 1d", "globalMax", 1, 2, (7,), 3)
global_pooling("global max 2d", "globalMax", 2, 2, (4, 5), 3)
global_pooling("global max 3d", "globalMax", 3, 2, (3, 4, 2), 2)

json.dump({"torch": torch.__version__, "cases": cases}, sys.stdout, indent=None, separators=(",", ":"))
