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
#
# A stride and a dilation are printed as a list, one number an axis, whether or not the axes agree; a padding is printed as
# the one number on every side, as the word for a border worked out, or as a flat pair for each axis where each end is
# stated apart. One case is not PyTorch's own layer at all and says so: an average pooling with a dilated window, which
# torch.nn has no layer for. Its numerator is a convolution by a kernel of ones and its divisor the same convolution over
# the mask of what is a value of the example, which is what an average over a dilated window is; ONNX's AveragePool-19 is
# the only reader that can ask for one.

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


def each(value, rank):
    # A setting written once for every axis, or once per axis already.
    return [int(one) for one in (value if isinstance(value, (list, tuple)) else [value] * rank)]


def tf_same(length, size, stride, dilation=1):
    # TensorFlow's rule, over what the window REACHES: a window of three at a dilation of two reaches over five.
    reach = dilation * (size - 1) + 1
    places = (length + stride - 1) // stride
    needed = max(0, (places - 1) * stride + reach - length)
    return needed // 2, needed - needed // 2


def border(mode, length, size, stride, dilation=1):
    # The border along one axis: (before, after).
    if mode == "same":
        return tf_same(length, size, stride, dilation)
    if mode == "causal":
        return dilation * (size - 1), 0
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


CONV = {1: F.conv1d, 2: F.conv2d, 3: F.conv3d}
CONV_T = {1: F.conv_transpose1d, 2: F.conv_transpose2d, 3: F.conv_transpose3d}
MAXP = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}
AVGP = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}
AD_AVG = {1: F.adaptive_avg_pool1d, 2: F.adaptive_avg_pool2d, 3: F.adaptive_avg_pool3d}
AD_MAX = {1: F.adaptive_max_pool1d, 2: F.adaptive_max_pool2d, 3: F.adaptive_max_pool3d}


def window_of(rank, sizes, stride, padding, dilation=1, rounds_up=False):
    # What every case prints of the window it walked.
    said = {"sizes": list(sizes), "stride": each(stride, rank), "dilation": each(dilation, rank)}
    if isinstance(padding, (list, tuple)):
        said["padding"] = [int(one) for one in padding]
    else:
        said["padding"] = padding
    if rounds_up:
        said["roundsUp"] = True
    return said


def borders_of(rank, extents, sizes, stride, padding, dilation):
    strides, dilations = each(stride, rank), each(dilation, rank)
    if isinstance(padding, (list, tuple)):
        return [(int(padding[2 * axis]), int(padding[2 * axis + 1])) for axis in range(rank)]
    return [border(padding, length, size, step, rate)
            for length, size, step, rate in zip(extents, sizes, strides, dilations)]


def convolution(name, rank, n, extents, in_channels, out_channels, sizes, stride, padding, dilation=1, groups=1):
    """A convolution, with a stride and a dilation of its own for each axis and its channels in groups."""
    strides, dilations = each(stride, rank), each(dilation, rank)
    x_last = torch.tensor(drawn(n, *extents, in_channels), requires_grad=True)
    weight = torch.tensor(drawn(out_channels, in_channels // groups, *sizes), requires_grad=True)
    bias = torch.tensor(drawn(out_channels), requires_grad=True)

    borders = borders_of(rank, extents, sizes, stride, padding, dilation)
    y_first = CONV[rank](padded(to_first(x_last), borders), weight, bias,
                         stride=tuple(strides), dilation=tuple(dilations), groups=groups)
    y_last = to_last(y_first)

    if isinstance(padding, int):
        check = CONV[rank](to_first(x_last), weight, bias, stride=tuple(strides), padding=padding,
                           dilation=tuple(dilations), groups=groups)
        assert torch.allclose(check, y_first, atol=1e-6), name

    g = torch.tensor(drawn(*y_last.shape))
    (y_last * g).sum().backward()

    # Our kernel: rows are (sizes..., channel in of the group), columns channel out.
    ours = weight.detach().permute(*range(2, 2 + rank), 1, 0).reshape(-1, out_channels)
    ours_grad = weight.grad.permute(*range(2, 2 + rank), 1, 0).reshape(-1, out_channels)

    cases.append({
        "name": name, "kind": "convolution", "rank": rank, "n": n, "extents": list(extents),
        "channels": in_channels, "outChannels": out_channels, "groups": groups,
        **window_of(rank, sizes, stride, padding, dilation),
        "input": flat(x_last.detach()), "kernel": flat(ours), "bias": flat(bias.detach()),
        "output": flat(y_last.detach()), "outShape": list(y_last.shape), "gradOutput": flat(g),
        "gradInput": flat(x_last.grad), "gradKernel": flat(ours_grad), "gradBias": flat(bias.grad),
    })


def transposed(name, rank, n, extents, in_channels, out_channels, sizes, stride, padding, output_padding=0,
               dilation=1, groups=1):
    """A transposed convolution: the one that gives back the places a convolution took away."""
    strides, dilations = each(stride, rank), each(dilation, rank)
    pads, extra = each(padding, rank), each(output_padding, rank)
    x_last = torch.tensor(drawn(n, *extents, in_channels), requires_grad=True)
    # PyTorch keeps a transposed kernel channels in first: (in, out/groups, sizes...).
    weight = torch.tensor(drawn(in_channels, out_channels // groups, *sizes), requires_grad=True)
    bias = torch.tensor(drawn(out_channels), requires_grad=True)

    y_first = CONV_T[rank](to_first(x_last), weight, bias, stride=tuple(strides), padding=tuple(pads),
                           output_padding=tuple(extra), dilation=tuple(dilations), groups=groups)
    y_last = to_last(y_first)

    g = torch.tensor(drawn(*y_last.shape))
    (y_last * g).sum().backward()

    # Our kernel for the fold: a row for every channel in, holding the window's places by the channels out of the group, so
    # one product of the example's own channels makes the patches a fold puts back.
    ours = weight.detach().permute(0, *range(2, 2 + rank), 1).reshape(in_channels, -1)
    ours_grad = weight.grad.permute(0, *range(2, 2 + rank), 1).reshape(in_channels, -1)

    cases.append({
        "name": name, "kind": "transposed", "rank": rank, "n": n, "extents": list(extents),
        "channels": in_channels, "outChannels": out_channels, "groups": groups,
        "outputPadding": extra,
        # A transposed convolution states its border, never works it out, so it is written at each end of each axis.
        **window_of(rank, sizes, stride, [side for pad in pads for side in (pad, pad)], dilation),
        "input": flat(x_last.detach()), "kernel": flat(ours), "bias": flat(bias.detach()),
        "output": flat(y_last.detach()), "outShape": list(y_last.shape), "gradOutput": flat(g),
        "gradInput": flat(x_last.grad), "gradKernel": flat(ours_grad), "gradBias": flat(bias.grad),
    })


def max_pooling(name, rank, n, extents, channels, sizes, stride, padding, dilation=1, rounds_up=False):
    strides, dilations = each(stride, rank), each(dilation, rank)
    x_last = torch.tensor(drawn(n, *extents, channels, low=-3.0, high=1.0), requires_grad=True)
    borders = borders_of(rank, extents, sizes, stride, padding, dilation)
    y_first = MAXP[rank](padded(to_first(x_last), borders, value=-math.inf), tuple(sizes), stride=tuple(strides),
                         dilation=tuple(dilations), ceil_mode=rounds_up)
    y_last = to_last(y_first)

    if isinstance(padding, int):
        check = MAXP[rank](to_first(x_last), tuple(sizes), stride=tuple(strides), padding=padding,
                           dilation=tuple(dilations), ceil_mode=rounds_up)
        assert torch.equal(check, y_first), name

    g = torch.tensor(drawn(*y_last.shape))
    (y_last * g).sum().backward()

    cases.append({
        "name": name, "kind": "max", "rank": rank, "n": n, "extents": list(extents), "channels": channels,
        **window_of(rank, sizes, stride, padding, dilation, rounds_up),
        "input": flat(x_last.detach()), "output": flat(y_last.detach()), "outShape": list(y_last.shape),
        "gradOutput": flat(g), "gradInput": flat(x_last.grad),
    })


def avg_pooling(name, rank, n, extents, channels, sizes, stride, padding, counts_padding, dilation=1, rounds_up=False):
    strides, dilations = each(stride, rank), each(dilation, rank)
    x_last = torch.tensor(drawn(n, *extents, channels, low=-3.0, high=1.0), requires_grad=True)
    borders = borders_of(rank, extents, sizes, stride, padding, dilation)
    x_first = padded(to_first(x_last), borders)
    volume = int(np.prod(sizes))
    undilated = all(rate == 1 for rate in dilations)

    if undilated:
        # PyTorch's own layer, as every release before this one took it: the sum over the window, divided by the window where
        # the border counts and by the values of the example where it does not.
        sums = AVGP[rank](x_first, tuple(sizes), stride=tuple(strides)) * volume

        if counts_padding:
            y_first = sums / volume
        else:
            real = padded(torch.ones_like(to_first(x_last).detach()), borders)
            y_first = sums / (AVGP[rank](real, tuple(sizes), stride=tuple(strides)) * volume)
    else:
        # torch.nn has no average pooling with a dilated window, and ONNX's AveragePool-19 is the one reader that asks for
        # one. A convolution by a kernel of ones IS the sum over the window, dilation and all, and the same convolution over
        # the mask of what is a value of the example IS the count it is divided by: PyTorch's arithmetic, said another way.
        ones_kernel = torch.ones(channels, 1, *sizes)
        shared = {"stride": tuple(strides), "dilation": tuple(dilations), "groups": channels}
        sums = CONV[rank](x_first, ones_kernel, None, **shared)
        real = torch.ones_like(x_first) if counts_padding else padded(torch.ones_like(to_first(x_last).detach()), borders)
        y_first = sums / CONV[rank](real, ones_kernel, None, **shared)

    y_last = to_last(y_first)

    if isinstance(padding, int) and undilated and not rounds_up:
        check = AVGP[rank](to_first(x_last), tuple(sizes), stride=tuple(strides), padding=padding,
                           count_include_pad=counts_padding)
        assert torch.allclose(check, y_first, atol=1e-6), name

    g = torch.tensor(drawn(*y_last.shape))
    (y_last * g).sum().backward()

    cases.append({
        "name": name, "kind": "average", "rank": rank, "n": n, "extents": list(extents), "channels": channels,
        "countsPadding": counts_padding,
        **window_of(rank, sizes, stride, padding, dilation, rounds_up),
        "input": flat(x_last.detach()), "output": flat(y_last.detach()), "outShape": list(y_last.shape),
        "gradOutput": flat(g), "gradInput": flat(x_last.grad),
    })


def rounding_pooling(name, kind, rank, n, extents, channels, sizes, stride, padding, counts_padding=False):
    """A pooling that also stands at a last place whose window overhangs the end: PyTorch's own ceil_mode."""
    x_last = torch.tensor(drawn(n, *extents, channels, low=-3.0, high=1.0), requires_grad=True)
    strides = each(stride, rank)
    shared = {"stride": tuple(strides), "padding": padding, "ceil_mode": True}

    if kind == "max":
        y_first = MAXP[rank](to_first(x_last), tuple(sizes), **shared)
    else:
        y_first = AVGP[rank](to_first(x_last), tuple(sizes), count_include_pad=counts_padding, **shared)

    y_last = to_last(y_first)
    g = torch.tensor(drawn(*y_last.shape))
    (y_last * g).sum().backward()

    said = {"name": name, "kind": kind, "rank": rank, "n": n, "extents": list(extents), "channels": channels,
            **window_of(rank, sizes, stride, padding, 1, True),
            "input": flat(x_last.detach()), "output": flat(y_last.detach()), "outShape": list(y_last.shape),
            "gradOutput": flat(g), "gradInput": flat(x_last.grad)}

    if kind == "average":
        said["countsPadding"] = counts_padding

    cases.append(said)


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


def adaptive(name, kind, rank, n, extents, channels, out_sizes):
    """A pooling to a size other than one place: its bins overlap and differ where the size does not divide."""
    x_last = torch.tensor(drawn(n, *extents, channels, low=-3.0, high=1.0), requires_grad=True)
    wanted = out_sizes[0] if rank == 1 else tuple(out_sizes)
    y_first = (AD_AVG if kind == "adaptiveAverage" else AD_MAX)[rank](to_first(x_last), wanted)
    y_last = to_last(y_first)

    g = torch.tensor(drawn(*y_last.shape))
    (y_last * g).sum().backward()

    cases.append({
        "name": name, "kind": kind, "rank": rank, "n": n, "extents": list(extents), "channels": channels,
        "outSizes": list(out_sizes),
        "input": flat(x_last.detach()), "output": flat(y_last.detach()), "outShape": list(y_last.shape),
        "gradOutput": flat(g), "gradInput": flat(x_last.grad),
    })


def padding_layer(name, rank, n, extents, channels, sides, value=0.0):
    """A layer that only pads: so many places of nothing before and after each axis, of one value."""
    x_last = torch.tensor(drawn(n, *extents, channels), requires_grad=True)
    y_last = to_last(padded(to_first(x_last), sides, value=value))

    g = torch.tensor(drawn(*y_last.shape))
    (y_last * g).sum().backward()

    cases.append({
        "name": name, "kind": "pad", "rank": rank, "n": n, "extents": list(extents), "channels": channels,
        "padding": [int(one) for pair in sides for one in pair], "value": value,
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

# --- a stride of its own for each axis ---
convolution("conv2d stride per axis", 2, 2, (7, 8), 2, 3, (3, 2), (2, 1), 0)
convolution("conv2d stride per axis stated", 2, 1, (6, 7), 2, 2, (3, 3), (1, 3), 1)
convolution("conv3d stride per axis", 3, 1, (5, 6, 7), 2, 2, (2, 3, 2), (1, 2, 3), 0)
max_pooling("max2d stride per axis", 2, 2, (7, 8), 2, (2, 2), (2, 1), 0)
avg_pooling("avg2d stride per axis", 2, 1, (7, 8), 2, (2, 3), (1, 2), 0, False)

# --- a dilated window ---
convolution("conv1d dilated", 1, 2, (12,), 2, 3, (3,), 1, 0, dilation=3)
convolution("conv1d dilated stated", 1, 1, (10,), 1, 2, (3,), 1, 2, dilation=2)
convolution("conv1d dilated same", 1, 1, (9,), 2, 2, (3,), 1, "same", dilation=2)
convolution("conv1d dilated causal", 1, 1, (9,), 2, 2, (3,), 1, "causal", dilation=2)
convolution("conv2d dilated per axis", 2, 2, (9, 9), 2, 2, (3, 2), 1, 0, dilation=(2, 3))
convolution("conv2d dilated strided", 2, 1, (11, 11), 1, 2, (3, 3), 2, 1, dilation=2)
convolution("conv3d dilated", 3, 1, (7, 7, 7), 2, 2, (2, 2, 2), 1, 0, dilation=2)
max_pooling("max2d dilated", 2, 1, (9, 9), 2, (3, 3), 1, 0, dilation=2)
max_pooling("max1d dilated strided", 1, 2, (11,), 2, (3,), 2, 1, dilation=2)
avg_pooling("avg1d dilated", 1, 2, (11,), 2, (3,), 1, 0, False, dilation=2)
avg_pooling("avg2d dilated counting", 2, 1, (9, 9), 2, (3, 3), 1, 1, True, dilation=2)

# --- a border stated at each end of each axis ---
convolution("conv1d border per end", 1, 2, (8,), 2, 2, (3,), 1, [1, 2])
convolution("conv2d border per end", 2, 2, (6, 6), 2, 2, (3, 3), 1, [1, 0, 0, 2])
convolution("conv3d border per end", 3, 1, (4, 4, 4), 1, 2, (2, 2, 2), 1, [1, 0, 0, 1, 1, 1])
max_pooling("max2d border per end", 2, 1, (6, 7), 2, (2, 2), 2, [1, 0, 0, 1])
avg_pooling("avg2d border per end leaving out", 2, 1, (6, 7), 2, (2, 2), 2, [1, 0, 0, 1], False)

# --- a pooling that rounds the number of places up ---
rounding_pooling("max1d rounding up", "max", 1, 2, (8,), 2, (3,), 2, 0)
rounding_pooling("max1d rounding up stated", "max", 1, 1, (8,), 2, (3,), 2, 1)
rounding_pooling("max2d rounding up per axis", "max", 2, 1, (7, 8), 2, (2, 3), 2, 0)
rounding_pooling("max2d rounding up", "max", 2, 1, (8, 8), 2, (3, 3), 2, 0)
rounding_pooling("max3d rounding up", "max", 3, 1, (6, 6, 6), 2, (3, 3, 3), 2, 0)
rounding_pooling("avg1d rounding up leaving out", "average", 1, 2, (8,), 2, (3,), 2, 0, False)
rounding_pooling("avg1d rounding up counting", "average", 1, 2, (8,), 2, (3,), 2, 1, True)
rounding_pooling("avg2d rounding up counting", "average", 2, 1, (8, 8), 2, (3, 3), 2, 1, True)
rounding_pooling("avg2d rounding up leaving out", "average", 2, 1, (8, 8), 2, (3, 3), 2, 1, False)
rounding_pooling("avg3d rounding up", "average", 3, 1, (6, 6, 6), 2, (3, 3, 3), 2, 0, False)

# --- channels in groups, and a depthwise and a separable convolution ---
convolution("conv1d groups", 1, 2, (9,), 4, 6, (3,), 1, 0, groups=2)
convolution("conv2d groups", 2, 2, (6, 6), 4, 6, (3, 3), 1, 1, groups=2)
convolution("conv2d groups strided", 2, 1, (7, 7), 6, 9, (3, 3), 2, 0, groups=3)
convolution("conv2d groups of eight", 2, 1, (6, 6), 8, 8, (3, 3), 1, 1, groups=8)
convolution("conv3d groups", 3, 1, (5, 5, 5), 4, 4, (2, 2, 2), 1, 0, groups=2)
convolution("conv1d depthwise", 1, 2, (9,), 3, 3, (3,), 1, 0, groups=3)
convolution("conv2d depthwise", 2, 2, (6, 7), 4, 4, (3, 3), 1, 1, groups=4)
convolution("conv2d depthwise multiplied", 2, 2, (6, 7), 3, 6, (3, 3), 1, 0, groups=3)
convolution("conv2d depthwise multiplied by three", 2, 1, (5, 5), 2, 6, (3, 3), 1, 0, groups=2)
convolution("conv2d depthwise strided", 2, 1, (8, 8), 4, 4, (3, 3), 2, 1, groups=4)
convolution("conv2d depthwise dilated", 2, 1, (9, 9), 3, 3, (3, 3), 1, 0, dilation=2, groups=3)
convolution("conv3d depthwise", 3, 1, (5, 5, 5), 3, 3, (2, 2, 2), 1, 0, groups=3)
convolution("conv2d pointwise", 2, 2, (5, 5), 6, 4, (1, 1), 1, 0)

# --- a transposed convolution ---
transposed("convT1d plain", 1, 2, (5,), 2, 3, (3,), 1, 0)
transposed("convT1d strided", 1, 2, (5,), 2, 2, (3,), 2, 1)
transposed("convT1d output padded", 1, 1, (5,), 3, 2, (3,), 2, 1, output_padding=1)
transposed("convT1d dilated", 1, 1, (6,), 2, 2, (3,), 1, 1, dilation=2)
transposed("convT1d dilated output padded", 1, 1, (6,), 2, 2, (3,), 1, 1, output_padding=2, dilation=3)
transposed("convT2d strided", 2, 2, (4, 5), 2, 3, (3, 3), 2, 1)
transposed("convT2d stride per axis", 2, 1, (4, 5), 2, 2, (3, 2), (2, 1), 0)
transposed("convT2d output padded", 2, 1, (4, 4), 2, 2, (3, 3), 2, 1, output_padding=1)
transposed("convT2d groups", 2, 1, (4, 4), 4, 6, (3, 3), 2, 0, groups=2)
transposed("convT3d strided", 3, 1, (3, 3, 4), 2, 2, (2, 2, 3), 2, 1)
transposed("convT3d plain", 3, 1, (3, 4, 3), 1, 2, (2, 2, 2), 1, 0)

# --- a pooling to a size other than one place ---
adaptive("adaptive avg 1d divides", "adaptiveAverage", 1, 2, (9,), 3, (3,))
adaptive("adaptive avg 1d overlaps", "adaptiveAverage", 1, 2, (7,), 3, (4,))
adaptive("adaptive avg 1d longer", "adaptiveAverage", 1, 1, (3,), 2, (5,))
adaptive("adaptive avg 2d overlaps", "adaptiveAverage", 2, 2, (9, 7), 2, (4, 3))
adaptive("adaptive avg 2d divides", "adaptiveAverage", 2, 1, (8, 6), 2, (4, 3))
adaptive("adaptive avg 3d overlaps", "adaptiveAverage", 3, 1, (5, 7, 4), 2, (2, 3, 3))
adaptive("adaptive max 1d overlaps", "adaptiveMax", 1, 2, (7,), 3, (4,))
adaptive("adaptive max 1d divides", "adaptiveMax", 1, 2, (10,), 2, (5,))
adaptive("adaptive max 2d overlaps", "adaptiveMax", 2, 2, (9, 7), 2, (4, 3))
adaptive("adaptive max 2d longer", "adaptiveMax", 2, 1, (3, 3), 2, (5, 4))
adaptive("adaptive max 3d overlaps", "adaptiveMax", 3, 1, (5, 7, 4), 2, (2, 3, 3))

# --- a layer that only pads ---
padding_layer("pad 1d", 1, 2, (6,), 3, [(1, 2)])
padding_layer("pad 2d", 2, 2, (4, 5), 2, [(1, 2), (3, 0)])
padding_layer("pad 2d even", 2, 1, (5, 5), 1, [(2, 2), (2, 2)])
padding_layer("pad 2d of a value", 2, 1, (4, 4), 2, [(1, 1), (0, 2)], value=-2.5)
padding_layer("pad 3d", 3, 1, (3, 4, 3), 2, [(1, 0), (0, 2), (1, 1)])

json.dump({"torch": torch.__version__, "cases": cases}, sys.stdout, indent=None, separators=(",", ":"))
