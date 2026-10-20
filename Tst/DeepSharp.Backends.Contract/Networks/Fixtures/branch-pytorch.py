# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Prints what PyTorch's own arithmetic makes of four networks that are not a straight line, on rows worked out from a formula: a
# residual block (the input added back after two linear layers), a dense block (every step's output laid beside the input and
# every step before it), a network that takes two blocks of one row's columns and adds what each makes of its own, and a network
# shaped like a U (a step that is laid beside an earlier one on the way up). branch-pytorch.json is what BranchContract holds each
# network to, on every engine: the values it gives, the gradient of the mean of its values times a coefficient with respect to
# the input and to every number the network learns, by the path DeepSharp gives each slot. Its output, when it was run, is
# branch-pytorch.txt beside it.
#
# Run once with PyTorch 2.14.1 for the CPU (no NumPy needed):
#
#     python branch-pytorch.py > branch-pytorch.txt
#
# DeepSharp lays a linear layer's weights out inputs by outputs, so the weights here are laid out so as well and multiplied the
# way DeepSharp multiplies them: rows times weights, plus the bias. Every number is worked out from its place, so the same
# numbers are made again on every machine.

import json
import os
import platform

import torch

torch.set_default_dtype(torch.float32)

here = os.path.dirname(os.path.abspath(__file__))


def weights(rows, columns, seed):
    """((7 x row + 3 x column + seed) mod 11 - 5) / 20, laid out inputs by outputs."""
    return torch.tensor(
        [[((7 * row + 3 * column + seed) % 11 - 5) / 20 for column in range(columns)] for row in range(rows)],
        dtype=torch.float32,
        requires_grad=True,
    )


def bias(width, seed):
    return torch.tensor([((5 * column + seed) % 7 - 3) / 40 + 0.0053 for column in range(width)], dtype=torch.float32, requires_grad=True)


def rows(count, width):
    """What a batch holds: ((3 x row + 5 x column) mod 9 - 4) / 8, and a little more, so no rectifier is handed exactly nothing."""
    return torch.tensor(
        [[((3 * row + 5 * column) % 9 - 4) / 8 + 0.0137 for column in range(width)] for row in range(count)], dtype=torch.float32, requires_grad=True
    )


def coefficients(shape):
    count = 1
    for length in shape:
        count *= length
    return torch.tensor([((3 * at) % 7 - 3) / 10 for at in range(count)], dtype=torch.float32).reshape(shape)


def flat(tensor):
    return [float(value) for value in tensor.detach().reshape(-1).tolist()]


def case(name, network, x, slots):
    """Runs the network, takes the gradient of the mean of its values times a coefficient, and keeps every number asked for."""
    out = network(x)
    loss = (out * coefficients(out.shape)).mean()
    loss.backward()
    return {
        "name": name,
        "rows": x.shape[0],
        "width": x.shape[1],
        "outputWidth": out.shape[1],
        "output": flat(out),
        "gradInput": flat(x.grad),
        "slots": {path: {"shape": list(tensor.shape), "values": flat(tensor), "grad": flat(tensor.grad)} for path, tensor in slots.items()},
    }


def linear(x, w, b):
    return x @ w + b


walks = []

# A residual block: the input, added to what two linear layers and a rectifier make of it.
w1, b1, w2, b2 = weights(6, 6, 0), bias(6, 0), weights(6, 6, 3), bias(6, 2)
walks.append(
    case(
        "residual",
        lambda x: x + linear(torch.relu(linear(x, w1, b1)), w2, b2),
        rows(4, 6),
        {"first.weight": w1, "first.bias": b1, "second.weight": w2, "second.bias": b2},
    )
)

# A dense block: each linear layer reads the input beside everything the layers before it made.
wa, ba = weights(5, 4, 1), bias(4, 1)
wb, bb = weights(9, 4, 4), bias(4, 3)
wo, bo = weights(13, 2, 2), bias(2, 5)


def dense_block(x):
    a = torch.relu(linear(x, wa, ba))
    b = torch.relu(linear(torch.cat([x, a], dim=1), wb, bb))
    return linear(torch.cat([x, a, b], dim=1), wo, bo)


walks.append(
    case(
        "dense block",
        dense_block,
        rows(4, 5),
        {"a.weight": wa, "a.bias": ba, "b.weight": wb, "b.bias": bb, "out.weight": wo, "out.bias": bo},
    )
)

# Two inputs, as the columns of one row: the first three and the last three, each through a layer of its own, then added.
wl, bl = weights(3, 4, 1), bias(4, 4)
wr, br = weights(3, 4, 5), bias(4, 6)
walks.append(
    case(
        "two inputs",
        lambda x: linear(x[:, 0:3], wl, bl) + linear(x[:, 3:6], wr, br),
        rows(4, 6),
        {"left.weight": wl, "left.bias": bl, "right.weight": wr, "right.bias": br},
    )
)

# A U: down through two layers, up through one, and the first step's output laid beside what comes up before the last layer.
wd1, bd1 = weights(6, 4, 2), bias(4, 2)
wd2, bd2 = weights(4, 2, 6), bias(2, 1)
wu, bu = weights(2, 4, 3), bias(4, 5)
wf, bf = weights(8, 3, 7), bias(3, 3)


def u_shape(x):
    first = torch.relu(linear(x, wd1, bd1))
    bottom = torch.relu(linear(first, wd2, bd2))
    up = torch.relu(linear(bottom, wu, bu))
    return linear(torch.cat([first, up], dim=1), wf, bf)


walks.append(
    case(
        "u shape",
        u_shape,
        rows(4, 6),
        {"down1.weight": wd1, "down1.bias": bd1, "down2.weight": wd2, "down2.bias": bd2, "up.weight": wu, "up.bias": bu, "last.weight": wf, "last.bias": bf},
    )
)

for walk in walks:
    print(walk["name"], "output", walk["output"][:4], "...")

with open(os.path.join(here, "branch-pytorch.json"), "w", encoding="utf-8", newline="\n") as written:
    json.dump({"torch": torch.__version__, "python": platform.python_version(), "walks": walks}, written, indent=1)
    written.write("\n")
