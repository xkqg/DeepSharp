# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# The numbers the contract holds every engine's earth mover's distance to: the loss of three rows of logits against
# their answers, and its gradient for every logit, as PyTorch works it out; and the same loss, row by row, as scipy's
# Wasserstein distance between the two distributions over the band positions 0, 1, 2, ... Its output, when it was run,
# is earth-movers.txt beside it.
#
# Run with PyTorch 2.14.1 for the CPU, scipy 1.18.1 and numpy 2.5.3 on Python 3.14.7:
#
#     python earth-movers.py
#
# The loss: each row's logits through a softmax are its shares; the distance between the cumulative shares and the
# cumulative answers, summed over every threshold but the last (where both are one), is the row's distance; the batch's
# loss is the mean of its rows'. With a remainder, the last answer is what is left of the whole, outside the bands'
# order: the bands are compared as shares of what they hold together, and the remainder by how far its share is off.

import platform

import numpy as np
import scipy
import torch
from scipy import stats

torch.set_printoptions(precision=10)

logits = [
    [0.3, -0.2, 0.5, 0.1, -0.4],
    [1.0, 0.2, -0.5, 0.0, 0.3],
    [-0.1, 0.4, 0.2, -0.3, 0.6],
]

ordered = [
    [0.1, 0.2, 0.3, 0.4, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0],
    [0.25, 0.25, 0.25, 0.25, 0.0],
]

with_remainder = [
    [0.1, 0.2, 0.3, 0.2, 0.2],
    [0.0, 0.5, 0.3, 0.0, 0.2],
    [0.3, 0.3, 0.2, 0.1, 0.1],
]


def distance(outputs, answers):
    shares = torch.softmax(outputs, dim=1)
    below = torch.cumsum(shares - answers, dim=1)[:, :-1]
    return below.abs().sum(dim=1).mean()


def distance_with_remainder(outputs, answers):
    shares = torch.softmax(outputs, dim=1)
    bands, held = shares[:, :-1], answers[:, :-1]
    shape = bands / bands.sum(dim=1, keepdim=True) - held / held.sum(dim=1, keepdim=True)
    below = torch.cumsum(shape, dim=1)[:, :-1]
    return (below.abs().sum(dim=1) + (shares[:, -1] - answers[:, -1]).abs()).mean()


def scipy_distance(outputs, answers, remainder):
    shares = np.exp(np.array(outputs, dtype=np.float64))
    shares = shares / shares.sum(axis=1, keepdims=True)
    answers = np.array(answers, dtype=np.float64)
    rows = []

    for predicted, held in zip(shares, answers):
        if remainder:
            places = np.arange(len(held) - 1)
            rows.append(stats.wasserstein_distance(places, places, predicted[:-1], held[:-1]) + abs(predicted[-1] - held[-1]))
        else:
            places = np.arange(len(held))
            rows.append(stats.wasserstein_distance(places, places, predicted, held))

    return float(np.mean(rows))


def show(name, loss, answers, remainder):
    outputs = torch.tensor(logits, dtype=torch.float32, requires_grad=True)
    value = loss(outputs, torch.tensor(answers, dtype=torch.float32))
    value.backward()

    print(f"{name} loss (PyTorch, float32): {value.item()!r}")
    print(f"{name} loss (scipy, float64):   {scipy_distance(logits, answers, remainder)!r}")
    print(f"{name} gradient of every logit, row by row:")

    for row in outputs.grad.tolist():
        print("    " + ", ".join(repr(each) for each in row))

    shares = torch.softmax(torch.tensor(logits, dtype=torch.float64), dim=1).numpy()
    print(f"{name} predictions, row by row:")

    for row in shares.tolist():
        print("    " + ", ".join(repr(each) for each in row))


print(f"torch {torch.__version__}, scipy {scipy.__version__}, numpy {np.__version__}, Python {platform.python_version()}")
show("ordered", distance, ordered, remainder=False)
show("remainder", distance_with_remainder, with_remainder, remainder=True)

# The smallest distance from a kink of each loss — a cumulative difference, or the remainder's, at nought — so a nudge of a
# hundredth of a logit, which moves a share by less than a three-hundredth, crosses none.
outputs = torch.softmax(torch.tensor(logits, dtype=torch.float64), dim=1)
print("ordered smallest |cumulative difference|:", torch.cumsum(outputs - torch.tensor(ordered, dtype=torch.float64), dim=1)[:, :-1].abs().min().item())
held = torch.tensor(with_remainder, dtype=torch.float64)
shape = outputs[:, :-1] / outputs[:, :-1].sum(dim=1, keepdim=True) - held[:, :-1] / held[:, :-1].sum(dim=1, keepdim=True)
print("remainder smallest |cumulative difference|:", torch.cumsum(shape, dim=1)[:, :-1].abs().min().item())
print("remainder smallest |remainder difference|:", (outputs[:, -1] - held[:, -1]).abs().min().item())
