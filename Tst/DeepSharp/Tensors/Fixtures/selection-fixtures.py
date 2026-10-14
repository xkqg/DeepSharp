# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Writes selection.json: pairs of single-precision numbers and what PyTorch makes of them — the larger of the two, the
# smaller of the two, and each of the first clamped between minus a bound and the bound — as the 32 bits of each number, so
# the tests can hold the library's own larger, smaller and clip to PyTorch's to the bit. Its output, when it was run, is
# selection.txt beside it.
#
# The numbers are the edges of the arithmetic: nought and minus nought, the largest and the smallest numbers a float holds,
# a bound and the float on either side of it, and two thousand ordinary ones from a seeded generator. None of them is not a
# number or infinite, because a clip refuses those by name before it takes a gradient.
#
# Run from this folder with PyTorch 2.14.1:
#
#     python selection-fixtures.py > selection.txt

import json
import sys

import torch

torch.manual_seed(20261014)

FLOAT_MAX = torch.finfo(torch.float32).max
TINY = torch.finfo(torch.float32).tiny
SMALLEST = float(torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)))

bounds = [0.5, 1e-3, 3.0, 1.0]


def around(bound):
    value = torch.tensor(bound, dtype=torch.float32)
    return [
        float(torch.nextafter(value, torch.tensor(float("-inf")))),
        float(value),
        float(torch.nextafter(value, torch.tensor(float("inf")))),
    ]


edges = [0.0, -0.0, 1.0, -1.0, 1e30, -1e30, FLOAT_MAX, -FLOAT_MAX, TINY, -TINY, SMALLEST, -SMALLEST, 0.1, -0.1]

for bound in bounds:
    for near in around(bound):
        edges.append(near)
        edges.append(-near)

first = []
second = []

for a in edges:
    for b in edges:
        first.append(a)
        second.append(b)

for a, b in zip(torch.randn(2000).tolist(), torch.randn(2000).tolist()):
    first.append(a)
    second.append(b)

a = torch.tensor(first, dtype=torch.float32)
b = torch.tensor(second, dtype=torch.float32)


def bits(values):
    return [int(each) & 0xFFFFFFFF for each in values.view(torch.int32).tolist()]


written = {
    "pairs": len(first),
    "first": bits(a),
    "second": bits(b),
    "maximum": bits(torch.maximum(a, b)),
    "minimum": bits(torch.minimum(a, b)),
    "bounds": bounds,
    "clamped": {repr(bound): bits(torch.clamp(a, -bound, bound)) for bound in bounds},
}

with open("selection.json", "w", encoding="utf-8", newline="\n") as file:
    json.dump(written, file, separators=(",", ":"))
    file.write("\n")

print(f"torch {torch.__version__}: {len(first)} pairs, {len(bounds)} bounds, written to selection.json", file=sys.stdout)
