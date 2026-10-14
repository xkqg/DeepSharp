# Prints PyTorch's values for a padding that fills the places it adds from the example itself - mirrored back, repeated from the
# edge, or wrapped round - and the gradient of a loss through it: border-pytorch.json, read by PaddingContract.
#
#   python border-pytorch.py > border-pytorch.json      (PyTorch 2.14.1, CPU, single precision)
#
# Examples are laid out with their channels last, as the seam lays them out, so every array printed here is flat in that order;
# PyTorch keeps the same numbers channels first and is handed them so. The places are stated in this library's order - the
# axes in their own order, a pair for each: before, then after - and turned here into the order torch.nn.functional.pad takes:
# the last axis first.
#
#   mirror   mode='reflect'    the edge is not repeated:    [0, 1, 2, 3] padded by 2 on each side is 2 1 0 1 2 3 2 1
#   repeat   mode='replicate'  the edge is repeated:        [0, 1, 2, 3] padded by 2 on each side is 0 0 0 1 2 3 3 3
#   wrap     mode='circular'   the example goes round:      [0, 1, 2, 3] padded by 2 on each side is 2 3 0 1 2 3 0 1
#
# The loss is the sum of the padded example times numbers drawn for it, so the gradient is those numbers added back onto the
# place each was taken from.

import json

import numpy as np
import torch
import torch.nn.functional as F

random = np.random.RandomState(20261014)
torch.manual_seed(20261014)

PYTORCH = {"mirror": "reflect", "repeat": "replicate", "wrap": "circular"}


def flat(array):
    return [float(value) for value in np.asarray(array, dtype=np.float32).reshape(-1)]


def case(name, mode, n, extents, channels, pads):
    rank = len(extents)
    shape = (n, *extents, channels)
    values = random.uniform(-1.0, 1.0, size=shape).astype(np.float32)

    # example, then the axes, then the channel: this library's layout; PyTorch's is example, channel, then the axes.
    x = torch.from_numpy(values).permute(0, rank + 1, *range(1, rank + 1)).contiguous().requires_grad_(True)

    # torch.nn.functional.pad takes the last axis first, and a pair for each: before, then after.
    order = []
    for axis in reversed(range(rank)):
        order += [pads[2 * axis], pads[2 * axis + 1]]

    y = F.pad(x, order, mode=PYTORCH[mode])
    out_shape = [y.shape[0], *y.shape[2:], y.shape[1]]
    weights = random.uniform(-1.0, 1.0, size=out_shape).astype(np.float32)
    w = torch.from_numpy(weights).permute(0, rank + 1, *range(1, rank + 1)).contiguous()
    (y * w).sum().backward()

    back = lambda tensor: tensor.detach().permute(0, *range(2, rank + 2), 1).contiguous().numpy()

    return {
        "name": name,
        "mode": mode,
        "rank": rank,
        "n": n,
        "extents": list(extents),
        "channels": channels,
        "pads": list(pads),
        "input": flat(values),
        "output": flat(back(y)),
        "outShape": out_shape,
        "gradOutput": flat(weights),
        "gradInput": flat(back(x.grad)),
    }


cases = [
    case("series-mirror", "mirror", 2, (5,), 2, (2, 1)),
    case("series-repeat", "repeat", 2, (5,), 2, (3, 2)),
    case("series-wrap", "wrap", 2, (5,), 2, (2, 3)),
    case("image-mirror", "mirror", 2, (4, 5), 2, (1, 2, 2, 1)),
    case("image-repeat", "repeat", 2, (4, 5), 2, (2, 1, 0, 3)),
    case("image-wrap", "wrap", 2, (4, 5), 2, (1, 2, 3, 1)),
    case("volume-mirror", "mirror", 1, (3, 4, 3), 2, (1, 1, 2, 0, 1, 2)),
    case("volume-repeat", "repeat", 1, (3, 4, 3), 2, (1, 2, 0, 1, 2, 2)),
    case("volume-wrap", "wrap", 1, (3, 4, 3), 2, (2, 1, 1, 1, 3, 0)),
]

print(json.dumps({"made": f"torch {torch.__version__}", "cases": cases}, indent=1))
