# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Prints what PyTorch's own arithmetic makes of a table model of the kind Gorishniy et al. describe (FT-Transformer): each column of a row
# turned into a token of its own - the column's value times a vector the layer learns, plus a vector it learns - the tokens read by two encoder
# layers, averaged over the columns, normalised, bent by a ReLU and put through a linear layer. table-pytorch.json is what ColumnTokensContract
# holds every engine to: the tokens alone and the whole model, the gradient of the mean of the answers times a coefficient with respect to the
# rows and to every number the model learns, by the name DeepSharp gives each slot. Its output, when it was run, is table-pytorch.txt beside it.
#
# Run once with PyTorch 2.14.1 for the CPU (no NumPy needed):
#
#     python table-pytorch.py > table-pytorch.txt
#
# PyTorch keeps a linear layer's and an attention's weights outputs by inputs; DeepSharp keeps them inputs by outputs. The tokens' weight and bias
# are kept as the formula writes them, a row of the embedding's width for each column.

import json
import os
import platform

import torch

torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)
torch.manual_seed(20261025)

here = os.path.dirname(os.path.abspath(__file__))
nn = torch.nn

N, COLUMNS, EMBED, HEADS, FEED = 5, 6, 8, 2, 16


def rows(count, width):
    return torch.tensor([[((3 * n + 5 * c) % 13 - 6) / 7 + 0.021 * (n + 1) for c in range(width)] for n in range(count)], dtype=torch.float32)


def coefficients(shape):
    count = 1
    for length in shape:
        count *= length
    return torch.tensor([((3 * at) % 7 - 3) / 10 for at in range(count)], dtype=torch.float32).reshape(shape)


def flat(tensor):
    return [float(value) for value in tensor.detach().reshape(-1).tolist()]


def ours(tensor):
    return tensor.t().contiguous() if tensor.dim() == 2 else tensor


class Tokens(nn.Module):
    def __init__(self):
        super().__init__()
        bound = EMBED ** -0.5
        self.weight = nn.Parameter(torch.empty(COLUMNS, EMBED).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.empty(COLUMNS, EMBED).uniform_(-bound, bound))

    def forward(self, x):
        return x[..., None] * self.weight + self.bias


class Table(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokens = Tokens()
        self.enc0 = nn.TransformerEncoderLayer(EMBED, HEADS, FEED, dropout=0.0, batch_first=True)
        self.enc1 = nn.TransformerEncoderLayer(EMBED, HEADS, FEED, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(EMBED)
        self.head = nn.Linear(EMBED, 1)

    def forward(self, x):
        looked = self.enc1(self.enc0(self.tokens(x)))
        return self.head(torch.relu(self.norm(looked.mean(dim=1))))


def kept(model, slot, tensor):
    """The tokens' numbers as they are written; every other matrix turned round to inputs by outputs."""
    return tensor if isinstance(model, Tokens) or slot.startswith("tokens.") else ours(tensor)


def case(name, model):
    x = rows(N, COLUMNS)
    x.requires_grad_(True)
    output = model(x)
    parameters = list(model.named_parameters())
    loss = (output * coefficients(output.shape)).mean()
    grads = torch.autograd.grad(loss, [x] + [parameter for _, parameter in parameters])
    print(name, "first numbers", flat(output)[:4])

    return {
        "name": name,
        "x": {"shape": list(x.shape), "values": flat(x)},
        "output": {"shape": list(output.shape), "values": flat(output)},
        "gradInput": flat(grads[0]),
        "slots": {
            slot: {"shape": list(kept(model, slot, parameter).shape), "values": flat(kept(model, slot, parameter)), "grad": flat(kept(model, slot, grads[1 + at]))}
            for at, (slot, parameter) in enumerate(parameters)
        },
    }


cases = [case("tokens", Tokens()), case("table", Table())]

with open(os.path.join(here, "table-pytorch.json"), "w", encoding="utf-8", newline="\n") as written:
    json.dump({"torch": torch.__version__, "python": platform.python_version(), "columns": COLUMNS, "embed": EMBED, "heads": HEADS, "feedForward": FEED, "cases": cases}, written, indent=1)
    written.write("\n")
