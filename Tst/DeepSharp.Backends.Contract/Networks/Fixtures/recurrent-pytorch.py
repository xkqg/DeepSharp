# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Prints what PyTorch's own arithmetic makes of recurrent layers - an LSTM and a GRU, of one layer and of two, in one direction and in
# both, over batches of series that are all as long as one another and over batches whose series are of unequal lengths - on series
# worked out from a formula. recurrent-pytorch.json is what RecurrentContract holds every engine to: the values (every step's, and the
# last state of each series at its own length), the gradient of the mean of those values times a coefficient with respect to the input and
# to every number the layer learns, by the name DeepSharp gives each slot. Its output, when it was run, is recurrent-pytorch.txt beside it.
#
# Run once with PyTorch 2.14.1 for the CPU (no NumPy needed):
#
#     python recurrent-pytorch.py > recurrent-pytorch.txt
#
# PyTorch keeps a layer's weights outputs by inputs, the four gates (LSTM: input, forget, cell, output; GRU: reset, update, new) one
# block after another along the outputs. DeepSharp keeps them inputs by outputs, as it keeps a linear layer's, the gates in the same
# order, so each weight here is PyTorch's turned round. A batch of unequal lengths is handed to PyTorch packed, so a series ends at its own
# length; the places of a shorter series beyond its end hold a million in what DeepSharp is handed, which PyTorch never reads.

import json
import os
import platform

import torch

torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)

here = os.path.dirname(os.path.abspath(__file__))
nn = torch.nn

N, T, INPUTS, HIDDEN = 3, 5, 3, 4


def series(count, steps, width):
    """((3 x series + 5 x step + 7 x channel) mod 11 - 5) / 6, plus a little, so no gate is handed exactly nothing."""
    return torch.tensor(
        [[[((3 * n + 5 * t + 7 * c) % 11 - 5) / 6 + 0.0137 for c in range(width)] for t in range(steps)] for n in range(count)],
        dtype=torch.float32,
    )


def coefficients(shape):
    count = 1
    for length in shape:
        count *= length
    return torch.tensor([((3 * at) % 7 - 3) / 10 for at in range(count)], dtype=torch.float32).reshape(shape)


def flat(tensor):
    return [float(value) for value in tensor.detach().reshape(-1).tolist()]


def ours(name, tensor):
    """The weights turned round to inputs by outputs; the biases as they are."""
    return tensor.t().contiguous() if name.startswith("weight") else tensor


def run(model, x, lengths):
    """What the layer makes of the batch: every step, zeros beyond a series' end, and the last state of each series."""
    if lengths is None:
        out, state = model(x)
    else:
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, state = model(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])

    h_n = state[0] if isinstance(state, tuple) else state
    last = torch.cat([h_n[-2], h_n[-1]], dim=1) if model.bidirectional else h_n[-1]
    return out, last


cases = []


def case(name, kind, layers, both, lengths):
    torch.manual_seed(20261020 + len(cases))
    model = (nn.LSTM if kind == "lstm" else nn.GRU)(INPUTS, HIDDEN, num_layers=layers, bidirectional=both, batch_first=True)
    x = series(N, T, INPUTS)

    if lengths is not None:
        for n, length in enumerate(lengths):
            x[n, length:, :] = 1e6

    x.requires_grad_(True)
    parameters = list(model.named_parameters())
    sequences, last = run(model, x, lengths)

    grads = {}
    for mode, value in (("sequences", sequences), ("last", last)):
        loss = (value * coefficients(value.shape)).mean()
        grads[mode] = torch.autograd.grad(loss, [x] + [parameter for _, parameter in parameters], retain_graph=True)

    mask = None
    if lengths is not None:
        mask = [[1.0 if t < length else 0.0 for t in range(T)] for length in lengths]

    cases.append(
        {
            "name": name,
            "kind": kind,
            "inputs": INPUTS,
            "hidden": HIDDEN,
            "layers": layers,
            "bothWays": both,
            "x": {"shape": list(x.shape), "values": flat(x)},
            "lengths": lengths,
            "mask": mask,
            "sequences": {"shape": list(sequences.shape), "values": flat(sequences)},
            "last": {"shape": list(last.shape), "values": flat(last)},
            "gradInput": {mode: flat(grads[mode][0]) for mode in grads},
            "slots": {
                name: {
                    "shape": list(ours(name, parameter).shape),
                    "values": flat(ours(name, parameter)),
                    "grad": {mode: flat(ours(name, grads[mode][1 + at])) for mode in grads},
                }
                for at, (name, parameter) in enumerate(parameters)
            },
        }
    )
    print(name, "last of the first series", flat(last)[:4])


case("lstm", "lstm", 1, False, None)
case("lstm of two layers in both directions", "lstm", 2, True, None)
case("gru", "gru", 1, False, None)
case("gru of two layers in both directions", "gru", 2, True, None)
case("lstm over series of unequal lengths, both directions", "lstm", 1, True, [5, 3, 2])
case("gru over series of unequal lengths, two layers", "gru", 2, False, [4, 5, 1])

with open(os.path.join(here, "recurrent-pytorch.json"), "w", encoding="utf-8", newline="\n") as written:
    json.dump({"torch": torch.__version__, "python": platform.python_version(), "cases": cases}, written, indent=1)
    written.write("\n")
