# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Prints what PyTorch's own arithmetic makes of attention - scaled dot-product attention over queries, keys and values handed in apart;
# nn.MultiheadAttention over a series, with and without a mask of the padding and with a causal mask; and nn.TransformerEncoderLayer,
# with the norm after the sums and before them - on series worked out from a formula. attention-pytorch.json is what AttentionContract
# holds every engine to: the values, the gradient of the mean of those values times a coefficient with respect to the input (to the
# queries, the keys and the values for the attention that takes them apart) and to every number the layer learns, by the name DeepSharp
# gives each slot. Its output, when it was run, is attention-pytorch.txt beside it, with what PyTorch makes of a series whose every key
# is padding.
#
# Run once with PyTorch 2.14.1 for the CPU (no NumPy needed):
#
#     python attention-pytorch.py > attention-pytorch.txt
#
# PyTorch keeps a layer's weights outputs by inputs; DeepSharp keeps them inputs by outputs, as it keeps a linear layer's, so each matrix
# here is PyTorch's turned round (the in_proj_weight's three blocks - query, key, value - one after another along the outputs, as
# PyTorch's are). PyTorch's key_padding_mask says which keys to leave out; the mask here says which keys take part, so it is the other
# way round. The sine positions are the formula of 'Attention Is All You Need', worked out here in Python's own arithmetic, since PyTorch
# has no module for them.

import json
import math
import os
import platform

import torch

torch.set_default_dtype(torch.float32)
torch.set_num_threads(1)

here = os.path.dirname(os.path.abspath(__file__))
nn = torch.nn
F = torch.nn.functional

N, T, E, HEADS, FEED = 3, 5, 8, 2, 16


def series(seed, count, steps, width):
    """((3 x series + 5 x step + 7 x channel + seed) mod 11 - 5) / 6, plus a little, so nothing is exactly nothing."""
    return torch.tensor(
        [[[((3 * n + 5 * t + 7 * c + seed) % 11 - 5) / 6 + 0.0137 * (1 + (n + c) % 3) for c in range(width)] for t in range(steps)] for n in range(count)],
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
    """A matrix turned round to inputs by outputs; a row as it is."""
    return tensor.t().contiguous() if tensor.dim() == 2 else tensor


def tensor_of(tensor):
    return {"shape": list(tensor.shape), "values": flat(tensor)}


def mask_of(lengths):
    """One where a step takes part, nought where it is padding: series by step."""
    return [[1.0 if t < length else 0.0 for t in range(T)] for length in lengths] if lengths is not None else None


def keys_left_out(lengths):
    """PyTorch's key_padding_mask: True where a key is left out."""
    return torch.tensor([[t >= length for t in range(T)] for length in lengths]) if lengths is not None else None


def causal_left_out():
    """PyTorch's attn_mask: True where a key may not be read - every key after the query."""
    return torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)


cases = []


def write(case):
    cases.append(case)
    print(case["name"], "first numbers", case["output"]["values"][:4])


def grads_of(output, leaves):
    loss = (output * coefficients(output.shape)).mean()
    return torch.autograd.grad(loss, leaves, retain_graph=True)


def split(x):
    return x.reshape(x.shape[0], x.shape[1], HEADS, E // HEADS).transpose(1, 2)


def merge(x):
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], E)


def attention_case(name, lengths, causal):
    q, k, v = series(1, N, T, E), series(4, N, T, E), series(9, N, T, E)
    for each in (q, k, v):
        each.requires_grad_(True)

    allowed = torch.ones(N, 1, T, T, dtype=torch.bool)
    if lengths is not None:
        allowed = allowed & ~keys_left_out(lengths).reshape(N, 1, 1, T)
    if causal:
        allowed = allowed & ~causal_left_out().reshape(1, 1, T, T)

    output = merge(F.scaled_dot_product_attention(split(q), split(k), split(v), attn_mask=allowed))
    gq, gk, gv = grads_of(output, [q, k, v])

    write(
        {
            "name": name,
            "kind": "attention",
            "embed": E,
            "heads": HEADS,
            "causal": causal,
            "q": tensor_of(q),
            "k": tensor_of(k),
            "v": tensor_of(v),
            "mask": mask_of(lengths),
            "output": tensor_of(output),
            "gradInput": {"q": flat(gq), "k": flat(gk), "v": flat(gv)},
            "slots": {},
        }
    )


def slots_of(model, grads, offset):
    return {
        name: {"shape": list(ours(name, parameter).shape), "values": flat(ours(name, parameter)), "grad": flat(ours(name, grads[offset + at]))}
        for at, (name, parameter) in enumerate(model.named_parameters())
    }


def mha_case(name, lengths, causal):
    torch.manual_seed(20261021 + len(cases))
    model = nn.MultiheadAttention(E, HEADS, dropout=0.0, batch_first=True)
    x = series(2, N, T, E)
    x.requires_grad_(True)
    output, _ = model(
        x, x, x,
        key_padding_mask=keys_left_out(lengths),
        attn_mask=causal_left_out() if causal else None,
        need_weights=False,
    )
    grads = grads_of(output, [x] + [parameter for _, parameter in model.named_parameters()])

    write(
        {
            "name": name,
            "kind": "multiHeadAttention",
            "embed": E,
            "heads": HEADS,
            "causal": causal,
            "x": tensor_of(x),
            "mask": mask_of(lengths),
            "output": tensor_of(output),
            "gradInput": {"x": flat(grads[0])},
            "slots": slots_of(model, grads, 1),
        }
    )


def encoder_case(name, lengths, norm_first, activation):
    torch.manual_seed(20261021 + len(cases))
    model = nn.TransformerEncoderLayer(E, HEADS, dim_feedforward=FEED, dropout=0.0, activation=activation, batch_first=True, norm_first=norm_first)
    model.train()
    x = series(3, N, T, E)
    x.requires_grad_(True)
    output = model(x, src_key_padding_mask=keys_left_out(lengths))
    grads = grads_of(output, [x] + [parameter for _, parameter in model.named_parameters()])

    write(
        {
            "name": name,
            "kind": "encoder",
            "embed": E,
            "heads": HEADS,
            "feedForward": FEED,
            "normFirst": norm_first,
            "activation": activation,
            "x": tensor_of(x),
            "mask": mask_of(lengths),
            "output": tensor_of(output),
            "gradInput": {"x": flat(grads[0])},
            "slots": slots_of(model, grads, 1),
        }
    )


attention_case("attention", None, False)
attention_case("attention with a padding mask", [5, 3, 2], False)
attention_case("attention with a causal mask", None, True)
attention_case("attention with both masks", [5, 4, 2], True)
attention_case("attention with a series of no step", [5, 0, 2], False)
mha_case("multi-head attention", None, False)
mha_case("multi-head attention with a padding mask", [5, 3, 2], False)
mha_case("multi-head attention with a causal mask", None, True)
mha_case("multi-head attention with both masks", [5, 4, 2], True)
mha_case("multi-head attention with a series of no step", [5, 0, 2], False)
encoder_case("encoder layer, norm after", None, False, "relu")
encoder_case("encoder layer, norm before", None, True, "relu")
encoder_case("encoder layer with a padding mask, norm after, gelu", [5, 3, 2], False, "gelu")
encoder_case("encoder layer with a padding mask, norm before", [5, 4, 2], True, "relu")

# The positions of 'Attention Is All You Need': sine at the even places and cosine at the odd, of the step over ten thousand to the power
# of the place's pair over the width.
positions = [[(math.sin if c % 2 == 0 else math.cos)(t / 10000 ** ((c - c % 2) / E)) for c in range(E)] for t in range(T)]

# What PyTorch makes of a series every key of which is padding: measured here, since the answer decides what DeepSharp does with it.
torch.manual_seed(1)
probe = nn.MultiheadAttention(E, HEADS, dropout=0.0, batch_first=True)
x = series(2, N, T, E)
left_out = torch.tensor([[False] * T, [True] * T, [False] * T])
output, _ = probe(x, x, x, key_padding_mask=left_out, need_weights=False)
print("every key of the second series left out: row of the second series", [float(value) for value in output[1, 0, :3].tolist()], "finite:", bool(torch.isfinite(output[1]).all()))
print("the other series are finite:", bool(torch.isfinite(output[0]).all()), bool(torch.isfinite(output[2]).all()))

with open(os.path.join(here, "attention-pytorch.json"), "w", encoding="utf-8", newline="\n") as written:
    json.dump({"torch": torch.__version__, "python": platform.python_version(), "positions": {"shape": [T, E], "values": [value for row in positions for value in row]}, "cases": cases}, written, indent=1)
    written.write("\n")
