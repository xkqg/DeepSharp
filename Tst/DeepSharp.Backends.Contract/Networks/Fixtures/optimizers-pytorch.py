# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Prints what PyTorch's own optimizers, gradient clipping and warm-up do on the walked rows the contract classes use
# (WalkedRows.cs): Titanic's first four training rows through a hidden layer of four, a rectifier and one output, judged by
# a binary cross-entropy on the logits. Every number the optimizer, loop and schedule tests hold AdamW, RMSprop, NAdam,
# the clipped step and the linear warm-up to comes from this script's output, optimizers-pytorch.txt beside it.
#
# Run once with PyTorch 2.14.1 for the CPU (no NumPy needed):
#
#     python optimizers-pytorch.py > optimizers-pytorch.txt
#
# Each walk prints the loss before every step and, after it, the parameters the tests look at, as the doubles the float32
# values are — DeepSharp lays a linear layer's weights out inputs by outputs, PyTorch outputs by inputs, so the hidden
# weights printed are turned round into DeepSharp's order.

import platform

import torch

torch.set_default_dtype(torch.float32)

passengers = torch.tensor(
    [
        [-0.75, -1.0, -0.4576526880264282, -0.9716978669166565, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        [-0.75, -1.0, -0.055541593581438065, -0.721728503704071, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [-1.0, -1.0, -0.3571248948574066, -0.969062864780426, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        [-0.75, -1.0, -0.13093742728233337, -0.7927113771438599, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)
survived = torch.tensor([[0.0], [1.0], [1.0], [1.0]], dtype=torch.float32)


def network():
    """The walked network: ((7 x row + 3 x column) mod 11 - 5) / 20 for the hidden weights, in DeepSharp's order."""
    hidden = torch.nn.Linear(14, 4)
    output = torch.nn.Linear(4, 1)
    with torch.no_grad():
        weights = torch.tensor(
            [[((7 * row + 3 * column) % 11 - 5) / 20 for column in range(4)] for row in range(14)], dtype=torch.float32
        )
        hidden.weight.copy_(weights.t())
        hidden.bias.copy_(torch.tensor([0.1, -0.1, 0.05, 0.0]))
        output.weight.copy_(torch.tensor([[-0.3, 0.2, 0.0, -0.2]]))
        output.bias.copy_(torch.tensor([0.2]))
    return hidden, output, torch.nn.Sequential(hidden, torch.nn.ReLU(), output)


def listed(tensor):
    return "[" + ", ".join(repr(value) for value in tensor.detach().reshape(-1).tolist()) + "]"


def show(hidden, output):
    weights = hidden.weight.detach().t().contiguous().reshape(-1)
    print("  output.weight ", listed(output.weight))
    print("  output.bias   ", listed(output.bias))
    print("  hidden.bias   ", listed(hidden.bias))
    print("  hidden.weight[0:4]  ", listed(weights[0:4]))
    print("  hidden.weight[4:8]  ", listed(weights[4:8]))
    print("  hidden.weight[28:32]", listed(weights[28:32]))


def walk(title, make, steps=3):
    hidden, output, model = network()
    optimizer = make(model.parameters())
    loss_of = torch.nn.BCEWithLogitsLoss()
    print(title)
    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_of(model(passengers), survived)
        loss.backward()
        optimizer.step()
        print(f" step {step + 1} loss {loss.item()!r}")
        show(hidden, output)
    for parameter in model.parameters():
        state = optimizer.state[parameter]
        if "mu_product" in state:
            print("  mu_product", repr(float(state["mu_product"])), "after", int(state["step"]), "steps")
            break
    print()


print(f"torch {torch.__version__}, Python {platform.python_version()}")
print()

walk("AdamW lr=0.01 weight_decay=0.1", lambda parameters: torch.optim.AdamW(parameters, lr=0.01, weight_decay=0.1))
walk("RMSprop lr=0.001", lambda parameters: torch.optim.RMSprop(parameters, lr=0.001))
walk("RMSprop lr=0.001 momentum=0.9", lambda parameters: torch.optim.RMSprop(parameters, lr=0.001, momentum=0.9))
walk("NAdam lr=0.01", lambda parameters: torch.optim.NAdam(parameters, lr=0.01))

# One step of SGD at a rate of a tenth, the gradients clipped by their whole norm first, as clip_grad_norm_ clips them.
for max_norm in (0.05, 10.0):
    hidden, output, model = network()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = torch.nn.BCEWithLogitsLoss()(model(passengers), survived)
    loss.backward()
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
    optimizer.step()
    print(f"SGD lr=0.1, clip_grad_norm_ max_norm={max_norm}")
    print(f"  total norm {total.item()!r}, coefficient {min(max_norm / (total.item() + 1e-6), 1.0)!r}")
    show(hidden, output)
    print()

defaults = {
    "AdamW": torch.optim.AdamW([torch.zeros(1)]).defaults,
    "RMSprop": torch.optim.RMSprop([torch.zeros(1)]).defaults,
    "NAdam": torch.optim.NAdam([torch.zeros(1)]).defaults,
}
for name, settings in defaults.items():
    print(name, "defaults", {key: settings[key] for key in ("lr", "betas", "alpha", "eps", "weight_decay", "momentum", "momentum_decay") if key in settings})
print()


def rates(scheduler, optimizer, epochs):
    seen = []
    for _ in range(epochs):
        seen.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return seen


warm = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
print("LinearLR start_factor=0.25 total_iters=4, lr 0.1:", rates(torch.optim.lr_scheduler.LinearLR(warm, start_factor=0.25, total_iters=4), warm, 8))

warm = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
print("LinearLR defaults (start 1/3, 5 epochs), lr 0.1:", rates(torch.optim.lr_scheduler.LinearLR(warm), warm, 7))

warm = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=0.1)
ramp = torch.optim.lr_scheduler.LinearLR(warm, start_factor=0.25, total_iters=4)
cosine = torch.optim.lr_scheduler.CosineAnnealingLR(warm, T_max=6, eta_min=0.001)
sequence = torch.optim.lr_scheduler.SequentialLR(warm, [ramp, cosine], milestones=[4])
print("SequentialLR(LinearLR 0.25 x4, CosineAnnealingLR T_max=6 eta_min=0.001), lr 0.1:", rates(sequence, warm, 12))
