# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Prints what PyTorch's own optimizers do with the options beyond their defaults - the weight decay added to the gradient of SGD,
# Adam, RMSprop and NAdam, SGD's dampening and Nesterov's momentum, AMSGrad for Adam and AdamW, and RMSprop's centred form -
# on the walked rows the contract classes use (WalkedRows.cs): Titanic's first four training rows through a hidden layer of
# four, a rectifier and one output, judged by a binary cross-entropy on the logits. optimizers-options-pytorch.json is what
# OptimizerContract holds each option to: the loss before every step, the parameters after the last, and the names PyTorch
# keeps its state under. Its output, when it was run, is optimizers-options-pytorch.txt beside it.
#
# Run once with PyTorch 2.14.1 for the CPU (no NumPy needed):
#
#     python optimizers-options-pytorch.py > optimizers-options-pytorch.txt
#
# DeepSharp lays a linear layer's weights out inputs by outputs, PyTorch outputs by inputs, so the hidden weights are printed
# turned round into DeepSharp's order. Every walk that is about an option is also walked without it, and the file says how far
# apart the two ended, so an option that changes nothing on these rows cannot stand as a test of itself.

import json
import os
import platform

import torch

torch.set_default_dtype(torch.float32)

here = os.path.dirname(os.path.abspath(__file__))

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


def values(tensor):
    return [float(value) for value in tensor.detach().reshape(-1).tolist()]


def walked(make, steps):
    hidden, output, model = network()
    optimizer = make(model.parameters())
    loss_of = torch.nn.BCEWithLogitsLoss()
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_of(model(passengers), survived)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    parameters = {
        "hidden.weight": values(hidden.weight.detach().t().contiguous()),
        "hidden.bias": values(hidden.bias),
        "output.weight": values(output.weight),
        "output.bias": values(output.bias),
    }
    state_keys = sorted({key for parameter in model.parameters() for key in optimizer.state[parameter].keys()})
    return losses, parameters, state_keys


def apart(one, other):
    return max(abs(a - b) for key in one for a, b in zip(one[key], other[key]))


# name, DeepSharp's name for the optimizer, the settings DeepSharp is handed, PyTorch's optimizer, steps and the walk it differs from.
walks = [
    ("sgd-decay", "sgd", 0.1, {"momentum": 0.9, "weightDecay": 0.1}, lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, weight_decay=0.1), 3,
     lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)),
    ("sgd-decay-without-momentum", "sgd", 0.1, {"weightDecay": 0.2}, lambda p: torch.optim.SGD(p, lr=0.1, weight_decay=0.2), 3,
     lambda p: torch.optim.SGD(p, lr=0.1)),
    ("sgd-dampening", "sgd", 0.1, {"momentum": 0.9, "dampening": 0.5}, lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, dampening=0.5), 3,
     lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)),
    ("sgd-nesterov", "sgd", 0.1, {"momentum": 0.9, "nesterov": True}, lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9, nesterov=True), 3,
     lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.9)),
    ("sgd-nesterov-decay", "sgd", 0.1, {"momentum": 0.8, "nesterov": True, "weightDecay": 0.05},
     lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.8, nesterov=True, weight_decay=0.05), 3,
     lambda p: torch.optim.SGD(p, lr=0.1, momentum=0.8, nesterov=True)),
    ("adam-decay", "adam", 0.01, {"weightDecay": 0.1}, lambda p: torch.optim.Adam(p, lr=0.01, weight_decay=0.1), 3,
     lambda p: torch.optim.Adam(p, lr=0.01)),
    ("adam-amsgrad", "adam", 0.1, {"betas": [0.9, 0.5], "amsgrad": True}, lambda p: torch.optim.Adam(p, lr=0.1, betas=(0.9, 0.5), amsgrad=True), 6,
     lambda p: torch.optim.Adam(p, lr=0.1, betas=(0.9, 0.5))),
    ("adamw-amsgrad", "adamw", 0.1, {"betas": [0.9, 0.5], "weightDecay": 0.1, "amsgrad": True},
     lambda p: torch.optim.AdamW(p, lr=0.1, betas=(0.9, 0.5), weight_decay=0.1, amsgrad=True), 6,
     lambda p: torch.optim.AdamW(p, lr=0.1, betas=(0.9, 0.5), weight_decay=0.1)),
    ("rmsprop-decay", "rmsprop", 0.001, {"weightDecay": 0.1}, lambda p: torch.optim.RMSprop(p, lr=0.001, weight_decay=0.1), 3,
     lambda p: torch.optim.RMSprop(p, lr=0.001)),
    ("rmsprop-centred", "rmsprop", 0.001, {"centered": True}, lambda p: torch.optim.RMSprop(p, lr=0.001, centered=True), 3,
     lambda p: torch.optim.RMSprop(p, lr=0.001)),
    ("rmsprop-centred-momentum-decay", "rmsprop", 0.001, {"centered": True, "momentum": 0.9, "weightDecay": 0.1},
     lambda p: torch.optim.RMSprop(p, lr=0.001, centered=True, momentum=0.9, weight_decay=0.1), 3,
     lambda p: torch.optim.RMSprop(p, lr=0.001, momentum=0.9)),
    ("nadam-decay", "nadam", 0.01, {"weightDecay": 0.1}, lambda p: torch.optim.NAdam(p, lr=0.01, weight_decay=0.1), 3,
     lambda p: torch.optim.NAdam(p, lr=0.01)),
]

made = f"torch {torch.__version__}, Python {platform.python_version()}"
print(made)
print()

fixture = {"made": made, "walks": []}

for name, kind, rate, settings, make, steps, plain in walks:
    losses, parameters, keys = walked(make, steps)
    _, without, _ = walked(plain, steps)
    gap = apart(parameters, without)
    print(f"{name}: {kind} lr={rate} {settings}, {steps} steps")
    print(f"  losses {losses!r}")
    print(f"  state {keys}")
    print(f"  farthest from the same walk without the option: {gap!r}")
    print()
    assert gap > 1e-4, f"{name}: the option changes nothing on these rows"
    fixture["walks"].append(
        {"name": name, "optimizer": kind, "rate": rate, "settings": settings, "steps": steps, "losses": losses, "parameters": parameters, "state": keys}
    )


# ---- the runs: what the loop does around an optimizer, on the same rows, full batches of four --------------------------------------
#
# Clipping each gradient value, then the whole norm; and an exponential moving average of the weights kept beside them. A step
# taken once every so many batches is the same run as the full batch when the batches are equal, so it is held to the walk without it.


def run(steps, rate, clip_value=None, clip_norm=None, decay=None):
    hidden, output, model = network()
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)
    loss_of = torch.nn.BCEWithLogitsLoss()
    averaged = None
    if decay is not None:
        averaged = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(decay))
    for _ in range(steps):
        optimizer.zero_grad()
        loss_of(model(passengers), survived).backward()
        if clip_value is not None:
            torch.nn.utils.clip_grad_value_(model.parameters(), clip_value)
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        if averaged is not None:
            averaged.update_parameters(model)

    def laid(source_hidden, source_output):
        return {
            "hidden.weight": values(source_hidden.weight.detach().t().contiguous()),
            "hidden.bias": values(source_hidden.bias),
            "output.weight": values(source_output.weight),
            "output.bias": values(source_output.bias),
        }

    result = laid(hidden, output)
    if averaged is not None:
        ema = averaged.module
        result_averaged = laid(ema[0], ema[2])
    else:
        result_averaged = None
    return result, result_averaged


runs = [
    ("plain", {"steps": 3, "rate": 0.1}),
    ("clip-value", {"steps": 3, "rate": 0.1, "clipValue": 0.02}),
    ("clip-value-then-norm", {"steps": 3, "rate": 0.1, "clipValue": 0.02, "clipNorm": 0.05}),
    ("average", {"steps": 3, "rate": 0.1, "average": 0.9}),
    ("average-and-clips", {"steps": 4, "rate": 0.1, "clipValue": 0.02, "clipNorm": 0.05, "average": 0.5}),
]

fixture["runs"] = []
reference, _ = run(3, 0.1)
for name, setting in runs:
    parameters, averaged = run(setting["steps"], setting["rate"], setting.get("clipValue"), setting.get("clipNorm"), setting.get("average"))
    gap = apart(parameters, reference) if name != "average-and-clips" else 1.0
    print(f"run {name}: {setting}; farthest from the plain run: {gap!r}")
    if name not in ("plain", "average"):
        assert gap > 1e-4, f"{name}: the option changes nothing on these rows"
    if averaged is not None:
        print(f"  the average is {apart(parameters, averaged)!r} from the weights")
        assert apart(parameters, averaged) > 1e-4
    fixture["runs"].append({"name": name, **setting, "parameters": parameters, "averaged": averaged})
print()


# ---- the rules: what a slot's numbers are put under beside being stepped ----------------------------------------------------------
#
# A penalty the loss adds (Keras's L1 and L2 regularizers: l1 x the sum of the absolute values, l2 x the sum of the squares, no
# half), a bound each output unit's numbers are held under after every step (Keras's MaxNorm, w x min(n, m) / (1e-7 + n) with n the
# norm over the inputs of a unit, the whole vector's for a vector), and a layer held still (requires_grad false). Full batches of
# four, SGD at a tenth; the loss each step is the loss the run reports, penalty included.


def ruled(steps, penalties=None, bounds=None, held=()):
    hidden, output, model = network()
    named = {"hidden.weight": hidden.weight, "hidden.bias": hidden.bias, "output.weight": output.weight, "output.bias": output.bias}
    for name in held:
        named[name].requires_grad_(False)
    optimizer = torch.optim.SGD([parameter for parameter in model.parameters() if parameter.requires_grad], lr=0.1)
    loss_of = torch.nn.BCEWithLogitsLoss()
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_of(model(passengers), survived)
        for name, (l1, l2) in (penalties or {}).items():
            loss = loss + l1 * named[name].abs().sum() + l2 * named[name].pow(2).sum()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        for name, bound in (bounds or {}).items():
            with torch.no_grad():
                parameter = named[name]
                # DeepSharp lays a linear layer's weights out inputs by outputs: a unit's numbers are a row of PyTorch's.
                norms = parameter.norm(dim=1, keepdim=True) if parameter.dim() == 2 else parameter.norm()
                parameter.mul_(torch.clamp(norms, 0, bound) / (1e-7 + norms))
    return losses, {
        "hidden.weight": values(hidden.weight.detach().t().contiguous()),
        "hidden.bias": values(hidden.bias),
        "output.weight": values(output.weight),
        "output.bias": values(output.bias),
    }


rule_runs = [
    ("penalty", {"steps": 3, "penalties": {"hidden.weight": [0.01, 0.02], "output.weight": [0.03, 0.0]}}),
    ("penalty-on-a-bias", {"steps": 3, "penalties": {"hidden.bias": [0.05, 0.0], "output.bias": [0.0, 0.1]}}),
    ("bound", {"steps": 3, "bounds": {"hidden.weight": 0.4, "output.weight": 0.2}}),
    ("bound-on-a-bias", {"steps": 3, "bounds": {"hidden.bias": 0.1, "output.bias": 0.15}}),
    ("held", {"steps": 3, "held": ["hidden.weight", "hidden.bias"]}),
    ("held-with-a-penalty-and-a-bound", {"steps": 3, "held": ["hidden.weight", "hidden.bias"], "penalties": {"output.weight": [0.02, 0.01]}, "bounds": {"output.weight": 0.25}}),
]

fixture["rules"] = []
for name, setting in rule_runs:
    losses, parameters = ruled(setting["steps"], setting.get("penalties"), setting.get("bounds"), setting.get("held", ()))
    plain_losses, plain = ruled(setting["steps"])
    gap = apart(parameters, plain)
    print(f"rules {name}: {setting}; farthest from the run without them: {gap!r}; losses {losses!r}")
    assert gap > 1e-4, f"{name}: the rules change nothing on these rows"
    fixture["rules"].append({"name": name, **setting, "losses": losses, "parameters": parameters})
print()

# ---- the rate shares: a rate of its own for a parameter ------------------------------------------------------------------------
#
# PyTorch's parameter groups, one group to a parameter: each group's learning rate is the optimizer's times the share the
# parameter has, the others' the whole. DeepSharp says it as a share of the rate in the rules of a parameter. Full batches of
# four through the walked network, three steps; every walk is also walked with every share at one, and the file says how far apart
# the two ended, so a share that changes nothing on these rows cannot stand as a test of itself.


def shared(make, steps, shares):
    hidden, output, model = network()
    named = {"hidden.weight": hidden.weight, "hidden.bias": hidden.bias, "output.weight": output.weight, "output.bias": output.bias}
    groups = [{"params": [parameter], "share": shares.get(name, 1.0)} for name, parameter in named.items()]
    optimizer = make(groups)
    loss_of = torch.nn.BCEWithLogitsLoss()
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_of(model(passengers), survived)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return losses, {
        "hidden.weight": values(hidden.weight.detach().t().contiguous()),
        "hidden.bias": values(hidden.bias),
        "output.weight": values(output.weight),
        "output.bias": values(output.bias),
    }


def grouped(optimizer, rate, **options):
    def make(groups):
        return optimizer([{"params": group["params"], "lr": rate * group["share"]} for group in groups], lr=rate, **options)

    return make


share_runs = [
    ("sgd-slow-hidden", "sgd", 0.1, {"momentum": 0.9}, grouped(torch.optim.SGD, 0.1, momentum=0.9), {"hidden.weight": 0.1, "hidden.bias": 0.1}),
    ("sgd-decay-fast-biases", "sgd", 0.1, {"weightDecay": 0.1}, grouped(torch.optim.SGD, 0.1, weight_decay=0.1), {"hidden.bias": 3.0, "output.bias": 3.0}),
    ("adam-slow-hidden", "adam", 0.05, {}, grouped(torch.optim.Adam, 0.05), {"hidden.weight": 0.1, "hidden.bias": 0.1}),
    ("adamw-slow-output", "adamw", 0.05, {"weightDecay": 0.1}, grouped(torch.optim.AdamW, 0.05, weight_decay=0.1), {"output.weight": 0.2, "output.bias": 0.2}),
    ("rmsprop-slow-hidden", "rmsprop", 0.01, {}, grouped(torch.optim.RMSprop, 0.01), {"hidden.weight": 0.1, "hidden.bias": 0.1}),
    ("nadam-fast-output", "nadam", 0.01, {}, grouped(torch.optim.NAdam, 0.01), {"output.weight": 4.0, "output.bias": 4.0}),
]

fixture["shares"] = []
for name, kind, rate, settings, make, shares in share_runs:
    losses, parameters = shared(make, 3, shares)
    _, whole = shared(make, 3, {})
    gap = apart(parameters, whole)
    print(f"shares {name}: {kind} lr={rate} {settings}, {shares}; farthest from the same walk at the whole rate: {gap!r}; losses {losses!r}")
    assert gap > 1e-4, f"{name}: the shares change nothing on these rows"
    fixture["shares"].append({"name": name, "optimizer": kind, "rate": rate, "settings": settings, "shares": shares, "steps": 3, "losses": losses, "parameters": parameters})
print()

with open(os.path.join(here, "optimizers-options-pytorch.json"), "w", encoding="utf-8", newline="\n") as file:
    json.dump(fixture, file, indent=1)
    file.write("\n")
