# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Prints the learning rate PyTorch's ReduceLROnPlateau takes at every epoch of a run, for the settings and the sequences of
# training losses below, and what PyTorch's SequentialLR does when it is handed one. TrainingPlateauTests holds the schedule
# that follows the training loss to these rates, epoch by epoch. plateau-pytorch.json is what it reads; the output of the run
# is plateau-pytorch.txt beside it.
#
# Run once with PyTorch 2.14.1 for the CPU (no NumPy needed):
#
#     python plateau-pytorch.py > plateau-pytorch.txt
#
# The rate of an epoch is the one the optimizer holds before that epoch's loss has been handed to the scheduler: the run asks
# for it, trains the epoch, and only then says how the epoch went.

import json
import os
import platform

import torch

here = os.path.dirname(os.path.abspath(__file__))

# A loss that falls, flattens, falls a little less than a ten-thousandth, and then climbs.
falling_then_flat = [1.0, 0.8, 0.64, 0.5, 0.45, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44,
                     0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44, 0.44]
noisy = [1.0, 0.7, 0.75, 0.6, 0.65, 0.58, 0.59, 0.57, 0.571, 0.5699, 0.5701, 0.56, 0.58, 0.55, 0.553, 0.549, 0.55, 0.55, 0.551, 0.5489,
         0.55, 0.5, 0.52, 0.51, 0.49, 0.495, 0.49, 0.49, 0.49, 0.4899, 0.49, 0.489, 0.4889, 0.4888, 0.4887, 0.4886, 0.4885, 0.4884, 0.4883, 0.4882]
tiny_gains = [1.0] + [1.0 * (1 - 5e-5) ** n for n in range(1, 30)]
climbing = [0.5 + 0.01 * n for n in range(30)]

cases = [
    {"name": "defaults", "initial": 0.1, "patience": 10, "factor": 0.1, "threshold": 1e-4, "cooldown": 0, "min_lr": 0.0, "losses": falling_then_flat},
    {"name": "patience two, a half", "initial": 0.1, "patience": 2, "factor": 0.5, "threshold": 1e-4, "cooldown": 0, "min_lr": 0.0, "losses": noisy},
    {"name": "cooldown three", "initial": 0.1, "patience": 1, "factor": 0.5, "threshold": 1e-4, "cooldown": 3, "min_lr": 0.0, "losses": falling_then_flat},
    {"name": "a least rate", "initial": 0.1, "patience": 1, "factor": 0.1, "threshold": 1e-4, "cooldown": 0, "min_lr": 0.01, "losses": falling_then_flat},
    {"name": "gains below the threshold", "initial": 0.1, "patience": 3, "factor": 0.5, "threshold": 1e-4, "cooldown": 0, "min_lr": 0.0, "losses": tiny_gains},
    {"name": "a wider threshold", "initial": 0.1, "patience": 3, "factor": 0.5, "threshold": 0.01, "cooldown": 0, "min_lr": 0.0, "losses": noisy},
    {"name": "a rate no further from nothing than epsilon", "initial": 1e-8, "patience": 1, "factor": 0.1, "threshold": 1e-4, "cooldown": 0, "min_lr": 0.0, "losses": falling_then_flat},
    {"name": "a loss that only climbs", "initial": 0.1, "patience": 4, "factor": 0.3, "threshold": 1e-4, "cooldown": 1, "min_lr": 0.0, "losses": climbing},
]


def rates(case):
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([parameter], lr=case["initial"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=case["factor"],
        patience=case["patience"],
        threshold=case["threshold"],
        threshold_mode="rel",
        cooldown=case["cooldown"],
        min_lr=case["min_lr"],
    )
    taken = []
    for loss in case["losses"]:
        taken.append(optimizer.param_groups[0]["lr"])
        scheduler.step(loss)
    return taken


walks = []
for case in cases:
    taken = rates(case)
    walks.append({**{key: value for key, value in case.items()}, "rates": taken})
    print(case["name"], taken)

# What SequentialLR makes of a ReduceLROnPlateau in its list, which a warm-up that hands over to a plateau would be.
parameter = torch.nn.Parameter(torch.zeros(1))
optimizer = torch.optim.SGD([parameter], lr=0.1)
try:
    torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=3),
            torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer),
        ],
        milestones=[3],
    )
    sequential = "accepted"
except Exception as refused:  # noqa: BLE001 - the message is what is kept
    sequential = f"{type(refused).__name__}: {refused}"
print("SequentialLR with a ReduceLROnPlateau:", sequential)

with open(os.path.join(here, "plateau-pytorch.json"), "w", encoding="utf-8", newline="\n") as written:
    json.dump({"torch": torch.__version__, "python": platform.python_version(), "sequential": sequential, "walks": walks}, written, indent=1)
    written.write("\n")
