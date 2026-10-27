# Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
#
# Not run by any build, any test or any package: read by nobody but a person, who runs it by hand with the three
# packages it imports installed (`pip install torch lion-pytorch`; sail-sg's Adan has no package on PyPI as of this
# writing, so clone github.com/sail-sg/Adan and put adan.py beside this script, or on the path, before running it).
#
# What it does: three steps of Lion, RAdam and Adan, each on the same single parameter and the same three gradients, so
# the numbers each prints can be pasted into OptimizerTests.cs as the golden values this library's own Lion, RAdam and
# Adan are asserted against — the same role plateau-pytorch.py already plays for TrainingPlateau.

import torch

try:
    from lion_pytorch import Lion
except ImportError:
    Lion = None

try:
    from adan import Adan
except ImportError:
    Adan = None

GRADIENTS = [0.5, -0.3, 0.2]


def three_steps(make_optimizer, name):
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))
    optimizer = make_optimizer([parameter])

    print(f"--- {name} ---")
    for step, value in enumerate(GRADIENTS):
        parameter.grad = torch.full_like(parameter, value)
        optimizer.step()
        optimizer.zero_grad()
        print(f"step {step + 1}: parameter = {parameter.detach().tolist()}")

    state = optimizer.state[parameter]
    print(f"final state keys: {sorted(state.keys())}")
    for key, held in state.items():
        if torch.is_tensor(held):
            print(f"  {key}: {held.tolist()}")


if __name__ == "__main__":
    if Lion is not None:
        three_steps(lambda parameters: Lion(parameters, lr=0.1, betas=(0.9, 0.99)), "Lion")
    else:
        print("lion-pytorch not installed: pip install lion-pytorch")

    three_steps(lambda parameters: torch.optim.RAdam(parameters, lr=0.1), "RAdam")

    if Adan is not None:
        three_steps(lambda parameters: Adan(parameters, lr=0.1, betas=(0.98, 0.92, 0.99)), "Adan")
    else:
        print("sail-sg/Adan's adan.py not found beside this script: clone github.com/sail-sg/Adan")
