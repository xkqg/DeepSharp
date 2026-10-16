# Prints PyTorch's values for the words a model is written in that 0.11.0 adds - the activations, the normalisations over groups
# of channels, over one example's channels and by the root mean square, the periodic embedding of a number, and the losses - with
# the gradient of a loss through each: vocabulary-pytorch.json, read by VocabularyContract.
#
#   python vocabulary-pytorch.py > vocabulary-pytorch.json      (PyTorch 2.14.1, CPU, single precision)
#
# Examples are laid out with their channels last, as the seam lays them out, so every array printed here is flat in that order;
# PyTorch's own layers take the channels first and are handed them so. A layer's loss is the sum of what it makes times numbers
# drawn for it ("gradOutput"), so the gradient of the example is those numbers sent back through the layer. A loss is PyTorch's
# own where PyTorch has one and its formula, written out, where it has none (the quantile, the focal loss as torchvision writes it,
# the hinge as Keras writes it, and the binary cross-entropy whose labels are softened as Keras softens them); the gradient is the
# loss's own, of the outputs.
#
# Every input holds a nought on purpose: the slope an activation takes at nought is PyTorch's, and a formula that rounds it off
# the other side would pass every other value.

import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

random = np.random.RandomState(20261016)
torch.manual_seed(20261016)


def flat(array):
    return [float(value) for value in np.asarray(array, dtype=np.float32).reshape(-1)]


def drawn(shape, low=-3.0, high=3.0, noughts=2):
    values = random.uniform(low, high, size=shape).astype(np.float32)
    places = values.reshape(-1)
    for at in random.choice(places.size, size=noughts, replace=False):
        places[at] = 0.0
    return values


def first(tensor):
    # channels last -> PyTorch's channels first
    rank = tensor.dim()
    return tensor.permute(0, rank - 1, *range(1, rank - 1)).contiguous() if rank > 2 else tensor


def last(tensor):
    # PyTorch's channels first -> channels last
    rank = tensor.dim()
    return tensor.permute(0, *range(2, rank), 1).contiguous() if rank > 2 else tensor


def through(name, kind, settings, values, module=None, function=None, channels_first=False, slots=()):
    x = torch.from_numpy(values.copy()).requires_grad_(True)
    handed = first(x) if channels_first else x
    y = module(handed) if module is not None else function(handed)
    y = last(y) if channels_first else y
    weights = random.uniform(-1.0, 1.0, size=tuple(y.shape)).astype(np.float32)
    (y * torch.from_numpy(weights)).sum().backward()

    case = {
        "name": name,
        "kind": kind,
        "settings": settings,
        "shape": list(values.shape),
        "input": flat(values),
        "output": flat(y.detach().numpy()),
        "outShape": list(y.shape),
        "gradOutput": flat(weights),
        "gradInput": flat(x.grad.numpy()),
        "slots": {},
    }
    for slot, parameter, laid in slots:
        case["slots"][slot] = {"value": flat(laid(parameter.detach())), "grad": flat(laid(parameter.grad))}
    return case


def as_is(tensor):
    return tensor.numpy()


activations = [
    through("gelu", "gelu", {"approximate": False}, drawn((3, 5)), function=lambda x: F.gelu(x)),
    through("gelu-tanh", "gelu", {"approximate": True}, drawn((3, 5)), function=lambda x: F.gelu(x, approximate="tanh")),
    through("silu", "silu", {}, drawn((3, 5)), function=F.silu),
    through("mish", "mish", {}, drawn((3, 5)), function=F.mish),
    through("leakyRelu", "leakyRelu", {"negativeSlope": 0.01}, drawn((3, 5)), function=lambda x: F.leaky_relu(x, 0.01)),
    through("leakyRelu-0.2", "leakyRelu", {"negativeSlope": 0.2}, drawn((2, 3, 4)), function=lambda x: F.leaky_relu(x, 0.2)),
    through("elu", "elu", {"alpha": 1.0}, drawn((3, 5)), function=lambda x: F.elu(x, 1.0)),
    through("elu-0.5", "elu", {"alpha": 0.5}, drawn((3, 5)), function=lambda x: F.elu(x, 0.5)),
    through("selu", "selu", {}, drawn((3, 5)), function=F.selu),
    through("softmax", "softmax", {}, drawn((2, 3, 4)), function=lambda x: F.softmax(x, dim=-1)),
]

shared = nn.PReLU(1, init=0.25)
with torch.no_grad():
    shared.weight.copy_(torch.tensor([0.3], dtype=torch.float32))
activations.append(through("prelu", "prelu", {"parameters": 1}, drawn((3, 5)), module=shared, slots=[("weight", shared.weight, as_is)]))

each = nn.PReLU(4)
with torch.no_grad():
    each.weight.copy_(torch.from_numpy(random.uniform(0.0, 0.5, size=4).astype(np.float32)))
activations.append(through("prelu-channels", "prelu", {"parameters": 4}, drawn((2, 3, 4)), module=each, channels_first=True, slots=[("weight", each.weight, as_is)]))


def affine(module, channels):
    with torch.no_grad():
        module.weight.copy_(torch.from_numpy(random.uniform(0.5, 1.5, size=channels).astype(np.float32)))
        if getattr(module, "bias", None) is not None:
            module.bias.copy_(torch.from_numpy(random.uniform(-0.5, 0.5, size=channels).astype(np.float32)))
    slots = [("weight", module.weight, as_is)]
    if getattr(module, "bias", None) is not None:
        slots.append(("bias", module.bias, as_is))
    return slots


norms = []
grouped = nn.GroupNorm(3, 6)
norms.append(through("groupNorm-image", "groupNorm", {"channels": 6, "groups": 3, "epsilon": 1e-5}, drawn((2, 3, 4, 6), noughts=0), module=grouped, channels_first=True, slots=affine(grouped, 6)))
grouped_rows = nn.GroupNorm(2, 6, eps=1e-3)
norms.append(through("groupNorm-rows", "groupNorm", {"channels": 6, "groups": 2, "epsilon": 1e-3}, drawn((4, 6), noughts=0), module=grouped_rows, slots=affine(grouped_rows, 6)))
instance = nn.InstanceNorm1d(3, affine=True)
norms.append(through("instanceNorm-series", "instanceNorm", {"channels": 3, "epsilon": 1e-5}, drawn((2, 5, 3), noughts=0), module=instance, channels_first=True, slots=affine(instance, 3)))
plain = nn.InstanceNorm2d(2)
norms.append(through("instanceNorm-plain", "instanceNorm", {"channels": 2, "epsilon": 1e-5, "scale": False, "shift": False}, drawn((2, 3, 3, 2), noughts=0), module=plain, channels_first=True))
rms = nn.RMSNorm(6)
norms.append(through("rmsNorm", "rmsNorm", {"features": 6, "epsilon": float(torch.finfo(torch.float32).eps)}, drawn((3, 6)), module=rms, slots=affine(rms, 6)))
rms_wide = nn.RMSNorm(4, eps=1e-6)
norms.append(through("rmsNorm-series", "rmsNorm", {"features": 4, "epsilon": 1e-6}, drawn((2, 3, 4)), module=rms_wide, slots=affine(rms_wide, 4)))


# The periodic embedding of Gorishniy, Rubachev and Babenko, "On Embeddings for Numerical Features in Tabular Deep Learning"
# (NeurIPS 2022), as their package rtdl_num_embeddings writes it: each feature v times frequencies c it learns, a whole turn
# round, its cosines then its sines, and a linear layer of each feature's own (or one shared by every feature, 'lite').
# The linear layer's numbers are printed as this library holds them: the cosines and
# sines of one feature by every feature's outputs for the features' own, and a dense layer's inputs by outputs when shared.
def periodic(name, features, frequencies, width, lite):
    values = drawn((6, features), low=-2.0, high=2.0)
    c = torch.from_numpy(random.normal(0.0, 0.3, size=(features, frequencies)).astype(np.float32)).requires_grad_(True)
    if lite:
        weight = torch.from_numpy(random.uniform(-0.3, 0.3, size=(2 * frequencies, width)).astype(np.float32)).requires_grad_(True)
        bias = torch.from_numpy(random.uniform(-0.3, 0.3, size=(width,)).astype(np.float32)).requires_grad_(True)
    else:
        weight = torch.from_numpy(random.uniform(-0.3, 0.3, size=(features, 2 * frequencies, width)).astype(np.float32)).requires_grad_(True)
        bias = torch.from_numpy(random.uniform(-0.3, 0.3, size=(features, width)).astype(np.float32)).requires_grad_(True)

    def forward(x):
        turned = 2 * math.pi * c * x[..., None]
        waves = torch.cat([torch.cos(turned), torch.sin(turned)], -1)
        made = waves @ weight + bias if lite else torch.einsum("...fi,fid->...fd", waves, weight) + bias
        return made.reshape(x.shape[0], features * width)

    if lite:
        laid_weight, laid_bias = (lambda w: w.numpy()), (lambda b: b.numpy())
    else:
        # kernel[i, f * width + j] = weight[f, i, j]; bias[f * width + j] = bias[f, j]
        laid_weight = lambda w: w.permute(1, 0, 2).reshape(2 * frequencies, features * width).numpy()
        laid_bias = lambda b: b.reshape(features * width).numpy()

    return through(
        name,
        "periodic",
        {"features": features, "frequencies": frequencies, "width": width, "lite": lite},
        values,
        function=forward,
        slots=[("frequencies", c, as_is), ("weight", weight, laid_weight), ("bias", bias, laid_bias)],
    )


embeddings = [periodic("periodic", 3, 4, 5, False), periodic("periodic-lite", 3, 4, 5, True)]


def loss(name, kind, settings, outputs, answers, function):
    o = torch.from_numpy(outputs.copy()).requires_grad_(True)
    a = torch.from_numpy(answers.copy())
    value = function(o, a)
    value.backward()
    return {
        "name": name,
        "kind": kind,
        "settings": settings,
        "shape": list(outputs.shape),
        "outputs": flat(outputs),
        "answers": flat(answers),
        "loss": float(value.detach()),
        "gradOutputs": flat(o.grad.numpy()),
    }


def pinball(q):
    return lambda o, a: torch.mean(torch.maximum(q * (a - o), (q - 1) * (a - o)))


def focal(gamma, alpha):
    # torchvision.ops.sigmoid_focal_loss, reduction='mean'
    def computed(o, a):
        p = torch.sigmoid(o)
        ce = F.binary_cross_entropy_with_logits(o, a, reduction="none")
        p_t = p * a + (1 - p) * (1 - a)
        value = ce * ((1 - p_t) ** gamma)
        if alpha is not None:
            value = (alpha * a + (1 - alpha) * (1 - a)) * value
        return value.mean()

    return computed


def hinge(o, a):
    # keras.losses.hinge: answers of nought or one are taken as minus one or one
    return torch.mean(torch.clamp(1 - (2 * a - 1) * o, min=0))


def softened_binary(epsilon):
    # keras.losses.binary_crossentropy(label_smoothing=epsilon): every answer moved towards a half
    return lambda o, a: F.binary_cross_entropy_with_logits(o, a * (1 - epsilon) + 0.5 * epsilon)


amounts = drawn((4, 3), noughts=0)
targets = random.uniform(-3.0, 3.0, size=(4, 3)).astype(np.float32)
logits = drawn((4, 3), noughts=0)
labels = (random.uniform(size=(4, 3)) > 0.5).astype(np.float32)
chances = random.uniform(size=(4, 3)).astype(np.float32)
shares = random.dirichlet(np.ones(3), size=4).astype(np.float32)
counts = random.poisson(2.0, size=(4, 3)).astype(np.float32)
rates = random.uniform(0.2, 3.0, size=(4, 3)).astype(np.float32)

losses = [
    loss("meanAbsoluteError", "meanAbsoluteError", {}, amounts, targets, F.l1_loss),
    loss("huber", "huber", {"delta": 1.0}, amounts, targets, lambda o, a: F.huber_loss(o, a, delta=1.0)),
    loss("huber-0.5", "huber", {"delta": 0.5}, amounts, targets, lambda o, a: F.huber_loss(o, a, delta=0.5)),
    loss("quantile-0.1", "quantile", {"quantile": 0.1}, amounts, targets, pinball(0.1)),
    loss("quantile-0.9", "quantile", {"quantile": 0.9}, amounts, targets, pinball(0.9)),
    loss("focal", "focal", {"gamma": 2.0, "alpha": 0.25}, logits, labels, focal(2.0, 0.25)),
    loss("focal-unbalanced", "focal", {"gamma": 1.5}, logits, labels, focal(1.5, None)),
    loss("hinge", "hinge", {}, logits, labels, hinge),
    loss("poisson", "poisson", {"logInput": True}, logits, counts, lambda o, a: F.poisson_nll_loss(o, a, log_input=True)),
    loss("poisson-rates", "poisson", {"logInput": False}, rates, counts, lambda o, a: F.poisson_nll_loss(o, a, log_input=False, eps=1e-8)),
    loss("crossEntropy-smoothed", "crossEntropy", {"smoothing": 0.1}, logits, shares, lambda o, a: F.cross_entropy(o, a, label_smoothing=0.1)),
    loss("binaryCrossEntropy-smoothed", "binaryCrossEntropy", {"smoothing": 0.1}, logits, labels, softened_binary(0.1)),
]

print(json.dumps({"made": f"torch {torch.__version__}", "layers": activations + norms + embeddings, "losses": losses}, indent=1))
