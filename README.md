# DeepSharp — deep learning in C#

DeepSharp lets you build and train neural networks in C#, on .NET 10.

You build a network the way you prefer. Stack the layers and let the library train them, or write the
forward pass yourself. Both give you the same model, trained by the same loop and saved to the same file.

**Nothing native to install.** The maths runs on .NET's own vector instructions, so your model travels
inside your application — a console tool, a service, a desktop app. No extra binaries per platform, and no
Python anywhere.

**It shows you what it is doing.** A training run draws its own loss curve, confusion matrix and
learning-rate schedule from the numbers it already has. No separate viewer to start, no browser needed.

```csharp
using DeepSharp.Tensors;

var maths = new CpuBackend();

var a = Tensor.From(new Shape(2, 2), [1f, 2f, 3f, 4f]);
var b = Tensor.From(new Shape(2, 2), [10f, 20f, 30f, 40f]);

var sum = maths.Add(a, b);   // 11, 22, 33, 44
```

## Version 0.1.0 — what is here today

This is the first release, and it is the foundation rather than the finished library. What it contains
works and is tested; everything above describes where it is going.

| | What it does |
|---|---|
| `Shape` | Says how big a tensor is — `2x3` is two rows of three. Tells you off straight away if the sizes do not match. |
| `Tensor` | The numbers themselves, laid out in that shape. Once made, it never changes, so it is safe to reuse. |
| `CpuBackend` | Does the arithmetic, using your processor's vector instructions. |
| `ITensorBackend` | The plug the arithmetic goes through, so it can be done elsewhere later without your model changing. |

Next: learning from mistakes (gradients), then the layers, the optimizers and the training loop. The
[changelog](CHANGELOG.md) records what each release actually added, and nothing is claimed before it is true.

## Why not TorchSharp or TensorFlow.NET?

Those hand you PyTorch's or TensorFlow's own interface, written in C#, with the original engine underneath.
That is the right choice when you need everything PyTorch can do and you do not mind shipping it: the
smallest build of that engine is 76 MB for each platform you support.

DeepSharp is for the other case. A model that ships inside your application, reads like C# instead of like
translated Python, and shows you its progress without a second tool. If the work later outgrows a processor,
the same model can hand its arithmetic to one of those engines — but it will never have to.

## Getting started

```
git clone https://github.com/xkqg/DeepSharp.git
cd DeepSharp
dotnet build DeepSharp.slnx -c Release
```

The [wiki](https://github.com/xkqg/DeepSharp/wiki) has the walkthrough, the design decisions, and what is
planned. [CONTRIBUTING.md](CONTRIBUTING.md) has the rules for changing anything here: a failing test first,
no warnings, and a coverage check that fails rather than reports.

## Licence

MIT. See [LICENSE](LICENSE).
