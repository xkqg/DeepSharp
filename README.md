# DeepSharp — deep learning in C#

DeepSharp is the C# layer over the engines that already exist. You describe, train and use a network in C#,
and the arithmetic runs on whichever engine suits the job — .NET's own vector maths out of the box, libtorch
through TorchSharp when the work gets bigger. Swapping between them does not change a line of your model.

**What DeepSharp adds is everything around the engine.** Getting your data in, the layers, the training
loop, the checkpoints, the metrics and the pictures. An engine gives you fast arithmetic; it does not give
you a way to describe a network in C#, feed it real data, watch it learn and save the result. That is the
part this library is for.

You build the network the way you prefer — stack the layers and let it train, or write the forward pass
yourself. Both give the same model, trained by the same loop, saved to the same file.

```csharp
using DeepSharp.Tensors;

var maths = new CpuBackend();

var a = Tensor.From(new Shape(2, 2), [1f, 2f, 3f, 4f]);
var b = Tensor.From(new Shape(2, 2), [10f, 20f, 30f, 40f]);

var sum = maths.Add(a, b);   // 11, 22, 33, 44
```

## Version 0.1.0 — what is here today

The first release is the foundation rather than the finished library. What it contains works and is tested;
everything above describes where it is going.

| | What it does |
|---|---|
| `Shape` | Says how big a tensor is — `2x3` is two rows of three. Tells you off straight away if the sizes do not match. |
| `Tensor` | The numbers themselves, laid out in that shape. Once made it never changes, so it is safe to reuse. |
| `ITensorBackend` | Which engine does the arithmetic. Your model is written against this, not against an engine. |
| `CpuBackend` | The engine that needs no installing: your processor's vector instructions, through .NET's own maths. |

Next: learning from mistakes (gradients), then the layers, the optimizers and the training loop. After that
a TorchSharp backend, so the same model can run its heavy work on libtorch. The [changelog](CHANGELOG.md)
records what each release actually added, and nothing is claimed before it is true.

## How this sits next to TorchSharp and TensorFlow.NET

Those are bindings: they hand you PyTorch's or TensorFlow's own interface, written in C#, with the original
engine underneath. They are excellent at being that, and DeepSharp is happy to use one.

What they do not give you is a library that reads like C#, a way to pour your data in, a training loop you
did not write yourself, or a picture of what happened. DeepSharp sits on top and provides those — and
because your model talks to a backend rather than to an engine, the choice of engine stays a choice.

The small print on that choice: the default backend needs nothing installed and travels inside your
application, while libtorch is 76 MB for every platform you ship to. You pick per project, not per library.

## Getting started

```
git clone https://github.com/xkqg/DeepSharp.git
cd DeepSharp
dotnet build DeepSharp.slnx -c Release
```

Runs on .NET 10. The [wiki](https://github.com/xkqg/DeepSharp/wiki) has the walkthrough, the design
decisions and what is planned. [CONTRIBUTING.md](CONTRIBUTING.md) has the rules for changing anything here:
a failing test first, no warnings, and a coverage check that fails rather than reports.

## Licence

MIT. See [LICENSE](LICENSE).
