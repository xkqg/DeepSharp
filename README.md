# DeepSharp — deep learning in C#

**The best of both worlds: TensorFlow's way of describing a network, PyTorch's way of running it.**

DeepSharp is the C# layer over the engines that already exist. You describe, train and use a network in C#,
and the arithmetic runs on whichever engine suits the job — .NET's own vector maths out of the box, libtorch
through TorchSharp when the work gets bigger. Swapping between them does not change a line of your model.

**What DeepSharp adds is everything around the engine.** Getting your data in, the layers, the training
loop, the checkpoints, the metrics and the pictures. An engine gives you fast arithmetic; it does not give
you a way to describe a network in C#, feed it real data, watch it learn and save the result.

```csharp
using DeepSharp.Tensors;

var maths = new CpuBackend();

var a = Tensor.From(new Shape(2, 2), [1f, 2f, 3f, 4f]);
var b = Tensor.From(new Shape(2, 2), [10f, 20f, 30f, 40f]);

var sum = maths.Add(a, b);   // 11, 22, 33, 44
```

## Pipeline-driven design

The course from raw data to a validated model is always the same — collect, add features, normalise, deal
with the gaps, split into training, validation and test, build, and check against data it has never seen.
DeepSharp asks you to **declare that course in advance as one artefact** rather than perform it, and then
replays it. The rule that makes it worth doing: **anything that learns from the data is fitted on the
training split alone and replayed unchanged.**

The wiki has both halves: [PDD](https://github.com/xkqg/DeepSharp/wiki/PDD) for the idea and the mistake it
removes, [Pipeline](https://github.com/xkqg/DeepSharp/wiki/Pipeline) for the verbs themselves.

## Version 0.1.0 — what is here today

The first release is the foundation rather than the finished library. What it contains works and is tested;
everything above describes where it is going, and the [changelog](CHANGELOG.md) records what each release
actually added.

| | What it does |
|---|---|
| `Shape` | Says how big a tensor is — `2x3` is two rows of three. Tells you off straight away if the sizes do not match. |
| `Tensor` | The numbers themselves, laid out in that shape. Once made it never changes, so it is safe to reuse. |
| `ITensorBackend` | Which engine does the arithmetic. Your model is written against this, not against an engine. |
| `CpuBackend` | The engine that needs no installing: your processor's vector instructions, through .NET's own maths. |
| `DeepSharp.Pipelines` | A second package: declare where the data comes from, how it is split and how its gaps are filled, then save that as a file and read it back unchanged. |

```csharp
using DeepSharp.Pipelines;

var declaration = Pdd.Create()
    .ReadCsv("btceur-1d.csv")                                   // declared, not opened
    .SplitByTime("timestamp", train: 0.70, validation: 0.15, test: 0.15)
    .FillMissing("trades", With.Mean)                           // only offered after the split
    .Declaration;

File.WriteAllText("btceur.pdd.json", declaration.ToJson());
```

Next come the features, the normalisers and the report, then gradients, the layers, the optimizers and the
training loop — see the [roadmap](https://github.com/xkqg/DeepSharp/wiki/Roadmap).

## Next to TorchSharp and TensorFlow.NET

Those are bindings: PyTorch's or TensorFlow's own interface written in C#, with the original engine
underneath. They are excellent at being that, and DeepSharp is happy to use one. What they do not give you
is a library that reads like C#, a way to pour your data in, a training loop you did not write yourself, or
a picture of what happened — and because your model talks to a backend rather than to an engine, the choice
of engine stays a choice. The [wiki](https://github.com/xkqg/DeepSharp/wiki) has the full comparison.

## Getting started

```
git clone https://github.com/xkqg/DeepSharp.git
cd DeepSharp
dotnet build DeepSharp.slnx -c Release
```

Runs on .NET 10. The [wiki](https://github.com/xkqg/DeepSharp/wiki) has the walkthrough, what you can build
with it, the design decisions and what is planned. [CONTRIBUTING.md](CONTRIBUTING.md) has the rules for
changing anything here: a failing test first, no warnings, and a coverage check that fails rather than
reports.

## Licence

MIT. See [LICENSE](LICENSE).
