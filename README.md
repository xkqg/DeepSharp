<img src="https://raw.githubusercontent.com/xkqg/DeepSharp/main/assets/icon.png" width="96" align="right" alt="" />

# DeepSharp — deep learning in C#

[![CI](https://github.com/xkqg/DeepSharp/actions/workflows/ci.yml/badge.svg)](https://github.com/xkqg/DeepSharp/actions/workflows/ci.yml)
[![NuGet](https://img.shields.io/nuget/v/DeepSharp)](https://www.nuget.org/packages/DeepSharp)
[![NuGet Downloads](https://img.shields.io/nuget/dt/DeepSharp)](https://www.nuget.org/packages/DeepSharp)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/xkqg/DeepSharp/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/xkqg/DeepSharp)](https://github.com/xkqg/DeepSharp)

**The best of both worlds: TensorFlow's way of describing a network, PyTorch's way of running it.**

DeepSharp is the C# layer over the engines that already exist: you describe, train and use a network in C#,
and the arithmetic runs on .NET's own vector maths out of the box or on a heavier engine later — swapping
between them does not change a line of your model. What it adds is everything around the engine: getting
your data in, the layers, the training loop, the checkpoints and the pictures.

**0.2.1 is the tensors and the data half.** What learns from them is next; the
[roadmap](https://github.com/xkqg/DeepSharp/wiki/Roadmap) says in which order, and the
[changelog](https://github.com/xkqg/DeepSharp/blob/main/CHANGELOG.md) records what each release added.

```
dotnet add package DeepSharp
dotnet add package DeepSharp.Pipelines
```

```csharp
using DeepSharp.Pipelines;

var prepared = Pdd.Create()
    .ReadCsv("btceur-1d.csv")                                // declared, not opened
    .SplitByTime("timestamp", train: 0.70, validation: 0.15) // test is the rest
    .FillMissing("trades", With.Mean)                        // only offered after the split
    .Normalise("close")
    .Build()
    .Run();
```

The course from raw data to a validated model is declared once as an artefact and replayed, and anything
that learns from the data is fitted on the training rows alone. That is the whole idea, and
[PDD](https://github.com/xkqg/DeepSharp/wiki/PDD) is where it is explained.

## Read on

| | |
|---|---|
| [Getting started](https://github.com/xkqg/DeepSharp/wiki/Getting-Started) | Install it, add two tensors, prepare a real file. |
| [PDD](https://github.com/xkqg/DeepSharp/wiki/PDD) | The idea this library is built around, and the mistake it removes. |
| [Pipeline](https://github.com/xkqg/DeepSharp/wiki/Pipeline) | Every verb in the order you write it: readers, features, the split, gaps, scales, the handover. |
| [Architecture](https://github.com/xkqg/DeepSharp/wiki/Architecture) | The design decisions, and what was deliberately left out. |
| [Next to TorchSharp and TensorFlow.NET](https://github.com/xkqg/DeepSharp/wiki#how-this-sits-next-to-torchsharp-and-tensorflownet) | What those give you, what they do not, and why the choice of engine stays a choice. |
| [Quality](https://github.com/xkqg/DeepSharp/wiki/Quality) | What has to be true before anything is allowed in. |
| [Roadmap](https://github.com/xkqg/DeepSharp/wiki/Roadmap) | What is next, and in which order. |
| [Contributing](https://github.com/xkqg/DeepSharp/blob/main/CONTRIBUTING.md) | A failing test first, no warnings, a coverage check that fails rather than reports. |
| [Security](https://github.com/xkqg/DeepSharp/wiki/Security) | What counts as a vulnerability here, and how to report one. |

## The packages

| | |
|---|---|
| `DeepSharp` | The tensors, their shape, and the backend the arithmetic runs on. |
| `DeepSharp.Pipelines` | The data half: readers, features, the split, gaps, scales, the handover — saved as a file and replayed. |
| `DeepSharp.Pipelines.DataFrame` | One reader for the long tail: a CSV, a database query, rows already in hand, through [MatPlotLibNet.DataFrame](https://www.nuget.org/packages/MatPlotLibNet.DataFrame). |
| `DeepSharp.Pipelines.Indicators` | Twelve indicators over a series as pipeline verbs, the arithmetic borrowed from [MatPlotLibNet](https://github.com/xkqg/MatPlotLibNet) rather than written again. |

Runs on .NET 10. MIT — see [LICENSE](https://github.com/xkqg/DeepSharp/blob/main/LICENSE).
