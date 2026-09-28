<img src="https://raw.githubusercontent.com/xkqg/DeepSharp/main/assets/icon.png" width="96" align="right" alt="" />

# DeepSharp — deep learning in C#

[![CI](https://github.com/xkqg/DeepSharp/actions/workflows/ci.yml/badge.svg)](https://github.com/xkqg/DeepSharp/actions/workflows/ci.yml)
[![NuGet](https://img.shields.io/nuget/v/DeepSharp)](https://www.nuget.org/packages/DeepSharp)
[![NuGet Downloads](https://img.shields.io/nuget/dt/DeepSharp)](https://www.nuget.org/packages/DeepSharp)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/xkqg/DeepSharp/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/xkqg/DeepSharp)](https://github.com/xkqg/DeepSharp)

**The best of three worlds: TensorFlow's way of describing a network, PyTorch's way of running it, and ML.NET's
way of learning from a table.**

DeepSharp is the C# layer over the engines that already exist: you describe, train and use a network in C#,
and the arithmetic runs on .NET's own vector maths out of the box or on a heavier engine later — swapping
between them does not change a line of your model. What it adds is everything around the engine: getting
your data in, the layers, the training loop, the checkpoints and the pictures. And a table that a tree learns
better than a network does not have to become a network: the same prepared data is meant for ML.NET's trainers
too.

**0.3.0 is the tensors, the data half, and a notebook to see the data in and choose its columns — in Verso, in an
application of your own, or in your browser from DeepSharp's own server.** What learns from them is next;
the [roadmap](https://github.com/xkqg/DeepSharp/wiki/Roadmap) says in which order, and the
[changelog](https://github.com/xkqg/DeepSharp/blob/main/CHANGELOG.md) records what each release added.

```
dotnet add package DeepSharp
dotnet add package DeepSharp.Pipelines
```

```csharp
using DeepSharp.Pipelines;

var prepared = Pdd.Create()
    .ReadCsv("btceur-1d.csv")                                // declared, not opened
    .Declare(schema => schema
        .Timestamp("timestamp")
        .Number("close")
        .Optional("trades", ColumnKind.Number))              // a column that may have gaps
    .SplitByTime("timestamp", train: 0.70, validation: 0.15) // test is the rest
    .FillMissing("trades", With.Mean)                        // only offered after the split
    .Normalise("close")
    .Build()
    .Run();
```

The course from raw data to a validated model is declared once as an artefact and replayed, and anything
that learns from the data is fitted on the training rows alone. That is the whole idea, and
[PDD](https://github.com/xkqg/DeepSharp/wiki/PDD) is where it is explained.

The steps and what they learned are one file: `prepared.ToJson()` writes it, and
`PreparedData.FromJson(text, StepCatalog.BuiltIn())` reads it back in a program that has never seen the data.
The catalog is the list of verbs the reader knows — add `.WithIndicators()` for a file that holds indicators,
which read the rows in their order and so need that order said first, with `.OrderBy("timestamp")`.

## A notebook to see it in

`DeepSharp.Verso.Notebooks` writes the same pipeline as a [Verso](https://www.versonotebooks.com/) notebook,
one block per step, each block the step's own JSON — edited as text, or field by field in Verso's properties
panel. "Show the data here" on a block runs the pipeline down to it and shows the rows there, each column
coloured over the training rows and every row marked with the part it lands in. A box on the grid leaves a
column out — it turns black — or makes it a category, and the notebook writes the step that does it. "Choose the
columns" lists every column of the source with its first values: tick it in or out, pick its kind, make it the
answer and set the answer's own values — or tick a range, and seventy bands of a flock are taken in, or made the
answer, with two ticks. What the blocks decide about their columns is saved beside the notebook, and the toolbar
takes a saved file over again, listing every change before it makes one and every saved decision it cannot make.
It also runs the whole pipeline and exports it as the same file the chain writes.

Install it from Verso's Extensions panel. The same package runs in Verso's VS Code extension, in the browser
editor `verso serve` opens, and inside an application of your own, through `DeepSharp.Verso.Api` — the
[Notebook](https://github.com/xkqg/DeepSharp/wiki/Notebook#where-it-runs) page says what each needs. Or let
DeepSharp's own server show it in your browser, with nothing else to install:

```
dotnet tool install --global DeepSharp.Verso.Serve
deepsharp-serve titanic.verso
```

It listens on this computer alone and answers only the address it prints, token and all.

A C# cell in the same notebook reads what the blocks declare, as text — it is there after "Show the data here" or
the toolbar's run, and taken back whenever the blocks may no longer make it:

```csharp
#r "nuget: DeepSharp.Pipelines.Indicators"
using DeepSharp.Pipelines;

if (Variables.TryGet<string>("deepsharp.pipeline", out var text))
{
    var folder = Variables.TryGet<string>("deepsharp.folder", out var saved) ? SourceFolder.Of(saved) : SourceFolder.WorkingDirectory;
    var declaration = PipelineDeclaration.FromJson(text, StepCatalog.BuiltIn().WithIndicators());
    var prepared = new Pipeline(declaration, rows: null, folder).Run();
}
```

## Read on

| | |
|---|---|
| [Getting started](https://github.com/xkqg/DeepSharp/wiki/Getting-Started) | Install it, add two tensors, prepare a real file. |
| [PDD](https://github.com/xkqg/DeepSharp/wiki/PDD) | The idea this library is built around, and the mistake it removes. |
| [Pipeline](https://github.com/xkqg/DeepSharp/wiki/Pipeline) | Every verb in the order you write it: readers, features, the split, gaps, scales, what a model is asked to predict, the handover. |
| [Notebook](https://github.com/xkqg/DeepSharp/wiki/Notebook) | A pipeline written block by block in Verso, and the data at any block. |
| [Architecture](https://github.com/xkqg/DeepSharp/wiki/Architecture) | The design decisions, and what was deliberately left out. |
| [Next to TorchSharp, TensorFlow.NET and ML.NET](https://github.com/xkqg/DeepSharp/wiki#how-this-sits-next-to-torchsharp-tensorflownet-and-mlnet) | What those give you, what they do not, why the choice of engine stays a choice, and where a trainer from ML.NET fits. |
| [Quality](https://github.com/xkqg/DeepSharp/wiki/Quality) | What has to be true before anything is allowed in. |
| [Roadmap](https://github.com/xkqg/DeepSharp/wiki/Roadmap) | What is next, and in which order. |
| [Contributing](https://github.com/xkqg/DeepSharp/blob/main/CONTRIBUTING.md) | A failing test first, no warnings, a coverage check that fails rather than reports. |
| [Security](https://github.com/xkqg/DeepSharp/wiki/Security) | What counts as a vulnerability here, and how to report one. |

## The packages

| | |
|---|---|
| `DeepSharp` | The tensors, their shape, and the backend the arithmetic runs on. |
| `DeepSharp.Pipelines` | The data half: readers, features, the split, gaps, scales, the answer in four kinds, the handover, and the column decisions saved on their own and taken over — saved as a file and replayed. |
| `DeepSharp.Pipelines.DataFrame` | One reader for the long tail: a CSV, a database query, rows already in hand — anything that fills Microsoft's DataFrame, `Microsoft.Data.Analysis`, reached through [MatPlotLibNet.DataFrame](https://www.nuget.org/packages/MatPlotLibNet.DataFrame). |
| `DeepSharp.Pipelines.Indicators` | Twelve indicators over a series as pipeline verbs, the arithmetic borrowed from [MatPlotLibNet](https://github.com/xkqg/MatPlotLibNet) rather than written again. |
| `DeepSharp.Verso.Notebooks` | A pipeline written as a [Verso](https://www.versonotebooks.com/) notebook, one block per step, with the data, a profile and a heatmap at any block, and its columns chosen from the grid or a list and saved beside it. It runs in Verso's VS Code extension, in `verso serve`, in DeepSharp's own server and in an application of your own. |
| `DeepSharp.Verso.Api` | An application of your own hosting the notebook: one open notebook for each file, however many views show it, with the notebook's parts registered by the package itself — so a program published as a single file has them too. Typing, running, a click on a block's controls and the toolbar's buttons take their turn one at a time; a cell is added after another or at the end, of any kind the engine has, taken away, moved past its neighbour or turned into another kind, each only where the notebook's layout allows it; as a cell's text is typed, its kernel offers what may come next and says what a word means, even while a run is under way; a new notebook is made as one block that reads a CSV file, never over a file that is there already; a run can be stopped, one that never ends or one that still waits for another notebook's C# run, a file a button hands over goes to whoever pressed it, the properties panel comes back field by field, the layout, the theme and the title are changed as Verso's editors change them, and what a person does to the dashboard's tiles goes to the layout's own part. The notebook opens and saves as Verso's browser editor does, writing nothing into it that the engine only falls back on; every view is told what changed, version by version — the cells, the run under way and what runs that no run owns, the toolbar, whether anything is unsaved, what became of the kernels, what the dashboard or the presentation draws, and what the notebook says of itself; a notebook no view shows can close by itself when nothing in it is unsaved, a close stops the run under way instead of waiting for it, and the cells, and what they show, come back as plain values. |
| `DeepSharp.Verso.Serve` | DeepSharp's own server: `deepsharp-serve`, a .NET tool, shows a notebook or a folder of them in your browser, on Verso's engine and built on `DeepSharp.Verso.Api`, with nothing else to install. It listens on this computer alone, answers only the address it prints, makes a change only for its own page, and writes into its folder only the notebooks it serves, what they save beside themselves, and a new notebook a page asks for, never over a file that is there; Ctrl+C ends it at once, whatever a notebook runs. Its page does what Verso's editor does, over one connection a tab, and carries everything it draws with, so it fetches nothing: the blocks, failures, JSON, CSV, progress, Mermaid diagrams and KaTeX formulas drawn as Verso draws them, a widget in a sandboxed frame, the dashboard and the presentation as the engine arranges them; the Metadata, Properties and View panels; Verso's keys, and what a cell's kernel offers and what a word means as its text is typed; the kernels' status, a dot while anything is unsaved, and the Stop where Run All stood. A cell is added, taken away once you say yes, moved or turned into another kind there, where the notebook's layout allows it; a file a button hands over arrives as a download; a dropped connection comes back by itself, keeping what was typed; and a folder's page makes a new notebook. |

Runs on .NET 8 and .NET 10. MIT — see [LICENSE](https://github.com/xkqg/DeepSharp/blob/main/LICENSE). The page
`deepsharp-serve` serves carries Mermaid and KaTeX, each under its own MIT licence, named in the tool's
[third-party notices](https://github.com/xkqg/DeepSharp/blob/main/Src/DeepSharp.Verso.Serve/THIRD-PARTY-NOTICES.txt).
