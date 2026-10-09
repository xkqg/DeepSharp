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
and the arithmetic runs on .NET's own vector maths out of the box or on libtorch, the engine under PyTorch, through
TorchSharp — swapping between them does not change a line of your model. What it adds is everything around the
engine: getting your data in, the layers, the training loop, the checkpoints and the pictures. And a table that a tree
learns better than a network does not have to become a network: the same prepared data is meant for ML.NET's trainers
too.

**0.8.1 is about optimizing: the numbers a network is trained with, searched for and measured by.** AdamW, RMSprop and
Nadam join Sgd and Adam, each held to PyTorch's own numbers; gradients can be clipped by their norm, a rate can warm up,
and a run can be watched epoch by epoch and stopped. `Study` tries many networks on rows a pipeline prepared, chooses by
the validation rows and keeps the test rows for the winner alone — `testSeed` fixes the test rows while the rest are dealt
again — and `Comparison` says, deal by deal and with its spread, whether a change helped. An answer that is a
distribution can be in an order: `emd`, `kl` and `rps` measure it, `earthMoversDistance` trains on it, and
`remainder` keeps what a flock that arrived short leaves; `target.numbers` is an answer of several free numbers. Two files
are one source with `read.join`, an `id` column travels with its rows and comes back beside the predictions, a
correlation is worked out once as numbers — Pearson's and Spearman's — and a profile can flag columns that keep one
another's order. A weekday written as a number is refused where it is declared instead of failing when the pipeline
runs, and a [step of your own](https://github.com/xkqg/DeepSharp/wiki/Writing-a-step-of-your-own) has a page. The
pipeline file is at version 8: everything written before still reads, and 0.8.0 refuses a file of the eighth by its
number. A network can walk more than an image now: convolutions along a series and through a volume — a series' window can be
causal — max and average poolings, global poolings and the dropouts of whole channels, each matched to PyTorch's own values and
gradients on every engine and read from Keras, ONNX and PyTorch files. They add one operation to the tensor seam, so an
engine written outside the library has one more to write; the changelog says which.

**0.8.0 lands what an exchange answers with as a file, and the pipeline reads it like any other.** A source that answers
differently every time it is asked cannot be read while a model trains: the pipeline would train on other numbers
tomorrow and still claim to be the same pipeline. `DeepSharp.Pipelines.Binance` fetches the candles of a market on
Binance over a closed window, once, and keeps them as a file beside a record of what was asked and what came back;
`await pipeline.ReadBinanceAsync(new BinanceCandles("BTCEUR", "1d", from, to))` lands the window where it is not landed
yet and adds an ordinary `.ReadCsv` of the landing, so what the pipeline's file names is a file and nothing of Binance.
The HTTP, the retries and the pacing are that package's alone — a project that only serves a trained model never carries
them — and it asks as a guest: an address's budget is shared with whatever else runs on it, so it goes slowly, waits
once the address has used half of what a minute allows, and stops the moment Binance says stop.

Earlier releases are in the [changelog](https://github.com/xkqg/DeepSharp/blob/main/CHANGELOG.md), which records what each one
added, and the [roadmap](https://github.com/xkqg/DeepSharp/wiki/Roadmap) says what is done and what is left out on purpose.

```
dotnet add package DeepSharp
dotnet add package DeepSharp.Pipelines
dotnet add package DeepSharp.Learners.Networks
```

```csharp
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

var pipeline = Pdd.Create()
    .ReadCsv("btceur-1d.csv")                                // declared, not opened
    .Declare(schema => schema
        .Timestamp("timestamp")
        .Number("close")
        .Optional("trades", ColumnKind.Number))              // a column that may have gaps
    .OrderBy("timestamp")
    .SettleGaps(gaps => gaps.Zero("trades"))                 // the gaps first: a quiet day traded nothing
    .AddFeature("turnover", "close", Arithmetic.Times, "trades") // then the features, worked out from settled columns
    .ScaleGiven(scale => scale                               // then the scaling, from bounds you know: no row decides
        .Between("turnover", 0, 2_000_000_000))              // them, so it stands here rather than below the split
    .SplitByTime("timestamp", train: 0.70, validation: 0.15, gap: 1) // then the split, just before what learns
    .Ahead("close", 1, AheadAs.Return)                       // the answer: tomorrow's return, from close as it was read
    .Drop("timestamp")                                       // what no model reads, before what learns
    .Normalise(scale => scale                                // and below it only what the training rows decide
        .Columns("close")                                    // the price's own scale, which no bound was given for
        .MaxAbs("trades"))                                   // a count, divided by its largest, so a quiet day stays nought
    .Report(report => report.Measure(Metric.Rmse).On(Part.Validation, Part.Test).As(Shown.Numbers))
    .WithTensorflow(network => network                       // and the network these rows train
        .Dense(8).Relu().Dense(1)                            // Keras's words: the widths come from the rows
        .Adam()
        .MeanSquaredError()
        .Run(seed: 42, epochs: 20))
    .Build();

var trained = pipeline.Train();                              // the course, from the file to the trained model

var file = trained.ToJson();                                 // the network and its pipeline, one file
```

`btceur-1d.csv` is a file of your own, and none ships here: a day a row of bitcoin's price in euros, as an exchange
exports its daily candles, under a header that names `timestamp`, `close` and `trades` — when the day starts, the price
it closed at, and how many trades it saw, empty on a day the exchange did not count them. `DeepSharp.Pipelines.Binance`
lands a file with those columns among its ten, as the section on reading the data shows. The samples below run the same
verbs on published data.

The course from raw data to a validated model is declared once as an artefact and replayed, and anything
that learns from the data is fitted on the training rows alone. That is the whole idea, and
[PDD](https://github.com/xkqg/DeepSharp/wiki/PDD) is where it is explained.

The network learns from the training rows and is judged by the validation rows; the test rows reach it only when the
report measures it, each measure beside what predicting the training rows' average would score. `trained.Predict(rows)`
answers rows that arrive later in the answer's own units — here, a return comes back as a price — and its `Unfamiliar`
names, for each row, the features it moves away from the one value every training row held there, which the network
learned nothing about; the report counts such rows in each part it measures. `TrainedNetwork.FromJson(file,
NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn().WithNetworks())` reads the network and its pipeline back in a program
that has never seen the data, refusing the network beside any other fit of its pipeline, however alike their columns
are.
`DeepSharp.Charts` draws the loss curve, the measures and a confusion matrix, as SVG, and `trained.Measures!.Report()`
renders the whole report as HTML.

The steps and what they learned are one file: `pipeline.Run().ToJson()` writes it, and
`PreparedData.FromJson(text, StepCatalog.BuiltIn())` reads it back in a program that has never seen the data.
The catalog is the list of verbs the reader knows — add `.WithIndicators()` for a file that holds indicators,
which read the rows in their order and so need that order said first, with `.OrderBy("timestamp")`; and `.WithParquet()`,
`.WithExcel()` or `.WithJson()` for one that reads its rows with those packages' readers.

## Starting from a course

A pipeline written from nothing starts with an empty chain and a reader who has to know which verb comes where. A course
says it: every step of a prepared pipeline in the order the steps belong, each present and waiting for what only you
know — the file to read, the columns to declare, the bounds a scale is given — and starting everything else as its verb
starts it. `PipelineCourse.Table` is the course for rows that do not depend on one another, a passenger list or a
customer file: read, declare, settle the gaps, add the features, scale from bounds you know, split, name the answer,
drop what is not needed, fill and scale what is learned, report, and name the network. `PipelineCourse.SeriesInTime` is
the same course for rows that follow each other in time, with three differences the rules make: the rows are put in order
first, they are divided along the clock, and the answer is the one that reads ahead, so the split keeps a gap as wide as
the answer reads.

A course is a value you say more of, and a file of its own. `course.Say("read.csv", …)` fills in a step,
`course.Waiting(catalog)` names every step that still waits and what it waits for, and `Pdd.From(course, catalog)` starts
a pipeline from a course that waits for nothing, refusing one that still does with every fault at its step. `Without`
leaves out the steps a pipeline does not need, and `Also` adds one more step of a verb, as a scale is written once a
column. `course.ToJson()` writes it and `PipelineCourse.FromJson(text, catalog)` reads it back through the door the pipeline's
file is read through; the file names the pipeline file's version, which does not move for it. In a
notebook, the toolbar's **Course for a table** and **Course for a series** write every step the blocks do not hold yet, in
one turn, each block as its verb's skeleton with `null` where only you can say — a block that waits does not read, so it
names the key it waits for and no rule is shown broken on a block nobody has touched. The wiki's
[Starting from a course](https://github.com/xkqg/DeepSharp/wiki/Starting-from-a-course) page has both courses step by
step, why each step stands where it does, and the code.

## Reading the data, and running it for the learner

`.ReadCsv(path)` is one of four readers of a file. `DeepSharp.Pipelines.Parquet` adds `.ReadParquet(path)`,
`DeepSharp.Pipelines.Excel` adds `.ReadExcel(path)` and `.ReadExcel(path, sheet)`, and `DeepSharp.Pipelines.Json` adds
`.ReadJson(path)`, for an array of records as an API hands them back; each is a verb of the pipeline's file as well,
replayed as the comma-separated file is, and the Titanic passenger list in each of the four formats gives the same rows
under the same keys in the same parts. `DeepSharp.Pipelines.DataFrame` reads Microsoft's data frame, and through it a
database query. A Parquet file and a database say what each column holds, and the proposal of kinds takes what they say.

A source that answers differently every time it is asked, such as an exchange's prices, is not read while a model trains:
the pipeline would train on other numbers tomorrow and still claim to be the same pipeline. It is fetched once, over a
closed window, and landed as a file that the pipeline then reads like any other. `DeepSharp.Pipelines.Binance` does that
for the candles of a market on Binance:

```csharp
using DeepSharp.Pipelines;

var from = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);
var candles = new BinanceCandles("BTCEUR", "1d", from, from.AddYears(1));   // a window with both ends, named by what it holds

var landed = await candles.LandAsync(SourceFolder.WorkingDirectory);       // asks Binance once, writes the candles and a record
```

`landed` is the name of a file in the working directory — `BTCEUR-1d-20240101T000000Z-20250101T000000Z.csv`, a row a day under
`timestamp`, `open`, `high`, `low`, `close`, `volume`, `quoteVolume`, `trades`, `takerBuyVolume` and
`takerBuyQuoteVolume`, every number as Binance wrote it — and a record beside it that says what was asked, when, of whom,
what came back page by page, and the fingerprint of the file. A pipeline written in code reads a relative path from the
working directory, so it reads that file like any other, with `.ReadCsv(landed)`; or
`await pipeline.ReadBinanceAsync(candles)` lands the window there and reads it in one line. The
pipeline's file names the file and nothing of Binance, so a program that only serves the trained model reads it without
the package, without HTTP and without a retry. Asking for the same window again asks Binance nothing: the file is held to
the fingerprint its record names and reused, and one that was changed since is refused rather than replaced.

The window runs from `from` up to, not including, `to`, and both ends are where a candle of that length begins — midnight
for a day, a Monday for a week, the first for a month — because Binance moves a start that is not, and a rounded window
would stop naming what it holds. The package goes as a guest. Binance counts what an address spends, and every
other program on the address spends from the same count, so it asks one request at a time, at least half a second apart,
waits for the next minute once the address has used half of what a minute allows whoever used it, and waits as long as
Binance asks when it asks, up to two minutes — a longer ask, a ban or a block is a stop. A landing writes only once the
whole window has been read, so a stop leaves nothing behind; a candle Binance does not have is counted in the record and
left out, never made up. What lands is Binance's data, under Binance's terms, and this package is not affiliated with
Binance. The wiki's [Live sources](https://github.com/xkqg/DeepSharp/wiki/Live-sources) page has the rest: what a landing
refuses, what its record holds, and the check that never reaches Binance.

A pipeline runs for the learner that learns from it. `Needs` says what that learner needs of its features:
`pipeline.RunFor(Needs.NoScale)` leaves out a step that only scales a feature, for a tree indifferent to scale, and
`RunFor(Needs.Categories)` also hands each category over as its place in the list the training rows held, for a learner
that splits on categories itself. A step an answer's way back runs through, one whose column a step below reads, and a
scale that refuses what it was not fitted on are always taken. `PreparedData.Skipped` names what a run left out, and its
file writes it, so a replay leaves out the same. On the wiki's Titanic pipeline a run for `Categories` hands a passenger
over as nine numbers where the network's run hands over fourteen, and the report measures both learners on the same
623, 133 and 135 rows. `Run()` takes every step, as it always did, and a learner is handed only a run it can take.

## On the engine you choose

The arithmetic runs on the light engine `DeepSharp` ships unless you name another, and naming it is one line:
`new FitOptions(seed: 42) { Backend = engine }` trains the network on it, judges it by the validation rows and takes the
report's measures there, and `trained.Predict(rows, engine)` serves on it. `DeepSharp.Backends.TorchSharp` is that
engine on libtorch — `TorchBackend.OnCpu()` on the processor, `TorchBackend.OnGpu(0)` on the first graphics card:

```
dotnet add package DeepSharp.Backends.TorchSharp
dotnet add package libtorch-cpu-win-x64 --version 2.10.0
```

The package brings TorchSharp 0.107.0 and nothing native, so your application brings libtorch: `libtorch-cpu-win-x64`,
`libtorch-cpu-linux-x64` or `libtorch-cpu-osx-arm64` 2.10.0 for the processor, `TorchSharp-cpu` for all three, or
`TorchSharp-cuda-windows` or `TorchSharp-cuda-linux` 0.107.0 for an NVIDIA card; made where there is none, the engine
names those packages. The processor's libtorch is 56.9 MB to 128.2 MB to download, depending on the platform, and a
card's runs to gigabytes, which is why no package of DeepSharp's brings it.

Which engine is the faster depends on the size of the work. Measured side by side on one machine, a Titanic training
step took 38.8 µs on the light engine and 336 µs on libtorch, which spends longer handing each small operation over
than the operation takes; a step of a small convolution took 12.1 ms on the light engine and 3.5 ms on libtorch on one
thread, 2.2 on sixteen; and a convolution over 256 images with 32 filters took 345 ms on the light engine, 27.5 on
libtorch on the processor and 5.0 on an RTX 5070 Ti. Every engine is held to the light one operation by operation and
step by step — a total within three roundings of the size of its terms — and not run by run, since two engines that add
up in another order drift apart over many steps: the networks sample's Titanic run kept the same epoch and got the same
0.815 of the test passengers right on all three, and ended with weights 0.08 from the light engine's on libtorch's
processor and 2.4e−7 from them on the card. The network holds no engine and its file names none, so a network trained
on one engine is served on another; `network.Predict(features, loss, engine)` is the one evaluation every door answers
rows through.

**Checkpoints** are taken every epoch, or only when the validation loss improves, and `CheckpointFile.Write(compiled,
prepared, checkpoint)` writes one as the network's file with what its run needs to go on. It records what the run went
under — the seed, the batch size, the early stopping, and the engine with the version and the device an engine that
implements `INamesItsVersionAndDevice` names — and a run that goes on from it under anything else is refused before
anything is put back, naming each difference: on the engine it was taken on, a run goes on to the bit, and on another
it would be another run under the same seed. `CheckpointFile.Read(text, NetworkCatalog.BuiltIn(), prepared)` reads one
back in one reading of its file, holding it to its pipeline, and `NetworkDocument.ReadCheckpoint` reads one kept under
keys of your own. The one file holds a network of up to about 45 million parameters, and a checkpoint under Adam one of
about 14 million; a network's file is read at 36 bytes a parameter, a checkpoint under Adam at 96.

## Reading what others trained

`network.Load(entries)` puts numbers trained somewhere else into a network by the path of each slot — `1.weight`,
`1.running_mean` — all of them or none, as PyTorch's `load_state_dict` does with `strict=True`, every fault named at once
in a `SlotLoadException`. `IImporter` is the seam a reader of another framework's files implements, and three packages
implement it, each handing back a `SavedNetwork` — the network and the loss it answers through:

```csharp
using DeepSharp.Import.Keras;
using DeepSharp.Import.Onnx;
using DeepSharp.Import.PyTorch;
using DeepSharp.Networks;
using DeepSharp.Tensors;

using var keras = File.OpenRead("titanic.keras");
var described = new KerasFile().Read(keras);                         // the network its file describes, and its loss

using var graph = File.OpenRead("titanic.onnx");
var exported = new OnnxFile(new BinaryCrossEntropy()).Read(graph);  // a graph names no loss, so it is handed one

using var state = File.OpenRead("titanic.pt");
var network = new Sequential().Dense(16).Relu().Dense(1).Lower(new Shape(14), new RandomStream(7));
var numbers = new TorchSaveFile(network, new BinaryCrossEntropy()).Read(state);   // numbers alone, into a network written here
```

`DeepSharp.Import.Keras` reads the `.keras` archive Keras 3 saves a model to and the `.h5` file it saved one to before;
`DeepSharp.Import.Onnx` the graphs `torch.onnx.export`, Keras's `model.export(format="onnx")` and tf2onnx write;
`DeepSharp.Import.PyTorch` a safetensors file, `SafetensorsFile`, or the file `torch.save(model.state_dict(), file)`
writes, `TorchSaveFile`. Each number is turned into the layout its slot keeps, as the layer that holds it says. The
Titanic network PyTorch trained answers the 135 test passengers within five roundings of a single-precision number of
PyTorch's chances, from a safetensors file, a `.pt` file or either exporter's graph, and the one Keras trained within
five of Keras's from its archive, its HDF5 file or its graph; what no network here is built of — a branch, a dilated
window, a stride of its own for each axis, a pooling that rounds up — is refused at the layer or the node that says it, every one at once. The pickle torch.save writes is a
program, so it is read by an interpreter that carries out only what PyTorch's own weights-only reader carries out, and
builds nothing a file names but what a state dictionary is made of: any other name is refused where the file names it,
before anything is looked up, built or run. A model read so is trained further behind a pipeline as any network is —
`described.Network.Compile(new Adam(0.001), described.Loss).Fit(prepared, options)` — and kept with that pipeline as the
one file.

## Samples

Two programs and a notebook in [Samples](https://github.com/xkqg/DeepSharp/tree/main/Samples) run all of this on
published data, and a test runs each as it stands:

```
dotnet run --project Samples/DeepSharp.Sample.Pipelines -c Release -f net10.0
dotnet run --project Samples/DeepSharp.Sample.Networks -c Release -f net10.0
```

The first asks the Titanic passenger list what each of its columns holds before anything is declared, has a profile say
what should not be there and how each is answered, prepares the passengers and a price series, and starts the
passengers' pipeline from the table's course, saying what waits before it fills it in. The second trains a
network on each of three datasets — whether a passenger survived, a price five days on, a day's bikes hour by hour —
has each pipeline's report measure it, saves it as its one file, reads it back and serves a row, and writes its charts.
The notebook, `Samples/titanic.verso`, writes the Titanic pipeline block by block; its C# cell trains a network on what
the blocks hand over, and its report block then draws the measures.

## A notebook to see it in

`DeepSharp.Verso.Notebooks` writes the same pipeline as a [Verso](https://www.versonotebooks.com/) notebook,
one block per step, each block the step's own JSON — edited as text, or field by field in Verso's properties
panel. The first block reads the rows from a comma-separated file, a Parquet file, a sheet of an Excel workbook or a
JSON file: the notebook brings those readers with it, as it brings the indicators and the charts. "Show the data
here" on a block runs the pipeline down to it and shows the rows there, each column
coloured over the training rows and every row marked with the part it lands in. A box on the grid leaves a
column out — it turns black — or makes it a category, and the notebook writes the step that does it. "Choose the
columns" lists every column of the source with its first values and what its cells propose it holds: tick it in, and
it comes in so, or out; pick its kind, make it the answer and set the answer's own values — or tick a range, and
seventy bands of a flock are taken in, or made the answer, with two ticks. A profile under a block names what should
not be there, and where the answer is a change to the columns — a column that hands a model the answer left out, a
fare of 0 said to stand for a gap — a box beside it makes it. What the blocks decide about their columns is saved
beside the notebook, and the toolbar takes a saved file over again, listing every change before it makes one and
every saved decision it cannot make. It also runs the whole pipeline and exports it as the same file the chain
writes, and where a notebook comes back from Jupyter or Markdown with its blocks as text — as Verso's browser editor
opens such a file — it makes blocks of them again.

Install it from Verso's Extensions panel with the notebook open: Verso loads a package installed there for a notebook
that names it among the extensions it needs, and installing it into the open notebook writes that notebook's name for
it, so it is loaded there from then on — install it the same way into each notebook that does not name it yet, the
sample among them. The same
package runs in Verso's VS Code extension, in the browser editor `verso serve` opens, and inside an application of your
own, through `DeepSharp.Verso.Api` — the [Notebook](https://github.com/xkqg/DeepSharp/wiki/Notebook#where-it-runs) page
says what each needs. Or let DeepSharp's own server show it in your browser, with nothing else to install — here the
sample, from the repository's folder:

```
dotnet tool install --global DeepSharp.Verso.Serve
deepsharp-serve Samples/titanic.verso
```

It listens on this computer alone and answers only the address it prints, token and all.

A C# cell in the same notebook reads what the blocks declare, as text — it is there after "Show the data here" or
the toolbar's run, and taken back whenever the blocks may no longer make it. The cell reads it through a catalog that
knows every verb a block can hold, so it brings the packages the notebook brings:

```csharp
#r "nuget: DeepSharp.Pipelines.Indicators"
#r "nuget: DeepSharp.Pipelines.Parquet"
#r "nuget: DeepSharp.Pipelines.Excel"
#r "nuget: DeepSharp.Pipelines.Json"
#r "nuget: DeepSharp.Learners.ML"
#r "nuget: DeepSharp.Learners.Networks"
using DeepSharp.Learners.ML;
using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;

if (Variables.TryGet<string>("deepsharp.pipeline", out var text))
{
    var folder = Variables.TryGet<string>("deepsharp.folder", out var saved) ? SourceFolder.Of(saved) : SourceFolder.WorkingDirectory;
    var catalog = StepCatalog.BuiltIn().WithIndicators().WithParquet().WithExcel().WithJson().WithNetworks().WithML();
    var declaration = PipelineDeclaration.FromJson(text, catalog);
    var prepared = new Pipeline(declaration, rows: null, folder).Run();
}
```

The notebook trains nothing, so a report block's measures come from the cell that trains a model: it hands back what the
report measured, `Variables.Set("deepsharp.predictions", trained.Measures!.PredictionsToJson())`, and "Show the data
here" at the report block measures those predictions on the notebook's own run of its blocks and draws them as the
report says, refusing predictions made behind any other fit — a block edited since the cell ran, say — with the words to
run the cell again. A cell that ends with `trained.Measures!.Report()` shows the same report, and one that ends with
`trained.History!.LossCurve()` the loss curve.

## Read on

| | |
|---|---|
| [Getting started](https://github.com/xkqg/DeepSharp/wiki/Getting-Started) | Install it, add two tensors, prepare a real file, train a network on it. |
| [Starting from a course](https://github.com/xkqg/DeepSharp/wiki/Starting-from-a-course) | Every step of a prepared pipeline in the order the steps belong, each waiting for what only you know: a table's and a series', as a value, a file and two buttons in the notebook. |
| [PDD](https://github.com/xkqg/DeepSharp/wiki/PDD) | The idea this library is built around, and the mistake it removes. |
| [Pipeline](https://github.com/xkqg/DeepSharp/wiki/Pipeline) | Every verb in the order you write it: readers, features, the split, gaps, scales, what a model is asked to predict, the handover, and a run for each learner. |
| [Live sources](https://github.com/xkqg/DeepSharp/wiki/Live-sources) | A source that answers differently every time it is asked, fetched once over a closed window and landed as a file: Binance's candles, what a landing is, what it refuses, and how it keeps to a budget it shares. |
| [Writing a step of your own](https://github.com/xkqg/DeepSharp/wiki/Writing-a-step-of-your-own) | A verb of your own that stands in a pipeline like the ones that ship, and the leak a step that learns from the answer can make. |
| [Searching for a network](https://github.com/xkqg/DeepSharp/wiki/Searching-for-a-network) | Which learning rate, how many units: a study that chooses by the validation rows and keeps the test rows for the winner, and a paired comparison of two pipelines. |
| [Networks](https://github.com/xkqg/DeepSharp/wiki/Networks) | Layers, losses, optimizers and the loop; a network in Keras's words or as code; trained behind a pipeline, measured, drawn and saved as one file. |
| [TorchSharp backend](https://github.com/xkqg/DeepSharp/wiki/TorchSharp-backend) | The engine on libtorch: what your application brings, what the engine is held to, and when it is the faster. |
| [Importing a model](https://github.com/xkqg/DeepSharp/wiki/Importing-a-model) | A model PyTorch, Keras or an ONNX exporter saved, read into the same network: what each reader reads, and what it refuses. |
| [Notebook](https://github.com/xkqg/DeepSharp/wiki/Notebook) | A pipeline written block by block in Verso, and the data at any block. |
| [Architecture](https://github.com/xkqg/DeepSharp/wiki/Architecture) | The design decisions, and what was deliberately left out. |
| [Next to TorchSharp, TensorFlow.NET and ML.NET](https://github.com/xkqg/DeepSharp/wiki#how-this-sits-next-to-torchsharp-tensorflownet-and-mlnet) | What those give you, what they do not, why the choice of engine stays a choice, and where a trainer from ML.NET fits. |
| [Quality](https://github.com/xkqg/DeepSharp/wiki/Quality) | What has to be true before anything is allowed in. |
| [Roadmap](https://github.com/xkqg/DeepSharp/wiki/Roadmap) | What is done, in the order each row needed the one above it, and what is left out on purpose. |
| [Contributing](https://github.com/xkqg/DeepSharp/blob/main/CONTRIBUTING.md) | A failing test first, no warnings, a coverage check that fails rather than reports. |
| [Security](https://github.com/xkqg/DeepSharp/wiki/Security) | What counts as a vulnerability here, and how to report one. |

## The packages

| | |
|---|---|
| `DeepSharp` | The tensors, their shape and the storage their values live on, the seam the arithmetic runs behind, what every engine refuses, the light engine on .NET's own vector maths, and the gradients worked out through it; the layers — dense, activations, dropout, normalisations, convolutions along a series, over an image and through a volume, poolings, global poolings and the dropouts of whole channels, their window padded as TensorFlow's 'same' — or, along a series, as Keras's 'causal' — if you say so — networks written as code or described in Keras's words, losses, optimizers — Sgd, Adam, AdamW, RMSprop and Nadam, derivable by an optimizer of your own — and learning-rate schedules, the training loop with early stopping, gradient clipping, an epoch hook, cancellation and checkpoints, a network written down as the kinds it is made of and the numbers it learned, and the load that puts numbers trained elsewhere into its slots. Brings System.Numerics.Tensors alone. |
| `DeepSharp.Pipelines` | The data half: readers and the kind each column's cells propose, features, the split, gaps, scales — one line naming the kind and the columns it holds for, landing the features between minus one and one unless the pipeline says otherwise — a profile that names what should not be there, the answer in five kinds, ordered or not, the report of what a trained model is measured by — amounts, classes and shares —, the learner a declaration is written for, the handover — a run for each learner, and every feature declared to land between minus one and one for a learner that needs it — and the column decisions saved on their own and taken over, a source that joins two files, an id that travels with its rows, a correlation worked out once as numbers, and a course to fill in — the steps of a prepared pipeline in the order they belong, each waiting for what only you know — saved as a file and replayed. Knows no tensor, and brings nothing but Microsoft's dependency-injection abstractions. |
| `DeepSharp.Pipelines.Parquet` | `.ReadParquet(path)`: an Apache Parquet file, which says what each of its columns holds. Brings Parquet.Net 6.1.0 and the compression libraries it reads with. |
| `DeepSharp.Pipelines.Excel` | `.ReadExcel(path)` and `.ReadExcel(path, sheet)`: a sheet of an `.xlsx`, `.xls` or `.xlsb` workbook, each cell as the sheet types it. Brings ExcelDataReader 3.9.0. |
| `DeepSharp.Pipelines.Json` | `.ReadJson(path)`: a JSON file holding an array of records, each value as the file writes it. Brings nothing: .NET reads JSON itself. |
| `DeepSharp.Pipelines.DataFrame` | One reader for the long tail: a CSV, a database query, rows already in hand — anything that fills Microsoft's DataFrame, `Microsoft.Data.Analysis`, reached through [MatPlotLibNet.DataFrame](https://www.nuget.org/packages/MatPlotLibNet.DataFrame). A CSV comes through as the text the file writes, and a query with the kinds the database gives its columns. |
| `DeepSharp.Pipelines.Indicators` | Twelve indicators over a series as pipeline verbs, the arithmetic borrowed from [MatPlotLibNet](https://github.com/xkqg/MatPlotLibNet) rather than written again. |
| `DeepSharp.Pipelines.Binance` | A live source, landed: `new BinanceCandles("BTCEUR", "1d", from, to)` over a closed window, `LandAsync` into a folder, or `await pipeline.ReadBinanceAsync(candles)` to land it and read it. The candles of a market on Binance become a file beside a record of what was asked and what came back, which the pipeline reads like any other file, by the fingerprint its record names; asking again asks Binance nothing. Brings Polly.Core 8.8.0, which does the retrying; that, the HTTP and the pacing stay in this package, since the pipeline's file names the landed file and nothing of Binance. Not affiliated with Binance; what lands is Binance's data, under Binance's terms. |
| `DeepSharp.Learners.Networks` | Where a network meets a pipeline: declared in the chain that prepares its rows — `.WithTorch(…)` or `.WithTensorflow(…)`, and `pipeline.Train()` runs it — or written as code and fitted by hand; trained on its training rows, judged by its validation rows, measured by its report, and saved with it as one file that refuses any other fit of it; what it predicts for rows served later comes back in the answer's own units, on whichever engine it is handed, naming what each row holds that the network learned nothing about; a search over networks that chooses by the validation rows and keeps the test rows for the winner, and a paired comparison of two pipelines; and a checkpoint is the same file with what the run needs to go on, refused to a run under another seed, batch size, early stopping, clip or engine. Brings the two packages it joins. |
| `DeepSharp.Learners.ML` | Where a trainer from ML.NET meets a pipeline: the chain says which trainer its rows are prepared for — `.WithML(trainer => trainer.FastTree())` — and the run made for it leaves out the steps a tree does without and writes down which, so a tree and a network are measured by one report on the same rows. This package declares that trainer and reads and writes the model it produces, and it carries no ML.NET at all, so a notebook, a server or an application that only reads a model never brings the library. |
| `DeepSharp.Learners.MLNet` | The half that carries ML.NET: it hands the prepared rows to Microsoft.ML, fits the tree the declaration names with `pipeline.TrainWithML()`, and writes the model beside the pipeline it was trained behind as one file. It names one trainer family on purpose — boosted trees and a forest — because a tree repeats from the seed the declaration carries, which is what a declaration meant to be replayed has to promise. Brings Microsoft.ML 5.0.0 and Microsoft.ML.FastTree 5.0.0. |
| `DeepSharp.Backends.TorchSharp` | The arithmetic on libtorch, on the processor or a graphics card: `TorchBackend.OnCpu()` or `TorchBackend.OnGpu(0)`, held to the same contract as the light engine, operation by operation. Brings TorchSharp 0.107.0; the application brings the libtorch it runs on. |
| `DeepSharp.Import.PyTorch` | A network PyTorch trained, read into the same network written here: a safetensors file, or the `.pt` file `torch.save(model.state_dict(), file)` writes, its pickle read as PyTorch's weights-only reader reads it. Brings Onnxify.Safetensors 0.3.11, a port of safetensors' own reader. |
| `DeepSharp.Import.Keras` | A model Keras 3 saved, as a `.keras` archive or an `.h5` file, read into a network built in Keras's words. Brings PureHDF 2.2.0, a managed HDF5 reader. |
| `DeepSharp.Import.Onnx` | An ONNX graph — PyTorch's, Keras's or tf2onnx's — lowered onto the layers it is. Brings OnnxSharp 0.3.2 and the Google.Protobuf 3.29.3 it reads with. |
| `DeepSharp.Charts` | The charts, as SVG, drawn with [MatPlotLibNet](https://github.com/xkqg/MatPlotLibNet) from what the training loop and the measures already keep: the loss curve, the learning rate, a confusion matrix, what was predicted against what was there, what was left over, every measure as bars beside the training rows' average, and a correlation as a heatmap; and the report, rendered once as HTML. |
| `DeepSharp.Verso.Notebooks` | A pipeline written as a [Verso](https://www.versonotebooks.com/) notebook, one block per step, the first reading a comma-separated, Parquet, Excel or JSON file, with the data, a profile and a heatmap at any block, and its columns chosen from the grid or a list — which shows what each column's cells propose — and saved beside it; a box beside a profile's alert gives its answer, and a report block draws the measures a C# cell hands back, and two buttons write the steps of a course into the blocks, each waiting for what only you know. It brings the charts, the indicators and the three readers, keeps a notebook of blocks in `.verso`, and makes blocks again of the steps another format kept as text. It runs in Verso's VS Code extension, in `verso serve`, in DeepSharp's own server and in an application of your own. |
| `DeepSharp.Verso.Api` | An application of your own hosting the notebook: one open notebook for each file, however many views show it, with the notebook's parts registered by the package itself — so a program published as a single file has them too. Typing, running, a click on a block's controls and the toolbar's buttons take their turn one at a time; a cell is added after another or at the end, of any kind the engine has, taken away, moved past its neighbour or turned into another kind, each only where the notebook's layout allows it; as a cell's text is typed, its kernel offers what may come next and says what a word means, even while a run is under way; a new notebook is made as one block that reads a CSV file, never over a file that is there already, and a pipeline block added to one arrives as the step of the course that belongs there instead of empty; a run can be stopped, one that never ends or one that still waits for another notebook's C# run, a file a button hands over goes to whoever pressed it, the properties panel comes back field by field, the layout, the theme and the title are changed as Verso's editors change them, and what a person does to the dashboard's tiles goes to the layout's own part. The notebook opens and saves as Verso's browser editor does, writing nothing into it that the engine only falls back on; every view is told what changed, version by version — the cells, the run under way and what runs that no run owns, the toolbar, whether anything is unsaved, what became of the kernels, what the dashboard or the presentation draws, and what the notebook says of itself; a notebook no view shows can close by itself when nothing in it is unsaved, a close stops the run under way instead of waiting for it, and the cells, and what they show, come back as plain values. Brings Verso's engine, 1.2.2. |
| `DeepSharp.Verso.Serve` | DeepSharp's own server: `deepsharp-serve`, a .NET tool, shows a notebook or a folder of them in your browser, on Verso's engine and built on `DeepSharp.Verso.Api`, with nothing else to install. It listens on this computer alone, answers only the address it prints, makes a change only for its own page, and writes into its folder only the notebooks it serves, what they save beside themselves, and a new notebook a page asks for, never over a file that is there; Ctrl+C ends it at once, whatever a notebook runs. Its page does what Verso's editor does, over one connection a tab, and carries everything it draws with, so it fetches nothing: the blocks, failures, JSON, CSV, progress, Mermaid diagrams and KaTeX formulas drawn as Verso draws them, a widget in a sandboxed frame, the dashboard and the presentation as the engine arranges them; the Metadata, Properties and View panels; Verso's keys, and what a cell's kernel offers and what a word means as its text is typed; the kernels' status, a dot while anything is unsaved, and the Stop where Run All stood. A cell is added, taken away once you say yes, moved or turned into another kind there, where the notebook's layout allows it; a file a button hands over arrives as a download; a dropped connection comes back by itself, keeping what was typed; and a folder's page makes a new notebook. |

`dotnet pack DeepSharp.slnx` makes all nineteen. They run on .NET 8 and .NET 10. MIT — see
[LICENSE](https://github.com/xkqg/DeepSharp/blob/main/LICENSE); each library a package brings comes under the licence its
own package states. The page `deepsharp-serve` serves carries Mermaid and KaTeX, each under its own MIT licence, and
DOMPurify under the Apache License 2.0, and the tool's
[third-party notices](https://github.com/xkqg/DeepSharp/blob/main/Src/DeepSharp.Verso.Serve/THIRD-PARTY-NOTICES.txt)
name those and every library the tool carries — Verso and its engine, the C# compiler, NuGet's client, Markdig, and the
readers' libraries — each with its version, where it comes from, its copyright and its licence.
