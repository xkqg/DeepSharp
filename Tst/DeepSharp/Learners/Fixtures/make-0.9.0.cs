// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

// Writes the files DeepSharp 0.9.0 — the packages on nuget.org, not this source — writes of a generated flock series, once,
// so the tests can read them as a person's application reads what an earlier release left. Nothing here is a real flock:
// every number is worked out from the row's place, so the files can be made again byte for byte.
//
// What 0.8.0's files could not hold, these do: a declared network under each of the five optimizers with a warm-up, a clip
// and early stopping; a checkpoint of a run that clipped its gradients, under each optimizer, with what the optimizer
// remembered; a network whose convolution is dilated and in groups, and one behind a layer that only pads with a value; and
// the words a declared network is written in, with every setting each takes. Everything a later release adds to a layer, an
// optimizer or the training part is written only when it says something, so these must read and be written again unchanged.
//
// Copy this file into an empty folder outside the repository (the repository's own build settings do not apply to it) and run:
//
//     dotnet run make-0.9.0.cs -- <folder the files go to>
//
// The files it leaves are those named flocks-0.9.0.*; the tests in FilesWrittenBy090Tests read them. Nothing in this folder is
// part of the build.

#:package DeepSharp@0.9.0
#:package DeepSharp.Pipelines@0.9.0
#:package DeepSharp.Learners.Networks@0.9.0

using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

var target = Path.GetFullPath(args.Length > 0 ? args[0] : ".");
Directory.CreateDirectory(target);
Directory.SetCurrentDirectory(target);   // a relative path in a pipeline is read from where the program runs

var Indented = new JsonSerializerOptions { WriteIndented = true };

const string Stem = "flocks-0.9.0";
var bands = new[] { "b1", "b2", "b3", "b4", "b5" };
var csv = Flocks(360);

File.WriteAllText(Path.Join(target, $"{Stem}.csv"), csv, new UTF8Encoding(false));

// Two rows that arrive later: the last of the series, served.
var lines = csv.TrimEnd('\n').Split('\n');
var served = $"{lines[0]}\n{lines[^2]}\n{lines[^1]}\n";

File.WriteAllText(Path.Join(target, $"{Stem}.served.csv"), served, new UTF8Encoding(false));

// ---- the series: five ordered weight bands as one answer, the flock's own count bringing it back as animals ----
Pipeline Series(Func<NetworkDeclaration, NetworkDeclaration> network) => Pdd.Create()
    .ReadCsv($"{Stem}.csv")
    .Declare(schema => schema
        .Timestamp("Date")
        .Integer("Seq").Category("Farm")
        .Number("Animals", "Age", "b1", "b2", "b3", "b4", "b5")
        .Optional("Temp", ColumnKind.Number))
    .OrderBy("Date", "Seq")
    .AddFeature("animalsPerDay", "Animals", Arithmetic.DividedBy, "Age")
    .TimeParts(parts => parts.Taking(TimePart.Month).Of("Date"))
    .TimePartsAsNumbers("Date", TimePart.Quarter)
    .Cyclical("Date", Period.MonthOfYear)
    .Profile("Farm", "Animals", "Age")
    .Correlation(["Animals", "Age"], Shown.Numbers)
    .SplitByTime("Date", train: 0.70, validation: 0.15, gap: 1)
    .Distribution(bands, scaleBy: "Animals")
    .NormaliseRow(Norm.L1, bands)
    .Drop("Date", "Seq", "Animals")
    .FillMissing(fill => fill.Median("Temp"))
    .EncodeCategories()
    .Normalise(scale => scale.Columns("Age", "animalsPerDay", "Date_quarter").MaxAbs("Temp"))
    .Report(report => report.Measure(Metric.Rmse, Metric.Mae, Metric.R2).On(Part.Validation, Part.Test).As(Shown.Numbers))
    .WithTorch(network)
    .Build();

// The five optimizers, each declared with a warm-up, a clip and early stopping: the words 0.9.0 wrote them in.
var optimizers = new (string Name, Func<NetworkDeclaration, NetworkDeclaration> Declared)[]
{
    ("sgd", network => network.Dense(8).Relu().Dense(5).Sgd(0.05, momentum: 0.9).CrossEntropy().WarmUp(2).ClipGradients(1.0).StoppingAfter(2).Run(seed: 20261012, epochs: 3, batch: 16)),
    ("adam", network => network.Dense(8).Relu().Dense(5).Adam(0.01).CrossEntropy().WarmUp(2).ClipGradients(1.0).StoppingAfter(2).Run(seed: 20261012, epochs: 3, batch: 16)),
    ("adamw", network => network.Dense(8).Relu().Dense(5).AdamW(0.01, weightDecay: 0.02).CrossEntropy().WarmUp(2).ClipGradients(1.0).StoppingAfter(2).Run(seed: 20261012, epochs: 3, batch: 16)),
    ("rmsprop", network => network.Dense(8).Relu().Dense(5).RmsProp(0.01, momentum: 0.5).CrossEntropy().WarmUp(2).ClipGradients(1.0).StoppingAfter(2).Run(seed: 20261012, epochs: 3, batch: 16)),
    ("nadam", network => network.Dense(8).Relu().Dense(5).Nadam(0.01).CrossEntropy().WarmUp(2).ClipGradients(1.0).StoppingAfter(2).Run(seed: 20261012, epochs: 3, batch: 16)),
};

// ---- a table: a label, a stratified split, a given scale ----
Pipeline Table() => Pdd.Create()
    .ReadCsv($"{Stem}.csv")
    .Declare(schema => schema
        .Category("Farm")
        .Boolean("Heavy")
        .Number("Animals", "Age"))
    .ScaleGiven(scale => scale.Between("Animals", 15000, 25000))
    .SplitStratified("Heavy", train: 0.70, validation: 0.15, seed: 7)
    .Target("Heavy")
    .EncodeCategories()
    .Normalise("Age")
    .Build();

// ---- a price-like series: the value rows ahead, as a return ----
Pipeline Ahead() => Pdd.Create()
    .ReadCsv($"{Stem}.csv")
    .Declare(schema => schema
        .Integer("Seq")
        .Number("Animals", "Age"))
    .OrderBy("Seq")
    .SplitByTime("Seq", train: 0.70, validation: 0.15, gap: 2)
    .Ahead("Animals", 2, AheadAs.Return)
    .Drop("Seq")
    .Normalise("Age")
    .Build();

var series = Series(network => network.Dense(8).Relu().Dense(5).Adam(0.01).CrossEntropy().Run(seed: 20261012, epochs: 3, batch: 16));
var table = Table();
var ahead = Ahead();

Write("series.pipeline.json", series.Declaration.ToJson());
Write("table.pipeline.json", table.Declaration.ToJson());
Write("ahead.pipeline.json", ahead.Declaration.ToJson());
Write("series.run.json", series.Run().ToJson());
Write("table.run.json", table.Run().ToJson());
Write("ahead.run.json", ahead.Run().ToJson());
Write("series.preset.json", PipelinePreset.Of(series.Declaration, header: null).ToJson());
Write("series.course.json", PipelineCourse.SeriesInTime.ToJson());
Write("table.course.json", PipelineCourse.Table.ToJson());

// ---- the network the series declares, trained behind it: its one file, its predictions, what it served ----
var engines = new Engines();
var trained = series.Train(engines);

Write("series.network.json", trained.ToJson());
Write("series.predictions.json", trained.Measures!.PredictionsToJson());

var told = trained.Predict(CsvRowSource.FromText(served));

// ---- each optimizer: the declaration, the network trained behind it, and a checkpoint of a run that clipped its gradients ----
var prepared = series.Run();
var facts = new JsonObject();
var optimizerFacts = new JsonObject();

foreach (var (name, declared) in optimizers)
{
    var pipeline = Series(declared);
    var fitted = pipeline.Train(engines);

    Write($"opt.{name}.pipeline.json", pipeline.Declaration.ToJson());
    Write($"opt.{name}.network.json", fitted.ToJson());

    var checkpointed = string.Empty;
    var compiled = new Sequential().Dense(8).Relu().Dense(5).Compile(OptimizerNamed(name), new CrossEntropy());

    // A checkpoint is written as the network stood when it was taken, so it is written then, at the end of the second epoch.
    var kept = compiled.Fit(prepared, new FitOptions(seed: 20261012)
    {
        Epochs = 4,
        BatchSize = 16,
        GradientClip = new GradientClip(1.0),
        Checkpoints = new Checkpoints(checkpoint => checkpointed = checkpoint.Epochs == 2 ? CheckpointFile.Write(compiled, prepared, checkpoint) : checkpointed),
    });

    Write($"opt.{name}.checkpoint.json", checkpointed);
    Write($"opt.{name}.fitted.network.json", kept.ToJson());

    optimizerFacts[name] = new JsonObject
    {
        ["keptEpoch"] = fitted.TrainedOn.Epoch,
        ["answers"] = Rows(fitted.Predict(CsvRowSource.FromText(served)).Answers),
        ["fittedEpochs"] = kept.History!.Epochs.Count,
        ["fittedAnswers"] = Rows(kept.Predict(CsvRowSource.FromText(served)).Answers),
        ["checkpointAnswers"] = Rows(TrainedNetwork.FromJson(checkpointed, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn().WithNetworks()).Predict(CsvRowSource.FromText(served)).Answers),
    };
}

// ---- a checkpoint of a network written in Keras's words and fitted by hand behind the same pipeline, as 0.8.0 wrote it ----
var checkpointedAdam = string.Empty;
var plain = new Sequential().Dense(5).Compile(new Adam(0.01), new CrossEntropy());
var plainFitted = plain.Fit(prepared, new FitOptions(seed: 20261012)
{
    Epochs = 4,
    BatchSize = 16,
    Checkpoints = new Checkpoints(checkpoint => checkpointedAdam = checkpoint.Epochs == 2 ? CheckpointFile.Write(plain, prepared, checkpoint) : checkpointedAdam),
});

Write("series.checkpoint.json", checkpointedAdam);
Write("series.fitted.network.json", plainFitted.ToJson());

// ---- what 0.9.0 added to a window: a window dilated and in groups, and a layer that only pads, with a value ----
// The series hands over eighteen features, laid out here as a three-by-three image of two channels. Every setting a window or
// a layer gained is written only when it says something, so these files hold that to the byte: network part 4.
TrainedNetwork Convolving(Func<Sequential, Sequential> network) =>
    network(new Sequential().Reshape(new Shape(3, 3, 2)))
        .Flatten()
        .Dense(5)
        .Compile(new Adam(0.01), new CrossEntropy())
        .Fit(prepared, new FitOptions(seed: 20261012) { Epochs = 2, BatchSize = 16 });

var dilated = Convolving(network => network.Conv2D(2, new Window(2, 2) { Dilation = 2 }, groups: 2).Relu());
var strided = Convolving(network => network.Conv2D(2, new Window(2, 2) { Stride = new Steps(1, 2) }).Relu());
var padded = Convolving(network => network.Pad2D(new Margins(new Sides(1, 0), new Sides(0, 1)), value: 0.5f).Conv2D(2, new Window(2, 2)).Relu());

Write("series.dilated.network.json", dilated.ToJson());
Write("series.strided.network.json", strided.ToJson());
Write("series.padded.network.json", padded.ToJson());

// ---- the verbs and keys this release knows, and the words a declared network is written in, so a later release can be held to them ----
var catalog = StepCatalog.BuiltIn().WithNetworks();
var verbs = new JsonArray();

foreach (var description in catalog.Descriptions)
{
    verbs.Add(new JsonObject
    {
        ["verb"] = description.Verb,
        ["since"] = description.Since,
        ["keys"] = Strings(description.Keys),
        ["required"] = Strings(description.Parameters.SelectMany(parameter => parameter.RequiredKeys)),
    });
}

Write("verbs.json", verbs.ToJsonString(Indented).ReplaceLineEndings("\n"));

JsonObject Words(IReadOnlyList<PartKind> kinds) => new(kinds.Select(kind => KeyValuePair.Create<string, JsonNode?>(kind.Name, Strings(kind.Settings.Select(setting => setting.Key)))));

var words = new JsonObject
{
    ["layers"] = Words(NetworkWords.Layers),
    ["optimizers"] = Words(NetworkWords.Optimizers),
    ["losses"] = Words(NetworkWords.Losses),
    ["schedules"] = Words(NetworkWords.Schedules),
    ["stopping"] = Words(NetworkWords.Stopping),
};

Write("words.json", words.ToJsonString(Indented).ReplaceLineEndings("\n"));

// ---- what 0.9.0 printed, which the tests hold this release to ----
var told2 = plainFitted.Predict(CsvRowSource.FromText(served));
var told3 = TrainedNetwork.FromJson(checkpointedAdam, NetworkCatalog.BuiltIn(), catalog).Predict(CsvRowSource.FromText(served));

facts["version"] = "0.9.0";
facts["pipelineVersion"] = PipelineDeclaration.Version;
facts["seriesEpochs"] = trained.History!.Epochs.Count;
facts["seriesKeptEpoch"] = trained.TrainedOn.Epoch;
facts["seriesAnswers"] = Rows(told.Answers);
facts["fittedAnswers"] = Rows(told2.Answers);
facts["checkpointAnswers"] = Rows(told3.Answers);
facts["fittedEpochs"] = plainFitted.History!.Epochs.Count;
facts["checkpointEpochs"] = 2;
facts["dilatedAnswers"] = Rows(dilated.Predict(CsvRowSource.FromText(served)).Answers);
facts["stridedAnswers"] = Rows(strided.Predict(CsvRowSource.FromText(served)).Answers);
facts["paddedAnswers"] = Rows(padded.Predict(CsvRowSource.FromText(served)).Answers);
facts["optimizers"] = optimizerFacts;
facts["measures"] = new JsonArray([.. trained.Measures!.Parts.Select(part => (JsonNode?)new JsonObject
{
    ["part"] = part.Part.ToString(),
    ["rows"] = part.Rows,
    ["values"] = new JsonArray([.. part.Values.Select(value => (JsonNode?)new JsonObject
    {
        ["metric"] = value.Metric.ToString(),
        ["value"] = Invariant(value.Value),
        ["baseline"] = Invariant(value.Baseline),
    })]),
})]);

Write("facts.json", facts.ToJsonString(Indented).ReplaceLineEndings("\n"));

Console.WriteLine($"wrote {Directory.GetFiles(target, $"{Stem}.*").Length} files to {target}");

void Write(string name, string text) => File.WriteAllText(Path.Join(target, $"{Stem}.{name}"), text, new UTF8Encoding(false));

static Optimizer OptimizerNamed(string name) => name switch
{
    "sgd" => new Sgd(0.05) { Momentum = 0.9 },
    "adam" => new Adam(0.01),
    "adamw" => new AdamW(0.01) { WeightDecay = 0.02 },
    "rmsprop" => new RmsProp(0.01) { Momentum = 0.5 },
    _ => new Nadam(0.01),
};

static string Invariant(double value) => value.ToString("R", CultureInfo.InvariantCulture);

static JsonArray Strings(IEnumerable<string> words) => new([.. words.Order(StringComparer.Ordinal).Select(word => (JsonNode?)JsonValue.Create(word))]);

static JsonArray Rows(IReadOnlyList<double[]> rows) => new([.. rows.Select(row => (JsonNode?)new JsonArray([.. row.Select(value => (JsonNode?)JsonValue.Create(Invariant(value)))]))]);

// Three flocks a day, so a moment holds several rows; five ordered weight bands that shift with the flock's age and add up to
// the flock; a temperature that is missing every seventeenth row.
static string Flocks(int rows)
{
    var text = new StringBuilder("Date,Seq,Farm,Animals,Age,Temp,Heavy,b1,b2,b3,b4,b5\n");
    var start = new DateTime(2024, 1, 1);

    for (var row = 0; row < rows; row++)
    {
        var animals = 20000 + (row * 37 % 900);
        var age = 28 + (row * 5 % 14);
        var temp = row % 17 == 0 ? string.Empty : (12 + (row * 7 % 90) / 10.0).ToString("0.0", CultureInfo.InvariantCulture);
        var weights = Enumerable.Range(0, 5).Select(band => 1.0 + Math.Max(0, 1.5 - Math.Abs(band - ((age - 28) / 3.5)))).ToArray();
        var total = weights.Sum();
        var counts = weights.Select(weight => (int)Math.Floor(animals * weight / total)).ToArray();

        counts[4] += animals - counts.Sum();

        text.Append(CultureInfo.InvariantCulture, $"{start.AddDays(row / 3):yyyy-MM-dd},{row},{"ABCD"[row % 4]},{animals},{age},{temp},{((counts[3] + counts[4]) * 2 > animals ? "True" : "False")},{string.Join(',', counts)}\n");
    }

    return text.ToString();
}
