// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

// Writes the files DeepSharp 0.11.0 - the packages on nuget.org, not this source - writes of a generated flock series, once,
// so the tests can read them as a person's application reads what an earlier release left. Nothing here is a real flock:
// every number is worked out from the row's place, so the files can be made again byte for byte.
//
// What 0.10.0's files could not hold, these do: everything 0.11.0 added - a number written as how far it fills a few bins, a
// key's history over its earlier training rows, how alike the parts of the split are; the activations, the normalisations,
// a category's vector and a periodic embedding a declared network is written in; a cross-entropy that softens its labels, a
// focal loss behind a table measured by F1, the area under the curve and the Brier score; and a run that keeps an average of
// its weights and is judged by it. Everything the release after it adds is written only when it says something, so these
// must read and be written again unchanged.
//
// Copy this file into an empty folder outside the repository (the repository's own build settings do not apply to it) and run:
//
//     dotnet run make-0.11.0.cs -- <folder the files go to>
//
// The files it leaves are those named flocks-0.11.0.*; the tests in FilesWrittenBy0110Tests read them. Nothing in this folder
// is part of the build.

#:package DeepSharp@0.11.0
#:package DeepSharp.Pipelines@0.11.0
#:package DeepSharp.Learners.Networks@0.11.0
#:property PublishAot=false

using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

var target = Path.GetFullPath(args.Length > 0 ? args[0] : ".");
Directory.CreateDirectory(target);
Directory.SetCurrentDirectory(target);   // a relative path in a pipeline is read from where the program runs

var Indented = new JsonSerializerOptions { WriteIndented = true };

const string Stem = "flocks-0.11.0";
var bands = new[] { "b1", "b2", "b3", "b4", "b5" };
var csv = Flocks(360);

File.WriteAllText(Path.Join(target, $"{Stem}.csv"), csv, new UTF8Encoding(false));

// Two rows that arrive later: the last of the series, served.
var lines = csv.TrimEnd('\n').Split('\n');
var served = $"{lines[0]}\n{lines[^2]}\n{lines[^1]}\n";

File.WriteAllText(Path.Join(target, $"{Stem}.served.csv"), served, new UTF8Encoding(false));

// ---- the series: five ordered weight bands as one answer, the flock's own count bringing it back as animals; a farm's history
// of its flocks, the temperature written as how far it fills four bins, and how alike the parts are ----
Pipeline Series(Func<NetworkDeclaration, NetworkDeclaration> network) => Pdd.Create()
    .ReadCsv($"{Stem}.csv")
    .Declare(schema => schema
        .Timestamp("Date")
        .Integer("Seq").Category("Farm")
        .Number("Animals", "Age", "b1", "b2", "b3", "b4", "b5")
        .Optional("Temp", ColumnKind.Number))
    .OrderBy("Date", "Seq")
    .AddFeature("animalsPerDay", "Animals", Arithmetic.DividedBy, "Age")
    .SplitByTime("Date", train: 0.70, validation: 0.15, gap: 1)
    .Distribution(bands, scaleBy: "Animals")
    .NormaliseRow(Norm.L1, bands)
    .History("Farm", "Age", 3, "Date", "Seq")
    .Likeness(0.2, "Age", "Farm", "Temp")
    .Drop("Date", "Seq", "Animals")
    .FillMissing(fill => fill.Median("Temp"))
    .PiecewiseLinear("Temp", 4, OutOfRange.Clip)
    .EncodeCategories()
    .Normalise(scale => scale.Columns("Age", "animalsPerDay", "Age_history_Farm"))
    .Report(report => report.Measure(Metric.Rmse, Metric.Mae, Metric.Mape).On(Part.Validation, Part.Test).As(Shown.Numbers))
    .WithTorch(network)
    .Build();

// What 0.11.0 let a declared network say, each in the words it wrote them in.
var declared = new (string Name, Func<NetworkDeclaration, NetworkDeclaration> Declared)[]
{
    // The activations and a normalisation by the root mean square, judged by a cross-entropy that softens its labels.
    ("vocabulary", network => network
        .Dense(8).Gelu().Dense(8).LeakyRelu().RmsNorm().Dense(8).Elu(0.5).Silu().Dense(5)
        .Adam(0.01).CrossEntropy(0.1).Run(seed: 20261017, epochs: 3, batch: 16)),

    // A farm's vector, over the columns its category was written down as.
    ("embedding", network => network.Embedding("Farm", 3).Dense(8).Mish().Dense(5).Adam(0.01).CrossEntropy().Run(seed: 20261017, epochs: 3, batch: 16)),

    // Every number turned round a few frequencies and written in a few values of its own.
    ("periodic", network => network.Periodic(0.1, 4, 3, lite: true).Dense(8).Selu().Dense(5).Adam(0.01).CrossEntropy().Run(seed: 20261017, epochs: 3, batch: 16)),

    // An average of the weights that early stopping judges by, and restores the best of.
    ("judged", network => network.Dense(8).Relu().Dense(5).Adam(0.01).CrossEntropy().Average(0.9).JudgedByAverage()
        .StoppingAfter(2, restoreBest: true).Run(seed: 20261017, epochs: 4, batch: 16)),
};

// ---- a table: a label, a stratified split, a given scale, a focal loss, and the measures of classes 0.11.0 added ----
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
    .Report(report => report.Measure(Metric.F1, Metric.Auc, Metric.Brier).On(Part.Validation, Part.Test).As(Shown.Numbers))
    .WithTorch(network => network.Dense(4).Relu().Dense(1).Adam(0.01).Focal(2, 0.25).Run(seed: 20261017, epochs: 3, batch: 16))
    .Build();

var series = Series(network => network.Dense(8).Relu().Dense(5).Adam(0.01).CrossEntropy().Run(seed: 20261017, epochs: 3, batch: 16));
var table = Table();
var engines = new Engines();
var catalog = StepCatalog.BuiltIn().WithNetworks();

Write("series.pipeline.json", series.Declaration.ToJson());
Write("series.run.json", series.Run().ToJson());
Write("table.pipeline.json", table.Declaration.ToJson());
Write("table.run.json", table.Run().ToJson());

// ---- the networks trained behind them: their one file, and what each served ----
var facts = new JsonObject();
var trained = series.Train(engines);
var tableTrained = table.Train(engines);

Write("series.network.json", trained.ToJson());
Write("series.predictions.json", trained.Measures!.PredictionsToJson());
Write("table.network.json", tableTrained.ToJson());

var declaredFacts = new JsonObject();

foreach (var (name, network) in declared)
{
    var pipeline = Series(network);
    var fitted = pipeline.Train(engines);

    Write($"{name}.pipeline.json", pipeline.Declaration.ToJson());
    Write($"{name}.network.json", fitted.ToJson());

    declaredFacts[name] = new JsonObject
    {
        ["keptEpoch"] = fitted.TrainedOn.Epoch,
        ["epochs"] = fitted.History!.Epochs.Count,
        ["answers"] = Rows(fitted.Predict(CsvRowSource.FromText(served)).Answers),
    };
}

// ---- a checkpoint of a run that keeps an average and is judged by it, as the run stood at the end of its second epoch ----
var prepared = series.Run();
var checkpointed = string.Empty;
var compiled = new Sequential().Dense(8).Gelu().Dense(5).Compile(new Adam(0.01), new CrossEntropy { Smoothing = 0.1 });
var kept = compiled.Fit(prepared, new FitOptions(seed: 20261017)
{
    Epochs = 4,
    BatchSize = 16,
    Average = 0.9,
    JudgedByAverage = true,
    EarlyStopping = new EarlyStopping { Patience = 2, RestoreBest = true },
    Checkpoints = new Checkpoints(checkpoint => checkpointed = checkpoint.Epochs == 2 ? CheckpointFile.Write(compiled, prepared, checkpoint) : checkpointed),
});

Write("judged.checkpoint.json", checkpointed);
Write("judged.fitted.network.json", kept.ToJson());

// ---- the verbs and keys this release knows, and the words a declared network is written in ----
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

// ---- what 0.11.0 printed, which the tests hold the next release to ----
facts["version"] = "0.11.0";
facts["pipelineVersion"] = PipelineDeclaration.Version;
facts["seriesKeptEpoch"] = trained.TrainedOn.Epoch;
facts["seriesAnswers"] = Rows(trained.Predict(CsvRowSource.FromText(served)).Answers);
facts["tableAnswers"] = Rows(tableTrained.Predict(CsvRowSource.FromText(served)).Answers);
facts["judgedFittedEpochs"] = kept.History!.Epochs.Count;
facts["judgedFittedAnswers"] = Rows(kept.Predict(CsvRowSource.FromText(served)).Answers);
facts["judgedCheckpointEpochs"] = 2;
facts["judgedCheckpointAnswers"] = Rows(TrainedNetwork.FromJson(checkpointed, NetworkCatalog.BuiltIn(), catalog).Predict(CsvRowSource.FromText(served)).Answers);
facts["declared"] = declaredFacts;
facts["measures"] = Measured(trained.Measures!);
facts["tableMeasures"] = Measured(tableTrained.Measures!);

Write("facts.json", facts.ToJsonString(Indented).ReplaceLineEndings("\n"));

Console.WriteLine($"wrote {Directory.GetFiles(target, $"{Stem}.*").Length} files to {target}");

void Write(string name, string text) => File.WriteAllText(Path.Join(target, $"{Stem}.{name}"), text, new UTF8Encoding(false));

static string Invariant(double value) => value.ToString("R", CultureInfo.InvariantCulture);

static JsonArray Strings(IEnumerable<string> words) => new([.. words.Order(StringComparer.Ordinal).Select(word => (JsonNode?)JsonValue.Create(word))]);

static JsonArray Rows(IReadOnlyList<double[]> rows) => new([.. rows.Select(row => (JsonNode?)new JsonArray([.. row.Select(value => (JsonNode?)JsonValue.Create(Invariant(value)))]))]);

static JsonArray Measured(Measures measures) => new([.. measures.Parts.Select(part => (JsonNode?)new JsonObject
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
