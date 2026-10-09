// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

// Writes the files DeepSharp 0.8.0 — the packages on nuget.org, not this source — writes of a generated flock series, once,
// so the tests can read them as a person's application reads what an earlier release left. Nothing here is a real flock:
// every number is worked out from the row's place, so the files can be made again byte for byte.
//
// Copy this file into an empty folder outside the repository (the repository's own build settings do not apply to it) and run:
//
//     dotnet run make-0.8.0.cs -- <folder the files go to>
//
// The files it leaves are those named flocks-0.8.0.*; the tests in FilesWrittenBy080Tests read them. Nothing in this folder is
// part of the build.

#:package DeepSharp@0.8.0
#:package DeepSharp.Pipelines@0.8.0
#:package DeepSharp.Learners.Networks@0.8.0

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

const string Stem = "flocks-0.8.0";
var bands = new[] { "b1", "b2", "b3", "b4", "b5" };
var csv = Flocks(360);

File.WriteAllText(Path.Join(target, $"{Stem}.csv"), csv, new UTF8Encoding(false));

// Two rows that arrive later: the last of the series, served.
var lines = csv.TrimEnd('\n').Split('\n');
var served = $"{lines[0]}\n{lines[^2]}\n{lines[^1]}\n";

File.WriteAllText(Path.Join(target, $"{Stem}.served.csv"), served, new UTF8Encoding(false));

// ---- the series: five ordered weight bands as one answer, the flock's own count bringing it back as animals ----
Pipeline Series() => Pdd.Create()
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
    .WithTensorflow(network => network.Dense(8).Relu().Dense(5).Adam(0.01).CrossEntropy().Run(seed: 20261009, epochs: 3, batch: 16))
    .Build();

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

var series = Series();
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

// ---- a checkpoint of a network written in Keras's words and fitted by hand behind the same pipeline ----
var prepared = series.Run();
var checkpointed = string.Empty;
var compiled = new Sequential().Dense(5).Compile(new Adam(0.01), new CrossEntropy());

// A checkpoint is written as the network stood when it was taken, so it is written then, at the end of the second epoch.
var fitted = compiled.Fit(prepared, new FitOptions(seed: 20261009)
{
    Epochs = 4,
    BatchSize = 16,
    Checkpoints = new Checkpoints(checkpoint => checkpointed = checkpoint.Epochs == 2 ? CheckpointFile.Write(compiled, prepared, checkpoint) : checkpointed),
});

Write("series.checkpoint.json", checkpointed);
Write("series.fitted.network.json", fitted.ToJson());

var told2 = fitted.Predict(CsvRowSource.FromText(served));

// A checkpoint serves as a trained network too: what it answers is what the network stood at when it was taken.
var told3 = TrainedNetwork.FromJson(checkpointed, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn().WithNetworks()).Predict(CsvRowSource.FromText(served));

// ---- the verbs and keys this release knows, so a later release can be held to them ----
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

// ---- what 0.8.0 printed, which the tests hold this release to ----
var facts = new JsonObject
{
    ["version"] = "0.8.0",
    ["pipelineVersion"] = PipelineDeclaration.Version,
    ["seriesEpochs"] = trained.History!.Epochs.Count,
    ["seriesKeptEpoch"] = trained.TrainedOn.Epoch,
    ["seriesAnswers"] = Rows(told.Answers),
    ["fittedAnswers"] = Rows(told2.Answers),
    ["checkpointAnswers"] = Rows(told3.Answers),
    ["fittedEpochs"] = fitted.History!.Epochs.Count,
    ["checkpointEpochs"] = 2,
    ["measures"] = new JsonArray([.. trained.Measures!.Parts.Select(part => (JsonNode?)new JsonObject
    {
        ["part"] = part.Part.ToString(),
        ["rows"] = part.Rows,
        ["values"] = new JsonArray([.. part.Values.Select(value => (JsonNode?)new JsonObject
        {
            ["metric"] = value.Metric.ToString(),
            ["value"] = Invariant(value.Value),
            ["baseline"] = Invariant(value.Baseline),
        })]),
    })]),
};

Write("facts.json", facts.ToJsonString(Indented).ReplaceLineEndings("\n"));

Console.WriteLine($"wrote {Directory.GetFiles(target, $"{Stem}.*").Length} files to {target}");

void Write(string name, string text) => File.WriteAllText(Path.Join(target, $"{Stem}.{name}"), text, new UTF8Encoding(false));

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
