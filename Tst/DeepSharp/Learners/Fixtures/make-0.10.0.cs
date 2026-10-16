// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

// Writes the files DeepSharp 0.10.0 â€” the packages on nuget.org, not this source â€” writes of a generated flock series, once,
// so the tests can read them as a person's application reads what an earlier release left. Nothing here is a real flock:
// every number is worked out from the row's place, so the files can be made again byte for byte.
//
// What 0.9.0's files could not hold, these do: everything 0.10.0 let a part be told â€” a dense layer without a bias or begun
// another way, a penalty, a bound and a hold on a layer's numbers, a normalisation without its scale or its shift, a padding
// that mirrors the example â€” the optimizers' weight decay, Nesterov's momentum, AMSGrad and the centred form, and a run that
// clips each gradient value, steps once every two batches and keeps an average of its weights while early stopping watches
// it; a checkpoint of that run; a network trained by a binary cross-entropy behind a table; and the words a declared network
// is written in. Everything the release after it adds is written only when it says something, so these must read and be
// written again unchanged.
//
// Copy this file into an empty folder outside the repository (the repository's own build settings do not apply to it) and run:
//
//     dotnet run make-0.10.0.cs -- <folder the files go to>
//
// The files it leaves are those named flocks-0.10.0.*; the tests in FilesWrittenBy0100Tests read them. Nothing in this folder
// is part of the build.

#:package DeepSharp@0.10.0
#:package DeepSharp.Pipelines@0.10.0
#:package DeepSharp.Learners.Networks@0.10.0
#:property PublishAot=false

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

const string Stem = "flocks-0.10.0";
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

// What 0.10.0 let a declared network say, each in the words it wrote them in.
var declared = new (string Name, Func<NetworkDeclaration, NetworkDeclaration> Declared)[]
{
    // The parts: a dense layer begun Glorot's way with a penalty and a bound, one without a bias, a normalisation without its
    // scale, one without its shift, and a layer held still.
    ("parts", network => network
        .Dense(8, weights: InitialiserWord.GlorotUniform).L2(0.001).MaxNorm(3.0).Relu()
        .BatchNorm(scale: false)
        .Dense(8, bias: false).L1(0.0005).Relu()
        .LayerNorm(scale: true, shift: false)
        .Dense(5).Held()
        .Adam(0.01).CrossEntropy().Run(seed: 20261016, epochs: 3, batch: 16)),

    // The optimizers' options.
    ("sgd", network => network.Dense(8).Relu().Dense(5).Sgd(0.05, momentum: 0.9).Nesterov().WeightDecay(0.001).CrossEntropy().Run(seed: 20261016, epochs: 3, batch: 16)),
    ("adam", network => network.Dense(8).Relu().Dense(5).Adam(0.01).AmsGrad().WeightDecay(0.001).CrossEntropy().Run(seed: 20261016, epochs: 3, batch: 16)),
    ("adamw", network => network.Dense(8).Relu().Dense(5).AdamW(0.01, weightDecay: 0.02).AmsGrad().CrossEntropy().Run(seed: 20261016, epochs: 3, batch: 16)),
    ("rmsprop", network => network.Dense(8).Relu().Dense(5).RmsProp(0.01, momentum: 0.5).Centered().WeightDecay(0.001).CrossEntropy().Run(seed: 20261016, epochs: 3, batch: 16)),
    ("nadam", network => network.Dense(8).Relu().Dense(5).Nadam(0.01).WeightDecay(0.001).CrossEntropy().Run(seed: 20261016, epochs: 3, batch: 16)),

    // The run's options: each value clipped, a step every two batches, and an average of the weights, while early stopping
    // that does not restore watches the run â€” the one way 0.10.0 lets an average and early stopping go together.
    ("run", network => network.Dense(8).Relu().Dense(5).Adam(0.01).CrossEntropy().ClipValue(0.5).StepEvery(2).Average(0.9)
        .StoppingAfter(2, restoreBest: false).Run(seed: 20261016, epochs: 4, batch: 16)),
};

// ---- a table: a label, a stratified split, a given scale, and a network a binary cross-entropy trains ----
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
    .WithTorch(network => network.Dense(4).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy().Run(seed: 20261016, epochs: 3, batch: 16))
    .Build();

var series = Series(network => network.Dense(8).Relu().Dense(5).Adam(0.01).CrossEntropy().Run(seed: 20261016, epochs: 3, batch: 16));
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

// ---- a checkpoint of a run that clips each value, steps every two batches, keeps an average and is watched by early stopping
// that does not restore, written as the run stood at the end of its second epoch ----
var prepared = series.Run();
var checkpointed = string.Empty;
var compiled = new Sequential().Dense(8).Relu().Dense(5).Compile(new Adam(0.01), new CrossEntropy());
var kept = compiled.Fit(prepared, new FitOptions(seed: 20261016)
{
    Epochs = 4,
    BatchSize = 16,
    ClipValue = 0.5,
    StepEvery = 2,
    Average = 0.9,
    EarlyStopping = new EarlyStopping { Patience = 2 },
    Checkpoints = new Checkpoints(checkpoint => checkpointed = checkpoint.Epochs == 2 ? CheckpointFile.Write(compiled, prepared, checkpoint) : checkpointed),
});

Write("averaged.checkpoint.json", checkpointed);
Write("averaged.fitted.network.json", kept.ToJson());

// ---- a padding that mirrors the example, as 0.10.0 wrote it: network part 5 ----
var mirrored = new Sequential().Reshape(new Shape(3, 3, 2))
    .Pad2D(new Margins(new Sides(1, 1), new Sides(1, 1)), Filling.Mirror)
    .Conv2D(2, new Window(2, 2)).Relu()
    .Flatten()
    .Dense(5)
    .Compile(new Adam(0.01), new CrossEntropy())
    .Fit(prepared, new FitOptions(seed: 20261016) { Epochs = 2, BatchSize = 16 });

Write("series.mirrored.network.json", mirrored.ToJson());

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

// ---- what 0.10.0 printed, which the tests hold the next release to ----
facts["version"] = "0.10.0";
facts["pipelineVersion"] = PipelineDeclaration.Version;
facts["seriesKeptEpoch"] = trained.TrainedOn.Epoch;
facts["seriesAnswers"] = Rows(trained.Predict(CsvRowSource.FromText(served)).Answers);
facts["tableAnswers"] = Rows(tableTrained.Predict(CsvRowSource.FromText(served)).Answers);
facts["averagedFittedEpochs"] = kept.History!.Epochs.Count;
facts["averagedFittedAnswers"] = Rows(kept.Predict(CsvRowSource.FromText(served)).Answers);
facts["averagedCheckpointEpochs"] = 2;
facts["averagedCheckpointAnswers"] = Rows(TrainedNetwork.FromJson(checkpointed, NetworkCatalog.BuiltIn(), catalog).Predict(CsvRowSource.FromText(served)).Answers);
facts["mirroredAnswers"] = Rows(mirrored.Predict(CsvRowSource.FromText(served)).Answers);
facts["declared"] = declaredFacts;
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
