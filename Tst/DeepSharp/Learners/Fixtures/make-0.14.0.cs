// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

// Writes the files DeepSharp 0.14.0 - the packages on nuget.org, not this source - writes of a generated flock series, once,
// so the tests can read them as a person's application reads what an earlier release left. Nothing here is a real flock:
// every number is worked out from the row's place, so the files can be made again byte for byte.
//
// What 0.13.0 files could not hold, these do: everything 0.14.0 added - a network that is not a straight line (a step that takes
// another's work, an add, two inputs read from one row), a rate that follows the training loss, a history over the last few places
// of an order, a text read a word at a time by a recurrent layer and by attention - beside everything 0.13.0 wrote, written again
// by 0.14.0. Everything the release after it adds is written only when it says something, so these must read and be written again
// unchanged.
//
// Copy this file into an empty folder outside the repository (the repository's own build settings do not apply to it) and run:
//
//     dotnet run make-0.14.0.cs -- <folder the files go to>
//
// The files it leaves are those named flocks-0.14.0.*; the tests in FilesWrittenBy0140Tests read them. Nothing in this folder
// is part of the build.

#:package DeepSharp@0.14.0
#:package DeepSharp.Pipelines@0.14.0
#:package DeepSharp.Learners.Networks@0.14.0
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

const string Stem = "flocks-0.14.0";
var bands = new[] { "b1", "b2", "b3", "b4", "b5" };
var csv = Flocks(360);

File.WriteAllText(Path.Join(target, $"{Stem}.csv"), csv, new UTF8Encoding(false));

// Two rows that arrive later: the last of the series, served.
var lines = csv.TrimEnd('\n').Split('\n');
var served = $"{lines[0]}\n{lines[^2]}\n{lines[^1]}\n";

File.WriteAllText(Path.Join(target, $"{Stem}.served.csv"), served, new UTF8Encoding(false));

// ---- the series: five ordered weight bands as one answer, the flock's own count bringing it back as animals; a farm's history
// of its flocks, the temperature written as how far it fills four bins, and how alike the parts are ----
Pipeline Series(Func<NetworkDeclaration, NetworkDeclaration> network, HistoryReach? reach = null) => Pdd.Create()
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
    .History("Farm", "Age", reach ?? new HistoryReach(3, 0), "Date", "Seq")
    .Likeness(0.2, "Age", "Farm", "Temp")
    .Watch(0.2, "Age", "Farm", "Temp")
    .Drop("Date", "Seq", "Animals")
    .FillMissing(fill => fill.Median("Temp"))
    .PiecewiseLinear("Temp", 4, OutOfRange.Clip)
    .EncodeCategories()
    .Normalise(scale => scale.Columns("Age", "animalsPerDay", "Age_history_Farm"))
    .Report(report => report.Measure(Metric.Rmse, Metric.Mae, Metric.Mape).On(Part.Validation, Part.Test).As(Shown.Numbers))
    .WithTorch(network)
    .Build();

// What 0.11.0 let a declared network say, written by 0.13.0, each in the words it wrote them in.
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

    // A part of a network that moves at a rate of its own, and a low-rank pair beside a held kernel.
    ("rateshare", network => network.Dense(8).RateShare(0.5).Relu().Dense(8).LowRank(2, 4).Relu().Dense(5).Adam(0.01).CrossEntropy().Run(seed: 20261017, epochs: 3, batch: 16)),

    // A variation of the training rows, which only a network that trains meets.
    ("noise", network => network.GaussianNoise(0.05, 3).Dense(8).Relu().Dense(5).Adam(0.01).CrossEntropy().Run(seed: 20261017, epochs: 3, batch: 16)),

    // An average of the weights that early stopping judges by, and restores the best of.
    ("judged", network => network.Dense(8).Relu().Dense(5).Adam(0.01).CrossEntropy().Average(0.9).JudgedByAverage()
        .StoppingAfter(2, restoreBest: true).Run(seed: 20261017, epochs: 4, batch: 16)),

    // 0.14.0: a step that takes the work of another, and an add of what two steps made.
    ("branch", network => network.Dense(8).Named("wide").Relu().Dense(8).Named("narrow").Add("wide", "narrow").Dense(5)
        .Adam(0.01).CrossEntropy().Run(seed: 20261017, epochs: 3, batch: 16)),

    // 0.14.0: two inputs read from the one row, each by the layers that are for it.
    ("inputs", network => network.Columns(0, 2).Named("left").Columns(2, 2).Takes("input").Named("right")
        .Dense(4).Takes("left").Named("fromLeft").Dense(4).Takes("right").Named("fromRight").Concatenate("fromLeft", "fromRight").Dense(5)
        .Adam(0.01).CrossEntropy().Run(seed: 20261017, epochs: 3, batch: 16)),

    // 0.14.0: a rate that follows the loss the run trained on.
    ("plateau", network => network.Dense(8).Relu().Dense(5).Adam(0.01).CrossEntropy().Plateau(patience: 1, factor: 0.5)
        .Run(seed: 20261017, epochs: 6, batch: 16)),
};

// 0.14.0: the last two places of a key's history, rather than every earlier flock.
var windowedNetwork = (Func<NetworkDeclaration, NetworkDeclaration>)(network => network.Dense(8).Relu().Dense(5).Adam(0.01).CrossEntropy().Run(seed: 20261017, epochs: 3, batch: 16));

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

// ---- a regression with rows set aside to calibrate an interval on ----
Pipeline Calibrated() => Pdd.Create()
    .ReadCsv($"{Stem}.csv")
    .Declare(schema => schema.Category("Farm").Number("Animals", "Age"))
    .ScaleGiven(scale => scale.Between("Animals", 15000, 25000))
    .Calibrate(0.20)
    .SplitAtRandom(0.50, 0.10, 5, testSeed: 9)
    .Target("Age")
    .EncodeCategories()
    .Report(report => report.Measure(Metric.Mae, Metric.Coverage, Metric.Width).On(Part.Validation, Part.Test).As(Shown.Numbers))
    .WithTorch(network => network.Dense(8).Relu().Dense(1).Adam(0.01).MeanSquaredError().Run(seed: 20261017, epochs: 3, batch: 16))
    .Build();

// ---- one slice of a small table: the third of three, dealt apart from the rest ----
Pipeline Sliced() => Pdd.Create()
    .ReadCsv($"{Stem}.csv")
    .Declare(schema => schema.Category("Farm").Boolean("Heavy").Number("Animals", "Age"))
    .ScaleGiven(scale => scale.Between("Animals", 15000, 25000))
    .SplitAtRandom(0.80, new Fold(3, 1), 42, 99)
    .Target("Heavy")
    .EncodeCategories()
    .Normalise("Age")
    .WithTorch(network => network.Dense(4).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy().Run(seed: 20261017, epochs: 3, batch: 16))
    .Build();

// ---- reviews cut into ids: the words of the training rows alone, a vector for each id, averaged ----
string[] praise = ["great", "lovely", "fine", "superb"];
string[] blame = ["awful", "dull", "bad", "boring"];
string[] filler = ["film", "plot", "actor", "scene", "music", "ending"];

var reviews = new StringBuilder("review,good\n");
var random = new Random(97);

for (var review = 0; review < 240; review++)
{
    var good = review % 2 == 0;
    var said = new List<string> { (good ? praise : blame)[random.Next(4)], filler[random.Next(6)], filler[random.Next(6)], (good ? praise : blame)[random.Next(4)] };

    reviews.Append(CultureInfo.InvariantCulture, $"{string.Join(' ', said.OrderBy(_ => random.Next()))},{(good ? 1 : 0)}\n");
}

File.WriteAllText(Path.Join(target, $"{Stem}.reviews.csv"), reviews.ToString(), new UTF8Encoding(false));
File.WriteAllText(Path.Join(target, $"{Stem}.reviews.served.csv"), "review,good\na great film with zxqv,1\nan awful plot with zxqv,0\n", new UTF8Encoding(false));

Pipeline Reviews(Func<NetworkDeclaration, NetworkDeclaration> network) => Pdd.Create()
    .ReadCsv($"{Stem}.reviews.csv")
    .Declare(schema => schema.Text("review").Integer("good"))
    .SplitAtRandom(0.60, 0.20, 3)
    .Tokenize("review", length: 8, vocabulary: 24)
    .Target("good")
    .Report(report => report.Measure(Metric.Accuracy).On(Part.Validation, Part.Test).As(Shown.Numbers))
    .WithTorch(network)
    .Build();

var reviewed14 = new (string Name, Func<NetworkDeclaration, NetworkDeclaration> Declared)[]
{
    // 0.14.0: the ids each keep a vector at their step, the padding becomes the mask, a recurrent layer reads them.
    ("gru", network => network.TokenEmbedding(vocabulary: 24, width: 4, keepSteps: true).Named("steps").Unpadded().Named("mask").Takes("input")
        .Gru(6).Takes("steps", "mask").Dense(1).Adam(0.05).BinaryCrossEntropy().Run(seed: 20261017, epochs: 3, batch: 16)),

    // 0.14.0: attention over the steps, past the padding the mask marks, with the positions of the steps.
    ("attention", network => network.TokenEmbedding(vocabulary: 24, width: 4, keepSteps: true).Named("steps").Unpadded().Named("mask").Takes("input")
        .SinePositions().Named("placed").Takes("steps").MultiHeadAttention(heads: 2).Named("attended").Takes("placed", "mask")
        .Flatten().Dense(1).Adam(0.05).BinaryCrossEntropy().Run(seed: 20261017, epochs: 3, batch: 16)),
};

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

// ---- the interval, the slice, the reviews and several networks answering as one ----
var calibrated = Calibrated();
var interval = calibrated.Train(engines).WithInterval(0.10);
var sliced = Sliced();
var reviewed = Reviews(network => network.TokenEmbedding(vocabulary: 24, width: 4).Dense(1).Adam(0.05).BinaryCrossEntropy().Run(seed: 20261017, epochs: 3, batch: 16));
var reviewsTrained = reviewed.Train(engines);
var reviewsServed = File.ReadAllText(Path.Join(target, $"{Stem}.reviews.served.csv"));
var step = (LearnNetworkStep)table.Declaration.Learner!;
var ensemble = table.RunFor(step.Needs).TrainEnsemble([step, step with { Seed = 2 }, step with { Seed = 3 }], new EnsembleOptions());

Write("interval.pipeline.json", calibrated.Declaration.ToJson());
Write("interval.network.json", interval.ToJson());
Write("fold.pipeline.json", sliced.Declaration.ToJson());
Write("fold.run.json", sliced.Run().ToJson());
Write("reviews.pipeline.json", reviewed.Declaration.ToJson());
Write("reviews.network.json", reviewsTrained.ToJson());
Write("ensemble.json", ensemble.ToJson());

// ---- 0.14.0: a text read by a recurrent layer and by attention, the windowed history, a run that lowers its rate on a plateau ----
foreach (var (name, network) in reviewed14)
{
    var pipeline = Reviews(network);
    var fitted = pipeline.Train(engines);

    Write($"{name}.pipeline.json", pipeline.Declaration.ToJson());
    Write($"{name}.network.json", fitted.ToJson());

    declaredFacts[name] = new JsonObject
    {
        ["keptEpoch"] = fitted.TrainedOn.Epoch,
        ["epochs"] = fitted.History!.Epochs.Count,
        ["answers"] = Rows(fitted.Predict(CsvRowSource.FromText(reviewsServed)).Answers),
    };
}

var windowed = Series(windowedNetwork, new HistoryReach(0, 2));
var windowedTrained = windowed.Train(engines);

Write("windowed.pipeline.json", windowed.Declaration.ToJson());
Write("windowed.run.json", windowed.Run().ToJson());
Write("windowed.network.json", windowedTrained.ToJson());
declaredFacts["windowed"] = new JsonObject
{
    ["keptEpoch"] = windowedTrained.TrainedOn.Epoch,
    ["epochs"] = windowedTrained.History!.Epochs.Count,
    ["answers"] = Rows(windowedTrained.Predict(CsvRowSource.FromText(served)).Answers),
};

// A run that lowers its rate when the training loss stops falling, as it stood at the end of its second epoch.
var plateauPrepared = series.Run();
var plateauCheckpoint = string.Empty;
var plateauCompiled = new Sequential().Dense(8).Relu().Dense(5).Compile(new Adam(0.01), new CrossEntropy());

plateauCompiled.Fit(plateauPrepared, new FitOptions(seed: 20261017)
{
    Epochs = 6,
    BatchSize = 16,
    Checkpoints = new Checkpoints(checkpoint => plateauCheckpoint = checkpoint.Epochs == 2 ? CheckpointFile.Write(plateauCompiled, plateauPrepared, checkpoint) : plateauCheckpoint),
});

Write("plateau.checkpoint.json", plateauCheckpoint);

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

// ---- what 0.13.0 printed, which the tests hold the next release to ----
facts["version"] = "0.14.0";
facts["pipelineVersion"] = PipelineDeclaration.Version;
facts["seriesKeptEpoch"] = trained.TrainedOn.Epoch;
facts["seriesAnswers"] = Rows(trained.Predict(CsvRowSource.FromText(served)).Answers);
facts["tableAnswers"] = Rows(tableTrained.Predict(CsvRowSource.FromText(served)).Answers);
facts["judgedFittedEpochs"] = kept.History!.Epochs.Count;
facts["judgedFittedAnswers"] = Rows(kept.Predict(CsvRowSource.FromText(served)).Answers);
facts["judgedCheckpointEpochs"] = 2;
facts["judgedCheckpointAnswers"] = Rows(TrainedNetwork.FromJson(checkpointed, NetworkCatalog.BuiltIn(), catalog).Predict(CsvRowSource.FromText(served)).Answers);
facts["declared"] = declaredFacts;
facts["intervalMargins"] = new JsonArray([.. interval.Interval!.Margins.Select(margin => (JsonNode?)JsonValue.Create(Invariant(margin)))]);
facts["intervalRows"] = interval.Interval.Rows;
facts["intervalAnswers"] = Rows(interval.Predict(CsvRowSource.FromText(served)).Answers);
facts["reviewsAnswers"] = Rows(reviewsTrained.Predict(CsvRowSource.FromText(reviewsServed)).Answers);
facts["ensembleAnswers"] = Rows(ensemble.Predict(CsvRowSource.FromText(served)).Answers);
facts["measures"] = Measured(trained.Measures!);
facts["watched"] = Watched(series.Run().Watched(CsvRowSource.FromText(served)));
facts["tableMeasures"] = Measured(tableTrained.Measures!);

Write("facts.json", facts.ToJsonString(Indented).ReplaceLineEndings("\n"));

Console.WriteLine($"wrote {Directory.GetFiles(target, $"{Stem}.*").Length} files to {target}");

void Write(string name, string text) => File.WriteAllText(Path.Join(target, $"{Stem}.{name}"), text, new UTF8Encoding(false));

static string Invariant(double value) => value.ToString("R", CultureInfo.InvariantCulture);

static JsonArray Strings(IEnumerable<string> words) => new([.. words.Order(StringComparer.Ordinal).Select(word => (JsonNode?)JsonValue.Create(word))]);

static JsonArray Rows(IReadOnlyList<double[]> rows) => new([.. rows.Select(row => (JsonNode?)new JsonArray([.. row.Select(value => (JsonNode?)JsonValue.Create(Invariant(value)))]))]);

// A window of served rows measured against the training rows the fit kept: each column's distance, whether it is above the
// bound, and how many rows each side held.
static JsonObject Watched(WindowLikeness window) => new()
{
    ["rows"] = window.Rows,
    ["columns"] = new JsonArray([.. window.Columns.Select(column => (JsonNode?)new JsonObject
    {
        ["column"] = column.Column,
        ["apart"] = Invariant(column.Apart),
        ["above"] = column.Above,
        ["trainingRows"] = column.TrainingRows,
        ["windowRows"] = column.WindowRows,
    })]),
};

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
