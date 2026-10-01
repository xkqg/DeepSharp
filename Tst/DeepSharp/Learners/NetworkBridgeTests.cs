// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json.Nodes;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;
using DeepSharp.Tests.Backends.Parts;
using DeepSharp.Tests.Networks;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A network meets a pipeline here: it learns from the training rows the pipeline hands over, each feature on one scale, is
/// judged by the validation rows, and never sees the test rows; the pipeline's report measures it; what it predicts for a
/// row served later comes back in the answer's own units; and the network and the pipeline it was trained behind are one
/// file, which refuses to be read beside any other fit of that pipeline.
/// </summary>
public class NetworkBridgeTests
{
    // Twenty rows t = 1…20 of x = sin t, whose answer is whether x is above nought; the answers of the given rows are 2,
    // which no binary cross-entropy could have meant. Ten rows train, five choose, five test.
    private static PreparedData Twenty(IEnumerable<int> twos, double validation = 0.25) =>
        Pdd.Create()
            .Read(
                CsvRowSource.FromText("t,x,y\n" + string.Join('\n', Enumerable.Range(1, 20).Select(t =>
                    string.Create(CultureInfo.InvariantCulture, $"{t},{Math.Sin(t)},{(twos.Contains(t) ? 2 : Math.Sin(t) > 0 ? 1 : 0)}"))) + "\n"),
                "twenty rows")
            .Declare(schema => schema.Integer("t", "y").Number("x"))
            .SplitByTime("t", 0.50, validation)
            .Normalise("x", Scale.MidRange)
            .Drop("t")
            .Target("y")
            .Build()
            .Run();

    private static CompiledNetwork Small() => new Sequential().Dense(4).Tanh().Dense(1).Compile(new Adam(0.05), new BinaryCrossEntropy());

    [Fact]
    public void ANetwork_LearnsFromTheTrainingRows_IsJudgedByTheValidationRows_AndNeverSeesTheTestRows()
    {
        // The last five rows answer 2, which the loss would refuse the moment it read one.
        var trained = Small().Fit(Twenty(twos: [16, 17, 18, 19, 20]), new FitOptions(seed: 3) { Epochs = 4 });

        Assert.Equal(4, trained.History!.Epochs.Count);
        Assert.All(trained.History.Epochs, epoch => Assert.NotNull(epoch.ValidationLoss));
        Assert.Equal(3, trained.TrainedOn.Seed);
        Assert.Equal(3, trained.TrainedOn.Epoch);
        Assert.Equal(["x"], trained.TrainedOn.Features);
        Assert.Equal(["y"], trained.TrainedOn.Answers);
        Assert.Equal("target", trained.TrainedOn.Output);
        Assert.Null(trained.Measures);
    }

    [Fact]
    public void ARowTheLossCouldNotHaveMeant_IsRefused_NamedAsItWasRead()
    {
        var wrong = Assert.Throws<ArgumentException>(() => Small().Fit(Twenty(twos: [4]), new FitOptions(seed: 3)));

        Assert.StartsWith("Row 4 cannot be trained on by this loss:", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EarlyStopping_WithNoValidationRows_IsRefused_AndARunWithoutItGoesOnUnjudged()
    {
        var unjudged = Twenty(twos: [], validation: 0);

        Assert.Throws<ArgumentException>(() => Small().Fit(unjudged, new FitOptions(seed: 3) { EarlyStopping = new EarlyStopping() }));
        Assert.All(Small().Fit(unjudged, new FitOptions(seed: 3)).History!.Epochs, epoch => Assert.Null(epoch.ValidationLoss));
    }

    [Fact]
    public void APipelineNamingNoAnswer_HasNothingToTrainANetworkOn()
    {
        var unanswered = Pdd.Create().Read(CsvRowSource.FromText("x\n0.5\n-0.5\n"), "two rows").Declare(schema => schema.Number("x")).SplitAtRandom(0.5, 0).Build().Run();

        Assert.Contains("names no answer", Assert.Throws<InvalidOperationException>(() => Small().Fit(unanswered, new FitOptions(seed: 3))).Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentNullException>(() => Small().Fit(null!, new FitOptions(seed: 3)));
        Assert.Throws<ArgumentNullException>(() => Small().Fit(unanswered, null!));
        Assert.Throws<ArgumentNullException>(() => ((CompiledNetwork)null!).Fit(unanswered, new FitOptions(seed: 3)));
    }

    [Fact]
    public void APipelineReadFromItsFile_HasNoRowsToTrainANetworkOn_AndFitSaysSo()
    {
        // A notebook's export, or any pipeline's file, keeps the fit and not the rows: fitting a network behind it says the
        // rows are missing and how to have them, rather than that the answer never reached the end of the pipeline.
        var exported = PreparedData.FromJson(Passengers(Repository.Data("titanic.csv")).ToJson(), StepCatalog.BuiltIn());

        var refused = Assert.Throws<InvalidOperationException>(() => Small().Fit(exported, new FitOptions(seed: 3)));

        Assert.StartsWith("This pipeline holds no rows: one read from its file", refused.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("reached the end", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ThePipelinesReport_MeasuresTheTrainedNetwork_OnEveryPartItNames()
    {
        var prepared = Pdd.Create()
            .Read(CsvRowSource.FromText("t,x,y\n" + string.Join('\n', Enumerable.Range(1, 20).Select(t => string.Create(CultureInfo.InvariantCulture, $"{t},{Math.Sin(t)},{(Math.Sin(t) > 0 ? 1 : 0)}"))) + "\n"), "twenty rows")
            .Declare(schema => schema.Integer("t", "y").Number("x"))
            .SplitByTime("t", 0.50, 0.25)
            .Normalise("x", Scale.MidRange)
            .Drop("t")
            .Target("y")
            .Report(report => report.Measure(Metric.Accuracy, Metric.Rmse).On(Part.Train, Part.Test).As(Shown.Numbers))
            .Build()
            .Run();

        var trained = Small().Fit(prepared, new FitOptions(seed: 3) { Epochs = 2 });
        var expected = prepared.Measure([.. new[] { Part.Train, Part.Test }.Select(part => Predicted(trained, prepared.Batch(part, Needs.OneScale)))]);

        Assert.Equal([Part.Train, Part.Test], trained.Measures!.Parts.Select(part => part.Part));
        Assert.Equal(expected.Parts.SelectMany(part => part.Values), trained.Measures.Parts.SelectMany(part => part.Values));
    }

    [Fact]
    public void WhatTheReportMeasuredANetworkFrom_HandedBackAsText_IsMeasuredAgainToTheSameMeasures()
    {
        // A notebook's C# cell hands the text back to the notebook, which runs the same pipeline and measures it there; the
        // run the network was trained behind measures it to the numbers the report took, what it learned nothing about too.
        var prepared = Pdd.Create()
            .Read(CsvRowSource.FromText("t,x,y\n" + string.Join('\n', Enumerable.Range(1, 20).Select(t => string.Create(CultureInfo.InvariantCulture, $"{t},{Math.Sin(t)},{(Math.Sin(t) > 0 ? 1 : 0)}"))) + "\n"), "twenty rows")
            .Declare(schema => schema.Integer("t", "y").Number("x"))
            .SplitByTime("t", 0.50, 0.25)
            .Normalise("x", Scale.MidRange)
            .Drop("t")
            .Target("y")
            .Report(report => report.Measure(Metric.Accuracy, Metric.Rmse).On(Part.Train, Part.Validation, Part.Test).As(Shown.Numbers))
            .Build()
            .Run();
        var trained = Small().Fit(prepared, new FitOptions(seed: 3) { Epochs = 2 });

        var again = prepared.MeasureAgain(trained.Measures!.PredictionsToJson());

        Assert.Equal(trained.Measures.Parts.SelectMany(part => part.Values), again.Parts.SelectMany(part => part.Values));
        Assert.Equal<int?>([0, 0, 0], again.Parts.Select(part => part.UnfamiliarRows));
    }

    [Fact]
    public void TheFirstPassenger_IsServedAsTheWalkedRow_AndComesBackAsTheChanceTheySurvived()
    {
        var prepared = Passengers(Repository.Data("titanic.csv"));
        var network = new LayerStack(
            new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias()), new Relu(), new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias()));
        var trained = TrainedNetwork.Of(network, new BinaryCrossEntropy(), prepared, seed: 1);
        var passenger = new InMemoryRowSource(["pclass", "sex", "age", "sibsp", "parch", "fare"], [["3", "male", "22.0", "1", "0", "7.25"]]);

        var predictions = trained.Predict(passenger);

        Assert.Equal(
            [-0.75, -1, -0.4576526765518975, -0.971697884875584, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            prepared.Served(passenger, Needs.OneScale).Features[0]);
        Assert.Equal(["survived"], predictions.AnswerNames);
        Assert.Equal([0], predictions.HandedInAt);
        Assert.Equal(1 / (1 + Math.Exp(-0.20000000298023224)), Assert.Single(Assert.Single(predictions.Answers)), 1e-6);
    }

    [Fact]
    public void AReturnFiveDaysOn_ComesBackAsAPrice_TheDaysCloseGrownByIt()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("apple.csv"))
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close", "AAPL.Volume"))
            .OrderBy("Date")
            .Cyclical("Date", Period.DayOfWeek)
            .SplitByTime("Date", train: 0.70, validation: 0.15, gap: 5)
            .Ahead("AAPL.Close", 5, AheadAs.Return)
            .Normalise("AAPL.Close", Scale.MidRange)
            .Normalise("AAPL.Volume", Scale.MidRange)
            .Drop("Date")
            .Build()
            .Run();
        var network = new LayerStack(new Dense(WalkedRows.TanhWeights(), WalkedRows.TanhBias()), new Tanh(), new Dense(WalkedRows.ReturnWeights(), WalkedRows.ReturnBias()));
        var trained = TrainedNetwork.Of(network, new MeanSquaredError(), prepared, seed: 1);
        var first = FirstLine(Repository.Data("apple.csv"), keep: _ => true);

        var served = prepared.Served(first, Needs.OneScale);
        var predicted = (double)network.Forward(Tensor.From(new Shape(1, served.Features[0].Length), [.. served.Features[0].Select(value => (float)value)]), Pass.Evaluation(new CpuBackend())).Values[0];

        Assert.Equal(127.830002 * (1 + predicted), Assert.Single(Assert.Single(trained.Predict(first).Answers)), 1e-9);
    }

    [Fact]
    public void TheFirstDaysShares_ComeBackAsItsBikes_SharingOutTheDaysTotal()
    {
        string[] hours = [.. Enumerable.Range(0, 24).Select(hour => string.Create(CultureInfo.InvariantCulture, $"h{hour:00}"))];
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("bikes.csv"))
            .Declare(schema => schema.Timestamp("dteday").Category("season", "weathersit").Boolean("holiday", "workingday").Number("temp", "atemp", "hum", "windspeed", "cnt").Number(hours))
            .Cyclical("dteday", Period.DayOfWeek)
            .Cyclical("dteday", Period.MonthOfYear)
            .SplitByTime("dteday", train: 0.70, validation: 0.15)
            .NormaliseRow(Norm.L1, hours)
            .EncodeCategories()
            .Normalise("temp", Scale.MidRange)
            .Normalise("atemp", Scale.MidRange)
            .Normalise("hum", Scale.MidRange)
            .Normalise("windspeed", Scale.MidRange)
            .Drop("dteday")
            .Distribution(hours, scaleBy: "cnt")
            .Drop("cnt")
            .Build()
            .Run();
        var network = new LayerStack(new Dense(
            Tensor.From(new Shape(21, 24), [.. Enumerable.Range(0, 21 * 24).Select(at => (((5 * (at / 24)) + (3 * (at % 24))) % 13 - 6) / 40f)]),
            Tensor.Zeros(new Shape(24))));
        var trained = TrainedNetwork.Of(network, new CrossEntropy(), prepared, seed: 1);

        var bikes = Assert.Single(trained.Predict(FirstLine(Repository.Data("bikes.csv"), keep: _ => true)).Answers);

        Assert.Equal(985, bikes.Sum(), 3);
        Assert.Equal([46.634762, 51.32066, 29.580357, 33.330074, 43.7691, 43.99994], bikes.Take(6).Select(count => Math.Round(count, 4)), new Close(1e-3));
    }

    [Fact]
    public void ATrainedNetworkSavedAndReadBack_PredictsTheSameNumbers_AndKeepsWhatItWasTrainedOn()
    {
        var prepared = Passengers(Repository.Data("titanic.csv"));
        var trained = new Sequential().Dense(8).Relu().Dense(1).Compile(new Adam(0.01), new BinaryCrossEntropy())
            .Fit(prepared, new FitOptions(seed: 20260929) { Epochs = 3 });
        var passengers = new InMemoryRowSource(["pclass", "sex", "age", "sibsp", "parch", "fare"], [["3", "male", "22.0", "1", "0", "7.25"], ["1", "female", "38", "1", "0", "71.2833"], ["2", "female", "", "0", "0", "13"]]);

        var json = trained.ToJson();
        var loaded = TrainedNetwork.FromJson(json, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn());

        Assert.Equal(trained.Predict(passengers).Answers, loaded.Predict(passengers).Answers);
        Assert.Equal(trained.TrainedOn.Features, loaded.TrainedOn.Features);
        Assert.Equal(trained.TrainedOn.TrainedBehind, loaded.TrainedOn.TrainedBehind);
        Assert.Equal(20260929, loaded.TrainedOn.Seed);
        Assert.Null(loaded.History);
        Assert.Null(loaded.Measures);
        Assert.Equal(json, loaded.ToJson());
        Assert.DoesNotContain("\"history\"", json, StringComparison.Ordinal);
    }

    [Fact]
    public void ACheckpoint_IsTheOneFileAndWhatTheRunNeedsToGoOn_AndGoesOnBitForBit()
    {
        var prepared = Passengers(Repository.Data("titanic.csv"));
        var straight = Small().Fit(prepared, new FitOptions(seed: 7) { Epochs = 4, BatchSize = 64 });
        var compiled = Small();
        var files = new List<string>();

        compiled.Fit(prepared, new FitOptions(seed: 7) { Epochs = 2, BatchSize = 64, Checkpoints = new Checkpoints(checkpoint => files.Add(CheckpointFile.Write(compiled, prepared, checkpoint))) });

        var resumed = CheckpointFile.Read(files[^1], NetworkCatalog.BuiltIn(), prepared);
        var goneOn = resumed.Compiled.Fit(prepared, new FitOptions(seed: 7) { Epochs = 4, BatchSize = 64, ResumeFrom = resumed.Checkpoint });
        var passenger = new InMemoryRowSource(["pclass", "sex", "age", "sibsp", "parch", "fare"], [["3", "male", "22.0", "1", "0", "7.25"]]);

        Assert.Equal(straight.History!.Epochs, goneOn.History!.Epochs);
        Assert.Equal(straight.Predict(passenger).Answers, goneOn.Predict(passenger).Answers);

        // A checkpoint is a whole trained network too: served as it stood at the end of its epoch.
        Assert.Equal(1, TrainedNetwork.FromJson(files[^1], NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn()).TrainedOn.Epoch);
    }

    [Fact]
    public void TheOneFileAndACheckpoint_EndEveryLineWithALineFeed_OnEverySystem()
    {
        // One line ending for the whole file, the one a pipeline's own file has on every system: the network's part is no
        // longer written with the line endings of the machine that wrote it beside a pipeline written with line feeds.
        var prepared = Passengers(Repository.Data("titanic.csv"));
        var compiled = Small();
        var files = new List<string>();

        var trained = compiled.Fit(prepared, new FitOptions(seed: 7) { Epochs = 1, Checkpoints = new Checkpoints(checkpoint => files.Add(CheckpointFile.Write(compiled, prepared, checkpoint))) });

        foreach (var file in (string[])[trained.ToJson(), Assert.Single(files)])
        {
            Assert.DoesNotContain('\r', file);
            Assert.True(file.Split('\n').Length > 100, "The file is indented, a key to a line.");
        }
    }

    [Fact]
    public void ACheckpointFile_GoneOnInBatchesOfSixteen_IsRefused_NamingTheBatchesItWasTakenIn()
    {
        var prepared = Passengers(Repository.Data("titanic.csv"));
        var compiled = Small();
        var files = new List<string>();
        compiled.Fit(prepared, new FitOptions(seed: 20260929) { Epochs = 2, Checkpoints = new Checkpoints(checkpoint => files.Add(CheckpointFile.Write(compiled, prepared, checkpoint))) });
        var resumed = CheckpointFile.Read(files[0], NetworkCatalog.BuiltIn(), prepared);

        var wrong = Assert.Throws<ArgumentException>(
            () => resumed.Compiled.Fit(prepared, new FitOptions(seed: 20260929) { Epochs = 4, BatchSize = 16, ResumeFrom = resumed.Checkpoint }));

        Assert.Equal(
            "The checkpoint was taken of a run in batches of 32, and going on in batches of 16 would take other rows into every step. (Parameter 'options')",
            wrong.Message);
    }

    [Fact]
    public void ACheckpointFile_GoneOnOnAnotherEngine_IsRefused_NamingTheEngineItWasTakenOnAndTheOneHanded_BeforeAnySlotIsPutBack()
    {
        var prepared = Passengers(Repository.Data("titanic.csv"));
        var compiled = Small();
        var files = new List<string>();
        compiled.Fit(prepared, new FitOptions(seed: 20260929) { Epochs = 2, Checkpoints = new Checkpoints(checkpoint => files.Add(CheckpointFile.Write(compiled, prepared, checkpoint))) });
        var resumed = CheckpointFile.Read(files[0], NetworkCatalog.BuiltIn(), prepared);
        var before = resumed.Compiled.Network.Slots().Select(named => named.Slot.Value).ToArray();

        var native = new NativeMemoryBackend();
        var light = new CpuBackend();

        var wrong = Assert.Throws<ArgumentException>(
            () => resumed.Compiled.Fit(prepared, new FitOptions(seed: 20260929) { Epochs = 4, Backend = native, ResumeFrom = resumed.Checkpoint }));

        Assert.Equal(
            $"The checkpoint was taken of a run on the engine 'cpu' {light.Version} on cpu, and going on under the engine 'nativememory' {native.Version} on cpu would round every step otherwise. (Parameter 'options')",
            wrong.Message);
        Assert.Equal(before, resumed.Compiled.Network.Slots().Select(named => named.Slot.Value));
        Assert.Equal(
            $$"""{"name":"cpu","version":"{{light.Version}}","device":"cpu"}""",
            JsonNode.Parse(files[0])!["training"]!["engine"]!.ToJsonString());
    }

    [Fact]
    public void ACheckpointFile_OfARunOnAnEngineOfSomebodyElsesThatNamesOnlyItself_RecordsItByName_AndGoesOnOnThatEngineAlone()
    {
        var prepared = Passengers(Repository.Data("titanic.csv"));
        var straight = Small().Fit(prepared, new FitOptions(seed: 7) { Epochs = 3, Backend = new CopyCountingBackend() });
        var compiled = Small();
        var files = new List<string>();
        compiled.Fit(prepared, new FitOptions(seed: 7)
        {
            Epochs = 2, Backend = new CopyCountingBackend(), Checkpoints = new Checkpoints(checkpoint => files.Add(CheckpointFile.Write(compiled, prepared, checkpoint))),
        });

        var refused = CheckpointFile.Read(files[^1], NetworkCatalog.BuiltIn(), prepared);
        var wrong = Assert.Throws<ArgumentException>(() => refused.Compiled.Fit(prepared, new FitOptions(seed: 7) { Epochs = 3, ResumeFrom = refused.Checkpoint }));
        var resumed = CheckpointFile.Read(files[^1], NetworkCatalog.BuiltIn(), prepared);
        var goneOn = resumed.Compiled.Fit(prepared, new FitOptions(seed: 7) { Epochs = 3, Backend = new CopyCountingBackend(), ResumeFrom = resumed.Checkpoint });

        Assert.Equal("""{"name":"copycounting"}""", JsonNode.Parse(files[^1])!["training"]!["engine"]!.ToJsonString());
        Assert.Contains(
            $"a run on the engine 'copycounting', and going on under the engine 'cpu' {new CpuBackend().Version} on cpu would round every step otherwise.",
            wrong.Message,
            StringComparison.Ordinal);
        Assert.Equal(straight.History!.Epochs, goneOn.History!.Epochs);
    }

    [Fact]
    public void ANetworkBesideAnotherFitOfItsPipeline_IsRefused_WhereItIsReadAndWhereItGoesOn()
    {
        var path = Path.Join(Directory.CreateTempSubdirectory("deepsharp-bridge-").FullName, "titanic.csv");
        File.Copy(Repository.Data("titanic.csv"), path);
        var fitted = Passengers(path);
        var compiled = Small();
        var files = new List<string>();
        var trained = compiled.Fit(fitted, new FitOptions(seed: 7) { Epochs = 1, Checkpoints = new Checkpoints(checkpoint => files.Add(CheckpointFile.Write(compiled, fitted, checkpoint))) });

        // The same declaration fitted again once the file has grown by a few passengers: every name is as it was, and the
        // numbers the fit learned are not.
        string[] grown = [.. File.ReadLines(path).Skip(1).Take(40)];
        File.AppendAllLines(path, grown);
        var refitted = Passengers(path);

        Assert.Equal(fitted.Batch(Part.Train, Needs.OneScale).FeatureNames, refitted.Batch(Part.Train, Needs.OneScale).FeatureNames);
        Assert.NotEqual(fitted.ToJson(), refitted.ToJson());

        var swapped = JsonNode.Parse(trained.ToJson())!;
        swapped["pipeline"] = JsonNode.Parse(refitted.ToJson());

        Assert.Contains("another fit", Assert.Throws<NetworkFileException>(() => TrainedNetwork.FromJson(swapped.ToJsonString(), NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn())).Message, StringComparison.Ordinal);
        Assert.Contains("another fit", Assert.Throws<ArgumentException>(() => CheckpointFile.Read(files[^1], NetworkCatalog.BuiltIn(), refitted)).Message, StringComparison.Ordinal);
        Assert.NotNull(CheckpointFile.Read(files[^1], NetworkCatalog.BuiltIn(), fitted).Checkpoint);

        // The file is read whole before it is held to the pipeline handed over: one that is itself wrong is refused for that,
        // at its place, whichever pipeline it is handed.
        var broken = JsonNode.Parse(files[^1])!;
        broken["training"]!["seed"] = "seven";

        var refused = Assert.Throws<NetworkFileException>(() => CheckpointFile.Read(broken.ToJsonString(), NetworkCatalog.BuiltIn(), refitted));

        Assert.Equal("The seed is the whole number the run was worked out from, under 'seed'.", Assert.Single(refused.Faults).Message);
    }

    [Fact]
    public void AFileWhoseNetworkWasTrainedOnOtherFeatures_OrForAnotherOutput_IsRefused()
    {
        var prepared = Passengers(Repository.Data("titanic.csv"));
        var file = JsonNode.Parse(Small().Fit(prepared, new FitOptions(seed: 7)).ToJson())!;
        var passenger = new InMemoryRowSource(["pclass", "sex", "age", "sibsp", "parch", "fare"], [["3", "male", "22.0", "1", "0", "7.25"]]);

        var renamed = file.DeepClone();
        renamed["network"]!["trainedOn"]!["features"]![11] = "sex_man";
        var readable = TrainedNetwork.FromJson(renamed.ToJsonString(), NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn());

        Assert.Contains("sex_man", Assert.Throws<InvalidOperationException>(() => readable.Predict(passenger)).Message, StringComparison.Ordinal);

        var labelled = file.DeepClone();
        labelled["network"]!["trainedOn"]!["output"] = "labels";
        var answered = file.DeepClone();
        answered["network"]!["trainedOn"]!["answers"]![0] = "died";

        Assert.Contains("'labels'", Assert.Throws<NetworkFileException>(() => TrainedNetwork.FromJson(labelled.ToJsonString(), NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn())).Message, StringComparison.Ordinal);
        Assert.Contains("died", Assert.Throws<NetworkFileException>(() => TrainedNetwork.FromJson(answered.ToJsonString(), NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn())).Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("""{"version": 2, "network": {}, "pipeline": {}}""", "version 2")]
    [InlineData("""{"version": 1, "network": {}, "pipeline": {}, "weights": {}}""", "'weights'")]
    [InlineData("""{"version": 1, "network": {}, "pipeline": {}, "weights": [1]}""", "'weights'")]
    [InlineData("""{"version": 1, "network": {}, "pipeline": {}, "pipeline": {}}""", "'pipeline' is written twice")]
    [InlineData("""[]""", "one JSON object")]
    public void AFileOfNoVersionThisReads_OrHoldingWhatNoPartIs_IsRefusedWhole(string json, string says)
    {
        Assert.Contains(says, Assert.Throws<NetworkFileException>(() => TrainedNetwork.FromJson(json, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn())).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EarlyStoppingThatRestoresTheBestEpoch_SavesTheBestEpoch_AndOneThatDoesNot_TheLast()
    {
        var prepared = Passengers(Repository.Data("titanic.csv"));
        var restored = Small().Fit(prepared, new FitOptions(seed: 7) { Epochs = 6, EarlyStopping = new EarlyStopping { Patience = 6, RestoreBest = true } });
        var kept = Small().Fit(prepared, new FitOptions(seed: 7) { Epochs = 6, EarlyStopping = new EarlyStopping { Patience = 6 } });

        Assert.Equal(restored.History!.BestEpoch, restored.TrainedOn.Epoch);
        Assert.Equal(kept.History!.Epochs[^1].Number, kept.TrainedOn.Epoch);
    }

    [Fact]
    public void NoRowsServed_AreNoPredictions()
    {
        var trained = Small().Fit(Passengers(Repository.Data("titanic.csv")), new FitOptions(seed: 7));

        var none = trained.Predict(new InMemoryRowSource(["pclass", "sex", "age", "sibsp", "parch", "fare"], []));

        Assert.Empty(none.Answers);
        Assert.Empty(none.HandedInAt);
        Assert.Throws<ArgumentNullException>(() => trained.Predict(null!));
    }

    [Fact]
    public void APipelineBesideTheNetworkThatNamesNoAnswer_IsRefused()
    {
        var file = JsonNode.Parse(Small().Fit(Passengers(Repository.Data("titanic.csv")), new FitOptions(seed: 7)).ToJson())!;
        var steps = file["pipeline"]!["declaration"]!.AsArray();
        steps.Remove(steps.Single(step => (string?)step!["step"] == "target"));

        var wrong = Assert.Throws<NetworkFileException>(() => TrainedNetwork.FromJson(file.ToJsonString(), NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn()));

        Assert.Contains("'target'", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACheckpointWhoseNetworkNamesNothingItWasTrainedOn_IsRefused()
    {
        var prepared = Passengers(Repository.Data("titanic.csv"));
        var compiled = Small();
        var files = new List<string>();
        compiled.Fit(prepared, new FitOptions(seed: 7) { Checkpoints = new Checkpoints(checkpoint => files.Add(CheckpointFile.Write(compiled, prepared, checkpoint))) });
        var file = JsonNode.Parse(files[0])!;
        file["network"]!.AsObject().Remove("trainedOn");

        Assert.Contains("trained on", Assert.Throws<NetworkFileException>(() => CheckpointFile.Read(file.ToJsonString(), NetworkCatalog.BuiltIn(), prepared)).Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentNullException>(() => CheckpointFile.Read(null!, NetworkCatalog.BuiltIn(), prepared));
        Assert.Throws<ArgumentNullException>(() => CheckpointFile.Read(files[0], null!, prepared));
        Assert.Throws<ArgumentNullException>(() => CheckpointFile.Read(files[0], NetworkCatalog.BuiltIn(), null!));
        Assert.Throws<ArgumentNullException>(() => CheckpointFile.Write(null!, prepared, null!));
        Assert.Throws<ArgumentNullException>(() => CheckpointFile.Write(compiled, null!, null!));
        Assert.Throws<ArgumentNullException>(() => CheckpointFile.Write(compiled, prepared, null!));
    }

    [Theory]
    [InlineData("""{"version": 1, "network": """, "stops being JSON")]
    [InlineData("""{"network": {}, "pipeline": {}}""", "names the whole number of the version")]
    [InlineData("""{"version": "one", "network": {}, "pipeline": {}}""", "names the whole number of the version")]
    [InlineData("""{"version": 1.5, "network": {}, "pipeline": {}}""", "names the whole number of the version")]
    public void AFileThatIsNotJson_OrNamesNoVersion_IsRefusedWhere(string json, string says)
    {
        Assert.Contains(says, Assert.Throws<NetworkFileException>(() => TrainedNetwork.FromJson(json, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn())).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFileWhoseNetworkNamesNothingItWasTrainedOn_IsNoTrainedNetwork()
    {
        var prepared = Passengers(Repository.Data("titanic.csv"));
        var file = JsonNode.Parse(Small().Fit(prepared, new FitOptions(seed: 7)).ToJson())!;
        file["network"]!.AsObject().Remove("trainedOn");

        Assert.Contains("trained on", Assert.Throws<NetworkFileException>(() => TrainedNetwork.FromJson(file.ToJsonString(), NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn())).Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentNullException>(() => TrainedNetwork.FromJson(null!, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn()));
        Assert.Throws<ArgumentNullException>(() => TrainedNetwork.FromJson("{}", null!, StepCatalog.BuiltIn()));
        Assert.Throws<ArgumentNullException>(() => TrainedNetwork.FromJson("{}", NetworkCatalog.BuiltIn(), null!));
    }

    [Theory]
    [InlineData(Needs.NoScale, "step 6, 'normalise'; step 7, 'normalise'; step 8, 'normalise'; step 9, 'normalise'")]
    [InlineData(Needs.Categories, "step 5, 'encode.categories'; step 6, 'normalise'")]
    public void ANetwork_IsRefusedARunMadeForALearnerThatDoesWithoutItsScalesOrItsCategories_NamingEachStep(Needs needs, string steps)
    {
        var run = WikiTitanic.In(WikiTitanic.DataFolder).RunFor(needs);

        var refused = Assert.Throws<InvalidOperationException>(() => Small().Fit(run, new FitOptions(seed: 3) { Epochs = 1 }));

        Assert.Contains(steps, refused.Message, StringComparison.Ordinal);
        Assert.Contains("Needs.OneScale", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACheckpoint_DoesNotGoOnBehindARunThatLeftOutItsScales()
    {
        // The 0.4.0 checkpoint goes on behind the run of every step (FilesWrittenBy040Tests); a run for a learner indifferent
        // to scale is another fit, whose rows reach a network unscaled.
        var checkpoint = File.ReadAllText(Path.Join(AppContext.BaseDirectory, "Learners", "Fixtures", "titanic-0.4.0.checkpoint.json"));
        var noScale = WikiTitanic.In(WikiTitanic.DataFolder).RunFor(Needs.NoScale);

        var refused = Assert.Throws<ArgumentException>(() => CheckpointFile.Read(checkpoint, NetworkCatalog.BuiltIn(), noScale));

        Assert.Contains("another fit", refused.Message, StringComparison.Ordinal);
        Assert.Equal(4, CheckpointFile.Read(checkpoint, NetworkCatalog.BuiltIn(), WikiTitanic.In(WikiTitanic.DataFolder).Run()).Checkpoint.Epochs);
    }

    [Fact]
    public void ANetworkInKerasWords_PaddedAsSameAndEndingInItsLossesSigmoid_IsTrainedBehindAPipeline_AndServedFromItsFileAlike()
    {
        // The passenger's fourteen features as an image of two rows by seven, through a window of three by two that walks two
        // places at a time and pads as 'same': one row and one column of nothing after the image. The last sigmoid is the
        // loss's, so the file holds none, and the network read back from it answers as the one trained.
        var prepared = WikiTitanic.In(WikiTitanic.DataFolder).Run();
        var passenger = new InMemoryRowSource(["pclass", "sex", "age", "sibsp", "parch", "fare"], [["3", "male", "22", "1", "0", "7.25"]]);
        var trained = new Sequential()
            .Reshape(new Shape(2, 7, 1))
            .Conv2D(3, new Window(3, 2) { Stride = 2, PaddingMode = PaddingMode.Same })
            .Relu()
            .Flatten()
            .Dense(1)
            .Sigmoid()
            .Compile(new Adam(0.01), new BinaryCrossEntropy())
            .Fit(prepared, new FitOptions(seed: 3) { Epochs = 2 });

        var file = trained.ToJson();
        var read = TrainedNetwork.FromJson(file, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn());

        Assert.Contains("\"padding\": \"same\"", file, StringComparison.Ordinal);
        Assert.DoesNotContain("\"sigmoid\"", file, StringComparison.Ordinal);
        Assert.Equal(new Window(3, 2) { Stride = 2, PaddingMode = PaddingMode.Same }, Assert.IsType<Conv2D>(Assert.IsType<LayerStack>(read.Network).Layers[1]).Window);
        Assert.Equal(trained.Predict(passenger).Answers, read.Predict(passenger).Answers);
    }

    // Titanic as the walks prepared it: fourteen features on one scale, and whether the passenger survived.
    private static PreparedData Passengers(string path) =>
        Pdd.Create()
            .ReadCsv(path)
            .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("pclass", "sex").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            .FillMissing("age", With.Median)
            .EncodeCategories()
            .Normalise("age", Scale.MidRange)
            .Normalise("fare", Scale.MidRange)
            .Normalise("sibsp", Scale.MidRange)
            .Normalise("parch", Scale.MidRange)
            .Target("survived")
            .Build()
            .Run();

    // The first row a file holds, with only the columns asked for: a row as it would be served.
    private static IRowSource FirstLine(string path, Func<string, bool> keep)
    {
        var lines = File.ReadLines(path).Take(2).Select(line => line.Split(',')).ToArray();
        int[] kept = [.. Enumerable.Range(0, lines[0].Length).Where(at => keep(lines[0][at]))];

        return new InMemoryRowSource([.. kept.Select(at => lines[0][at])], [[.. kept.Select(at => (string?)lines[1][at])]]);
    }

    private static PartPredictions Predicted(TrainedNetwork trained, Batch batch)
    {
        var features = Tensor.From(new Shape(batch.RowCount, batch.Width), [.. batch.Features.SelectMany(row => row.Select(value => (float)value))]);
        var outputs = trained.Loss.Predictions(trained.Network.Forward(features, Pass.Evaluation(new CpuBackend())), new CpuBackend()).Values.ToArray();
        var width = outputs.Length / batch.RowCount;

        return new PartPredictions(batch, [.. Enumerable.Range(0, batch.RowCount).Select(row => outputs.Skip(row * width).Take(width).Select(value => (double)value).ToArray())]);
    }

    // Numbers equal within a tolerance.
    private sealed class Close(double within) : IEqualityComparer<double>
    {
        public bool Equals(double x, double y) => Math.Abs(x - y) <= within;

        public int GetHashCode(double obj) => 0;
    }
}
