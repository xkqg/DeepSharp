// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;
using DeepSharp.Tests.Backends.Parts;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// The engine is the caller's, handed in and owned by nothing. The one a run is handed trains the network, judges it by
/// the validation rows and measures it as the pipeline's report declares; a network serves on the engine its caller hands
/// it, trained here or read from its file alike, and on a light one of its own unless said; and the file names no engine.
/// </summary>
public class NetworkEngineTests
{
    [Fact]
    public void TheEngineARunIsHanded_MeasuresItsReport_EveryPassOfIt()
    {
        var compiled = Small();
        var reported = Twenty(reported: true);
        var measured = new CountingBackend();
        var unmeasured = new CountingBackend();

        compiled.Fit(reported, new FitOptions(seed: 3) { Epochs = 2, Backend = measured });
        Small().Fit(Twenty(reported: false), new FitOptions(seed: 3) { Epochs = 2, Backend = unmeasured });

        // The same run without a report trains and judges on the engine it is handed; what the report adds to it is one
        // evaluation pass over each part it names, through the loss's activation, and every operation of those reached the
        // engine the run was handed.
        Assert.True(unmeasured.Operations > 0);
        Assert.Equal(
            OnePassOver(compiled.Network, compiled.Loss, reported.Batch(Part.Train, Needs.OneScale).Features)
            + OnePassOver(compiled.Network, compiled.Loss, reported.Batch(Part.Test, Needs.OneScale).Features),
            measured.Operations - unmeasured.Operations);
    }

    [Fact]
    public void TheReportsPartsAreMeasuredAChunkAtATime_AtTheRunsBatchSize_NotOneWholePassRegardlessOfItsSize()
    {
        var compiled = Small();
        var reported = Twenty(reported: true);
        var measured = new CountingBackend();
        var unmeasured = new CountingBackend();

        compiled.Fit(reported, new FitOptions(seed: 3) { Epochs = 2, BatchSize = 3, Backend = measured });
        Small().Fit(Twenty(reported: false), new FitOptions(seed: 3) { Epochs = 2, BatchSize = 3, Backend = unmeasured });

        var perPass = OnePassOver(compiled.Network, compiled.Loss, reported.Batch(Part.Train, Needs.OneScale).Features);

        // Ten training rows chunked at three rows a chunk: four passes (3, 3, 3, 1). Five test rows: two passes (3, 2).
        // At the run's default batch size of thirty-two both parts would be one pass each, as
        // TheEngineARunIsHanded_MeasuresItsReport_EveryPassOfIt pins; a smaller batch size chunks the report too.
        Assert.Equal((4 + 2) * perPass, measured.Operations - unmeasured.Operations);
    }

    [Fact]
    public void AnEngine_ChangesWhereTheArithmeticRuns_NotWhatItComesTo_NorTheFile()
    {
        var reported = Twenty(reported: true);

        var counted = Small().Fit(reported, new FitOptions(seed: 3) { Epochs = 2, Backend = new CountingBackend() });
        var light = Small().Fit(reported, new FitOptions(seed: 3) { Epochs = 2 });

        Assert.Equal(light.History!.Epochs, counted.History!.Epochs);
        Assert.Equal(light.Measures!.Parts.SelectMany(part => part.Values), counted.Measures!.Parts.SelectMany(part => part.Values));
        Assert.Equal(light.ToJson(), counted.ToJson());
    }

    [Fact]
    public void ANetwork_ServesOnTheEngineItsCallerHandsIt_OnePassOfIt_AnsweringAsOnTheLightOne()
    {
        var trained = Small().Fit(Passengers(), new FitOptions(seed: 7));
        var engine = new CountingBackend();

        var served = trained.Predict(Travellers(), engine);

        Assert.Equal(OnePassOver(trained.Network, trained.Loss, trained.Prepared.Served(Travellers(), Needs.OneScale).Features), engine.Operations);
        Assert.Equal(trained.Predict(Travellers()).Answers, served.Answers);
        Assert.Equal([0, 1, 2], served.HandedInAt);
    }

    [Fact]
    public void AHundredServedRows_AreAnsweredInFourPassesOfThirtyTwo_BitEqualToOneWholePassOnCpuBackend()
    {
        var trained = Small().Fit(Passengers(), new FitOptions(seed: 7));
        var travellers = OneHundredTravellers();
        var counted = new CountingBackend();

        var served = trained.Predict(travellers, counted);

        var servedFeatures = trained.Prepared.Served(travellers, Needs.OneScale).Features;
        var perPass = OnePassOver(trained.Network, trained.Loss, servedFeatures);

        // A hundred rows chunked at thirty-two: four passes (32, 32, 32, 4) — not the one pass a hundred rows made before
        // serving chunked, however many rows were handed in.
        Assert.Equal(4 * perPass, counted.Operations);

        var whole = trained.Network.Predict(
            Tensor.From(new Shape(servedFeatures.Count, servedFeatures[0].Length), [.. servedFeatures.SelectMany(row => row.Select(value => (float)value))]),
            trained.Loss,
            new CpuBackend());

        // Chunked or not, an evaluation pass on CpuBackend answers the same rows the same way, to the bit.
        Assert.Equal(whole.Values.ToArray(), served.Answers.Select(row => (float)Assert.Single(row)).ToArray());
    }

    [Fact]
    public void ANetworkReadFromItsFile_ServesOnTheEngineItsCallerHandsIt()
    {
        var trained = Small().Fit(Passengers(), new FitOptions(seed: 7));
        var loaded = TrainedNetwork.FromJson(trained.ToJson(), NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn());
        var engine = new CountingBackend();

        var served = loaded.Predict(Travellers(), engine);

        Assert.Equal(OnePassOver(loaded.Network, loaded.Loss, loaded.Prepared.Served(Travellers(), Needs.OneScale).Features), engine.Operations);
        Assert.Equal(trained.Predict(Travellers()).Answers, served.Answers);
    }

    [Fact]
    public void NoRowsServed_RunNothingOnTheEngine_AndServingIsRefusedNoEngineOrNoRows()
    {
        var trained = Small().Fit(Passengers(), new FitOptions(seed: 7));
        var engine = new CountingBackend();

        var none = trained.Predict(new InMemoryRowSource(["pclass", "sex", "age", "sibsp", "parch", "fare"], []), engine);

        Assert.Empty(none.Answers);
        Assert.Equal(0, engine.Operations);
        Assert.Equal("backend", Assert.Throws<ArgumentNullException>(() => trained.Predict(Travellers(), null!)).ParamName);
        Assert.Equal("rows", Assert.Throws<ArgumentNullException>(() => trained.Predict(null!, engine)).ParamName);
    }

    [Fact]
    public void OptionsThatNameNoEngine_HoldTheLightOne_WhichTheRunAndItsReportAreHanded()
    {
        // Which engine a run takes when none is named is said in one place, the options: they hold the light one from the
        // start, the loop and the report both take the engine the options hold, and nothing handed in brings the light
        // one back.
        var unsaid = new FitOptions(seed: 3);
        var counting = new CountingBackend();

        Assert.IsType<CpuBackend>(unsaid.Backend);
        Assert.Same(unsaid.Backend, unsaid.Backend);
        Assert.IsType<CpuBackend>(new FitOptions(seed: 3) { Backend = null }.Backend);
        Assert.Same(counting, new FitOptions(seed: 3) { Backend = counting }.Backend);

        var reported = Twenty(reported: true);
        var light = Small().Fit(reported, unsaid);
        var named = Small().Fit(reported, new FitOptions(seed: 3) { Backend = unsaid.Backend });

        Assert.Equal(named.History!.Epochs, light.History!.Epochs);
        Assert.Equal(named.Measures!.Parts.SelectMany(part => part.Values), light.Measures!.Parts.SelectMany(part => part.Values));
        Assert.Equal(named.ToJson(), light.ToJson());
    }

    [Fact]
    public void ANetworkWithItsLoss_AnswersRowsByOneEvaluationPass_TheOneEveryDoorAnswersBy()
    {
        // The one evaluation a network here answers rows by — an evaluation pass on the engine handed in, through the loss's
        // activation — written once, in the core: a compiled network predicts by it, and a trained one serves by it.
        var trained = Small().Fit(Passengers(), new FitOptions(seed: 7));
        var served = trained.Prepared.Served(Travellers(), Needs.OneScale).Features;
        var features = Tensor.From(new Shape(served.Count, served[0].Length), [.. served.SelectMany(row => row.Select(value => (float)value))]);
        var engine = new CountingBackend();

        var answered = trained.Network.Predict(features, trained.Loss, engine);

        Assert.Equal(OnePassOver(trained.Network, trained.Loss, served), engine.Operations);
        Assert.Equal(trained.Network.Compile(new Adam(), trained.Loss).Predict(features, new CpuBackend()).Values.ToArray(), answered.Values.ToArray());
        Assert.Equal(answered.Values.ToArray().Select(value => (double)value), trained.Predict(Travellers(), engine).Answers.Select(row => Assert.Single(row)));
        Assert.Equal("features", Assert.Throws<ArgumentNullException>(() => trained.Network.Predict(null!, trained.Loss, engine)).ParamName);
        Assert.Equal("loss", Assert.Throws<ArgumentNullException>(() => trained.Network.Predict(features, null!, engine)).ParamName);
        Assert.Equal("backend", Assert.Throws<ArgumentNullException>(() => trained.Network.Predict(features, trained.Loss, null!)).ParamName);
    }

    [Fact]
    public void AnEngineOfSomebodyElses_KeepsWhatARunMakesOnItsOwnStorage_EachCopiedOutOnceAtMost_AndTheRunComesToWhatTheLightOneDoes()
    {
        var reported = Twenty(reported: true);
        var engine = new CopyCountingBackend();

        var trained = Small().Fit(reported, new FitOptions(seed: 3) { Epochs = 2, Backend = engine });
        var file = trained.ToJson();
        var light = Small().Fit(reported, new FitOptions(seed: 3) { Epochs = 2 });

        // Every number the network learned stands where the engine keeps it, and the file — which reads each of them twice,
        // once to check it and once to write it — copied each out once; nothing the run made was copied out more than once,
        // however often the loop, the report and the file read it.
        var slots = trained.Network.Slots().Select(named => named.Slot.Value.Storage).ToArray();
        Assert.NotEmpty(slots);
        Assert.All(slots, storage => Assert.Equal(1, Assert.IsType<CopyCountingStorage>(storage).Copies));
        Assert.All(engine.Made, storage => Assert.InRange(storage.Copies, 0, 1));
        Assert.True(engine.TakenIn > 0, "The rows are handed to the engine from this machine's memory, and it takes them in.");

        // Where the arithmetic ran changes nothing of what it came to.
        Assert.Equal(light.History!.Epochs, trained.History!.Epochs);
        Assert.Equal(light.Measures!.Parts.SelectMany(part => part.Values), trained.Measures!.Parts.SelectMany(part => part.Values));
        Assert.Equal(light.ToJson(), file);
    }

    [Fact]
    public void ANetwork_ServesOnAnEngineOfSomebodyElses_ReadingItsAnswersBackOnce()
    {
        var trained = Small().Fit(Passengers(), new FitOptions(seed: 7));
        var engine = new CopyCountingBackend();

        var served = trained.Predict(Travellers(), engine);

        Assert.Equal(trained.Predict(Travellers()).Answers, served.Answers);
        Assert.Equal(1, engine.Made[^1].Copies);
        Assert.All(engine.Made, storage => Assert.InRange(storage.Copies, 0, 1));
    }

    private static CompiledNetwork Small() => new Sequential().Dense(4).Tanh().Dense(1).Compile(new Adam(0.05), new BinaryCrossEntropy());

    // How many operations one evaluation pass over the rows makes, through the loss's activation: what measuring or serving
    // them costs an engine.
    private static int OnePassOver(Network network, Loss loss, IReadOnlyList<double[]> rows)
    {
        var counted = new CountingBackend();
        var features = Tensor.From(new Shape(rows.Count, rows[0].Length), [.. rows.SelectMany(row => row.Select(value => (float)value))]);

        loss.Predictions(network.Forward(features, Pass.Evaluation(counted)), counted);

        return counted.Operations;
    }

    // Twenty rows t = 1…20 of x = sin t, whose answer is whether x is above nought: ten train, five judge, five test; and,
    // when reported, a report measuring the network on the training rows and the test rows.
    private static PreparedData Twenty(bool reported)
    {
        var declared = Pdd.Create()
            .Read(
                CsvRowSource.FromText("t,x,y\n" + string.Join('\n', Enumerable.Range(1, 20).Select(t =>
                    string.Create(CultureInfo.InvariantCulture, $"{t},{Math.Sin(t)},{(Math.Sin(t) > 0 ? 1 : 0)}"))) + "\n"),
                "twenty rows")
            .Declare(schema => schema.Integer("t", "y").Number("x"))
            .SplitByTime("t", 0.50, 0.25)
            .Normalise("x", Scale.MidRange)
            .Drop("t")
            .Target("y");

        return (reported ? declared.Report(report => report.Measure(Metric.Accuracy, Metric.Rmse).On(Part.Train, Part.Test).As(Shown.Numbers)) : declared)
            .Build()
            .Run();
    }

    // Titanic as the walks prepared it: fourteen features on one scale, and whether the passenger survived.
    private static PreparedData Passengers() =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
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

    // Three passengers served after training, the third with no age given.
    private static InMemoryRowSource Travellers() =>
        new(
            ["pclass", "sex", "age", "sibsp", "parch", "fare"],
            [["3", "male", "22.0", "1", "0", "7.25"], ["1", "female", "38", "1", "0", "71.2833"], ["2", "female", "", "0", "0", "13"]]);

    // A hundred passengers served after training, every one a class and a sex the training rows held, each row's age,
    // siblings, parents and fare spread out so no two are alike.
    private static InMemoryRowSource OneHundredTravellers() =>
        new(
            ["pclass", "sex", "age", "sibsp", "parch", "fare"],
            [.. Enumerable.Range(0, 100).Select(row => (IReadOnlyList<string?>)
            [
                ((row % 3) + 1).ToString(CultureInfo.InvariantCulture),
                row % 2 == 0 ? "male" : "female",
                (20 + (row % 40)).ToString(CultureInfo.InvariantCulture),
                (row % 3).ToString(CultureInfo.InvariantCulture),
                (row % 2).ToString(CultureInfo.InvariantCulture),
                (10 + (row % 90)).ToString(CultureInfo.InvariantCulture),
            ])]);

    // An engine of the test's own: the light engine's arithmetic, with every operation it is handed counted.
    private sealed class CountingBackend : ITensorBackend
    {
        private readonly CpuBackend _inner = new();

        public int Operations { get; private set; }

        public string Name => "counting";

        public Tensor Add(Tensor left, Tensor right) => Counted(_inner.Add(left, right));

        public Tensor Subtract(Tensor left, Tensor right) => Counted(_inner.Subtract(left, right));

        public Tensor Multiply(Tensor left, Tensor right) => Counted(_inner.Multiply(left, right));

        public Tensor MatMul(Tensor left, Tensor right) => Counted(_inner.MatMul(left, right));

        public Tensor Transpose(Tensor matrix) => Counted(_inner.Transpose(matrix));

        public Tensor AddRow(Tensor matrix, Tensor row) => Counted(_inner.AddRow(matrix, row));

        public Tensor SumRows(Tensor matrix) => Counted(_inner.SumRows(matrix));

        public Tensor Mean(Tensor values) => Counted(_inner.Mean(values));

        public Tensor Scale(Tensor values, Tensor factor) => Counted(_inner.Scale(values, factor));

        public Tensor Fill(Shape shape, float value) => Counted(_inner.Fill(shape, value));

        public Tensor Relu(Tensor values) => Counted(_inner.Relu(values));

        public Tensor Positive(Tensor values) => Counted(_inner.Positive(values));

        public Tensor Tanh(Tensor values) => Counted(_inner.Tanh(values));

        public Tensor Sigmoid(Tensor values) => Counted(_inner.Sigmoid(values));

        public Tensor Exp(Tensor values) => Counted(_inner.Exp(values));

        public Tensor Log(Tensor values) => Counted(_inner.Log(values));

        public Tensor Sqrt(Tensor values) => Counted(_inner.Sqrt(values));

        public Tensor Softplus(Tensor values) => Counted(_inner.Softplus(values));

        public Tensor Divide(Tensor left, Tensor right) => Counted(_inner.Divide(left, right));

        public Tensor LogSoftmax(Tensor matrix) => Counted(_inner.LogSoftmax(matrix));

        public Tensor Reshape(Tensor values, Shape shape) => Counted(_inner.Reshape(values, shape));

        public Tensor Unfold(Tensor images, Window window) => Counted(_inner.Unfold(images, window));

        public Tensor Fold(Tensor patches, Shape images, Window window) => Counted(_inner.Fold(patches, images, window));

        private Tensor Counted(Tensor result)
        {
            Operations++;

            return result;
        }
    }
}
