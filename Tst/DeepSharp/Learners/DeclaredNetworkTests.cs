// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tests.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// The network declared in the pipeline that trains it: the same course from the rows to a validated model, written down
/// once and replayed. A declaration says which layers, which way the error is carried back, when the run stops and on
/// which engine it runs — and training it gives the network the same words written as code give, number for number,
/// because the declaration is lowered onto that model and nothing else runs.
/// </summary>
public class DeclaredNetworkTests
{
    private static FittingBuilder Passengers() => Pdd.Create()
        .ReadCsv(Repository.Data("titanic.csv"))
        .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("sex").Optional("age", ColumnKind.Number).Number("fare"))
        .SplitStratified("survived", 0.70, 0.15)
        .FillMissing("age", With.Median)
        .EncodeCategories()
        .Normalise("age", Scale.MidRange)
        .Normalise("fare", Scale.MidRange)
        .Normalise("sibsp", Scale.MidRange)
        .Normalise("parch", Scale.MidRange)
        .Target("survived");

    private static Pipeline Declared() => Passengers()
        .WithTorch(network => network
            .Dense(4).Relu().Dense(1)
            .Adam(0.01)
            .BinaryCrossEntropy()
            .Run(seed: 20260929, epochs: 3))
        .Build();

    [Fact]
    public void ADeclaredNetwork_IsTheNetworkTheSameWordsWrittenAsCodeGive_NumberForNumber()
    {
        // The one test that holds the declared door to the written one: same layers, same optimizer, same loss, same run,
        // so the same numbers come out. A declaration that lowered onto anything else would show here at the first weight.
        var declared = Declared().Train();
        var written = new Sequential().Dense(4).Relu().Dense(1)
            .Compile(new Adam(0.01), new BinaryCrossEntropy())
            .Fit(Passengers().Build().RunFor(Needs.OneScale), new FitOptions(20260929) { Epochs = 3 });

        // The network itself, number for number. The pipelines differ by the one step that declares the network, so the
        // digest each network records of the fit it was trained behind differs with them — which is the point of it.
        Assert.Equal(Network(written), Network(declared));
        Assert.Equal(written.History!.Epochs.Select(epoch => epoch.Loss), declared.History!.Epochs.Select(epoch => epoch.Loss));
    }

    // The network itself as its file writes it: its layers and every number in them. What it records of the pipeline it
    // was trained behind is left out here, because that is exactly what differs between the two doors.
    private static IEnumerable<string> Network(TrainedNetwork trained)
    {
        var network = System.Text.Json.JsonDocument.Parse(trained.ToJson()).RootElement.GetProperty("network");

        return new[] { "layers", "parameters", "state" }
            .Where(part => network.TryGetProperty(part, out _))
            .Select(part => network.GetProperty(part).GetRawText());
    }

    [Fact]
    public void TheDeclarationCarriesTheNetworkAndTheRun_SoTheFileSaysWhatWasTrained()
    {
        var step = Assert.IsType<LearnNetworkStep>(Declared().Declaration.Learner);

        Assert.Equal("learn.network", step.Verb);
        Assert.Equal(["dense", "relu", "dense"], step.Layers.Select(layer => layer.Kind));
        Assert.Equal(4, step.Layers[0].Whole("units"));
        Assert.Equal("adam", step.Optimizer.Kind);
        Assert.Equal(0.01, step.Optimizer.Number("rate"));
        Assert.Equal("binaryCrossEntropy", step.Loss.Kind);
        Assert.Equal("never", step.Stopping.Kind);
        Assert.Equal(Needs.OneScale, step.Needs);
        Assert.Equal(3, step.Epochs);
        Assert.Equal(20260929, step.Seed);
    }

    [Fact]
    public void ADeclaredNetwork_GoesThroughTheFileAndComesBackTheSame()
    {
        var declaration = Declared().Declaration;
        var again = PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn().WithNetworks());

        Assert.Equal(declaration, again);
        Assert.Equal(declaration.ToJson(), again.ToJson());
    }

    [Fact]
    public void TheTwoVocabularies_WriteTheNumbersEachLibraryLeavesUnsaid()
    {
        // TensorFlow's words and PyTorch's lower onto one model; where a caller says nothing, each library's own number
        // is written down, so the file says which run was asked for rather than which library asked for it.
        var keras = Assert.IsType<LearnNetworkStep>(Passengers().WithTensorflow(network => network.Dense(1).Adam().MeanSquaredError().Run(1)).Build().Declaration.Learner);
        var torch = Assert.IsType<LearnNetworkStep>(Passengers().WithTorch(network => network.Dense(1).Adam().MeanSquaredError().Run(1)).Build().Declaration.Learner);

        Assert.Equal(1e-7, keras.Optimizer.Number("epsilon"));
        Assert.Equal(1e-8, torch.Optimizer.Number("epsilon"));
        Assert.Equal(0.001, keras.Optimizer.Number("rate"));
        Assert.Equal(0.001, torch.Optimizer.Number("rate"));
    }

    [Fact]
    public void TheEngineIsNamedInTheDeclaration_AndTheRunHappensOnTheOneThatName_IsGivenTo()
    {
        // The declaration names an engine; which engine that name stands for is the application's to say, so a model
        // declared once runs on the light engine here and on libtorch where libtorch is handed over under the same name.
        var pipeline = Passengers()
            .WithTorch(network => network.Dense(1).Adam(0.01).MeanSquaredError().Run(seed: 20260929, epochs: 1).On("ours"))
            .Build();

        Assert.Equal("ours", Assert.IsType<LearnNetworkStep>(pipeline.Declaration.Learner).Engine);
        Assert.NotNull(pipeline.Train(new Engines().Use("ours", new CpuBackend())).History);
        Assert.Equal("light", Assert.IsType<LearnNetworkStep>(Declared().Declaration.Learner).Engine);
    }

    [Fact]
    public void AnEngineNobodyNamed_IsRefusedByName()
    {
        var pipeline = Passengers()
            .WithTorch(network => network.Dense(1).Adam().MeanSquaredError().Run(1).On("elsewhere"))
            .Build();

        var refused = Assert.Throws<InvalidOperationException>(() => pipeline.Train());

        Assert.Contains("elsewhere", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ADeclaredRunStopsEarlyWhenItSaysSo_AndItsReportIsMeasured()
    {
        var trained = Passengers()
            .Report(report => report.Measure(Metric.Accuracy).On(Part.Train, Part.Test).As(Shown.Numbers))
            .WithTensorflow(network => network
                .Dense(4).Relu().Dense(1)
                .Adam(0.01)
                .BinaryCrossEntropy()
                .Run(seed: 20260929, epochs: 50)
                .StoppingAfter(patience: 1, restoreBest: true))
            .Build()
            .Train();

        Assert.NotNull(trained.Measures);
        Assert.True(trained.History!.Epochs.Count <= 50);
    }

    [Fact]
    public void APipelineThatNamesNoLearner_HasNothingToTrain()
    {
        var refused = Assert.Throws<InvalidOperationException>(() => Passengers().Build().Train());

        Assert.Contains("names no learner", refused.Message, StringComparison.Ordinal);
    }
    [Fact]
    public void EveryWordADeclaredNetworkIsWrittenIn_BecomesTheLayerTheSavedNetworkWritesUnderThatName()
    {
        // The words and the network's own file speak one language, so a declaration lowered here and a network saved
        // there are the same thing under the same names.
        var described = new[]
        {
            new PartDeclaration("dense", [new PartSetting("units", PartValue.Of(2))]),
            new PartDeclaration("relu", []),
            new PartDeclaration("tanh", []),
            new PartDeclaration("sigmoid", []),
            new PartDeclaration("dropout", [new PartSetting("rate", PartValue.Of(0.5))]),
            new PartDeclaration("batchNorm", [new PartSetting("momentum", PartValue.Of(0.9)), new PartSetting("epsilon", PartValue.Of(1e-5))]),
            new PartDeclaration("layerNorm", [new PartSetting("epsilon", PartValue.Of(1e-5))]),
        }.Described().Lower(new Shape(3), new RandomStream(1));

        Assert.Equal(
            ["Dense", "Relu", "Tanh", "Sigmoid", "Dropout", "BatchNorm", "LayerNorm"],
            described.HeldLayers().Select(held => held.Layer.GetType().Name));
    }

    [Fact]
    public void EveryOptimizerLossAndStoppingAWordNames_BecomesTheOneTheCoreHas()
    {
        var sgd = Assert.IsType<Sgd>(new PartDeclaration("sgd", [new PartSetting("rate", PartValue.Of(0.5)), new PartSetting("momentum", PartValue.Of(0.25))]).Moved());
        var adam = Assert.IsType<Adam>(NetworkWords.Optimizers[1].Declared().Moved());

        Assert.Equal(0.25, sgd.Momentum);
        Assert.Equal(1e-8, adam.Epsilon);
        Assert.IsType<MeanSquaredError>(new PartDeclaration("meanSquaredError", []).Judged());
        Assert.IsType<CrossEntropy>(new PartDeclaration("crossEntropy", []).Judged());
        Assert.IsType<BinaryCrossEntropy>(new PartDeclaration("binaryCrossEntropy", []).Judged());
        Assert.Null(new PartDeclaration("never", []).Stops());

        var stops = new PartDeclaration("patience", [
            new PartSetting("patience", PartValue.Of(3)),
            new PartSetting("least", PartValue.Of(0.25)),
            new PartSetting("best", PartValue.Of(false)),
        ]).Stops();

        Assert.Equal(3, stops!.Patience);
        Assert.Equal(0.25, stops.MinDelta);
        Assert.False(stops.RestoreBest);
    }

    [Fact]
    public void AWordTheseVocabulariesDoNotKnow_IsRefusedNamingTheOnesTheyDo()
    {
        Assert.Throws<ArgumentNullException>(() => ((IReadOnlyList<PartDeclaration>)null!).Described());
        Assert.Contains("'dense'", Assert.Throws<NotSupportedException>(() => new[] { new PartDeclaration("transformer", []) }.Described()).Message, StringComparison.Ordinal);
        Assert.Contains("'adam'", Assert.Throws<NotSupportedException>(() => new PartDeclaration("rmsprop", []).Moved()).Message, StringComparison.Ordinal);
        Assert.Contains("'crossEntropy'", Assert.Throws<NotSupportedException>(() => new PartDeclaration("hinge", []).Judged()).Message, StringComparison.Ordinal);
        Assert.Contains("'never'", Assert.Throws<NotSupportedException>(() => new PartDeclaration("whenever", []).Stops()).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheWordsOfBothVocabularies_WriteTheNumbersEachLibraryLeavesUnsaid_ForEveryWordThatHasOne()
    {
        var keras = Assert.IsType<LearnNetworkStep>(Passengers()
            .WithTensorflow(network => network.Dense(2).Tanh().Sigmoid().Dropout(0.25).BatchNorm().LayerNorm().Dense(1).Sgd().CrossEntropy().Run(1))
            .Build().Declaration.Learner);
        var torch = Assert.IsType<LearnNetworkStep>(Passengers()
            .WithTorch(network => network.Dense(2).BatchNorm().LayerNorm().Dense(1).Sgd(momentum: 0.5).CrossEntropy().Run(1))
            .Build().Declaration.Learner);

        Assert.Equal(["dense", "tanh", "sigmoid", "dropout", "batchNorm", "layerNorm", "dense"], keras.Layers.Select(layer => layer.Kind));
        Assert.Equal(0.99, keras.Layers[4].Number("momentum"));
        Assert.Equal(1e-3, keras.Layers[5].Number("epsilon"));
        Assert.Equal(0.01, keras.Optimizer.Number("rate"));
        Assert.Equal(0.9, torch.Layers[1].Number("momentum"));
        Assert.Equal(1e-5, torch.Layers[2].Number("epsilon"));
        Assert.Equal(0.001, torch.Optimizer.Number("rate"));
        Assert.Equal(0.5, torch.Optimizer.Number("momentum"));
        Assert.Equal("crossEntropy", torch.Loss.Kind);
    }

    [Fact]
    public void AStepWrittenByHand_IsHeldToTheSameWordsAndNumbersAsOneTheChainWrote()
    {
        var step = new LearnNetworkStep(
            [new PartDeclaration("dense", [new PartSetting("units", PartValue.Of(1))])],
            NetworkWords.Optimizers[1].Declared(),
            NetworkWords.Losses[0].Declared());

        Assert.Equal(LearnNetworkStep.LightEngine, step.Engine);
        Assert.Equal("never", step.Stopping.Kind);
        Assert.Equal(100, step.Epochs);
        Assert.Equal(32, step.Batch);
        Assert.Equal(20260929, step.Seed);
        Assert.Throws<ArgumentOutOfRangeException>(() => step with { Epochs = 0 });
        Assert.Throws<ArgumentOutOfRangeException>(() => step with { Batch = 0 });
        Assert.Throws<ArgumentException>(() => step with { Engine = " " });
        Assert.Throws<ArgumentException>(() => step with { Stopping = new PartDeclaration("whenever", []) });
        Assert.Equal(step, step with { });
        Assert.Equal(step.GetHashCode(), (step with { }).GetHashCode());
        Assert.NotEqual(step, step with { Seed = 1 });
        Assert.NotEqual(step, step with { Epochs = 1 });
        Assert.NotEqual(step, step with { Batch = 1 });
        Assert.NotEqual(step, step with { Engine = "torch" });
        Assert.NotEqual(step, step with { Stopping = NetworkWords.Stopping[1].Declared() });
        Assert.NotEqual(step, new LearnNetworkStep(step.Layers, NetworkWords.Optimizers[0].Declared(), step.Loss));
        Assert.NotEqual(step, new LearnNetworkStep(step.Layers, step.Optimizer, NetworkWords.Losses[1].Declared()));
        Assert.NotEqual(step, new LearnNetworkStep([.. step.Layers, new PartDeclaration("relu", [])], step.Optimizer, step.Loss));
        Assert.False(step.Equals(null));
    }

    [Fact]
    public void AnEngineGivenTwoNames_AndANameGivenNothing_AreToldApart()
    {
        var engines = new Engines().Use("ours", new CpuBackend());

        Assert.IsType<CpuBackend>(engines.Named("ours"));
        Assert.IsType<CpuBackend>(engines.Named(LearnNetworkStep.LightEngine));
        Assert.Throws<ArgumentException>(() => engines.Use("ours", new CpuBackend()));
        Assert.Throws<ArgumentException>(() => engines.Use(" ", new CpuBackend()));
        Assert.Throws<ArgumentNullException>(() => engines.Use("other", null!));
        Assert.Contains("'ours'", Assert.Throws<InvalidOperationException>(() => engines.Named("elsewhere")).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void APipelineThatNamesALearnerOfAnotherKind_IsNotTrainedHere()
    {
        var declaration = new PipelineDeclaration([
            new ReadCsvStep("titanic.csv"),
            new DeclareStep([new ColumnDeclaration("survived", ColumnKind.Integer, false), new ColumnDeclaration("fare", ColumnKind.Number, false)]),
            new SplitAtRandomStep(new SplitShares(0.7, 0.15, 0.15), 1),
            new TargetStep("survived"),
            new LearnStumpStep(1),
        ]);

        var refused = Assert.Throws<InvalidOperationException>(() => new Pipeline(declaration).Train());

        Assert.Contains("learn.stump", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NothingIsTrained_AndNothingDeclares()
    {
        Assert.Throws<ArgumentNullException>(() => ((Pipeline)null!).Train());
        Assert.Throws<ArgumentNullException>(() => ((PreparedData)null!).Train());
        Assert.Throws<ArgumentNullException>(() => Passengers().WithTorch(null!));
        Assert.Throws<ArgumentNullException>(() => ((FittingBuilder)null!).WithTorch(network => network));
        Assert.Throws<ArgumentNullException>(() => Passengers().WithTensorflow(null!));
        Assert.Throws<ArgumentNullException>(() => ((FittingBuilder)null!).WithTensorflow(network => network));
    }

    [Fact]
    public void AChainThatSaysEveryNumberItself_WritesThoseRatherThanTheOnesItsLibraryLeaves()
    {
        // The other side of the two vocabularies: a number a caller names is the number written, whichever words the
        // chain is in, so neither library's own is reached for.
        var step = Assert.IsType<LearnNetworkStep>(Passengers()
            .WithTensorflow(network => network
                .Dense(2).Dropout(0.5).BatchNorm(momentum: 0.5, epsilon: 0.25).LayerNorm(epsilon: 0.125).Dense(1)
                .Adam(rate: 0.5, epsilon: 0.0625)
                .MeanSquaredError()
                .Run(seed: 1, epochs: 2, batch: 8))
            .Build().Declaration.Learner);
        var sgd = Assert.IsType<LearnNetworkStep>(Passengers()
            .WithTorch(network => network.Dense(1).Sgd(rate: 0.25, momentum: 0.125).MeanSquaredError().Run(1))
            .Build().Declaration.Learner);

        Assert.Equal(0.5, step.Layers[1].Number("rate"));
        Assert.Equal(0.5, step.Layers[2].Number("momentum"));
        Assert.Equal(0.25, step.Layers[2].Number("epsilon"));
        Assert.Equal(0.125, step.Layers[3].Number("epsilon"));
        Assert.Equal(0.5, step.Optimizer.Number("rate"));
        Assert.Equal(0.0625, step.Optimizer.Number("epsilon"));
        Assert.Equal(2, step.Epochs);
        Assert.Equal(8, step.Batch);
        Assert.Equal(0.25, sgd.Optimizer.Number("rate"));
        Assert.Equal(0.125, sgd.Optimizer.Number("momentum"));
    }

}
