// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using DeepSharp.Tests.Networks;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// A network written down: its layers by the names they are registered under and the settings they are rebuilt from,
/// every slot's numbers by its path, its loss by name — and, for a checkpoint, everything a run needs to go on. Read back
/// through a catalog of the kinds a reader knows, it is the same network to the last bit; anything else in the file is
/// refused, every fault at its line and column.
/// </summary>
public abstract class NetworkDocumentContract(ITensorBackend engine)
{
    private readonly ITensorBackend _backend = engine;

    [Fact]
    public void ANetworkWrittenAndReadBack_IsTheSameNetwork_ToTheLastBit()
    {
        var network = new LayerStack(
            new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias()), new Relu(), new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias()));

        var read = NetworkDocument.ReadNetwork(Written(network, new BinaryCrossEntropy()), "network", NetworkCatalog.BuiltIn());

        Assert.IsType<LayerStack>(read.Network);
        Assert.IsType<BinaryCrossEntropy>(read.Loss);
        Assert.Equal(Bits(network), Bits(read.Network));
        Assert.Equal(
            network.Forward(WalkedRows.Passengers(), Pass.Evaluation(_backend)).Values.ToArray(),
            read.Network.Forward(WalkedRows.Passengers(), Pass.Evaluation(_backend)).Values.ToArray());
    }

    [Fact]
    public void EveryKindOfLayer_IsWrittenAndReadBack_WithItsSettingsAndItsRunningStatistics()
    {
        var draws = new RandomStream(9).Draw("initialise:test", 0, 0);
        var network = new LayerStack(
            new Reshape(new Shape(4, 4, 1)),
            new Conv2D(1, 2, new Window(3, 3) { Stride = 1, Padding = 1 }, draws),
            new BatchNorm(2) { Momentum = 0.2, Epsilon = 1e-3 },
            new Tanh(),
            new Flatten(),
            new LayerNorm(32) { Epsilon = 1e-4 },
            new Dropout(0.3),
            new Dense(32, 3, draws),
            new Sigmoid());
        var rows = Tensor.From(new Shape(3, 16), [.. Enumerable.Range(0, 48).Select(at => MathF.Sin(at))]);
        network.Forward(rows, Pass.Training(_backend, new RandomStream(1), 0, 0));

        var read = NetworkDocument.ReadNetwork(Written(network, new MeanSquaredError()), "network", NetworkCatalog.BuiltIn());
        var stack = Assert.IsType<LayerStack>(read.Network);

        Assert.Equal(Bits(network), Bits(stack));
        Assert.Equal(0.2, Assert.IsType<BatchNorm>(stack.Layers[2]).Momentum);
        Assert.Equal(1e-3, ((BatchNorm)stack.Layers[2]).Epsilon);
        Assert.Equal(1e-4, Assert.IsType<LayerNorm>(stack.Layers[5]).Epsilon);
        Assert.Equal(0.3, Assert.IsType<Dropout>(stack.Layers[6]).Rate);
        Assert.Equal(new Window(3, 3) { Padding = 1 }, Assert.IsType<Conv2D>(stack.Layers[1]).Window);
        Assert.Equal(
            network.Forward(rows, Pass.Evaluation(_backend)).Values.ToArray(),
            stack.Forward(rows, Pass.Evaluation(_backend)).Values.ToArray());
    }

    [Fact]
    public void AConvolutionPaddedAsSame_IsWrittenWithTheWord_AndReadBackPaddedSo()
    {
        var window = new Window(4, 3) { Stride = 2, PaddingMode = PaddingMode.Same };
        var network = new LayerStack(new Conv2D(2, 3, window, new RandomStream(9).Draw("initialise:test", 0, 0)), new Flatten());
        var images = Tensor.From(new Shape(2, 7, 8, 2), [.. Enumerable.Range(0, 224).Select(at => MathF.Sin(at * 0.3f))]);

        var text = Written(network, new MeanSquaredError());
        var read = Assert.IsType<LayerStack>(NetworkDocument.ReadNetwork(text, "network", NetworkCatalog.BuiltIn()).Network);

        Assert.Contains("\"padding\": \"same\"", text, StringComparison.Ordinal);
        Assert.Equal(window, Assert.IsType<Conv2D>(read.Layers[0]).Window);
        Assert.Equal(Bits(network), Bits(read));
        Assert.Equal(
            network.Forward(images, Pass.Evaluation(_backend)).Values.ToArray(),
            read.Forward(images, Pass.Evaluation(_backend)).Values.ToArray());
    }

    [Fact]
    public void EveryLayerThatWalksASeries_AnImageOrAVolume_IsWrittenAndReadBack_WithItsWindowAndItsSettings()
    {
        var draws = new RandomStream(9).Draw("initialise:test", 0, 0);
        var line = new LayerStack(
            new Conv1D(2, 3, new Window1D(3) { Stride = 2, Padding = 1 }, draws),
            new MaxPool1D(new Window1D(2) { Stride = 2 }),
            new SpatialDropout1D(0.3),
            new Conv1D(3, 2, new Window1D(2) { PaddingMode = PaddingMode.Causal }, draws),
            new AvgPool1D(new Window1D(2) { Stride = 1, PaddingMode = PaddingMode.Same }) { CountsPadding = true },
            new GlobalMaxPool1D());
        var plane = new LayerStack(
            new Conv2D(1, 2, new Window(3, 3) { Stride = 1, PaddingMode = PaddingMode.Same }, draws),
            new MaxPool2D(new Window(2, 2) { Stride = 2, Padding = 1 }),
            new SpatialDropout2D(0.2),
            new AvgPool2D(new Window(2, 2) { Stride = 2 }),
            new GlobalAvgPool2D { KeepsAxes = true });
        var volume = new LayerStack(
            new Conv3D(1, 2, new Window3D(2, 2, 2) { Stride = 1, Padding = 1 }, draws),
            new MaxPool3D(new Window3D(2, 2, 2) { Stride = 2 }),
            new SpatialDropout3D(0.1),
            new AvgPool3D(new Window3D(2, 2, 2) { Stride = 1 }),
            new GlobalMaxPool3D { KeepsAxes = true });

        AssertWrittenAndReadBack(line, Tensor.From(new Shape(2, 9, 2), [.. Enumerable.Range(0, 36).Select(at => MathF.Sin(at))]));
        AssertWrittenAndReadBack(plane, Tensor.From(new Shape(2, 6, 6, 1), [.. Enumerable.Range(0, 72).Select(at => MathF.Sin(at))]));
        AssertWrittenAndReadBack(volume, Tensor.From(new Shape(2, 4, 4, 4, 1), [.. Enumerable.Range(0, 128).Select(at => MathF.Sin(at))]));

        var read = NetworkDocument.ReadNetwork(Written(line, new MeanSquaredError()), "network", NetworkCatalog.BuiltIn());
        var layers = Assert.IsType<LayerStack>(read.Network).Layers;

        Assert.Equal(new Window1D(3) { Stride = 2, Padding = 1 }, Assert.IsType<Conv1D>(layers[0]).Window);
        Assert.Equal(new Window1D(2) { Stride = 2 }, Assert.IsType<MaxPool1D>(layers[1]).Window);
        Assert.Equal(0.3, Assert.IsType<SpatialDropout1D>(layers[2]).Rate);
        Assert.Equal(PaddingMode.Causal, Assert.IsType<Conv1D>(layers[3]).Window.PaddingMode);
        Assert.True(Assert.IsType<AvgPool1D>(layers[4]).CountsPadding);
        Assert.False(Assert.IsType<GlobalMaxPool1D>(layers[5]).KeepsAxes);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EveryGlobalPooling_IsWrittenAndReadBack_WithTheAxesItKeepsOrDrops(bool keepsAxes)
    {
        var series = Tensor.From(new Shape(2, 5, 3), [.. Enumerable.Range(0, 30).Select(at => MathF.Sin(at))]);
        var images = Tensor.From(new Shape(2, 4, 4, 3), [.. Enumerable.Range(0, 96).Select(at => MathF.Sin(at))]);
        var volumes = Tensor.From(new Shape(2, 3, 3, 3, 3), [.. Enumerable.Range(0, 162).Select(at => MathF.Sin(at))]);

        AssertWrittenAndReadBack(new LayerStack(new GlobalMaxPool1D { KeepsAxes = keepsAxes }), series);
        AssertWrittenAndReadBack(new LayerStack(new GlobalAvgPool1D { KeepsAxes = keepsAxes }), series);
        AssertWrittenAndReadBack(new LayerStack(new GlobalMaxPool2D { KeepsAxes = keepsAxes }), images);
        AssertWrittenAndReadBack(new LayerStack(new GlobalAvgPool2D { KeepsAxes = keepsAxes }), images);
        AssertWrittenAndReadBack(new LayerStack(new GlobalMaxPool3D { KeepsAxes = keepsAxes }), volumes);
        AssertWrittenAndReadBack(new LayerStack(new GlobalAvgPool3D { KeepsAxes = keepsAxes }), volumes);
    }

    [Fact]
    public void ASettingThatHoldsItsDefault_IsLeftOutOfTheFile_SoOnlyWhatIsSaidIsWritten()
    {
        var keeping = Written(new LayerStack(new AvgPool2D(new Window(2, 2)) { CountsPadding = true }, new GlobalMaxPool2D { KeepsAxes = true }), new MeanSquaredError());
        var leaving = Written(new LayerStack(new AvgPool2D(new Window(2, 2)), new GlobalMaxPool2D()), new MeanSquaredError());
        var padded = Written(new LayerStack(new Conv1D(1, 1, new Window1D(2) { PaddingMode = PaddingMode.Causal }, new RandomStream(9).Draw("initialise:test", 0, 0))), new MeanSquaredError());

        Assert.Contains("\"countsPadding\": true", keeping, StringComparison.Ordinal);
        Assert.Contains("\"keepsAxes\": true", keeping, StringComparison.Ordinal);
        Assert.DoesNotContain("countsPadding", leaving, StringComparison.Ordinal);
        Assert.DoesNotContain("keepsAxes", leaving, StringComparison.Ordinal);
        Assert.Contains("\"padding\": \"causal\"", padded, StringComparison.Ordinal);
    }

    // The stack written and read back is the same stack to the last bit, and answers a batch as it did.
    private void AssertWrittenAndReadBack(LayerStack network, Tensor batch)
    {
        var read = Assert.IsType<LayerStack>(NetworkDocument.ReadNetwork(Written(network, new MeanSquaredError()), "network", NetworkCatalog.BuiltIn()).Network);

        Assert.Equal(network.Layers.Select(layer => layer.GetType()), read.Layers.Select(layer => layer.GetType()));
        Assert.Equal(Bits(network), Bits(read));
        Assert.Equal(
            network.Forward(batch, Pass.Evaluation(_backend)).Values.ToArray(),
            read.Forward(batch, Pass.Evaluation(_backend)).Values.ToArray());
    }

    [Fact]
    public void AConvolutionInANetworksPartOfTheFirstVersion_IsReadWithTheBorderItStates()
    {
        // 0.4.0 wrote every window's border as the number of rows and columns on each side, in the first version's part.
        var text = Written(new LayerStack(new Conv2D(1, 1, new Window(2, 2) { Padding = 1 }, new RandomStream(9).Draw("initialise:test", 0, 0))), new MeanSquaredError())
            .Replace("\"version\": 2", "\"version\": 1", StringComparison.Ordinal);

        var read = Assert.IsType<LayerStack>(NetworkDocument.ReadNetwork(text, "network", NetworkCatalog.BuiltIn()).Network);

        Assert.Contains("\"padding\": 1", text, StringComparison.Ordinal);
        Assert.Equal(new Window(2, 2) { Padding = 1 }, Assert.IsType<Conv2D>(read.Layers[0]).Window);
    }

    [Fact]
    public void TheValuesAFloatCanHoldAtItsEdges_SurviveTheRoundTripBitForBit()
    {
        float[] edges = [float.Epsilon, float.MaxValue, -0f, 1f / 3, 1e-45f, -float.MaxValue, 0.1f, 16777216f, 1.17549435e-38f, -2.5e-40f];
        var network = new LayerStack(new Dense(Tensor.From(new Shape(10, 1), edges), Tensor.Zeros(new Shape(1))));

        var read = NetworkDocument.ReadNetwork(Written(network, new MeanSquaredError()), "network", NetworkCatalog.BuiltIn());

        Assert.Equal(Bits(network), Bits(read.Network));
    }

    [Fact]
    public void TheLoss_IsWrittenByTheNameItIsRegisteredUnder()
    {
        var text = Written(new LayerStack(new Relu()), new CrossEntropy());

        Assert.Contains("\"crossEntropy\"", text, StringComparison.Ordinal);
        Assert.IsType<CrossEntropy>(NetworkDocument.ReadNetwork(text, "network", NetworkCatalog.BuiltIn()).Loss);
    }

    [Fact]
    public void TheDistanceAlongAnOrder_IsWrittenWithWhatIsLeftOfTheWhole_OnlyWhenThereIsSuch()
    {
        var left = Written(new LayerStack(new Relu()), new EarthMoversDistance(remainder: true));
        var whole = Written(new LayerStack(new Relu()), new EarthMoversDistance());

        Assert.Contains("\"earthMoversDistance\"", left, StringComparison.Ordinal);
        Assert.Contains("\"remainder\": true", left, StringComparison.Ordinal);
        Assert.DoesNotContain("remainder", whole, StringComparison.Ordinal);
        Assert.True(Assert.IsType<EarthMoversDistance>(NetworkDocument.ReadNetwork(left, "network", NetworkCatalog.BuiltIn()).Loss).Remainder);
        Assert.False(Assert.IsType<EarthMoversDistance>(NetworkDocument.ReadNetwork(whole, "network", NetworkCatalog.BuiltIn()).Loss).Remainder);
        Assert.False(Assert.IsType<EarthMoversDistance>(NetworkDocument.ReadNetwork(
            left.Replace("\"remainder\": true", "\"remainder\": false", StringComparison.Ordinal), "network", NetworkCatalog.BuiltIn()).Loss).Remainder);

        var refused = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(
            left.Replace("\"remainder\": true", "\"remainder\": 3", StringComparison.Ordinal), "network", NetworkCatalog.BuiltIn()));

        Assert.Contains("'remainder' is true or false here", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ANetworkWrittenAsCode_IsReadBackByItsRegisteredName()
    {
        var network = new Written2(new RandomStream(3).Draw("initialise:test", 0, 0));
        var text = Written(network, new MeanSquaredError());

        var read = NetworkDocument.ReadNetwork(text, "network", NetworkCatalog.BuiltIn().Register<Written2>());

        Assert.IsType<Written2>(read.Network);
        Assert.Equal(Bits(network), Bits(read.Network));
    }

    [Fact]
    public void AKindNoOneRegistered_IsRefused_NamingThePackageItComesFrom()
    {
        var text = Written(new Written2(new RandomStream(3).Draw("initialise:test", 0, 0)), new MeanSquaredError());

        var wrong = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(text, "network", NetworkCatalog.BuiltIn()));

        Assert.Contains("written2", wrong.Message, StringComparison.Ordinal);
        Assert.Contains(typeof(Written2).Assembly.GetName().Name!, wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AKindNothingKnows_IsRefused_NamingTheNearestOne()
    {
        var text = Written(new LayerStack(new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias())), new MeanSquaredError())
            .Replace("\"dense\"", "\"dens\"", StringComparison.Ordinal);

        var wrong = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(text, "network", NetworkCatalog.BuiltIn()));

        Assert.Contains("'dens'", wrong.Message, StringComparison.Ordinal);
        Assert.Contains("'dense'", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AMissingSlot_AnExtraSlot_AndAWrongShape_AreEachRefusedAtTheirLineAndColumn()
    {
        var text = Written(new LayerStack(new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias())), new MeanSquaredError());
        var lines = text.Split('\n');
        var parameters = Array.FindIndex(lines, line => line.Contains("\"parameters\"", StringComparison.Ordinal));
        var shape = Array.FindIndex(lines, parameters, line => line.Contains("\"shape\"", StringComparison.Ordinal));
        var bias = Array.FindIndex(lines, parameters, line => line.Contains("\"0.bias\"", StringComparison.Ordinal));
        var broken = string.Join('\n', lines.Select((line, at) =>
            at == bias ? line.Replace("\"0.bias\"", "\"0.biass\"", StringComparison.Ordinal)
            : at == shape ? line.Replace("[", "[ 2,", StringComparison.Ordinal)
            : line));

        var wrong = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(broken, "network", NetworkCatalog.BuiltIn()));

        Assert.Equal(
            [
                new NetworkFileFault(parameters + 1, lines[parameters].IndexOf('"', StringComparison.Ordinal) + 1, "'0.bias' is missing: every slot of the network is written."),
                new NetworkFileFault(shape + 1, lines[shape].IndexOf('"', StringComparison.Ordinal) + 1, "'0.weight' is a 4x1 slot here, and is written as 2x4x1."),
                new NetworkFileFault(bias + 1, lines[bias].IndexOf('"', StringComparison.Ordinal) + 1, "'0.biass' is no slot of this network."),
            ],
            wrong.Faults);
    }

    [Fact]
    public void ASlotOfAnotherShapeThanItsLayerHolds_IsRefused()
    {
        var text = Written(new LayerStack(new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias())), new MeanSquaredError());
        var broken = text.Replace("\"inputs\": 4", "\"inputs\": 5", StringComparison.Ordinal);

        var wrong = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(broken, "network", NetworkCatalog.BuiltIn()));

        Assert.Contains("0.weight", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFileOfANewerVersion_IsRefusedWhole()
    {
        var newer = NetworkDocument.Version + 1;
        var text = Written(new LayerStack(new Relu()), new MeanSquaredError())
            .Replace("\"version\": 2", $"\"version\": {newer}", StringComparison.Ordinal);

        var wrong = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(text, "network", NetworkCatalog.BuiltIn()));

        Assert.Single(wrong.Faults);
        Assert.Contains($"version {newer}", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheNetworksPart_IsWrittenAsItsSecondVersion_AndOneOfTheFirst_IsReadAsSayingNothingOfWhatHeldOneValue()
    {
        // The second version records which features held one value on every training row; the first — 0.4.0's — did not,
        // and is read as saying nothing of it, which is not the same as saying that none did. The third says nothing new of a
        // network, so a network's part is still written as the second, which 0.8.0 reads.
        var text = Written(writer =>
        {
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, Weighing(), new BinaryCrossEntropy(), Said(unvaried: null));
        });
        var first = text.Replace("\"version\": 2", "\"version\": 1", StringComparison.Ordinal);

        Assert.Equal(3, NetworkDocument.Version);
        Assert.Contains("\"version\": 2", text, StringComparison.Ordinal);
        Assert.Null(NetworkDocument.ReadNetwork(first, "network", NetworkCatalog.BuiltIn()).TrainedOn!.Unvaried);
    }

    [Fact]
    public void WhichFeaturesHeldOneValue_IsWrittenWithWhatANetworkWasTrainedOn_InTheFeaturesOrder_AndReadBack_NoneAsNone_AndUnsaidAsUnsaid()
    {
        var some = Read(Said(new Dictionary<string, double> { ["alone"] = 1, ["age"] = -0.5 }));
        var none = Read(Said(new Dictionary<string, double>()));
        var unsaid = Read(Said(unvaried: null));

        Assert.Equal([new KeyValuePair<string, double>("age", -0.5), new("alone", 1)], some.TrainedOn!.Unvaried!);
        Assert.Empty(none.TrainedOn!.Unvaried!);
        Assert.Null(unsaid.TrainedOn!.Unvaried);
    }

    [Theory]
    [InlineData("sex", 0.0, "'sex'")]
    [InlineData("fare", double.NaN, "'fare'")]
    [InlineData("fare", double.PositiveInfinity, "'fare'")]
    public void WhichFeaturesHeldOneValue_NamingWhatIsNoFeature_OrAValueThatIsNoFiniteNumber_CannotBeWritten(string feature, double value, string named)
    {
        var wrong = Assert.Throws<InvalidOperationException>(() => Read(Said(new Dictionary<string, double> { [feature] = value })));

        Assert.Contains(named, wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AValueThatIsNotAFiniteNumber_CannotBeWritten()
    {
        var network = new LayerStack(new Dense(Tensor.From(new Shape(1, 1), [float.NaN]), Tensor.Zeros(new Shape(1))));

        var wrong = Assert.Throws<InvalidOperationException>(() => Written(network, new MeanSquaredError()));

        Assert.Contains("0.weight", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnOptimizerWrittenOutsideTheLibrary_IsKeptByACheckpoint_NamedByItsFile_AndTheRunGoesOnAsIfItHadNeverStopped()
    {
        var straight = Rich();
        var stopped = Rich();
        var kept = new List<Checkpoint>();

        straight.Compile(new CountingSteps(0.02), new MeanSquaredError())
            .Fit(Rows(96), Rows(24, 7), new FitOptions(seed: 5) { Backend = _backend, Epochs = 6, BatchSize = 16 });

        var compiled = stopped.Compile(new CountingSteps(0.02), new MeanSquaredError());
        compiled.Fit(Rows(96), Rows(24, 7), new FitOptions(seed: 5) { Backend = _backend, Epochs = 3, BatchSize = 16, Checkpoints = new Checkpoints(kept.Add) });

        // An optimizer is named by its file only once it is registered, and a file naming one nobody registered is refused.
        var text = Checkpointed(compiled, kept[^1]);
        var known = NetworkCatalog.BuiltIn();
        var refused = Assert.Throws<NetworkFileException>(() =>
            NetworkDocument.ReadTraining(text, "training", known, NetworkDocument.ReadNetwork(text, "network", known)));

        Assert.Contains("countingSteps", refused.Message, StringComparison.Ordinal);

        var catalog = NetworkCatalog.BuiltIn().Register<CountingSteps>();
        var resumed = NetworkDocument.ReadTraining(text, "training", catalog, NetworkDocument.ReadNetwork(text, "network", catalog));
        var optimizer = Assert.IsType<CountingSteps>(resumed.Compiled.Optimizer);
        var weight = resumed.Compiled.Network.Parameters().First();

        // Six steps an epoch, three epochs: what it remembered of the weight crossed in the file.
        Assert.Equal(18, optimizer.StepsOf(weight));

        resumed.Compiled.Fit(Rows(96), Rows(24, 7), new FitOptions(seed: 5) { Backend = _backend, Epochs = 6, BatchSize = 16, ResumeFrom = resumed.Checkpoint });

        Assert.Equal(Bits(straight), Bits(resumed.Compiled.Network));
        Assert.Equal(36, optimizer.StepsOf(weight));
    }

    [Fact]
    public void ACheckpointWrittenAndReadBack_ResumesTheRunAsIfItHadNeverStopped()
    {
        var straight = Rich();
        var stopped = Rich();
        var kept = new List<Checkpoint>();
        var options = new FitOptions(seed: 5) { Backend = _backend, Epochs = 6, BatchSize = 16, EarlyStopping = new EarlyStopping { Patience = 10, RestoreBest = true } };

        straight.Compile(new Adam(0.01), new MeanSquaredError(), new CosineDecay(6, 0.001)).Fit(Rows(96), Rows(24, 7), options);
        var compiled = stopped.Compile(new Adam(0.01), new MeanSquaredError(), new CosineDecay(6, 0.001));
        compiled.Fit(Rows(96), Rows(24, 7), new FitOptions(seed: 5)
        {
            Backend = _backend,
            Epochs = 3, BatchSize = 16, EarlyStopping = new EarlyStopping { Patience = 10, RestoreBest = true }, Checkpoints = new Checkpoints(kept.Add),
        });

        var text = Written(writer =>
        {
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, compiled.Network, compiled.Loss);
            writer.WritePropertyName("training");
            NetworkDocument.WriteTraining(writer, compiled, kept[^1]);
        });
        var catalog = NetworkCatalog.BuiltIn();
        var saved = NetworkDocument.ReadNetwork(text, "network", catalog);
        var resumed = NetworkDocument.ReadTraining(text, "training", catalog, saved);

        Assert.IsType<Adam>(resumed.Compiled.Optimizer);
        Assert.IsType<CosineDecay>(resumed.Compiled.Schedule);
        Assert.Equal(3, resumed.Checkpoint.Epochs);
        resumed.Compiled.Fit(Rows(96), Rows(24, 7), new FitOptions(seed: 5)
        {
            Backend = _backend,
            Epochs = 6, BatchSize = 16, EarlyStopping = new EarlyStopping { Patience = 10, RestoreBest = true }, ResumeFrom = resumed.Checkpoint,
        });

        Assert.Equal(Bits(straight), Bits(resumed.Compiled.Network));
    }

    [Fact]
    public void ACheckpoint_RecordsTheBatchSizeAndTheEarlyStoppingItsRunWentUnder_AndARunGoneOnFromItUnderOthers_IsRefused()
    {
        var kept = new List<Checkpoint>();
        var compiled = Rich().Compile(new Adam(0.01), new MeanSquaredError());
        compiled.Fit(Rows(96), Rows(24, 7), new FitOptions(seed: 5)
        {
            Backend = _backend,
            Epochs = 2, BatchSize = 16, EarlyStopping = new EarlyStopping { Patience = 10, MinDelta = 0.001, RestoreBest = true }, Checkpoints = new Checkpoints(kept.Add),
        });

        var text = Checkpointed(compiled, kept[^1]);
        var training = JsonNode.Parse(text)!["training"]!.AsObject();
        var catalog = NetworkCatalog.BuiltIn();
        var resumed = NetworkDocument.ReadTraining(text, "training", catalog, NetworkDocument.ReadNetwork(text, "network", catalog));
        var wrong = Assert.Throws<ArgumentException>(() => resumed.Compiled.Fit(Rows(96), Rows(24, 7), new FitOptions(seed: 5)
        {
            Backend = _backend,
            Epochs = 4, EarlyStopping = new EarlyStopping { Patience = 10, MinDelta = 0.001, RestoreBest = true }, ResumeFrom = resumed.Checkpoint,
        }));

        Assert.Equal(
            ["version", "seed", "batchSize", "earlyStopping", "engine", "optimizer", "schedule", "memory", "judgement", "history"],
            training.Select(key => key.Key));
        Assert.Equal(2, (int)training["version"]!);
        Assert.Equal(16, (int)training["batchSize"]!);
        Assert.Equal("""{"patience":10,"minDelta":0.001,"restoreBest":true}""", training["earlyStopping"]!.ToJsonString());
        Assert.Contains("in batches of 16, and going on in batches of 32", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACheckpoint_RecordsTheEngineItsRunWasOn_AsTheEngineNamesItself_AndARunGoneOnFromItsFileOnAnother_IsRefused()
    {
        var kept = new List<Checkpoint>();
        var compiled = Rich().Compile(new Adam(0.01), new MeanSquaredError());
        compiled.Fit(Rows(96), Rows(24, 7), new FitOptions(seed: 5) { Backend = _backend, Epochs = 2, BatchSize = 16, Checkpoints = new Checkpoints(kept.Add) });

        var text = Checkpointed(compiled, kept[^1]);
        var catalog = NetworkCatalog.BuiltIn();
        var resumed = NetworkDocument.ReadTraining(text, "training", catalog, NetworkDocument.ReadNetwork(text, "network", catalog));
        var wrong = Assert.Throws<ArgumentException>(() => resumed.Compiled.Fit(Rows(96), Rows(24, 7), new FitOptions(seed: 5)
        {
            Backend = new ElsewhereBackend(_backend), Epochs = 4, BatchSize = 16, ResumeFrom = resumed.Checkpoint,
        }));

        // The engine's name, and its version and the device it works on where it names them; the network's part names none.
        var engine = JsonNode.Parse(text)!["training"]!["engine"]!.AsObject();
        Assert.Equal(_backend.Name, (string)engine["name"]!);
        Assert.Equal((_backend as INamesItsVersionAndDevice)?.Version, (string?)engine["version"]);
        Assert.Equal((_backend as INamesItsVersionAndDevice)?.Device, (string?)engine["device"]);
        Assert.DoesNotContain("engine", JsonNode.Parse(text)!["network"]!.AsObject().Select(key => key.Key));
        Assert.Contains($"a run on the engine {ElsewhereBackend.NamedAs(_backend)}, and going on under the engine {ElsewhereBackend.Named}", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACheckpointOfARunWhoseGradientsWereClipped_IsWrittenAsTheThirdVersion_WithTheNorm_AndGoesOnFromItsFileBitForBit_OnlyUnderThatClip()
    {
        // The third version is the one that can say a clip, and is written only where there is one: a library that reads up
        // to the second names the version it does not read, rather than a key it does not know.
        var straight = Rich();
        var stopped = Rich();
        var kept = new List<Checkpoint>();

        straight.Compile(new Adam(0.01), new MeanSquaredError()).Fit(Rows(96), Rows(24, 7), Options(6));
        var compiled = stopped.Compile(new Adam(0.01), new MeanSquaredError());
        compiled.Fit(Rows(96), Rows(24, 7), Options(3, checkpoints: new Checkpoints(kept.Add)));

        var text = Checkpointed(compiled, kept[^1]);
        var training = JsonNode.Parse(text)!["training"]!.AsObject();
        var catalog = NetworkCatalog.BuiltIn();
        var resumed = NetworkDocument.ReadTraining(text, "training", catalog, NetworkDocument.ReadNetwork(text, "network", catalog));
        var refused = Assert.Throws<ArgumentException>(() => resumed.Compiled.Fit(Rows(96), Rows(24, 7), new FitOptions(seed: 5)
        {
            Backend = _backend, Epochs = 6, BatchSize = 16, ResumeFrom = resumed.Checkpoint,
        }));

        resumed.Compiled.Fit(Rows(96), Rows(24, 7), Options(6, from: resumed.Checkpoint));

        Assert.Equal(3, (int)training["version"]!);
        Assert.Equal(2, (int)JsonNode.Parse(text)!["network"]!["version"]!);
        Assert.Equal(
            ["version", "seed", "batchSize", "earlyStopping", "clipNorm", "engine", "optimizer", "schedule", "memory", "judgement", "history"],
            training.Select(key => key.Key));
        Assert.Equal(0.25, (double)training["clipNorm"]!);
        Assert.Contains("clipped to a norm of 0.25, and going on without clipping them", refused.Message, StringComparison.Ordinal);
        Assert.Equal(Bits(straight), Bits(resumed.Compiled.Network));

        FitOptions Options(int epochs, Checkpoints? checkpoints = null, Checkpoint? from = null) => new(seed: 5)
        {
            Backend = _backend, Epochs = epochs, BatchSize = 16, GradientClip = new GradientClip(0.25), Checkpoints = checkpoints, ResumeFrom = from,
        };
    }

    [Fact]
    public void ACheckpointOfARunWithNoEarlyStopping_SaysSoWithNull()
    {
        var compiled = new LayerStack(new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias())).Compile(new Sgd(0.05), new MeanSquaredError());
        var kept = new List<Checkpoint>();
        compiled.Fit(Rows(8, width: 4), validation: null, new FitOptions(seed: 1) { Backend = _backend, Checkpoints = new Checkpoints(kept.Add) });

        var training = JsonNode.Parse(Checkpointed(compiled, kept[^1]))!["training"]!.AsObject();

        Assert.Equal(32, (int)training["batchSize"]!);
        Assert.True(training.TryGetPropertyValue("earlyStopping", out var earlyStopping));
        Assert.Null(earlyStopping);
    }

    [Fact]
    public void ACheckpointOfTheFirstVersion_RecordsNone_IsWrittenAgainAsThatVersionWroteIt_AndGoesOnInAnyBatches_OnAnyEngine()
    {
        // The first version, which 0.4.0 wrote, records neither the batch size nor the early stopping nor the engine: read, it
        // says nothing of them, and written again it still says nothing, rather than a value nobody recorded.
        var kept = new List<Checkpoint>();
        var compiled = Rich().Compile(new Adam(0.01), new MeanSquaredError());
        compiled.Fit(Rows(96), Rows(24, 7), new FitOptions(seed: 5) { Backend = _backend, Epochs = 2, BatchSize = 16, Checkpoints = new Checkpoints(kept.Add) });
        var file = JsonNode.Parse(Checkpointed(compiled, kept[^1]))!;
        var part = file["training"]!.AsObject();
        part.Remove("batchSize");
        part.Remove("earlyStopping");
        part.Remove("engine");
        part["version"] = 1;
        var catalog = NetworkCatalog.BuiltIn();
        var first = file.ToJsonString();
        var read = NetworkDocument.ReadTraining(first, "training", catalog, NetworkDocument.ReadNetwork(first, "network", catalog));

        var again = Checkpointed(read.Compiled, read.Checkpoint);
        var training = JsonNode.Parse(again)!["training"]!.AsObject();
        var reread = NetworkDocument.ReadTraining(again, "training", catalog, NetworkDocument.ReadNetwork(again, "network", catalog));
        var history = reread.Compiled.Fit(
            Rows(96), Rows(24, 7), new FitOptions(seed: 5) { Backend = new ElsewhereBackend(_backend), Epochs = 3, BatchSize = 8, ResumeFrom = reread.Checkpoint });

        Assert.Equal(1, (int)training["version"]!);
        Assert.False(training.ContainsKey("batchSize"));
        Assert.False(training.ContainsKey("earlyStopping"));
        Assert.False(training.ContainsKey("engine"));
        Assert.Equal([0, 1, 2], history.Epochs.Select(epoch => epoch.Number));
    }

    [Fact]
    public void EveryOptimizerAndSchedule_IsWrittenWithItsSettings_AndReadBack()
    {
        var network = new LayerStack(new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias()));
        var kept = new List<Checkpoint>();
        var catalog = NetworkCatalog.BuiltIn();

        foreach (var (optimizer, schedule) in new (Optimizer, LearningRateSchedule)[]
                 {
                     (new Sgd(0.05) { Momentum = 0.9 }, new StepDecay(3, 0.5)),
                     (new Adam(0.02) { Betas = new Betas(0.8, 0.99), Epsilon = 1e-6 }, new ExponentialDecay(0.95)),
                     (new Sgd(), new ConstantRate()),
                     (new AdamW(), new LinearWarmup(3, 0.5, new CosineDecay(4, 0.01))),
                     (new Nadam(), new LinearWarmup(2)),
                 })
        {
            var compiled = network.Compile(optimizer, new MeanSquaredError(), schedule);
            compiled.Fit(Rows(8, width: 4), validation: null, new FitOptions(seed: 1) { Backend = _backend, Checkpoints = new Checkpoints(kept.Add) });

            var text = Written(writer =>
            {
                writer.WritePropertyName("network");
                NetworkDocument.WriteNetwork(writer, network, compiled.Loss);
                writer.WritePropertyName("training");
                NetworkDocument.WriteTraining(writer, compiled, kept[^1]);
            });
            var resumed = NetworkDocument.ReadTraining(text, "training", catalog, NetworkDocument.ReadNetwork(text, "network", catalog));

            Assert.Equal(optimizer.GetType(), resumed.Compiled.Optimizer.GetType());
            Assert.Equal(optimizer.Rate, resumed.Compiled.Optimizer.Rate);
            Assert.Equal(schedule.GetType(), resumed.Compiled.Schedule.GetType());
            Assert.Equal(schedule.RateAt(4, 0.1), resumed.Compiled.Schedule.RateAt(4, 0.1));
        }
    }

    [Fact]
    public void TheOptimizersPyTorchHasBeyondSgdAndAdam_AreWrittenWithEverySetting_AndReadBackHoldingThem()
    {
        var network = new LayerStack(new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias()));
        var catalog = NetworkCatalog.BuiltIn();

        var adamW = Assert.IsType<AdamW>(Resumed(new AdamW(0.02) { Betas = new Betas(0.8, 0.99), Epsilon = 1e-6, WeightDecay = 0.05 }).Optimizer);
        var rmsProp = Assert.IsType<RmsProp>(Resumed(new RmsProp(0.03) { Alpha = 0.9, Epsilon = 1e-7, Momentum = 0.5 }).Optimizer);
        var nadam = Assert.IsType<Nadam>(Resumed(new Nadam(0.04) { Betas = new Betas(0.85, 0.98), Epsilon = 1e-5, MomentumDecay = 0.002 }).Optimizer);

        Assert.Equal([0.02, 0.8, 0.99, 1e-6, 0.05], new[] { adamW.Rate, adamW.Betas.First, adamW.Betas.Second, adamW.Epsilon, adamW.WeightDecay });
        Assert.Equal([0.03, 0.9, 1e-7, 0.5], new[] { rmsProp.Rate, rmsProp.Alpha, rmsProp.Epsilon, rmsProp.Momentum });
        Assert.Equal([0.04, 0.85, 0.98, 1e-5, 0.002], new[] { nadam.Rate, nadam.Betas.First, nadam.Betas.Second, nadam.Epsilon, nadam.MomentumDecay });

        CompiledNetwork Resumed(Optimizer optimizer)
        {
            var kept = new List<Checkpoint>();
            var compiled = network.Compile(optimizer, new MeanSquaredError());
            compiled.Fit(Rows(8, width: 4), validation: null, new FitOptions(seed: 1) { Backend = _backend, Checkpoints = new Checkpoints(kept.Add) });
            var text = Checkpointed(compiled, kept[^1]);

            return NetworkDocument.ReadTraining(text, "training", catalog, NetworkDocument.ReadNetwork(text, "network", catalog)).Compiled;
        }
    }

    [Theory]
    [InlineData("adamw")]
    [InlineData("rmsprop")]
    [InlineData("nadam")]
    public void ARunOfEachNewOptimizer_WrittenToACheckpointsFileAndReadBack_GoesOnBitForBit(string kind)
    {
        // What each remembers crosses the file under PyTorch's names — Nadam's product of momentums worked out again from the
        // steps — and the run goes on to where the run that never stopped came.
        var straight = Rich();
        var stopped = Rich();
        var kept = new List<Checkpoint>();

        straight.Compile(Optimizer(kind), new MeanSquaredError()).Fit(Rows(96), Rows(24, 7), new FitOptions(seed: 5) { Backend = _backend, Epochs = 6, BatchSize = 16 });
        var compiled = stopped.Compile(Optimizer(kind), new MeanSquaredError());
        compiled.Fit(Rows(96), Rows(24, 7), new FitOptions(seed: 5) { Backend = _backend, Epochs = 3, BatchSize = 16, Checkpoints = new Checkpoints(kept.Add) });

        var text = Checkpointed(compiled, kept[^1]);
        var catalog = NetworkCatalog.BuiltIn();
        var resumed = NetworkDocument.ReadTraining(text, "training", catalog, NetworkDocument.ReadNetwork(text, "network", catalog));
        resumed.Compiled.Fit(Rows(96), Rows(24, 7), new FitOptions(seed: 5) { Backend = _backend, Epochs = 6, BatchSize = 16, ResumeFrom = resumed.Checkpoint });

        Assert.Equal(kind, JsonNode.Parse(text)!["training"]!["optimizer"]!["kind"]!.GetValue<string>());
        Assert.Equal(Bits(straight), Bits(resumed.Compiled.Network));

        static Optimizer Optimizer(string kind) => kind switch
        {
            "adamw" => new AdamW(0.01) { WeightDecay = 0.1 },
            "rmsprop" => new RmsProp(0.001) { Momentum = 0.9 },
            _ => new Nadam(0.01),
        };
    }

    [Fact]
    public void WhatANetworkWasTrainedOn_IsWrittenWithIt_AndReadBack()
    {
        var trainedOn = new TrainedOn
        {
            Features = ["age", "fare"],
            Answers = ["survived"],
            Output = "target",
            TrainedBehind = new string('a', 64),
            Seed = 20260929,
            Epoch = 11,
        };
        var network = new LayerStack(new Dense(Tensor.From(new Shape(2, 1), [0.5f, -0.5f]), Tensor.Zeros(new Shape(1))));

        var read = NetworkDocument.ReadNetwork(
            Written(writer =>
            {
                writer.WritePropertyName("network");
                NetworkDocument.WriteNetwork(writer, network, new BinaryCrossEntropy(), trainedOn);
            }),
            "network",
            NetworkCatalog.BuiltIn());

        Assert.Equal(["age", "fare"], read.TrainedOn!.Features);
        Assert.Equal(["survived"], read.TrainedOn.Answers);
        Assert.Equal("target", read.TrainedOn.Output);
        Assert.Equal(new string('a', 64), read.TrainedOn.TrainedBehind);
        Assert.Equal(20260929, read.TrainedOn.Seed);
        Assert.Equal(11, read.TrainedOn.Epoch);
        Assert.Null(NetworkDocument.ReadNetwork(Written(network, new BinaryCrossEntropy()), "network", NetworkCatalog.BuiltIn()).TrainedOn);
    }

    [Fact]
    public void TheNamesACatalogKnows_AreTheOnesThisPackageShips_AndARegistrationOfOneTwice_IsRefused()
    {
        var catalog = NetworkCatalog.BuiltIn();

        Assert.Equal(
            ["adam", "adamw", "avgpool1d", "avgpool2d", "avgpool3d", "batchNorm", "binaryCrossEntropy", "constant", "conv1d", "conv2d", "conv3d", "cosineDecay", "crossEntropy", "dense",
             "dropout", "earthMoversDistance", "exponentialDecay", "flatten", "globalavgpool1d", "globalavgpool2d", "globalavgpool3d", "globalmaxpool1d", "globalmaxpool2d", "globalmaxpool3d", "layerNorm", "linearWarmup", "maxpool1d", "maxpool2d",
             "maxpool3d", "meanSquaredError", "nadam", "relu", "reshape", "rmsprop", "sgd", "sigmoid", "spatialdropout1d", "spatialdropout2d", "spatialdropout3d", "stack", "stepDecay", "tanh"],
            catalog.Names.Order(StringComparer.Ordinal));
        Assert.Throws<ArgumentException>(() => catalog.Register<Dense>());
    }

    private static string Written(Network network, Loss loss) => Written(writer =>
    {
        writer.WritePropertyName("network");
        NetworkDocument.WriteNetwork(writer, network, loss);
    });

    // A network and a checkpoint of it, written beside each other as a checkpoint's file holds them.
    private static string Checkpointed(CompiledNetwork compiled, Checkpoint checkpoint) => Written(writer =>
    {
        writer.WritePropertyName("network");
        NetworkDocument.WriteNetwork(writer, compiled.Network, compiled.Loss);
        writer.WritePropertyName("training");
        NetworkDocument.WriteTraining(writer, compiled, checkpoint);
    });

    // A network of one dense layer weighing three features, trained on them, written with what it was trained on and read back.
    private static SavedNetwork Read(TrainedOn trainedOn) =>
        NetworkDocument.ReadNetwork(
            Written(writer =>
            {
                writer.WritePropertyName("network");
                NetworkDocument.WriteNetwork(writer, Weighing(), new BinaryCrossEntropy(), trainedOn);
            }),
            "network",
            NetworkCatalog.BuiltIn());

    private static LayerStack Weighing() => new(new Dense(Tensor.From(new Shape(3, 1), [0.5f, -0.5f, 0.25f]), Tensor.Zeros(new Shape(1))));

    private static TrainedOn Said(IReadOnlyDictionary<string, double>? unvaried) => new()
    {
        Features = ["age", "fare", "alone"],
        Answers = ["survived"],
        Output = "target",
        TrainedBehind = new string('a', 64),
        Seed = 20260929,
        Epoch = 11,
        Unvaried = unvaried,
    };

    private static string Written(Action<Utf8JsonWriter> body)
    {
        using var stream = new MemoryStream();

        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            body(writer);
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(stream.ToArray());
    }

    private static int[][] Bits(Layer network) =>
        [.. network.Slots().Select(slot => slot.Slot.Value.Values.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray())];

    private static LayerStack Rich() => new(
        new Dense(3, 8, new RandomStream(4).Draw("initialise:0", 0, 0)), new BatchNorm(8), new Relu(), new Dropout(0.25),
        new Dense(8, 1, new RandomStream(4).Draw("initialise:4", 0, 0)));

    private static TrainingData Rows(int count, int offset = 0, int width = 3)
    {
        var features = Enumerable.Range(offset * width, count * width).Select(at => MathF.Sin(at)).ToArray();
        var answers = Enumerable.Range(0, count).Select(row => 2 * features[row * width]).ToArray();

        return new TrainingData(Tensor.From(new Shape(count, width), features), Tensor.From(new Shape(count, 1), answers));
    }

    // A network written as code, registered under a name of its own.
    private sealed class Written2 : Network, ISaved<Written2>
    {
        private readonly Dense _first;
        private readonly Dense _second;

        public Written2(Draws draws)
        {
            _first = AddLayer("first", new Dense(3, 2, draws));
            _second = AddLayer("second", new Dense(2, 1, draws));
        }

        public static string Name => "written2";

        public static Written2 Rebuild(JsonElement settings, Rebuilding rebuilding) => new(rebuilding.Draws);

        public void WriteSettings(Utf8JsonWriter writer)
        {
        }

        protected override Tensor Compute(Tensor input, Pass pass) => _second.Forward(pass.Backend.Tanh(_first.Forward(input, pass)), pass);
    }

    // An optimizer written outside the library: plain steps against the gradient, counting the steps each parameter took.
    private sealed class CountingSteps(double rate) : Optimizer(rate), ISaved<CountingSteps>
    {
        private readonly Dictionary<Parameter, int> _steps = [];

        public static string Name => "countingSteps";

        public static CountingSteps Rebuild(JsonElement settings, Rebuilding rebuilding) => new(rebuilding.Number(settings, "rate"));

        public void WriteSettings(Utf8JsonWriter writer) => writer.WriteNumber("rate", Rate);

        public int StepsOf(Parameter parameter) => _steps.GetValueOrDefault(parameter);

        protected override SlotMemory? MemoryOf(Parameter parameter) =>
            _steps.TryGetValue(parameter, out var steps) ? new SlotMemory(steps, new Dictionary<string, Tensor>()) : null;

        protected override void Recall(Parameter parameter, SlotMemory memory) => _steps[parameter] = memory.Steps;

        protected override Tensor Moved(Parameter parameter, Tensor gradient, double rate, ITensorBackend backend)
        {
            _steps[parameter] = StepsOf(parameter) + 1;

            return backend.Subtract(parameter.Value, backend.Scale(gradient, Scalar(backend, rate)));
        }
    }
}
