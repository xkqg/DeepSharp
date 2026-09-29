// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A network written down: its layers by the names they are registered under and the settings they are rebuilt from,
/// every slot's numbers by its path, its loss by name — and, for a checkpoint, everything a run needs to go on. Read back
/// through a catalog of the kinds a reader knows, it is the same network to the last bit; anything else in the file is
/// refused, every fault at its line and column.
/// </summary>
public class NetworkDocumentTests
{
    private readonly ITensorBackend _backend = new CpuBackend();

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
        Assert.Contains("DeepSharp.Tests", wrong.Message, StringComparison.Ordinal);
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
        var text = Written(new LayerStack(new Relu()), new MeanSquaredError()).Replace("\"version\": 1", "\"version\": 2", StringComparison.Ordinal);

        var wrong = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(text, "network", NetworkCatalog.BuiltIn()));

        Assert.Single(wrong.Faults);
        Assert.Contains("version 2", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AValueThatIsNotAFiniteNumber_CannotBeWritten()
    {
        var network = new LayerStack(new Dense(Tensor.From(new Shape(1, 1), [float.NaN]), Tensor.Zeros(new Shape(1))));

        var wrong = Assert.Throws<InvalidOperationException>(() => Written(network, new MeanSquaredError()));

        Assert.Contains("0.weight", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACheckpointWrittenAndReadBack_ResumesTheRunAsIfItHadNeverStopped()
    {
        var straight = Rich();
        var stopped = Rich();
        var kept = new List<Checkpoint>();
        var options = new FitOptions(seed: 5) { Epochs = 6, BatchSize = 16, EarlyStopping = new EarlyStopping { Patience = 10, RestoreBest = true } };

        straight.Compile(new Adam(0.01), new MeanSquaredError(), new CosineDecay(6, 0.001)).Fit(Rows(96), Rows(24, 7), options);
        var compiled = stopped.Compile(new Adam(0.01), new MeanSquaredError(), new CosineDecay(6, 0.001));
        compiled.Fit(Rows(96), Rows(24, 7), new FitOptions(seed: 5)
        {
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
            Epochs = 6, BatchSize = 16, EarlyStopping = new EarlyStopping { Patience = 10, RestoreBest = true }, ResumeFrom = resumed.Checkpoint,
        });

        Assert.Equal(Bits(straight), Bits(resumed.Compiled.Network));
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
                 })
        {
            var compiled = network.Compile(optimizer, new MeanSquaredError(), schedule);
            compiled.Fit(Rows(8, width: 4), validation: null, new FitOptions(seed: 1) { Checkpoints = new Checkpoints(kept.Add) });

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
            ["adam", "batchNorm", "binaryCrossEntropy", "constant", "conv2d", "cosineDecay", "crossEntropy", "dense", "dropout", "exponentialDecay",
             "flatten", "layerNorm", "meanSquaredError", "relu", "reshape", "sgd", "sigmoid", "stack", "stepDecay", "tanh"],
            catalog.Names.Order(StringComparer.Ordinal));
        Assert.Throws<ArgumentException>(() => catalog.Register<Dense>());
    }

    private static string Written(Network network, Loss loss) => Written(writer =>
    {
        writer.WritePropertyName("network");
        NetworkDocument.WriteNetwork(writer, network, loss);
    });

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
}
