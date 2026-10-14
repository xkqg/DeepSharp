// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// What a run does around its optimizer, held to what only the library's own door can see: a window of batches added up and
/// averaged by hand lands where the loop's does, a window cut short by the end of an epoch is stepped there, an average of
/// the weights and the options that went with it are written into a checkpoint's training part as the fifth version and read
/// back into a run that goes on bit for bit — and a part of an earlier version, or one that says half of it, is refused by name.
/// </summary>
public class RunOptionsTests
{
    private static readonly ITensorBackend Backend = new CpuBackend();

    [Fact]
    public void AStepTakenOnceEveryThreeBatches_IsTheMeanOfThreeBatchesStepped_TheWindowCutShortByTheEpochsEndStepped_AsItStands()
    {
        // Four rows in batches of one make four batches an epoch; every three, a step after the third and one after the
        // fourth, which the end of the epoch cuts to a window of one — and no window goes across the end of an epoch.
        var network = Network();
        var fitted = Network();
        var sgd = new Sgd(0.1);

        network.Compile(sgd, new BinaryCrossEntropy()).Fit(Rows(), validation: null, new FitOptions(seed: 7) { Epochs = 2, BatchSize = 1, StepEvery = 3 });

        var stream = new RandomStream(7);
        var manual = new Sgd(0.1);

        for (var epoch = 0; epoch < 2; epoch++)
        {
            var order = stream.Draw("shuffle", epoch, 0).Permutation(4);
            var sums = new Dictionary<Tensor, Tensor>(ReferenceEqualityComparer.Instance);
            var counted = 0;

            for (var step = 0; step < 4; step++)
            {
                var batch = Rows().Rows(order.AsSpan(step, 1));
                var recording = new RecordingBackend(Backend);
                var loss = new BinaryCrossEntropy().Of(fitted.Forward(batch.Features, Pass.Training(recording, stream, epoch, step)), batch.Answers, recording);
                var gradients = recording.GradientsOf(loss, fitted.Parameters().Select(parameter => parameter.Value));

                foreach (var parameter in fitted.Parameters())
                {
                    sums[parameter.Value] = sums.TryGetValue(parameter.Value, out var sum) ? Backend.Add(sum, gradients[parameter.Value]) : gradients[parameter.Value];
                }

                counted++;

                if (counted == 3 || step == 3)
                {
                    var mean = new Gradients(sums.ToDictionary(each => each.Key, each => counted == 1 ? each.Value : Backend.Scale(each.Value, Backend.Fill(new Shape(), (float)(1.0 / counted)))));

                    manual.Step(fitted.Parameters(), mean, 0.1, Backend);
                    sums.Clear();
                    counted = 0;
                }
            }
        }

        Assert.Equal(Bits(fitted), Bits(network));
    }

    [Fact]
    public void ClippingEachValue_BoundsEveryGradientValueByTheBound_AndLeavesTheOnesInsideItAlone()
    {
        var parameter = new Dense(Tensor.Zeros(new Shape(1, 4)), Tensor.Zeros(new Shape(4))).Weight;
        var gradients = new Gradients(new Dictionary<Tensor, Tensor>(ReferenceEqualityComparer.Instance) { [parameter.Value] = Tensor.From(new Shape(1, 4), [0.5f, -0.5f, 0.01f, -0.02f]) });

        var clipped = gradients.Clamped([parameter], 0.02, Backend)[parameter.Value];

        Assert.Equal<float[]>([0.02f, -0.02f, 0.01f, -0.02f], clipped.Values.ToArray());
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void AGradientThatIsNoFiniteNumber_IsRefusedByNameBeforeItsValuesAreClipped(float wrong)
    {
        var parameter = new Dense(Tensor.Zeros(new Shape(1, 2)), Tensor.Zeros(new Shape(2))).Weight;
        var gradients = new Gradients(new Dictionary<Tensor, Tensor>(ReferenceEqualityComparer.Instance) { [parameter.Value] = Tensor.From(new Shape(1, 2), [0.5f, wrong]) });

        var refused = Assert.Throws<InvalidOperationException>(() => gradients.Clamped([parameter], 0.02, Backend));

        Assert.Contains(parameter.Name, refused.Message, StringComparison.Ordinal);
        Assert.Contains("not a finite number", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ARunWithNoneOfTheOptions_IsWrittenAsItAlwaysWas_AndTheOnesThatSaySomethingNeedTheFifthVersion()
    {
        Assert.Equal(2, TrainingVersion(new FitOptions(seed: 3) { Epochs = 1, BatchSize = 2 }));
        Assert.Equal(3, TrainingVersion(new FitOptions(seed: 3) { Epochs = 1, BatchSize = 2, GradientClip = new GradientClip(1) }));
        Assert.Equal(5, TrainingVersion(new FitOptions(seed: 3) { Epochs = 1, BatchSize = 2, ClipValue = 0.5 }));
        Assert.Equal(5, TrainingVersion(new FitOptions(seed: 3) { Epochs = 1, BatchSize = 2, StepEvery = 2 }));
        Assert.Equal(5, TrainingVersion(new FitOptions(seed: 3) { Epochs = 1, BatchSize = 2, Average = 0.9 }));
        Assert.Equal(5, TrainingVersion(new FitOptions(seed: 3) { Epochs = 1, BatchSize = 2, GradientClip = new GradientClip(1), ClipValue = 0.5 }));
    }

    [Fact]
    public void TheOptionsOfARun_AreWrittenIntoTheTrainingPart_AndReadBackIntoARunThatGoesOnBitForBit()
    {
        var options = new FitOptions(seed: 3) { Epochs = 2, BatchSize = 1, StepEvery = 3, ClipValue = 0.02, GradientClip = new GradientClip(0.5), Average = 0.9 };
        var compiled = Network().Compile(new Adam(0.05), new BinaryCrossEntropy());
        var (written, kept) = Written(compiled, options);

        using var document = JsonDocument.Parse(written);
        var training = document.RootElement.GetProperty("training");

        Assert.Equal(5, training.GetProperty("version").GetInt32());
        Assert.Equal(0.02, training.GetProperty("clipValue").GetDouble());
        Assert.Equal(3, training.GetProperty("stepEvery").GetInt32());
        Assert.Equal(0.9, training.GetProperty("average").GetDouble());
        Assert.Equal(compiled.Network.Slots().Where(named => named.Slot is Parameter).Select(named => named.Path).Order(StringComparer.Ordinal), training.GetProperty("averaged").EnumerateObject().Select(each => each.Name).Order(StringComparer.Ordinal));

        var read = NetworkDocument.ReadCheckpoint(written, "network", "training", NetworkCatalog.BuiltIn());

        Assert.Equal(written, WrittenAgain(read.Run.Compiled, read.Run.Checkpoint));

        // The run goes on from the file as it does from the checkpoint in memory.
        var straight = Network();
        var onwards = read.Run.Compiled;

        straight.Compile(new Adam(0.05), new BinaryCrossEntropy()).Fit(Rows(), validation: null, WithEpochs(options, 4));
        onwards.Fit(Rows(), validation: null, WithEpochs(options, 4, read.Run.Checkpoint));

        Assert.Equal(Bits(straight), Bits(onwards.Network));
        Assert.NotNull(kept);
    }

    [Fact]
    public void ARunWithNoOption_WritesNoneOfTheirKeys()
    {
        var (written, _) = Written(Network().Compile(new Adam(0.05), new BinaryCrossEntropy()), new FitOptions(seed: 3) { Epochs = 1, BatchSize = 2 });

        foreach (var key in new[] { "clipValue", "stepEvery", "\"average\"", "averaged" })
        {
            Assert.DoesNotContain(key, written, StringComparison.Ordinal);
        }
    }

    [Theory]
    [InlineData("clipValue", 2)]
    [InlineData("stepEvery", 2)]
    [InlineData("average", 2)]
    [InlineData("clipValue", 4)]
    [InlineData("stepEvery", 3)]
    public void AnOptionInATrainingPartOfAnEarlierVersion_IsRefusedByTheVersionThatCannotSayIt(string key, int version)
    {
        var options = new FitOptions(seed: 3) { Epochs = 1, BatchSize = 2, ClipValue = 0.02, StepEvery = 2, Average = 0.9 };
        var (written, _) = Written(Network().Compile(new Adam(0.05), new BinaryCrossEntropy()), options);
        var earlier = written.Replace("\"version\":5", $"\"version\":{version}", StringComparison.Ordinal);

        var refused = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadCheckpoint(earlier, "network", "training", NetworkCatalog.BuiltIn()));

        Assert.Contains(key, refused.Message, StringComparison.Ordinal);
        Assert.Contains("fifth", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnAverageWithoutItsTensors_AndTensorsWithoutTheirAverage_AreRefusedByName()
    {
        var options = new FitOptions(seed: 3) { Epochs = 1, BatchSize = 2, Average = 0.9 };
        var (written, _) = Written(Network().Compile(new Adam(0.05), new BinaryCrossEntropy()), options);
        var withoutTensors = JsonNode(written, training => training.Remove("averaged"));
        var withoutAverage = JsonNode(written, training => training.Remove("average"));

        Assert.Contains("averaged", Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadCheckpoint(withoutTensors, "network", "training", NetworkCatalog.BuiltIn())).Message, StringComparison.Ordinal);
        Assert.Contains("averaged", Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadCheckpoint(withoutAverage, "network", "training", NetworkCatalog.BuiltIn())).Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("clipValue", "0")]
    [InlineData("clipValue", "\"x\"")]
    [InlineData("stepEvery", "0")]
    [InlineData("stepEvery", "1.5")]
    [InlineData("average", "1")]
    [InlineData("average", "0")]
    public void AnOptionWrittenAsSomethingItIsNot_IsRefusedWhereItStands(string key, string wrong)
    {
        var options = new FitOptions(seed: 3) { Epochs = 1, BatchSize = 2, ClipValue = 0.02, StepEvery = 2, Average = 0.9 };
        var (written, _) = Written(Network().Compile(new Adam(0.05), new BinaryCrossEntropy()), options);
        var broken = JsonNode(written, training => training[key] = System.Text.Json.Nodes.JsonNode.Parse(wrong));

        Assert.Contains(key, Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadCheckpoint(broken, "network", "training", NetworkCatalog.BuiltIn())).Message, StringComparison.Ordinal);
    }

    private static int TrainingVersion(FitOptions options)
    {
        var (written, _) = Written(Network().Compile(new Sgd(0.1), new BinaryCrossEntropy()), options);

        using var document = JsonDocument.Parse(written);

        return document.RootElement.GetProperty("training").GetProperty("version").GetInt32();
    }

    private static LayerStack Network() => new(
        new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias()), new Relu(), new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias()));

    private static TrainingData Rows() => new(WalkedRows.Passengers(), WalkedRows.Survived());

    private static FitOptions WithEpochs(FitOptions options, int epochs, Checkpoint? from = null) => new(options.Seed)
    {
        Epochs = epochs,
        BatchSize = options.BatchSize,
        StepEvery = options.StepEvery,
        ClipValue = options.ClipValue,
        GradientClip = options.GradientClip,
        Average = options.Average,
        ResumeFrom = from,
    };

    // A run of the options given, and the network and the training part as one file holds them at the end of its last epoch — taken
    // as a checkpoint is, while the network is as the checkpoint left it.
    private static (string Text, Checkpoint Checkpoint) Written(CompiledNetwork compiled, FitOptions options)
    {
        Checkpoint? kept = null;
        string? text = null;
        var run = new FitOptions(options.Seed)
        {
            Epochs = options.Epochs,
            BatchSize = options.BatchSize,
            StepEvery = options.StepEvery,
            ClipValue = options.ClipValue,
            GradientClip = options.GradientClip,
            Average = options.Average,
            Checkpoints = new Checkpoints(checkpoint =>
            {
                kept = checkpoint;
                text = WrittenAgain(compiled, checkpoint);
            }),
        };

        compiled.Fit(Rows(), validation: null, run);

        return (text!, kept!);
    }

    [Theory]
    [InlineData(null, 0.5, null, null, "whose gradients were never clipped value by value, and going on clipping them value by value to 0.5 would move every step otherwise.")]
    [InlineData(0.5, null, null, null, "whose gradients were clipped value by value to 0.5, and going on without clipping them that way would move every step otherwise.")]
    [InlineData(0.5, 0.25, null, null, "whose gradients were clipped value by value to 0.5, and going on clipping them value by value to 0.25 would move every step otherwise.")]
    [InlineData(0.5, 0.5, null, null, null)]
    [InlineData(null, null, null, 0.9, "that kept no average of its weights, and going on keeping one at 0.9 would end it holding other weights.")]
    [InlineData(null, null, 0.9, null, "that kept an average of its weights at 0.9, and going on without one would end it holding other weights.")]
    [InlineData(null, null, 0.9, 0.5, "that kept an average of its weights at 0.9, and going on keeping one at 0.5 would end it holding other weights.")]
    [InlineData(null, null, 0.9, 0.9, null)]
    public void AClipOfEachValueAndAnAverage_AreEachNamedByWhatTheyDo_WhenARunGoesOnUnderOthers(double? takenClip, double? handedClip, double? takenAverage, double? handedAverage, string? says)
    {
        var taken = new Pace(32, null) { ClipValue = takenClip, Average = takenAverage };
        var handed = new FitOptions(seed: 1) { ClipValue = handedClip, Average = handedAverage };

        var unlike = taken.Unlike(handed).ToArray();

        if (says is null)
        {
            Assert.Empty(unlike);
        }
        else
        {
            Assert.Equal($"The checkpoint was taken of a run {says}", Assert.Single(unlike));
        }
    }

    [Fact]
    public void AWindowOfBatches_IsNamedByHowOftenAStepWasTaken_WhenARunGoesOnUnderAnother()
    {
        var unlike = new Pace(32, null) { StepEvery = 3 }.Unlike(new FitOptions(seed: 1) { StepEvery = 2 }).ToArray();

        Assert.Equal(
            "The checkpoint was taken of a run that took a step once every 3 batches, and going on taking one once every 2 batches would take other rows into every step.",
            Assert.Single(unlike));
    }

    private static string WrittenAgain(CompiledNetwork compiled, Checkpoint checkpoint)
    {
        var buffer = new System.Buffers.ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, compiled.Network, compiled.Loss);
            writer.WritePropertyName("training");
            NetworkDocument.WriteTraining(writer, compiled, checkpoint);
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }

    // A file's text with its training part changed.
    private static string JsonNode(string text, Action<System.Text.Json.Nodes.JsonObject> change)
    {
        var root = System.Text.Json.Nodes.JsonNode.Parse(text)!.AsObject();

        change(root["training"]!.AsObject());

        return root.ToJsonString();
    }

    private static int[][] Bits(Layer network) =>
        [.. network.Parameters().Select(parameter => parameter.Value.Values.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray())];
}
