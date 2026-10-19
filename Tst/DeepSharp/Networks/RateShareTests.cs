// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A rate of its own for a part of a network: a share of the run's rate that a parameter's numbers move at, said where every
/// other rule of a parameter is said — in <see cref="SlotRules"/> — and so by the doors the rules already have. Held to what
/// only the library's own door can see: a step that moves a parameter by the rate times its share on every optimizer, a file
/// that writes the share as the seventh version of a network's part and writes every network that says none as it always was,
/// and a run that goes on from a checkpoint under the shares it was taken under and is refused under others.
/// </summary>
public class RateShareTests
{
    private static readonly ITensorBackend Backend = new CpuBackend();

    public static TheoryData<string> Kinds => ["sgd", "sgdMomentum", "adam", "adamw", "rmsprop", "nadam"];

    [Fact]
    public void ARuleThatSaysNoShare_IsTheDefaultRule_AndMovesTheNumbersAtTheWholeRate()
    {
        Assert.Null(new SlotRules().RateShare);
        Assert.True(new SlotRules().IsDefault);
        Assert.False(new SlotRules { RateShare = 0.5 }.IsDefault);
        Assert.NotEqual(new SlotRules(), new SlotRules { RateShare = 0.5 });
        Assert.True(new SlotRules { RateShare = 0.5 }.Learns);
    }

    [Theory]
    [InlineData(-1.0)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void AShareThatIsNotANumberAboveNothing_IsRefused(double wrong) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new SlotRules { RateShare = wrong });

    [Fact]
    public void AShareOfNothing_IsRefused_PointingToTheHold()
    {
        var refused = Assert.Throws<ArgumentOutOfRangeException>(() => new SlotRules { RateShare = 0 });

        Assert.Contains("Held", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [MemberData(nameof(Kinds))]
    public void AParameterWithAShare_IsMovedAtTheRateTimesTheShare_AndAnotherBesideItAtTheWholeRate_OnEveryOptimizer(string kind)
    {
        var shared = Made(1);
        var whole = Made(2);
        var optimizer = Make(kind, 0.1);

        shared.Rules = new SlotRules { RateShare = 0.5 };

        // Three steps, so what an optimizer remembers of a parameter between steps is used: it is kept for each parameter's own
        // slot, and a share changes the rate a step moves it by, never what is remembered.
        for (var step = 0; step < 3; step++)
        {
            optimizer.Step([shared, whole], Grads(step, shared, whole), 0.1, Backend);
        }

        Assert.Equal(Reference(kind, 0.05, 1), Bits(shared));
        Assert.Equal(Reference(kind, 0.1, 2), Bits(whole));
    }

    [Fact]
    public void AShareAboveOne_MovesTheNumbersFurtherThanTheWholeRate()
    {
        var larger = Made(1);
        var plain = Made(1);

        larger.Rules = new SlotRules { RateShare = 2 };

        new Sgd(0.1).Step([larger], Grads(0, larger), 0.1, Backend);
        new Sgd(0.1).Step([plain], Grads(0, plain), 0.1, Backend);

        var start = Made(1).Value.Values.ToArray();
        var stepped = plain.Value.Values.ToArray();

        for (var at = 0; at < start.Length; at++)
        {
            Assert.Equal(start[at] - (2 * (start[at] - stepped[at])), larger.Value.Values[at], 5);
        }
    }

    [Fact]
    public void ALayersShare_GoesOnEveryNumberItKeeps_AndSayingOneTakesItOff()
    {
        var (hidden, output, _) = Network();

        hidden.ScaleRate(0.25);
        output.Weight.Rules = new SlotRules { L2 = 0.1 };

        Assert.Equal(0.25, hidden.Weight.Rules.RateShare);
        Assert.Equal(0.25, hidden.Bias!.Rules.RateShare);
        Assert.Null(output.Weight.Rules.RateShare);

        hidden.ScaleRate(1);

        Assert.Null(hidden.Weight.Rules.RateShare);
        Assert.Equal(default, hidden.Bias!.Rules);
        Assert.Throws<ArgumentOutOfRangeException>(() => hidden.ScaleRate(0));
    }

    [Fact]
    public void TheShareWord_PutsTheShareOnTheLayerTheWordBeforeItMakes_AndKeepsWhatTheOtherRuleWordsPut()
    {
        var stack = new Sequential()
            .Dense(4).RateShare(0.1).Ruled(new SlotRules { L2 = 0.2 })
            .Relu()
            .Dense(2).Ruled(new SlotRules { L1 = 0.3, MaxNorm = 2 }).RateShare(0.5).Held()
            .Dense(1)
            .Lower(new Shape(3), new RandomStream(1));
        var dense = stack.Layers.OfType<Dense>().ToArray();

        Assert.Equal(new SlotRules { L2 = 0.2, RateShare = 0.1 }, dense[0].Weight.Rules);
        Assert.Equal(new SlotRules { RateShare = 0.1 }, dense[0].Bias!.Rules);
        Assert.Equal(new SlotRules { L1 = 0.3, MaxNorm = 2, RateShare = 0.5, Held = true }, dense[1].Weight.Rules);
        Assert.Equal(new SlotRules { RateShare = 0.5, Held = true }, dense[1].Bias!.Rules);
        Assert.Equal(default, dense[2].Weight.Rules);
    }

    [Fact]
    public void ARuleWordAfterTheShareWord_DoesNotWipeTheShare_AndTheShareWordAfterItDoesNotWipeThePenalty()
    {
        var stack = new Sequential()
            .Dense(2).RateShare(0.4).Ruled(new SlotRules { L2 = 0.1 }, new SlotRules { L1 = 0.2 })
            .Dense(1).Ruled(new SlotRules { L1 = 0.5 }).RateShare(0.4)
            .Lower(new Shape(3), new RandomStream(1));
        var dense = stack.Layers.OfType<Dense>().ToArray();

        Assert.Equal(new SlotRules { L2 = 0.1, RateShare = 0.4 }, dense[0].Weight.Rules);
        Assert.Equal(new SlotRules { L1 = 0.2, RateShare = 0.4 }, dense[0].Bias!.Rules);
        Assert.Equal(new SlotRules { L1 = 0.5, RateShare = 0.4 }, dense[1].Weight.Rules);
        Assert.Throws<InvalidOperationException>(() => new Sequential().RateShare(0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Sequential().Dense(1).RateShare(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Sequential().Dense(1).RateShare(double.NaN));
    }

    [Fact]
    public void AShareWordForALayerThatHoldsNoNumber_IsMadeAsItWas()
    {
        var stack = new Sequential().Dense(2).Relu().RateShare(0.5).Dense(1).Lower(new Shape(3), new RandomStream(1));

        Assert.IsType<Relu>(stack.Layers[1]);
    }

    [Fact]
    public void ANetworkThatSaysNoShare_IsWrittenAsItWasWritten_AndOneThatOnlyHoldsAParameterIsStillTheFifth()
    {
        var plain = Written(Network().Network);
        var (hidden, _, network) = Network();

        hidden.Hold();

        var held = Written(network);

        Assert.Contains("\"version\":2", plain, StringComparison.Ordinal);
        Assert.DoesNotContain("rateShare", plain, StringComparison.Ordinal);
        Assert.Contains("\"version\":5", held, StringComparison.Ordinal);
        Assert.DoesNotContain("rateShare", held, StringComparison.Ordinal);
    }

    [Fact]
    public void ANetworkWhoseParameterHasAShare_IsWrittenAsTheSeventhVersion_TheShareBesideTheNumbers_AndReadBackAsItWas()
    {
        var (hidden, output, network) = Network();

        hidden.Weight.Rules = new SlotRules { RateShare = 0.1, L2 = 0.02 };
        output.Bias!.Rules = new SlotRules { RateShare = 3, Held = true };

        var text = Written(network);
        using var document = JsonDocument.Parse(text);
        var part = document.RootElement.GetProperty("network");
        var parameters = part.GetProperty("parameters");

        Assert.Equal(7, part.GetProperty("version").GetInt32());
        Assert.Equal(0.1, parameters.GetProperty("0.weight").GetProperty("rateShare").GetDouble());
        Assert.Equal(0.02, parameters.GetProperty("0.weight").GetProperty("l2").GetDouble());
        Assert.False(parameters.GetProperty("0.bias").TryGetProperty("rateShare", out _));
        Assert.Equal(3, parameters.GetProperty("2.bias").GetProperty("rateShare").GetDouble());

        var read = Assert.IsType<LayerStack>(NetworkDocument.ReadNetwork(text, "network", NetworkCatalog.BuiltIn()).Network);
        var layers = read.Layers.OfType<Dense>().ToArray();

        Assert.Equal(new SlotRules { RateShare = 0.1, L2 = 0.02 }, layers[0].Weight.Rules);
        Assert.Equal(default, layers[0].Bias!.Rules);
        Assert.Equal(new SlotRules { RateShare = 3, Held = true }, layers[1].Bias!.Rules);
        Assert.Equal(text, Written(read));
    }

    [Fact]
    public void AShareInAPartOfAnEarlierVersion_IsRefusedByTheVersionThatCannotSayIt_NamingTheSeventh()
    {
        var (hidden, _, network) = Network();

        hidden.Weight.Rules = new SlotRules { RateShare = 0.1 };

        var earlier = Written(network).Replace("\"version\":7", "\"version\":6", StringComparison.Ordinal);
        var refused = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(earlier, "network", NetworkCatalog.BuiltIn()));

        Assert.Contains("'rateShare'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("seventh", refused.Message, StringComparison.Ordinal);
        Assert.Contains("sixth", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("0")]
    [InlineData("-1")]
    [InlineData("\"x\"")]
    [InlineData("true")]
    public void AShareWrittenAsSomethingItIsNot_IsRefusedWhereItStands(string wrong)
    {
        var (hidden, _, network) = Network();

        hidden.Weight.Rules = new SlotRules { RateShare = 0.1 };

        var root = JsonNode.Parse(Written(network))!.AsObject();

        root["network"]!["parameters"]!["0.weight"]!["rateShare"] = JsonNode.Parse(wrong);

        var refused = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(root.ToJsonString(), "network", NetworkCatalog.BuiltIn()));

        Assert.Contains("'rateShare'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ARunWithShares_ThatStopsAndGoesOnFromAFile_IsTheRunThatNeverStopped_BitForBit()
    {
        var straight = Network();
        var stopped = Network();

        Share(straight);
        Share(stopped);

        straight.Network.Compile(new Adam(0.05), new BinaryCrossEntropy()).Fit(Rows(), Rows(), Options(5));

        string? file = null;
        var compiled = stopped.Network.Compile(new Adam(0.05), new BinaryCrossEntropy());

        compiled.Fit(Rows(), Rows(), Options(2, new Checkpoints(checkpoint => file = WrittenWith(compiled, checkpoint))));

        var read = NetworkDocument.ReadCheckpoint(file!, "network", "training", NetworkCatalog.BuiltIn());

        Assert.Equal(0.1, Assert.IsType<LayerStack>(read.Run.Compiled.Network).Layers.OfType<Dense>().First().Weight.Rules.RateShare);

        read.Run.Compiled.Fit(Rows(), Rows(), Options(5, from: read.Run.Checkpoint));

        Assert.Equal(AllBits(straight.Network), AllBits(read.Run.Compiled.Network));

        FitOptions Options(int epochs, Checkpoints? checkpoints = null, Checkpoint? from = null) => new(seed: 5)
        {
            Epochs = epochs, BatchSize = 2, Checkpoints = checkpoints, ResumeFrom = from,
        };
    }

    [Fact]
    public void ARunWithAShare_ChangesTheNumbersItLearns_ComparedWithoutOne()
    {
        var shared = Network();
        var plain = Network();

        Share(shared);
        shared.Network.Compile(new Sgd(0.1), new BinaryCrossEntropy()).Fit(Rows(), validation: null, new FitOptions(seed: 1) { Epochs = 3, BatchSize = 2 });
        plain.Network.Compile(new Sgd(0.1), new BinaryCrossEntropy()).Fit(Rows(), validation: null, new FitOptions(seed: 1) { Epochs = 3, BatchSize = 2 });

        Assert.NotEqual(AllBits(plain.Network), AllBits(shared.Network));
    }

    [Fact]
    public void ARunGoneOnUnderOtherShares_ThanItsCheckpointWasTakenUnder_IsRefusedNamingWhatWasShared()
    {
        var first = Network();
        var kept = new List<Checkpoint>();

        first.Hidden.ScaleRate(0.1);
        first.Network.Compile(new Sgd(0.1), new BinaryCrossEntropy()).Fit(Rows(), validation: null, new FitOptions(seed: 1) { Epochs = 1, BatchSize = 2, Checkpoints = new Checkpoints(kept.Add) });

        var other = Network();

        other.Output.ScaleRate(0.5);

        var refused = Assert.Throws<ArgumentException>(
            () => other.Network.Compile(new Sgd(0.1), new BinaryCrossEntropy()).Fit(Rows(), validation: null, new FitOptions(seed: 1) { Epochs = 2, BatchSize = 2, ResumeFrom = kept[0] }));

        Assert.Contains("'0.weight'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'2.weight'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("rate", refused.Message, StringComparison.Ordinal);

        // And the same shares are gone on under without a word.
        var same = Network();

        same.Hidden.ScaleRate(0.1);
        same.Network.Compile(new Sgd(0.1), new BinaryCrossEntropy()).Fit(Rows(), validation: null, new FitOptions(seed: 1) { Epochs = 2, BatchSize = 2, ResumeFrom = kept[0] });
    }

    [Fact]
    public void ARunGoneOnWithAShareWhereItsCheckpointHadNone_IsRefused()
    {
        var first = Network();
        var kept = new List<Checkpoint>();

        first.Network.Compile(new Sgd(0.1), new BinaryCrossEntropy()).Fit(Rows(), validation: null, new FitOptions(seed: 1) { Epochs = 1, BatchSize = 2, Checkpoints = new Checkpoints(kept.Add) });

        var shared = Network();

        shared.Hidden.ScaleRate(0.5);

        Assert.Throws<ArgumentException>(
            () => shared.Network.Compile(new Sgd(0.1), new BinaryCrossEntropy()).Fit(Rows(), validation: null, new FitOptions(seed: 1) { Epochs = 2, BatchSize = 2, ResumeFrom = kept[0] }));
    }

    private static Optimizer Make(string kind, double rate) => kind switch
    {
        "sgd" => new Sgd(rate),
        "sgdMomentum" => new Sgd(rate) { Momentum = 0.9, WeightDecay = 0.01 },
        "adam" => new Adam(rate),
        "adamw" => new AdamW(rate),
        "rmsprop" => new RmsProp(rate),
        "nadam" => new Nadam(rate),
        _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, null),
    };

    // A parameter worked from its place, so the same place is the same numbers.
    private static Parameter Made(int place) =>
        new($"p{place}", Tensor.From(new Shape(2, 3), [.. Enumerable.Range(0, 6).Select(at => ((place * 7) + (at * 3)) % 11 / 10f - 0.4f)]));

    // The gradients of one step: other numbers each step, the same for the same place.
    private static Gradients Grads(int step, params Parameter[] parameters) => new(parameters.ToDictionary(
        parameter => parameter.Value,
        parameter => Tensor.From(parameter.Value.Shape, [.. Enumerable.Range(0, parameter.Value.Shape.Count).Select(at => (((step + 2) * (at + 3)) % 7 / 7f) - 0.3f)])));

    // What a parameter without a share comes to under three steps at the given rate: the number a share is held to.
    private static int[] Reference(string kind, double rate, int place)
    {
        var plain = Made(place);
        var optimizer = Make(kind, rate);

        for (var step = 0; step < 3; step++)
        {
            optimizer.Step([plain], Grads(step, plain), rate, Backend);
        }

        return Bits(plain);
    }

    private static int[] Bits(Parameter parameter) => [.. parameter.Value.Values.ToArray().Select(BitConverter.SingleToInt32Bits)];

    private static int[][] AllBits(Layer network) =>
        [.. network.Parameters().Select(parameter => parameter.Value.Values.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray())];

    private static void Share((Dense Hidden, Dense Output, LayerStack Network) walked)
    {
        walked.Hidden.ScaleRate(0.1);
        walked.Output.Weight.Rules = new SlotRules { RateShare = 2 };
    }

    private static (Dense Hidden, Dense Output, LayerStack Network) Network()
    {
        var hidden = new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias());
        var output = new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias());

        return (hidden, output, new LayerStack(hidden, new Relu(), output));
    }

    private static TrainingData Rows() => new(WalkedRows.Passengers(), WalkedRows.Survived());

    private static string Written(Network network)
    {
        var buffer = new ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, network, new MeanSquaredError());
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }

    private static string WrittenWith(CompiledNetwork compiled, Checkpoint checkpoint)
    {
        var buffer = new ArrayBufferWriter<byte>();

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
}
