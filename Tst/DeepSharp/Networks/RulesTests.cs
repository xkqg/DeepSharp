// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// What the learning does to a parameter's numbers beside stepping them, held to what only the library's own door can see: the
/// rules written beside the numbers in a network's file, as the fifth version and as nothing at all where there are none; a
/// parameter held still that no optimizer remembers anything of and no step moves; a run that goes on from a checkpoint under
/// the rules it was taken under, bit for bit, and is refused under others; and a bound that a transposed convolution cannot be
/// put under is refused by name.
/// </summary>
public class RulesTests
{
    private static readonly ITensorBackend Backend = new CpuBackend();

    [Fact]
    public void ANetworkWhoseParametersLearnAsTheyAlwaysDid_IsWrittenAsItAlwaysWas_WithNoneOfTheRulesKeys()
    {
        var text = Written(Network().Network);

        Assert.Contains("\"version\":2", text, StringComparison.Ordinal);

        foreach (var key in new[] { "learns", "\"l1\"", "\"l2\"", "maxNorm" })
        {
            Assert.DoesNotContain(key, text, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void TheRulesOfAParameter_AreWrittenBesideItsNumbers_AsTheFifthVersion_AndReadBackAsTheyWere()
    {
        var (hidden, output, network) = Network();

        hidden.Weight.Rules = new SlotRules { L1 = 0.01, L2 = 0.02, MaxNorm = 0.5 };
        hidden.Bias!.Rules = new SlotRules { Held = true };
        output.Weight.Rules = new SlotRules { L2 = 0.1 };

        var text = Written(network);
        using var document = JsonDocument.Parse(text);
        var parameters = document.RootElement.GetProperty("network").GetProperty("parameters");

        Assert.Equal(5, document.RootElement.GetProperty("network").GetProperty("version").GetInt32());
        Assert.Equal(0.01, parameters.GetProperty("0.weight").GetProperty("l1").GetDouble());
        Assert.Equal(0.02, parameters.GetProperty("0.weight").GetProperty("l2").GetDouble());
        Assert.Equal(0.5, parameters.GetProperty("0.weight").GetProperty("maxNorm").GetDouble());
        Assert.False(parameters.GetProperty("0.bias").GetProperty("learns").GetBoolean());
        Assert.False(parameters.GetProperty("0.bias").TryGetProperty("l1", out _));
        Assert.False(parameters.GetProperty("2.bias").TryGetProperty("learns", out _));
        Assert.Equal(0.1, parameters.GetProperty("2.weight").GetProperty("l2").GetDouble());

        var read = Assert.IsType<LayerStack>(NetworkDocument.ReadNetwork(text, "network", NetworkCatalog.BuiltIn()).Network);
        var layers = read.Layers.OfType<Dense>().ToArray();

        Assert.Equal(new SlotRules { L1 = 0.01, L2 = 0.02, MaxNorm = 0.5 }, layers[0].Weight.Rules);
        Assert.Equal(new SlotRules { Held = true }, layers[0].Bias!.Rules);
        Assert.Equal(default, layers[1].Bias!.Rules);
        Assert.Equal(new SlotRules { L2 = 0.1 }, layers[1].Weight.Rules);
        Assert.Equal(text, Written(read));
    }

    [Theory]
    [InlineData("l1")]
    [InlineData("learns")]
    [InlineData("maxNorm")]
    public void RulesInAPartOfAnEarlierVersion_AreRefusedByTheVersionThatCannotSayThem(string key)
    {
        var (hidden, _, network) = Network();

        hidden.Weight.Rules = new SlotRules { L1 = 0.01, MaxNorm = 0.5, Held = true };

        var earlier = Written(network).Replace("\"version\":5", "\"version\":4", StringComparison.Ordinal);
        var refused = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(earlier, "network", NetworkCatalog.BuiltIn()));

        Assert.Contains($"'{key}'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("fifth", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("l1", "-1")]
    [InlineData("l2", "\"x\"")]
    [InlineData("maxNorm", "0")]
    [InlineData("learns", "true")]
    [InlineData("learns", "1")]
    public void RulesWrittenAsSomethingTheyAreNot_AreRefusedWhereTheyStand(string key, string wrong)
    {
        var (hidden, _, network) = Network();

        hidden.Weight.Rules = new SlotRules { L1 = 0.01, MaxNorm = 0.5, Held = true, L2 = 0.2 };

        var root = JsonNode.Parse(Written(network))!.AsObject();

        root["network"]!["parameters"]!["0.weight"]![key] = JsonNode.Parse(wrong);

        var refused = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(root.ToJsonString(), "network", NetworkCatalog.BuiltIn()));

        Assert.Contains($"'{key}'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ARulesKeyBesideARunningStatistic_IsNotWrittenThere_AndIsRefused()
    {
        var norm = new LayerStack(new Dense(2, 2, Draws()), new BatchNorm(2));
        var root = JsonNode.Parse(Written(norm))!.AsObject();

        root["network"]!["state"]!["1.running_mean"]!["learns"] = false;

        var refused = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(root.ToJsonString(), "network", NetworkCatalog.BuiltIn()));

        Assert.Contains("learns", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AParameterThatIsHeld_IsMovedByNoStep_AndNoOptimizerRemembersAnythingOfIt()
    {
        var (hidden, output, network) = Network();
        var before = hidden.Weight.Value.Values.ToArray();
        var adam = new Adam(0.05);

        hidden.Hold();

        network.Compile(adam, new BinaryCrossEntropy()).Fit(Rows(), validation: null, new FitOptions(seed: 1) { Epochs = 3, BatchSize = 2 });

        Assert.Equal(before, hidden.Weight.Value.Values.ToArray());
        Assert.Null(adam.KeptOf(hidden.Weight));
        Assert.Null(adam.KeptOf(hidden.Bias!));
        Assert.NotNull(adam.KeptOf(output.Weight));
        Assert.False(hidden.Weight.Learns);

        hidden.Release();

        Assert.True(hidden.Weight.Learns);
        Assert.True(hidden.Bias!.Learns);
    }

    [Fact]
    public void ARunUnderAPenaltyABoundAndAHeldLayer_ThatStopsAndGoesOnFromAFile_IsTheRunThatNeverStopped_BitForBit()
    {
        var straight = Network();
        var stopped = Network();
        var onwards = Network();

        Rule(straight);
        Rule(stopped);

        straight.Network.Compile(new Adam(0.05), new BinaryCrossEntropy()).Fit(Rows(), Rows(), Options(5));

        string? file = null;
        var compiled = stopped.Network.Compile(new Adam(0.05), new BinaryCrossEntropy());

        compiled.Fit(Rows(), Rows(), Options(2, new Checkpoints(checkpoint => file = WrittenWith(compiled, checkpoint))));

        var read = NetworkDocument.ReadCheckpoint(file!, "network", "training", NetworkCatalog.BuiltIn());

        Assert.False(Assert.IsType<LayerStack>(read.Run.Compiled.Network).Layers.OfType<Dense>().First().Weight.Learns);

        read.Run.Compiled.Fit(Rows(), Rows(), Options(5, from: read.Run.Checkpoint));

        Assert.Equal(Bits(straight.Network), Bits(read.Run.Compiled.Network));
        Assert.NotEqual(Bits(onwards.Network), Bits(straight.Network));

        FitOptions Options(int epochs, Checkpoints? checkpoints = null, Checkpoint? from = null) => new(seed: 5)
        {
            Epochs = epochs, BatchSize = 2, Checkpoints = checkpoints, ResumeFrom = from,
        };
    }

    [Fact]
    public void ARunGoneOnUnderOtherRules_ThanItsCheckpointWasTakenUnder_IsRefusedNamingWhatWasHeld()
    {
        var first = Network();
        var kept = new List<Checkpoint>();

        first.Hidden.Hold();
        first.Network.Compile(new Sgd(0.1), new BinaryCrossEntropy()).Fit(Rows(), validation: null, new FitOptions(seed: 1) { Epochs = 1, BatchSize = 2, Checkpoints = new Checkpoints(kept.Add) });

        // The network goes on as it is, and the checkpoint is gone on from by a network with a different layer held.
        var other = Network();

        other.Output.Hold();

        var refused = Assert.Throws<ArgumentException>(
            () => other.Network.Compile(new Sgd(0.1), new BinaryCrossEntropy()).Fit(Rows(), validation: null, new FitOptions(seed: 1) { Epochs = 2, BatchSize = 2, ResumeFrom = kept[0] }));

        Assert.Contains("'0.weight'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'2.weight'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("held still", refused.Message, StringComparison.Ordinal);

        // And the same rules are gone on under without a word.
        var same = Network();

        same.Hidden.Hold();
        same.Network.Compile(new Sgd(0.1), new BinaryCrossEntropy()).Fit(Rows(), validation: null, new FitOptions(seed: 1) { Epochs = 2, BatchSize = 2, ResumeFrom = kept[0] });
    }

    [Fact]
    public void ABoundOnATransposedConvolution_IsRefusedByName_ForItsKernelIsNotLaidOutByUnit()
    {
        var draws = new RandomStream(1).Draw("initialise:test", 0, 0);
        var network = new LayerStack(new Conv2DTranspose(new Channels { In = 1, Out = 2 }, new Window(2, 2), draws), new Flatten(), new Dense(18, 1, draws));
        var transposed = Assert.IsType<Conv2DTranspose>(network.Layers[0]);

        transposed.Weight.Rules = new SlotRules { MaxNorm = 1 };

        var refused = Assert.Throws<ArgumentException>(() => network.Compile(new Sgd(0.1), new MeanSquaredError())
            .Fit(new TrainingData(Tensor.Zeros(new Shape(2, 2, 2, 1)), Tensor.Zeros(new Shape(2, 1))), validation: null, new FitOptions(seed: 1)));

        Assert.Contains("Conv2DTranspose", refused.Message, StringComparison.Ordinal);
        Assert.Contains("unit", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnOptimizerBoundsOnlyTheParametersItWasHandedWithABound_AndTheOthersAreMovedAsTheyAlwaysWere()
    {
        var plain = Network();
        var bounded = Network();

        bounded.Output.Weight.Rules = new SlotRules { MaxNorm = 100 };
        plain.Network.Compile(new Sgd(0.1), new BinaryCrossEntropy()).Fit(Rows(), validation: null, new FitOptions(seed: 1) { Epochs = 2, BatchSize = 4 });
        bounded.Network.Compile(new Sgd(0.1), new BinaryCrossEntropy()).Fit(Rows(), validation: null, new FitOptions(seed: 1) { Epochs = 2, BatchSize = 4 });

        // A bound the numbers never reach leaves them as they were, to a millionth: the scale is one less a hair.
        foreach (var (one, other) in plain.Network.Parameters().Zip(bounded.Network.Parameters()))
        {
            Assert.InRange(one.Value.Values.ToArray().Zip(other.Value.Values.ToArray(), (was, now) => Math.Abs(was - now)).Max(), 0, 1e-6);
        }
    }

    private static void Rule((Dense Hidden, Dense Output, LayerStack Network) walked)
    {
        walked.Hidden.Hold();
        walked.Output.Weight.Rules = new SlotRules { L1 = 0.01, L2 = 0.02, MaxNorm = 0.3 };
        walked.Output.Bias!.Rules = new SlotRules { MaxNorm = 0.4 };
    }

    private static (Dense Hidden, Dense Output, LayerStack Network) Network()
    {
        var hidden = new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias());
        var output = new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias());

        return (hidden, output, new LayerStack(hidden, new Relu(), output));
    }

    private static TrainingData Rows() => new(WalkedRows.Passengers(), WalkedRows.Survived());

    private static Draws Draws() => new RandomStream(5).Draw("initialise:0", 0, 0);

    private static string Written(Network network)
    {
        var buffer = new System.Buffers.ArrayBufferWriter<byte>();

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

    private static int[][] Bits(Layer network) =>
        [.. network.Parameters().Select(parameter => parameter.Value.Values.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray())];
}
