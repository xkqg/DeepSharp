// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using DeepSharp.Tests.Networks;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// How a layer's numbers begin, and a layer that has no bias: a linear layer or a convolution of any of the six kinds built
/// without one holds its kernel alone, passes what it is handed through the kernel alone, writes that it has no bias into its
/// file as the version that can say so, and reads back without one. A layer that says nothing of how it begins begins as it
/// always did, to the bit — the same draws, in the same order, from the same stream.
/// </summary>
public abstract class BeginningContract(ITensorBackend engine)
{
    private readonly ITensorBackend _backend = engine;

    private static Draws DrawsFor(string purpose) => new RandomStream(42).Draw(purpose, 0, 0);

    [Fact]
    public void ABeginningThatSaysNothingOfHowALayerStarts_StartsItAsItAlwaysStarted_ToTheBit()
    {
        var always = new Dense(14, 16, DrawsFor("initialise:0"));
        var said = new Dense(14, 16, new Beginning(DrawsFor("initialise:0")));
        var conv = new Conv2D(new Channels { In = 1, Out = 2 }, new Window(3, 3), new Beginning(DrawsFor("initialise:0")));
        var convAlways = new Conv2D(1, 2, new Window(3, 3), DrawsFor("initialise:0"));

        Assert.Equal(Bits(always.Weight.Value), Bits(said.Weight.Value));
        Assert.Equal(Bits(always.Bias!.Value), Bits(said.Bias!.Value));
        Assert.Equal(Bits(convAlways.Weight.Value), Bits(conv.Weight.Value));
        Assert.Equal(Bits(convAlways.Bias!.Value), Bits(conv.Bias!.Value));
    }

    [Fact]
    public void ABeginningOfNothingButDraws_SaysNothingMore()
    {
        var beginning = new Beginning(DrawsFor("initialise:0"));

        Assert.False(beginning.WithoutBias);
        Assert.Null(beginning.Weights);
        Assert.Null(beginning.Bias);
        Assert.Equal(Bits(new Dense(3, 2, DrawsFor("initialise:0")).Weight.Value), Bits(new Dense(3, 2, beginning).Weight.Value));
    }

    [Fact]
    public void ABeginning_CanSayHowTheWeightsStartAndHowTheBiasDoes()
    {
        var dense = new Dense(3, 2, new Beginning(DrawsFor("initialise:0")) { Weights = new Ones(), Bias = new Constant(0.5) });
        var conv = new Conv2D(new Channels { In = 1, Out = 2 }, new Window(2, 2), new Beginning(DrawsFor("initialise:0")) { Weights = new Zeros(), Bias = new Ones() });

        Assert.All(dense.Weight.Value.Values.ToArray(), value => Assert.Equal(1f, value));
        Assert.All(dense.Bias!.Value.Values.ToArray(), value => Assert.Equal(0.5f, value));
        Assert.All(conv.Weight.Value.Values.ToArray(), value => Assert.Equal(0f, value));
        Assert.All(conv.Bias!.Value.Values.ToArray(), value => Assert.Equal(1f, value));
    }

    [Fact]
    public void ABeginningThatWantsNoBiasAndSaysHowToStartOne_IsRefused_AsIsOneWithNoDraws()
    {
        var contradicts = new Beginning(DrawsFor("initialise:0")) { WithoutBias = true, Bias = new Zeros() };

        Assert.Throws<ArgumentException>(() => new Dense(3, 2, contradicts));
        Assert.Throws<ArgumentException>(() => new Conv2D(new Channels { In = 1, Out = 2 }, new Window(2, 2), contradicts));
        Assert.Throws<ArgumentNullException>(() => new Dense(3, 2, default(Beginning)));
        Assert.Throws<ArgumentNullException>(() => new Conv2D(new Channels { In = 1, Out = 2 }, new Window(2, 2), default(Beginning)));
    }

    [Fact]
    public void ADenseWithoutABias_KeepsItsWeightAlone_StartedAsWithABias()
    {
        var with = new Dense(14, 16, DrawsFor("initialise:0"));
        var without = new Dense(14, 16, new Beginning(DrawsFor("initialise:0")) { WithoutBias = true });

        Assert.Null(without.Bias);
        Assert.Equal(["weight"], without.Slots().Select(slot => slot.Path));
        Assert.Equal([without.Weight], without.Parameters());
        Assert.Equal(Bits(with.Weight.Value), Bits(without.Weight.Value));
    }

    [Fact]
    public void ADenseWithoutABias_PassesRowsThroughTheMatrixAlone_AsOneWhoseBiasIsNothingDoes()
    {
        var without = new Dense(WalkedRows.HiddenWeights(), null);
        var nothing = new Dense(WalkedRows.HiddenWeights(), Tensor.Zeros(new Shape(4)));

        Assert.Equal(
            Bits(nothing.Forward(WalkedRows.Passengers(), Pass.Evaluation(_backend))),
            Bits(without.Forward(WalkedRows.Passengers(), Pass.Evaluation(_backend))));
    }

    [Fact]
    public void TheGradientThroughADenseWithoutABias_IsTheWeightsOwn_AsWithABiasOfNothing()
    {
        var without = new Dense(WalkedRows.HiddenWeights(), null);
        var nothing = new Dense(WalkedRows.HiddenWeights(), Tensor.Zeros(new Shape(4)));

        Assert.Equal(Bits(GradientOf(nothing, nothing.Weight.Value)), Bits(GradientOf(without, without.Weight.Value)));
        Assert.Equal(Bits(GradientOf(without, without.Weight.Value)), Bits(GradientOf(without, without.Weight.Value)));
    }

    [Theory]
    [InlineData("conv1d")]
    [InlineData("conv2d")]
    [InlineData("conv3d")]
    [InlineData("conv1dtranspose")]
    [InlineData("conv2dtranspose")]
    [InlineData("conv3dtranspose")]
    public void AConvolutionWithoutABias_KeepsItsKernelAlone_StartedAsWithABias(string kind)
    {
        var with = Convolution(kind, new Beginning(DrawsFor("initialise:0")));
        var without = Convolution(kind, new Beginning(DrawsFor("initialise:0")) { WithoutBias = true });

        Assert.NotNull(with.Bias);
        Assert.Null(without.Bias);
        Assert.Equal(["weight"], without.Slots().Select(slot => slot.Path));
        Assert.Equal(["weight", "bias"], with.Slots().Select(slot => slot.Path));
        Assert.Equal(Bits(with.Weight.Value), Bits(without.Weight.Value));
    }

    [Theory]
    [InlineData("conv1d")]
    [InlineData("conv2d")]
    [InlineData("conv3d")]
    [InlineData("conv1dtranspose")]
    [InlineData("conv2dtranspose")]
    [InlineData("conv3dtranspose")]
    public void AConvolutionStartedAtAKernelAndNoBias_IsTheOneStartedAtThatKernelAndABiasOfNothing_InWhatItMakes(string kind)
    {
        var drawn = Convolution(kind, new Beginning(DrawsFor("initialise:0")));
        var channels = new Channels { In = 1, Out = 2 };
        var without = ConvolutionAt(kind, drawn.Weight.Value, null, channels);
        var nothing = ConvolutionAt(kind, drawn.Weight.Value, Tensor.Zeros(new Shape(2)), channels);
        var input = Input(kind);

        Assert.Null(without.Bias);
        Assert.Equal(
            Bits(nothing.Forward(input, Pass.Evaluation(_backend))),
            Bits(without.Forward(input, Pass.Evaluation(_backend))));
    }

    [Fact]
    public void ALayerWithoutABias_IsWrittenSayingSo_AsTheFifthVersion_AndReadBackWithoutOne()
    {
        var network = new LayerStack(
            new Dense(3, 2, new Beginning(DrawsFor("initialise:0")) { WithoutBias = true }),
            new Relu(),
            new Dense(2, 1, DrawsFor("initialise:2")));

        var text = Written(network);
        var read = Assert.IsType<LayerStack>(NetworkDocument.ReadNetwork(text, "network", NetworkCatalog.BuiltIn()).Network);

        Assert.Contains("\"bias\": false", text, StringComparison.Ordinal);
        Assert.Equal(5, VersionOf(text));
        Assert.Null(Assert.IsType<Dense>(read.Layers[0]).Bias);
        Assert.NotNull(Assert.IsType<Dense>(read.Layers[2]).Bias);
        Assert.Equal(Bits(network), Bits(read));
    }

    [Theory]
    [InlineData("conv1d")]
    [InlineData("conv2d")]
    [InlineData("conv3d")]
    [InlineData("conv1dtranspose")]
    [InlineData("conv2dtranspose")]
    [InlineData("conv3dtranspose")]
    public void AConvolutionWithoutABias_IsWrittenAsTheFifthVersion_AndReadBackWithoutOne(string kind)
    {
        var network = new LayerStack(Convolution(kind, new Beginning(DrawsFor("initialise:0")) { WithoutBias = true }));
        var text = Written(network);
        var read = Assert.IsType<LayerStack>(NetworkDocument.ReadNetwork(text, "network", NetworkCatalog.BuiltIn()).Network);

        Assert.Equal(5, VersionOf(text));
        Assert.Null(Assert.IsAssignableFrom<Convolution>(read.Layers[0]).Bias);
        Assert.Equal(Bits(network), Bits(read));
    }

    [Fact]
    public void ALayerWithABias_IsWrittenAsItAlwaysWas_NoMentionOfOneAndTheSecondVersion()
    {
        var text = Written(new LayerStack(new Dense(3, 2, DrawsFor("initialise:0")), new Conv2D(1, 2, new Window(2, 2), DrawsFor("initialise:1"))));

        Assert.DoesNotContain("\"bias\": false", text, StringComparison.Ordinal);
        Assert.Equal(2, VersionOf(text));
    }

    [Fact]
    public void AFileThatNamesABiasOfALayerWithoutOne_OrLeavesOutTheBiasOfOneWithIt_IsRefusedBySlotLoad_ByPath()
    {
        var without = Written(new LayerStack(new Dense(3, 2, new Beginning(DrawsFor("initialise:0")) { WithoutBias = true })));
        var withBias = Written(new LayerStack(new Dense(3, 2, DrawsFor("initialise:0"))));

        // Say the first has a bias, which it has no slot for: the layer reads as bearing none, and the slot is no slot of it.
        var extra = without.Replace("\"parameters\": {", "\"parameters\": {\n    \"0.bias\": { \"shape\": [2], \"values\": [0.0, 0.0] },", StringComparison.Ordinal);
        var refusedExtra = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(extra, "network", NetworkCatalog.BuiltIn()));

        Assert.Contains("'0.bias' is no slot of this network.", refusedExtra.Message, StringComparison.Ordinal);

        // Say the second has none: it reads as bearing one, and the bias is missing.
        var node = System.Text.Json.Nodes.JsonNode.Parse(withBias)!;

        node["network"]!["parameters"]!.AsObject().Remove("0.bias");

        var refusedMissing = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(node.ToJsonString(), "network", NetworkCatalog.BuiltIn()));

        Assert.Contains("'0.bias' is missing: every slot of the network is written.", refusedMissing.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ARunTrainsAndResumesALayerWithoutABias_ItsOptimizerRemembersTheWeightAlone()
    {
        var compiled = new LayerStack(new Dense(1, 1, new Beginning(DrawsFor("initialise:0")) { WithoutBias = true }))
            .Compile(new Adam(0.05), new MeanSquaredError());
        Checkpoint? kept = null;
        var rows = new TrainingData(
            Tensor.From(new Shape(4, 1), [0.1f, 0.2f, 0.3f, 0.4f]),
            Tensor.From(new Shape(4, 1), [0.2f, 0.4f, 0.6f, 0.8f]));

        var options = new FitOptions(seed: 3) { Epochs = 2, BatchSize = 2, Backend = _backend, Checkpoints = new Checkpoints(checkpoint => kept = checkpoint) };

        compiled.Fit(rows, validation: null, options);

        var text = Checkpointed(compiled, kept!);
        using var document = JsonDocument.Parse(text);
        var memory = document.RootElement.GetProperty("training").GetProperty("memory");

        Assert.Equal(["0.weight"], memory.EnumerateObject().Select(entry => entry.Name));

        var read = NetworkDocument.ReadCheckpoint(text, "network", "training", NetworkCatalog.BuiltIn());

        Assert.Null(Assert.IsType<Dense>(Assert.IsType<LayerStack>(read.Run.Compiled.Network).Layers[0]).Bias);
    }

    [Fact]
    public void TheWordsOfADescription_CanSayHowALayerStarts_OrThatItHasNoBias_AndTheWeightsStartWhereTheyDidWithOne()
    {
        var said = new Sequential()
            .Input(new Shape(36))
            .Reshape(new Shape(6, 6, 1))
            .Conv2D(4, new Window(3, 3), new Beginning { WithoutBias = true })
            .Flatten()
            .Dense(8, new Beginning { WithoutBias = true })
            .Dense(3, new Beginning { Weights = new Ones(), Bias = new Constant(0.25) })
            .Lower(new Shape(36), new RandomStream(3));
        var plain = new Sequential()
            .Input(new Shape(36))
            .Reshape(new Shape(6, 6, 1))
            .Conv2D(4, new Window(3, 3))
            .Flatten()
            .Dense(8)
            .Dense(3)
            .Lower(new Shape(36), new RandomStream(3));

        var convolution = Assert.IsType<Conv2D>(said.Layers[1]);
        var hidden = Assert.IsType<Dense>(said.Layers[3]);
        var output = Assert.IsType<Dense>(said.Layers[4]);

        Assert.Null(convolution.Bias);
        Assert.Null(hidden.Bias);
        Assert.All(output.Weight.Value.Values.ToArray(), value => Assert.Equal(1f, value));
        Assert.All(output.Bias!.Value.Values.ToArray(), value => Assert.Equal(0.25f, value));
        Assert.Equal(Bits(Assert.IsType<Conv2D>(plain.Layers[1]).Weight.Value), Bits(convolution.Weight.Value));
        Assert.Equal(Bits(Assert.IsType<Dense>(plain.Layers[3]).Weight.Value), Bits(hidden.Weight.Value));
    }

    [Fact]
    public void EveryConvolutionWord_CanBeToldItHasNoBias_AlongOneAxisTwoAndThree_PlainDepthwiseAndTransposed()
    {
        var without = new Beginning { WithoutBias = true };
        var series = new Sequential().Conv1D(4, new Window1D(2), without).Conv1D(4, new Window1D(2), 2, without).DepthwiseConv1D(2, new Window1D(2), without).Conv1DTranspose(3, new Window1D(2), without)
            .Lower(new Shape(8, 2), new RandomStream(5));
        var image = new Sequential().Conv2D(4, new Window(2, 2), without).Conv2D(4, new Window(2, 2), 2, without).DepthwiseConv2D(2, new Window(2, 2), without).Conv2DTranspose(3, new Window(2, 2), without)
            .Lower(new Shape(8, 8, 2), new RandomStream(5));
        var volume = new Sequential().Conv3D(4, new Window3D(2, 2, 2), without).Conv3D(4, new Window3D(2, 2, 2), 2, without).DepthwiseConv3D(2, new Window3D(2, 2, 2), without).Conv3DTranspose(3, new Window3D(2, 2, 2), without)
            .Lower(new Shape(6, 6, 6, 2), new RandomStream(5));

        foreach (var layer in series.Layers.Concat(image.Layers).Concat(volume.Layers))
        {
            Assert.Null(Assert.IsAssignableFrom<Convolution>(layer).Bias);
        }
    }

    [Fact]
    public void TheNormalisationWords_CanBeToldTheyKeepNoScaleOrNoShift()
    {
        var lowered = new Sequential()
            .Dense(4)
            .BatchNorm(new Affine { WithoutScale = true })
            .BatchNorm(0.9, 1e-3, new Affine { WithoutShift = true })
            .LayerNorm(new Affine { WithoutScale = true, WithoutShift = true })
            .LayerNorm(1e-3, new Affine { WithoutShift = true })
            .Lower(new Shape(3), new RandomStream(5));

        Assert.Null(Assert.IsType<BatchNorm>(lowered.Layers[1]).Weight);
        Assert.Null(Assert.IsType<BatchNorm>(lowered.Layers[2]).Bias);
        Assert.NotNull(Assert.IsType<BatchNorm>(lowered.Layers[2]).Weight);
        Assert.Equal(0.1, Assert.IsType<BatchNorm>(lowered.Layers[2]).Momentum, 12);
        Assert.Null(Assert.IsType<LayerNorm>(lowered.Layers[3]).Weight);
        Assert.Null(Assert.IsType<LayerNorm>(lowered.Layers[3]).Bias);
        Assert.Null(Assert.IsType<LayerNorm>(lowered.Layers[4]).Bias);
        Assert.Equal(1e-3, Assert.IsType<LayerNorm>(lowered.Layers[4]).Epsilon);
    }

    [Fact]
    public void ADescriptionIsTold_HowItsLayersStart_NeverFromWhichDraws_AndAContradictionIsRefused()
    {
        var draws = DrawsFor("initialise:0");
        var named = new Beginning(draws);
        var contradicts = new Beginning { WithoutBias = true, Bias = new Zeros() };

        Assert.Contains("draws", Assert.Throws<ArgumentException>(() => new Sequential().Dense(4, named)).Message, StringComparison.Ordinal);
        Assert.Contains("draws", Assert.Throws<ArgumentException>(() => new Sequential().Conv2D(4, new Window(2, 2), named)).Message, StringComparison.Ordinal);
        Assert.Contains("draws", Assert.Throws<ArgumentException>(() => new Sequential().DepthwiseConv2D(2, new Window(2, 2), named)).Message, StringComparison.Ordinal);
        Assert.Contains("draws", Assert.Throws<ArgumentException>(() => new Sequential().Conv2DTranspose(2, new Window(2, 2), named)).Message, StringComparison.Ordinal);
        Assert.Contains("no bias", Assert.Throws<ArgumentException>(() => new Sequential().Dense(4, contradicts)).Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentException>(() => new Sequential().Conv1D(4, new Window1D(2), contradicts));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Sequential().Dense(0, new Beginning { WithoutBias = true }));
    }

    [Fact]
    public void ADescriptionCompiledWithLayersThatHaveNoBias_IsFittedAndWrittenAsTheFifthVersion()
    {
        var compiled = new Sequential().Dense(3, new Beginning { WithoutBias = true }).Relu().Dense(1)
            .Compile(new Sgd(0.05), new MeanSquaredError());
        var rows = new TrainingData(
            Tensor.From(new Shape(4, 2), [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f]),
            Tensor.From(new Shape(4, 1), [0.2f, 0.4f, 0.6f, 0.8f]));

        var history = compiled.Fit(rows, validation: null, new FitOptions(seed: 3) { Epochs = 2, BatchSize = 2, Backend = _backend });
        var text = Written(compiled.Network);

        Assert.Equal(2, history.Epochs.Count);
        Assert.Contains("\"bias\": false", text, StringComparison.Ordinal);
        Assert.Equal(5, VersionOf(text));
        Assert.Null(Assert.IsType<Dense>(Assert.IsType<LayerStack>(compiled.Network).Layers[0]).Bias);
    }

    private static Layer ConvolutionLayer(string kind, Beginning beginning, Channels channels) => kind switch
    {
        "conv1d" => new Conv1D(channels, new Window1D(2), beginning),
        "conv2d" => new Conv2D(channels, new Window(2, 2), beginning),
        "conv3d" => new Conv3D(channels, new Window3D(2, 2, 2), beginning),
        "conv1dtranspose" => new Conv1DTranspose(channels, new Window1D(2), beginning),
        "conv2dtranspose" => new Conv2DTranspose(channels, new Window(2, 2), beginning),
        _ => new Conv3DTranspose(channels, new Window3D(2, 2, 2), beginning),
    };

    private static Convolution Convolution(string kind, Beginning beginning) =>
        (Convolution)ConvolutionLayer(kind, beginning, new Channels { In = 1, Out = 2 });

    private static Convolution ConvolutionAt(string kind, Tensor kernel, Tensor? bias, Channels channels) => kind switch
    {
        "conv1d" => new Conv1D(kernel, bias, new Window1D(2), channels),
        "conv2d" => new Conv2D(kernel, bias, new Window(2, 2), channels),
        "conv3d" => new Conv3D(kernel, bias, new Window3D(2, 2, 2), channels),
        "conv1dtranspose" => new Conv1DTranspose(kernel, bias, new Window1D(2), channels),
        "conv2dtranspose" => new Conv2DTranspose(kernel, bias, new Window(2, 2), channels),
        _ => new Conv3DTranspose(kernel, bias, new Window3D(2, 2, 2), channels),
    };

    // A batch of one example the window of two can stand on, a place of one channel each.
    private static Tensor Input(string kind)
    {
        var rank = int.Parse(kind.Replace("conv", string.Empty, StringComparison.Ordinal)[..1], System.Globalization.CultureInfo.InvariantCulture);
        var lengths = Enumerable.Repeat(3, rank).ToArray();
        var count = lengths.Aggregate(1, (all, length) => all * length);

        return Tensor.From(new Shape([1, .. lengths, 1]), [.. Enumerable.Range(0, count).Select(at => ((at % 7) - 3) / 4f)]);
    }

    private Tensor GradientOf(Dense layer, Tensor wrt)
    {
        var recording = new RecordingBackend(_backend);
        var output = layer.Forward(WalkedRows.Passengers(), Pass.Evaluation(recording));
        var loss = recording.Mean(recording.Multiply(output, output));

        return recording.GradientsOf(loss, [wrt])[wrt];
    }

    private static int[] Bits(Tensor tensor) => [.. tensor.Values.ToArray().Select(BitConverter.SingleToInt32Bits)];

    private static int[][] Bits(Layer network) => [.. network.Slots().Select(slot => Bits(slot.Slot.Value))];

    private static int VersionOf(string text)
    {
        using var document = JsonDocument.Parse(text);

        return document.RootElement.GetProperty("network").GetProperty("version").GetInt32();
    }

    private static string Written(Network network) => Written(writer =>
    {
        writer.WritePropertyName("network");
        NetworkDocument.WriteNetwork(writer, network, new MeanSquaredError());
    });

    private static string Checkpointed(CompiledNetwork compiled, Checkpoint checkpoint) => Written(writer =>
    {
        writer.WritePropertyName("network");
        NetworkDocument.WriteNetwork(writer, compiled.Network, compiled.Loss);
        writer.WritePropertyName("training");
        NetworkDocument.WriteTraining(writer, compiled, checkpoint);
    });

    private static string Written(Action<Utf8JsonWriter> body)
    {
        var buffer = new System.Buffers.ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            body(writer);
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }
}
