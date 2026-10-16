// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// The words a model is written in that 0.11.0 adds — the smooth activations a transformer is made of, the leaky, parametric
/// and self-normalising ones, softmax as a layer, the normalisations over groups of channels, over one example's channels and by
/// the root mean square, the periodic embedding of a number, and the losses — each held to PyTorch's own numbers: what it makes
/// of an example, the gradient of a loss through it, and, for a loss, its value and its gradient.
/// </summary>
/// <remarks>
/// The numbers are what Fixtures/vocabulary-pytorch.py printed. Every example holds a nought, so the slope an activation takes
/// there is PyTorch's too: a layer written so that its gradient at nought is rounded off to the other side passes every other value.
/// </remarks>
/// <param name="engine">The backend the passes run on.</param>
public abstract class VocabularyContract(ITensorBackend engine)
{
    private readonly ITensorBackend _backend = engine;

    private static readonly JsonElement Reference = JsonDocument
        .Parse(File.ReadAllText(Path.Join(Repository.Root, "Tst", "DeepSharp.Backends.Contract", "Networks", "Fixtures", "vocabulary-pytorch.json")))
        .RootElement;

    public static TheoryData<string> Layers { get; } = [.. Reference.GetProperty("layers").EnumerateArray().Select(each => each.GetProperty("name").GetString()!)];

    public static TheoryData<string> Losses { get; } = [.. Reference.GetProperty("losses").EnumerateArray().Select(each => each.GetProperty("name").GetString()!)];

    [Theory]
    [MemberData(nameof(Layers), MemberType = typeof(VocabularyContract))]
    public void ALayer_IsPyTorchsForTheSameInput(string name)
    {
        var (network, input, expected) = LayerCase(name);

        var output = network.Forward(input, Pass.Evaluation(_backend));

        Assert.Equal(ShapeOf(expected, "outShape"), output.Shape);
        AssertClose(Floats(expected, "output"), output.Values);
    }

    [Theory]
    [MemberData(nameof(Layers), MemberType = typeof(VocabularyContract))]
    public void TheGradientOfALossThroughALayer_IsPyTorchs_ForTheExampleAndForEveryNumberItLearns(string name)
    {
        var (network, input, expected) = LayerCase(name);
        var recording = new RecordingBackend(_backend);
        var weights = Tensor.From(ShapeOf(expected, "outShape"), Floats(expected, "gradOutput"));
        var slots = network.Slots().ToDictionary(named => named.Path, named => named.Slot.Value);

        // The loss is the mean of what the layer makes times numbers drawn for it, so every gradient is PyTorch's over the count.
        var loss = recording.Mean(recording.Multiply(network.Forward(input, Pass.Evaluation(recording)), weights));
        var gradients = recording.GradientsOf(loss, [input, .. slots.Values]);
        var scale = (float)weights.Values.Length;

        AssertClose([.. Floats(expected, "gradInput").Select(value => value / scale)], gradients[input].Values);

        foreach (var slot in expected.GetProperty("slots").EnumerateObject())
        {
            AssertClose([.. Floats(slot.Value, "grad").Select(value => value / scale)], gradients[slots[$"0.{slot.Name}"]].Values);
        }
    }

    [Theory]
    [MemberData(nameof(Layers), MemberType = typeof(VocabularyContract))]
    public void ALayer_IsWrittenIntoANetworksFile_AndReadBackMakingTheSameNumbers(string name)
    {
        var (network, input, _) = LayerCase(name);
        var text = Written(network, new MeanSquaredError());
        var read = NetworkDocument.ReadNetwork(text, "network", NetworkCatalog.BuiltIn()).Network;

        Assert.Equal(Written(read, new MeanSquaredError()), text);
        Assert.Equal(2, VersionOf(text));
        Assert.Equal(network.Forward(input, Pass.Evaluation(_backend)).Values.ToArray(), read.Forward(input, Pass.Evaluation(_backend)).Values.ToArray());
    }

    [Theory]
    [MemberData(nameof(Losses), MemberType = typeof(VocabularyContract))]
    public void ALoss_AndItsGradient_ArePyTorchs(string name)
    {
        var expected = Reference.GetProperty("losses").EnumerateArray().Single(each => each.GetProperty("name").GetString() == name);
        var shape = ShapeOf(expected, "shape");
        var outputs = Tensor.From(shape, Floats(expected, "outputs"));
        var answers = Tensor.From(shape, Floats(expected, "answers"));
        var recording = new RecordingBackend(_backend);

        var loss = LossOf(expected).Of(outputs, answers, recording);
        var gradients = recording.GradientsOf(loss, [outputs]);

        AssertClose([(float)expected.GetProperty("loss").GetDouble()], loss.Values);
        AssertClose(Floats(expected, "gradOutputs"), gradients[outputs].Values);
    }

    [Theory]
    [MemberData(nameof(Losses), MemberType = typeof(VocabularyContract))]
    public void ALoss_IsWrittenIntoANetworksFile_AndReadBackAsItWas(string name)
    {
        var expected = Reference.GetProperty("losses").EnumerateArray().Single(each => each.GetProperty("name").GetString() == name);
        var loss = LossOf(expected);
        var text = Written(new LayerStack(new Dense(Tensor.Zeros(new Shape(3, 3)), null)), loss);
        var read = NetworkDocument.ReadNetwork(text, "network", NetworkCatalog.BuiltIn());

        Assert.Equal(loss.GetType(), read.Loss.GetType());
        Assert.Equal(Written(read.Network, read.Loss), text);
    }

    [Fact]
    public void ALeakyOrParametricRectifier_SlopesAtNoughtAsItDoesBelowIt_AndAnExponentialOneAsItsAlpha()
    {
        var nought = Tensor.Zeros(new Shape(1, 1));

        Assert.Equal(0.2f, SlopeAt(new LeakyRelu(0.2), nought), 6);
        Assert.Equal(0.25f, SlopeAt(new PRelu(1), nought), 6);
        Assert.Equal(0.5f, SlopeAt(new Elu(0.5), nought), 6);
        Assert.Equal(1.0507009873554805f * 1.6732632423543772f, SlopeAt(new Selu(), nought), 5);
        Assert.Equal(0.5f, SlopeAt(new Gelu(), nought), 6);
    }

    [Fact]
    public void AnExponentialRectifier_OfLargeSizesEitherWay_IsFinite_AndSoIsItsGradient()
    {
        var large = Tensor.From(new Shape(1, 4), [1000f, -1000f, 88.8f, -88.8f]);

        foreach (Layer layer in new Layer[] { new Elu(), new Selu(), new Gelu(), new Gelu(approximate: true), new Silu(), new Mish() })
        {
            var recording = new RecordingBackend(_backend);
            var output = layer.Forward(large, Pass.Evaluation(recording));
            var gradient = recording.GradientsOf(recording.Mean(output), [large])[large];

            Assert.All(output.Values.ToArray(), value => Assert.True(float.IsFinite(value), $"{layer.GetType().Name} made {value}."));
            Assert.All(gradient.Values.ToArray(), value => Assert.True(float.IsFinite(value), $"{layer.GetType().Name} sent back {value}."));
        }
    }

    [Fact]
    public void AFocalLoss_OfAnAnswerItIsCertainOf_IsFinite_AndSoIsItsGradient()
    {
        var outputs = Tensor.From(new Shape(1, 4), [100f, -100f, 100f, -100f]);
        var answers = Tensor.From(new Shape(1, 4), [1f, 0f, 0f, 1f]);
        var recording = new RecordingBackend(_backend);

        var loss = new Focal(2) { Alpha = 0.25 }.Of(outputs, answers, recording);
        var gradient = recording.GradientsOf(loss, [outputs])[outputs];

        Assert.True(float.IsFinite(loss.Values[0]));
        Assert.All(gradient.Values.ToArray(), value => Assert.True(float.IsFinite(value), $"The focal loss sent back {value}."));
    }

    [Fact]
    public void ALossAppliesTheActivationItIsWrittenWith_AndNoOther()
    {
        Assert.True(new BinaryCrossEntropy().Applies(new Sigmoid()));
        Assert.True(new Focal(2).Applies(new Sigmoid()));
        Assert.True(new CrossEntropy().Applies(new Softmax()));
        Assert.True(new EarthMoversDistance().Applies(new Softmax()));
        Assert.False(new BinaryCrossEntropy().Applies(new Softmax()));
        Assert.False(new CrossEntropy().Applies(new Sigmoid()));
        Assert.False(new MeanSquaredError().Applies(new Sigmoid()));
        Assert.False(new MeanSquaredError().Applies(new Softmax()));
        Assert.False(new Hinge().Applies(new Sigmoid()));
        Assert.False(new Poisson(logInput: true).Applies(new Softmax()));
    }

    [Fact]
    public void EveryOutputActivation_IsWhatAPredictionGoesThrough()
    {
        var outputs = Tensor.From(new Shape(1, 3), [-1f, 0f, 2f]);

        Assert.Equal<float[]>([-1f, 0f, 2f], new MeanSquaredError().Predictions(outputs, _backend).Values.ToArray());
        AssertClose([MathF.Exp(-1f), 1f, MathF.Exp(2f)], new Poisson(logInput: true).Predictions(outputs, _backend).Values);
        Assert.Equal<float[]>([-1f, 0f, 2f], new Poisson(logInput: false).Predictions(outputs, _backend).Values.ToArray());
        Assert.Equal<float[]>([0f, 0f, 1f], new Hinge().Predictions(outputs, _backend).Values.ToArray());
        AssertClose([1 / (1 + MathF.Exp(1f)), 0.5f, 1 / (1 + MathF.Exp(-2f))], new Focal(2).Predictions(outputs, _backend).Values);
        Assert.Equal(OutputActivation.Exp, new Poisson(logInput: true).Activation);
        Assert.Equal(OutputActivation.Margin, new Hinge().Activation);
    }

    [Fact]
    public void ALossWhoseLabelsAreSoftened_IsWrittenAsTheSixthVersion_AndOneWhoseLabelsAreNot_AsItAlwaysWas()
    {
        var network = new LayerStack(new Dense(Tensor.Zeros(new Shape(3, 3)), Tensor.Zeros(new Shape(3))));

        Assert.Equal(6, VersionOf(Written(network, new CrossEntropy { Smoothing = 0.1 })));
        Assert.Equal(6, VersionOf(Written(network, new BinaryCrossEntropy { Smoothing = 0.1 })));
        Assert.Equal(2, VersionOf(Written(network, new CrossEntropy())));
        Assert.DoesNotContain("smoothing", Written(network, new BinaryCrossEntropy()), StringComparison.Ordinal);
    }

    [Fact]
    public void TheSettingsOfAWord_AreRefusedWhereTheyCannotBeMeant()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new LeakyRelu(-0.1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Elu(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PRelu(0));
        Assert.Throws<ArgumentException>(() => new GroupNorm(6, 4));
        Assert.Throws<ArgumentOutOfRangeException>(() => new RmsNorm(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Huber(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Quantile(1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Quantile(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Focal(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Focal(2) { Alpha = 1.5 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new CrossEntropy { Smoothing = 1 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new Periodic(3, new Periodicity(4, 0, 5), Draws()));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Periodic(3, new Periodicity(0, 0.1, 5), Draws()));
        Assert.NotNull(new Hinge().Refusal([0.5]));
        Assert.NotNull(new Poisson(logInput: true).Refusal([-1]));
        Assert.Null(new Hinge().Refusal([0, 1]));
    }

    [Fact]
    public void APeriodicEmbedding_StartsItsFrequenciesFromANormalCurveOfTheSpreadItIsTold_CutThreeSpreadsOut()
    {
        var layer = new Periodic(20, new Periodicity(50, 0.1, 4), Draws());
        var frequencies = layer.Frequencies.Value.Values.ToArray();

        Assert.Equal(new Shape(20, 50), layer.Frequencies.Value.Shape);
        Assert.All(frequencies, value => Assert.InRange(value, -0.3f, 0.3f));
        Assert.InRange(Math.Sqrt(frequencies.Average(value => value * (double)value)), 0.09, 0.11);
    }

    [Fact]
    public void AnEmbedding_PutsTheVectorOfTheRowsCategoryWhereItsBlockStood_AndKeepsTheRestOfTheRowInOrder()
    {
        // Two rows of five columns, the category in columns 1, 3 and 4 (not side by side): the first row holds the second
        // category, the second row the third. Each vector is two numbers.
        var network = new LayerStack(new Embedding(5, [1, 3, 4], 2, Draws()));
        var vectors = Tensor.From(new Shape(3, 2), [10f, 11f, 20f, 21f, 30f, 31f]);
        network.Load([new SlotEntry("0.weight", vectors, "test")]);
        var rows = Tensor.From(new Shape(2, 5), [7f, 0f, 8f, 1f, 0f, -7f, 0f, -8f, 0f, 1f]);

        var made = network.Forward(rows, Pass.Evaluation(_backend));

        Assert.Equal(new Shape(2, 4), made.Shape);
        Assert.Equal<float[]>([7f, 20f, 21f, 8f, -7f, 30f, 31f, -8f], made.Values.ToArray());
    }

    [Fact]
    public void TheGradientOfAnEmbedding_ReachesOnlyTheVectorsOfTheCategoriesTheRowsHold_AsPyTorchsLookUpSendsItBack()
    {
        var layer = new Embedding(4, [0, 1, 2], 2, Draws());
        var rows = Tensor.From(new Shape(3, 4), [0f, 1f, 0f, 5f, 0f, 1f, 0f, 6f, 1f, 0f, 0f, 7f]);
        var weights = Tensor.From(new Shape(3, 3), [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f]);
        var recording = new RecordingBackend(_backend);

        var loss = recording.Mean(recording.Multiply(layer.Forward(rows, Pass.Evaluation(recording)), weights));
        var gradient = recording.GradientsOf(loss, [layer.Weight.Value])[layer.Weight.Value];

        // Rows one and two hold the second category, row three the first, none the third: each vector's gradient is the weights
        // of the places it stood at, added over the rows that hold it, over the nine values.
        Assert.Equal<float[]>([7f / 9, 8f / 9, (1f + 4f) / 9, (2f + 5f) / 9, 0f, 0f], gradient.Values.ToArray());
    }

    [Fact]
    public void AnEmbedding_StartsAsPyTorchsDoes_AndIsWrittenSayingItsBlock()
    {
        var layer = new Embedding(6, [2, 3, 4, 5], 3, Draws());
        var text = Written(new LayerStack(layer), new MeanSquaredError());
        var read = Assert.IsType<Embedding>(Assert.IsType<LayerStack>(NetworkDocument.ReadNetwork(text, "network", NetworkCatalog.BuiltIn()).Network).Layers[0]);

        Assert.Equal(new Shape(4, 3), layer.Weight.Value.Shape);
        Assert.Equal(5, layer.Outputs);
        Assert.Equal([2, 3, 4, 5], read.Block);
        Assert.Equal(2, VersionOf(text));
        Assert.Throws<ArgumentException>(() => new Embedding(6, [2], 3, Draws()));
        Assert.Throws<ArgumentException>(() => new Embedding(6, [2, 2], 3, Draws()));
        Assert.Throws<ArgumentException>(() => new Embedding(6, [2, 6], 3, Draws()));
        Assert.Throws<ArgumentException>(() => layer.Forward(Tensor.Zeros(new Shape(1, 5)), Pass.Evaluation(_backend)));
    }

    private (LayerStack Network, Tensor Input, JsonElement Expected) LayerCase(string name)
    {
        var expected = Reference.GetProperty("layers").EnumerateArray().Single(each => each.GetProperty("name").GetString() == name);
        var network = new LayerStack(LayerOf(expected));
        var given = expected.GetProperty("slots").EnumerateObject()
            .Select(slot => new SlotEntry($"0.{slot.Name}", Tensor.From(network.Slots().Single(named => named.Path == $"0.{slot.Name}").Slot.Value.Shape, Floats(slot.Value, "value")), "vocabulary-pytorch.json"))
            .ToList();

        // A slot the reference leaves at its start - a normalisation's scale and shift where PyTorch keeps none - stays as it began.
        given.AddRange(network.Slots().Where(named => given.All(entry => entry.Path != named.Path)).Select(named => new SlotEntry(named.Path, named.Slot.Value, "start")));
        network.Load(given);

        return (network, Tensor.From(ShapeOf(expected, "shape"), Floats(expected, "input")), expected);
    }

    private static Layer LayerOf(JsonElement expected)
    {
        var settings = expected.GetProperty("settings");
        double Number(string key) => settings.GetProperty(key).GetDouble();
        int Whole(string key) => settings.GetProperty(key).GetInt32();
        bool Yes(string key) => !settings.TryGetProperty(key, out var said) || said.GetBoolean();

        return expected.GetProperty("kind").GetString() switch
        {
            "gelu" => new Gelu(settings.GetProperty("approximate").GetBoolean()),
            "silu" => new Silu(),
            "mish" => new Mish(),
            "selu" => new Selu(),
            "leakyRelu" => new LeakyRelu(Number("negativeSlope")),
            "elu" => new Elu(Number("alpha")),
            "softmax" => new Softmax(),
            "prelu" => new PRelu(Whole("parameters")),
            "groupNorm" => new GroupNorm(Whole("channels"), Whole("groups")) { Epsilon = Number("epsilon") },
            "instanceNorm" => new InstanceNorm(Whole("channels"), new Affine { WithoutScale = !Yes("scale"), WithoutShift = !Yes("shift") }) { Epsilon = Number("epsilon") },
            "rmsNorm" => new RmsNorm(Whole("features")) { Epsilon = Number("epsilon") },
            "periodic" => new Periodic(Whole("features"), new Periodicity(Whole("frequencies"), 0.1, Whole("width")) { Lite = settings.GetProperty("lite").GetBoolean() }, Draws()),
            var other => throw new InvalidOperationException($"vocabulary-pytorch.json names a layer '{other}' this contract does not build."),
        };
    }

    private static Loss LossOf(JsonElement expected)
    {
        var settings = expected.GetProperty("settings");
        double Number(string key) => settings.GetProperty(key).GetDouble();

        return expected.GetProperty("kind").GetString() switch
        {
            "meanAbsoluteError" => new MeanAbsoluteError(),
            "huber" => new Huber(Number("delta")),
            "quantile" => new Quantile(Number("quantile")),
            "focal" => new Focal(Number("gamma")) { Alpha = settings.TryGetProperty("alpha", out var alpha) ? alpha.GetDouble() : null },
            "hinge" => new Hinge(),
            "poisson" => new Poisson(settings.GetProperty("logInput").GetBoolean()),
            "crossEntropy" => new CrossEntropy { Smoothing = Number("smoothing") },
            "binaryCrossEntropy" => new BinaryCrossEntropy { Smoothing = Number("smoothing") },
            var other => throw new InvalidOperationException($"vocabulary-pytorch.json names a loss '{other}' this contract does not build."),
        };
    }

    // The gradient of a layer's one output at a value of one input.
    private float SlopeAt(Layer layer, Tensor at)
    {
        var recording = new RecordingBackend(_backend);

        return recording.GradientsOf(recording.Mean(layer.Forward(at, Pass.Evaluation(recording))), [at])[at].Values[0];
    }

    private static Draws Draws() => new RandomStream(20261016).Draw("initialise:0", 0, 0);

    private static Shape ShapeOf(JsonElement written, string key) => new([.. written.GetProperty(key).EnumerateArray().Select(length => length.GetInt32())]);

    private static float[] Floats(JsonElement written, string key) => [.. written.GetProperty(key).EnumerateArray().Select(value => (float)value.GetDouble())];

    private static string Written(Network network, Loss loss)
    {
        var buffer = new System.Buffers.ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, network, loss);
            writer.WriteEndObject();
        }

        return System.Text.Encoding.UTF8.GetString(buffer.WrittenSpan);
    }

    private static int VersionOf(string text)
    {
        using var document = JsonDocument.Parse(text);

        return document.RootElement.GetProperty("network").GetProperty("version").GetInt32();
    }

    private static void AssertClose(float[] expected, ReadOnlySpan<float> actual)
    {
        Assert.Equal(expected.Length, actual.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], actual[at], Math.Max(2e-5, Math.Abs(expected[at]) * 2e-5));
        }
    }
}
