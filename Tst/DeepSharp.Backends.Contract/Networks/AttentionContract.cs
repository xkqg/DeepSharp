// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// Attention — scaled dot-product attention over queries, keys and values handed in apart, and multi-head attention over a series, with a mask
/// of the padding and a causal mask — gives PyTorch's values, and the gradient of a loss through it is PyTorch's, to the input and to every
/// number the layer learns, on every engine. A series of no step to read is made nothing, as PyTorch makes it, and never a number that is not
/// finite. A layer is written down and read back as the very layer it was, and the sine positions are the formula of 'Attention Is All You Need'.
/// </summary>
/// <param name="engine">The engine the arithmetic runs on.</param>
public abstract class AttentionContract(ITensorBackend engine)
{
    private static readonly JsonElement Fixture = JsonDocument
        .Parse(File.ReadAllText(Path.Join(Repository.Root, "Tst", "DeepSharp.Backends.Contract", "Networks", "Fixtures", "attention-pytorch.json")))
        .RootElement;

    private readonly ITensorBackend _backend = engine;

    public static TheoryData<string> AttentionCases { get; } = Named("attention");

    public static TheoryData<string> MultiHeadCases { get; } = Named("multiHeadAttention");

    public static TheoryData<string> EncoderCases { get; } = Named("encoder");

    public static TheoryData<string> Masked { get; } =
        [.. Fixture.GetProperty("cases").EnumerateArray().Where(each => each.GetProperty("kind").GetString() != "encoder" && each.GetProperty("mask").ValueKind != JsonValueKind.Null).Select(each => each.GetProperty("name").GetString()!)];

    [Theory]
    [MemberData(nameof(AttentionCases))]
    public void AttentionOverQueriesKeysAndValues_IsPyTorchs_AndSoIsTheGradientToEachOfThem(string name)
    {
        var made = Case(name);
        var layer = new Attention(new Attending(made.GetProperty("heads").GetInt32()) { Causal = made.GetProperty("causal").GetBoolean() });
        var recording = new RecordingBackend(_backend);
        var (q, k, v) = (Held(made, "q"), Held(made, "k"), Held(made, "v"));
        var output = layer.Forward(new QueryKeyValue(q, k, v), Mask(made), Pass.Evaluation(recording));
        var loss = recording.Mean(recording.Multiply(output, Coefficients(output.Shape)));
        var gradients = recording.GradientsOf(loss, [q, k, v]);

        Assert.Equal(Shape(made.GetProperty("output")), output.Shape);
        AssertClose("the values", Values(made.GetProperty("output").GetProperty("values")), output.Values);
        AssertClose("the gradient to the queries", Values(made.GetProperty("gradInput").GetProperty("q")), gradients[q].Values);
        AssertClose("the gradient to the keys", Values(made.GetProperty("gradInput").GetProperty("k")), gradients[k].Values);
        AssertClose("the gradient to the values", Values(made.GetProperty("gradInput").GetProperty("v")), gradients[v].Values);
    }

    [Theory]
    [MemberData(nameof(MultiHeadCases))]
    public void MultiHeadAttentionOverASeries_IsPyTorchs_AndSoIsTheGradientToTheSeriesAndToEveryNumberTheLayerLearns(string name)
    {
        var made = Case(name);
        var (layer, network) = MultiHead(made);
        var recording = new RecordingBackend(_backend);
        var x = Held(made, "x");
        var output = Run(layer, x, made, recording);
        var loss = recording.Mean(recording.Multiply(output, Coefficients(output.Shape)));
        var gradients = recording.GradientsOf(loss, [x, .. network.Parameters().Select(parameter => parameter.Value)]);

        Assert.Equal(Shape(made.GetProperty("output")), output.Shape);
        AssertClose("the values", Values(made.GetProperty("output").GetProperty("values")), output.Values);
        AssertClose("the gradient to the series", Values(made.GetProperty("gradInput").GetProperty("x")), gradients[x].Values);

        foreach (var named in network.Slots())
        {
            AssertClose($"the gradient to {named.Path}", Values(made.GetProperty("slots").GetProperty(named.Path["attn.".Length..]).GetProperty("grad")), gradients[named.Slot.Value].Values);
        }
    }

    [Theory]
    [MemberData(nameof(EncoderCases))]
    public void AnEncoderBlockMadeOfWords_IsPyTorchsEncoderLayer_AndSoIsTheGradientToTheSeriesAndToEveryNumberItLearns(string name)
    {
        var made = Case(name);
        var shape = new EncoderShape(made.GetProperty("heads").GetInt32(), made.GetProperty("feedForward").GetInt32())
        {
            NormFirst = made.GetProperty("normFirst").GetBoolean(),
            Gelu = made.GetProperty("activation").GetString() == "gelu",
            Masked = made.GetProperty("mask").ValueKind != JsonValueKind.Null,
        };
        var embed = made.GetProperty("embed").GetInt32();
        var words = Sequential.EncoderWords(shape, embed);
        var network = shape.Masked ? words.Lower(new Shape(5, embed), new Shape(5), new RandomStream(7)) : words.Lower(new Shape(5, embed), new RandomStream(7));
        var block = network;

        network.Load([.. made.GetProperty("slots").EnumerateObject().Select(slot => new SlotEntry(
            slot.Name,
            Tensor.From(Shape(slot.Value), Values(slot.Value.GetProperty("values"))),
            "PyTorch"))]);

        var recording = new RecordingBackend(_backend);
        var x = Held(made, "x");
        var pass = Pass.Evaluation(recording);
        var output = Mask(made) is { } mask ? block.Forward(x, mask, pass) : block.Forward(x, pass);
        var loss = recording.Mean(recording.Multiply(output, Coefficients(output.Shape)));
        var gradients = recording.GradientsOf(loss, [x, .. network.Parameters().Select(parameter => parameter.Value)]);

        Assert.Equal(Shape(made.GetProperty("output")), output.Shape);
        AssertClose("the values", Values(made.GetProperty("output").GetProperty("values")), output.Values);
        AssertClose("the gradient to the series", Values(made.GetProperty("gradInput").GetProperty("x")), gradients[x].Values);

        foreach (var named in network.Slots())
        {
            AssertClose($"the gradient to {named.Path}", Values(made.GetProperty("slots").GetProperty(named.Path).GetProperty("grad")), gradients[named.Slot.Value].Values);
        }

        Assert.Equal(made.GetProperty("slots").EnumerateObject().Select(slot => slot.Name).Order(), network.Slots().Select(slot => slot.Path).Order());
    }

    [Theory]
    [MemberData(nameof(Masked))]
    public void ASeriesOfNoStepToRead_IsMadeNothingOf_AndNoNumberIsEverNotFinite(string name)
    {
        var made = Case(name);
        var output = made.GetProperty("kind").GetString() == "attention"
            ? new Attention(new Attending(made.GetProperty("heads").GetInt32()) { Causal = made.GetProperty("causal").GetBoolean() })
                .Forward(new QueryKeyValue(Held(made, "q"), Held(made, "k"), Held(made, "v")), Mask(made), Pass.Evaluation(_backend))
            : Run(MultiHead(made).Layer, Held(made, "x"), made, _backend);

        Assert.All(output.Values.ToArray(), value => Assert.True(float.IsFinite(value)));
    }

    [Fact]
    public void ASeriesOfNoStepToRead_ReadsNothingAndSoAttendsToNothing()
    {
        var made = Case("attention with a series of no step");
        var output = new Attention(new Attending(made.GetProperty("heads").GetInt32()))
            .Forward(new QueryKeyValue(Held(made, "q"), Held(made, "k"), Held(made, "v")), Mask(made), Pass.Evaluation(_backend));
        var steps = output.Shape[1] * output.Shape[2];

        Assert.All(output.Values.ToArray()[steps..(2 * steps)], value => Assert.Equal(0f, value));
    }

    [Theory]
    [MemberData(nameof(AttentionCases))]
    [MemberData(nameof(MultiHeadCases))]
    public void ALayer_IsWrittenDownAndReadBackAsTheLayerItWas(string name)
    {
        var made = Case(name);
        var isAttention = made.GetProperty("kind").GetString() == "attention";
        Layer layer = isAttention
            ? new Attention(new Attending(made.GetProperty("heads").GetInt32()) { Causal = made.GetProperty("causal").GetBoolean() })
            : MultiHead(made).Layer;
        var network = isAttention ? new LayerStack([new Step(layer, "attn")]) : MultiHead(made).Network;
        var written = Written(network);
        var read = NetworkDocument.ReadNetwork(written, "network", NetworkCatalog.BuiltIn());
        var again = ((LayerStack)read.Network).Layers[0];

        Assert.Equal(layer.GetType(), again.GetType());
        Assert.Equal(network.Slots().Select(slot => slot.Path), read.Network.Slots().Select(slot => slot.Path));
        Assert.Equal(written, Written(read.Network));
    }

    [Fact]
    public void ADropoutOfTheWeights_DrawsTheSameOnEveryEngine_AndOnlyWhileTheLayerTrains()
    {
        var made = Case("multi-head attention");
        var (layer, _) = MultiHead(made, dropout: 0.4);
        var x = Held(made, "x");
        var stream = new RandomStream(11);
        var here = layer.Forward(x, Pass.Training(_backend, stream, 0, 0)).Values.ToArray();
        var oracle = layer.Forward(x, Pass.Training(new CpuBackend(), stream, 0, 0)).Values.ToArray();
        var evaluated = layer.Forward(x, Pass.Evaluation(_backend)).Values.ToArray();
        var again = layer.Forward(x, Pass.Training(_backend, stream, 0, 0)).Values.ToArray();

        Assert.Equal(oracle.Length, here.Length);
        Assert.All(here.Zip(oracle, (one, other) => Math.Abs(one - other)), difference => Assert.InRange(difference, 0, 2e-5));
        Assert.Equal(here, again);
        Assert.NotEqual(evaluated, here);
        Assert.Equal(Shape(made.GetProperty("output")).Count, evaluated.Length);
    }

    [Fact]
    public void ALayerRefusesWhatIsNotASeriesOfItsOwnWidth_AndAMaskOfAnotherShape()
    {
        var made = Case("multi-head attention");
        var (layer, _) = MultiHead(made);
        var pass = Pass.Evaluation(_backend);

        Assert.Throws<ArgumentException>(() => layer.Forward(Tensor.Zeros(new Shape(2, 8)), pass));
        Assert.Throws<ArgumentException>(() => layer.Forward(Tensor.Zeros(new Shape(2, 5, 7)), pass));
        Assert.Throws<ArgumentException>(() => layer.Forward(Tensor.Zeros(new Shape(2, 5, 8)), Tensor.Zeros(new Shape(2, 4)), pass));
        Assert.Throws<ArgumentException>(() => new MultiHeadAttention(8, new Attending(3), new RandomStream(1).Draw("a", 0, 0)));
        Assert.Throws<ArgumentException>(() => new Attention(new Attending(3)).Forward(new QueryKeyValue(Tensor.Zeros(new Shape(2, 5, 8)), Tensor.Zeros(new Shape(2, 5, 8)), Tensor.Zeros(new Shape(2, 5, 8))), null, pass));
        Assert.Throws<ArgumentException>(() => new Attention(new Attending(2)).Forward(new QueryKeyValue(Tensor.Zeros(new Shape(2, 5, 8)), Tensor.Zeros(new Shape(2, 4, 8)), Tensor.Zeros(new Shape(2, 5, 8))), null, pass));
    }

    [Fact]
    public void ThePositions_AreTheSineAndCosineOfAStepOverTenThousandToThePowerOfThePairOverTheWidth()
    {
        var positions = Fixture.GetProperty("positions");
        var layer = new SinePositions();
        var nothing = Tensor.Zeros(new Shape(2, 5, 8));
        var made = layer.Forward(nothing, Pass.Evaluation(_backend));
        var each = made.Values.ToArray();

        Assert.Equal(Shape(positions), new Shape(5, 8));
        AssertClose("the positions of the first series", Values(positions.GetProperty("values")), each.AsSpan(0, 40));
        AssertClose("the positions of the second series", Values(positions.GetProperty("values")), each.AsSpan(40, 40));
    }

    [Fact]
    public void ThePositions_AreAddedToWhatTheyAreHanded_AndTheGradientPassesThroughUnchanged()
    {
        var x = Held(Case("multi-head attention"), "x");
        var recording = new RecordingBackend(_backend);
        var output = new SinePositions().Forward(x, Pass.Evaluation(recording));
        var loss = recording.Mean(recording.Multiply(output, Coefficients(output.Shape)));
        var gradients = recording.GradientsOf(loss, [x]);
        var bare = new SinePositions().Forward(Tensor.Zeros(x.Shape), Pass.Evaluation(_backend)).Values.ToArray();

        AssertClose("the series with its positions", x.Values.ToArray().Zip(bare, (one, other) => one + other).ToArray(), output.Values);
        AssertClose("the gradient to the series", Values(Coefficients(output.Shape).Values.ToArray().Select(value => value / output.Shape.Count)), gradients[x].Values);
        Assert.Throws<ArgumentException>(() => new SinePositions().Forward(Tensor.Zeros(new Shape(5, 8)), Pass.Evaluation(_backend)));
    }

    // The layer a case describes, held by a stack under the name the slots go by, with PyTorch's numbers loaded into it.
    private (MultiHeadAttention Layer, LayerStack Network) MultiHead(JsonElement made, double dropout = 0)
    {
        var how = new Attending(made.GetProperty("heads").GetInt32()) { Causal = made.GetProperty("causal").GetBoolean(), Dropout = dropout };
        var layer = new MultiHeadAttention(made.GetProperty("embed").GetInt32(), how, new RandomStream(7).Draw("attention", 0, 0));
        var network = new LayerStack([new Step(layer, "attn")]);

        network.Load([.. made.GetProperty("slots").EnumerateObject().Select(slot => new SlotEntry(
            $"attn.{slot.Name}",
            Tensor.From(Shape(slot.Value), Values(slot.Value.GetProperty("values"))),
            "PyTorch"))]);

        return (layer, network);
    }

    private static Tensor Run(MultiHeadAttention layer, Tensor x, JsonElement made, ITensorBackend backend)
    {
        var pass = Pass.Evaluation(backend);

        return Mask(made) is { } mask ? layer.Forward(x, mask, pass) : layer.Forward(x, pass);
    }

    private static Tensor? Mask(JsonElement made)
    {
        if (made.GetProperty("mask").ValueKind == JsonValueKind.Null)
        {
            return null;
        }

        var rows = made.GetProperty("mask").EnumerateArray().ToArray();

        return Tensor.From(new Shape(rows.Length, rows[0].GetArrayLength()), [.. rows.SelectMany(row => row.EnumerateArray().Select(value => (float)value.GetDouble()))]);
    }

    private static Tensor Held(JsonElement made, string name) => Tensor.From(Shape(made.GetProperty(name)), Values(made.GetProperty(name).GetProperty("values")));

    private static TheoryData<string> Named(string kind) =>
        [.. Fixture.GetProperty("cases").EnumerateArray().Where(each => each.GetProperty("kind").GetString() == kind).Select(each => each.GetProperty("name").GetString()!)];

    private static JsonElement Case(string name) => Fixture.GetProperty("cases").EnumerateArray().Single(each => each.GetProperty("name").GetString() == name);

    private static Shape Shape(JsonElement tensor) => new([.. tensor.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]);

    private static float[] Values(JsonElement numbers) => [.. numbers.EnumerateArray().Select(value => (float)value.GetDouble())];

    private static float[] Values(IEnumerable<float> numbers) => [.. numbers];

    private static Tensor Coefficients(Shape shape) => Tensor.From(shape, [.. Enumerable.Range(0, shape.Count).Select(at => ((((3 * at) % 7) - 3) / 10f))]);

    private static string Written(Network network)
    {
        using var stream = new MemoryStream();

        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, network, new MeanSquaredError());
            writer.WriteEndObject();
        }

        return System.Text.Encoding.UTF8.GetString(stream.ToArray()).Replace("\r\n", "\n", StringComparison.Ordinal);
    }

    // A gradient or a value within what two different engines doing the same sums in a different order may differ by.
    private static void AssertClose(string what, float[] expected, ReadOnlySpan<float> actual)
    {
        Assert.Equal(expected.Length, actual.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.True(Math.Abs(expected[at] - actual[at]) <= 2e-5 + (1e-4 * Math.Abs(expected[at])), $"{what}, number {at}: PyTorch {expected[at]}, here {actual[at]}.");
        }
    }
}
