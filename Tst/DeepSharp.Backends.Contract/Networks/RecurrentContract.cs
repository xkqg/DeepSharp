// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// A layer that reads a series a step at a time and carries a state along it — an LSTM, a GRU — of one layer or of two, in one direction or
/// in both, gives PyTorch's values for every step and for the last state of each series, and the gradient of a loss through them is PyTorch's,
/// to the input and to every number the layer learns, on every engine. A batch of series of unequal lengths is read each as it would be read
/// alone: a step the mask says is no step of a series leaves its state as it was and makes nothing, and a series read backwards starts from
/// its own end. A layer is written down and read back as the very layer it was.
/// </summary>
/// <param name="engine">The engine the arithmetic runs on.</param>
public abstract class RecurrentContract(ITensorBackend engine)
{
    private static readonly JsonElement Fixture = JsonDocument
        .Parse(File.ReadAllText(Path.Join(Repository.Root, "Tst", "DeepSharp.Backends.Contract", "Networks", "Fixtures", "recurrent-pytorch.json")))
        .RootElement;

    private readonly ITensorBackend _backend = engine;

    public static TheoryData<string> Cases { get; } = [.. Fixture.GetProperty("cases").EnumerateArray().Select(each => each.GetProperty("name").GetString()!)];

    [Theory]
    [MemberData(nameof(Cases))]
    public void EveryStepOfEverySeries_IsPyTorchs_AndSoIsTheGradientToTheInputAndToEveryNumberTheLayerLearns(string name)
    {
        var made = Case(name);
        var (layer, network) = Layer(made, sequences: true);
        var recording = new RecordingBackend(_backend);
        var x = Series(made);
        var output = Run(layer, x, made, recording);
        var loss = recording.Mean(recording.Multiply(output, Coefficients(output.Shape)));
        var gradients = recording.GradientsOf(loss, [x, .. layer.Parameters().Select(parameter => parameter.Value)]);

        Assert.Equal(Shape(made.GetProperty("sequences")), output.Shape);
        AssertClose("the values", Values(made.GetProperty("sequences").GetProperty("values")), output.Values);
        AssertClose("the gradient to the input", Values(made.GetProperty("gradInput").GetProperty("sequences")), gradients[x].Values);

        foreach (var named in network.Slots())
        {
            var slot = made.GetProperty("slots").GetProperty(named.Path["rnn.".Length..]);

            AssertClose($"the gradient to {named.Path}", Values(slot.GetProperty("grad").GetProperty("sequences")), gradients[named.Slot.Value].Values);
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void TheLastStateOfEverySeries_IsAtItsOwnLength_AndTheGradientThroughItIsPyTorchs(string name)
    {
        var made = Case(name);
        var (layer, network) = Layer(made, sequences: false);
        var recording = new RecordingBackend(_backend);
        var x = Series(made);
        var output = Run(layer, x, made, recording);
        var loss = recording.Mean(recording.Multiply(output, Coefficients(output.Shape)));
        var gradients = recording.GradientsOf(loss, [x, .. layer.Parameters().Select(parameter => parameter.Value)]);

        Assert.Equal(Shape(made.GetProperty("last")), output.Shape);
        AssertClose("the last states", Values(made.GetProperty("last").GetProperty("values")), output.Values);
        AssertClose("the gradient to the input", Values(made.GetProperty("gradInput").GetProperty("last")), gradients[x].Values);

        foreach (var named in network.Slots())
        {
            AssertClose($"the gradient to {named.Path}", Values(made.GetProperty("slots").GetProperty(named.Path["rnn.".Length..]).GetProperty("grad").GetProperty("last")), gradients[named.Slot.Value].Values);
        }
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void ALayer_IsWrittenDownAndReadBackAsTheLayerItWas(string name)
    {
        var made = Case(name);
        var (layer, network) = Layer(made, sequences: true);
        var written = Written(network);
        var read = NetworkDocument.ReadNetwork(written, "network", NetworkCatalog.BuiltIn());
        var again = Assert.IsAssignableFrom<Recurrent>(((LayerStack)read.Network).Layers[0]);
        var x = Series(made);

        Assert.Equal(network.Slots().Select(slot => slot.Path), read.Network.Slots().Select(slot => slot.Path));
        Assert.Equal(
            Run(layer, x, made, _backend).Values.ToArray(),
            Run(again, x, made, _backend).Values.ToArray());
        Assert.Equal(written, Written(read.Network));
    }

    [Fact]
    public void TheGatesAreInPyTorchsOrder_SoAWeightScaledGateByGateGivesPyTorchsValuesOnlyInThatOrder()
    {
        var made = Case("lstm");
        var (layer, network) = Layer(made, sequences: true);
        var x = Series(made);
        var asLoaded = Run(layer, x, made, _backend).Values.ToArray();

        // Swapping the forget gate's block and the cell gate's block of the first weight changes the answer: the order is part of what is loaded.
        var weight = network.Slots().Single(slot => slot.Path == "rnn.weight_ih_l0").Slot.Value;
        var swapped = weight.Values.ToArray();
        var hidden = made.GetProperty("hidden").GetInt32();
        var rows = weight.Shape[0];

        for (var row = 0; row < rows; row++)
        {
            for (var unit = 0; unit < hidden; unit++)
            {
                (swapped[(row * 4 * hidden) + hidden + unit], swapped[(row * 4 * hidden) + (2 * hidden) + unit]) = (swapped[(row * 4 * hidden) + (2 * hidden) + unit], swapped[(row * 4 * hidden) + hidden + unit]);
            }
        }

        network.Load([.. network.Slots().Select(named => new SlotEntry(named.Path, named.Path == "rnn.weight_ih_l0" ? Tensor.From(weight.Shape, swapped) : named.Slot.Value, "gates swapped"))]);

        Assert.NotEqual(asLoaded, Run(layer, x, made, _backend).Values.ToArray());
    }

    [Fact]
    public void AnUnequalBatchWithPaddingOfAMillion_NeverMakesANumberThatIsNotFinite()
    {
        var made = Case("lstm over series of unequal lengths, both directions");
        var (layer, _) = Layer(made, sequences: true);
        var output = Run(layer, Series(made), made, _backend);

        Assert.All(output.Values.ToArray(), value => Assert.True(float.IsFinite(value)));
    }

    [Fact]
    public void ASeriesOfNoStepToRead_MakesNothingAndLeavesItsStateAtNought()
    {
        var made = Case("gru over series of unequal lengths, two layers");
        var (layer, _) = Layer(made, sequences: false);
        var series = Series(made);
        var none = Tensor.Zeros(new Shape(series.Shape[0], series.Shape[1]));
        var output = layer.Forward(series, none, Pass.Evaluation(_backend));

        Assert.All(output.Values.ToArray(), value => Assert.Equal(0f, value));
    }

    [Fact]
    public void ALayerRefusesWhatIsNotABatchOfSeriesOfItsOwnWidth_AndAMaskOfAnotherShape()
    {
        var made = Case("gru");
        var (layer, _) = Layer(made, sequences: false);
        var pass = Pass.Evaluation(_backend);

        Assert.Throws<ArgumentException>(() => layer.Forward(Tensor.Zeros(new Shape(2, 3)), pass));
        Assert.Throws<ArgumentException>(() => layer.Forward(Tensor.Zeros(new Shape(2, 5, 7)), pass));
        Assert.Throws<ArgumentException>(() => layer.Forward(Tensor.Zeros(new Shape(2, 5, 3)), Tensor.Zeros(new Shape(2, 4)), pass));
    }

    // The layer a case describes, held by a stack under the name the slots go by, with PyTorch's numbers loaded into it.
    private (Recurrent Layer, LayerStack Network) Layer(JsonElement made, bool sequences)
    {
        var recurrence = new Recurrence(made.GetProperty("inputs").GetInt32(), made.GetProperty("hidden").GetInt32())
        {
            Layers = made.GetProperty("layers").GetInt32(),
            BothWays = made.GetProperty("bothWays").GetBoolean(),
            Sequences = sequences,
        };
        var draws = new RandomStream(7).Draw("recurrent", 0, 0);
        Recurrent layer = made.GetProperty("kind").GetString() == "lstm" ? new Lstm(recurrence, draws) : new Gru(recurrence, draws);
        var network = new LayerStack([new Step(layer, "rnn")]);

        network.Load([.. made.GetProperty("slots").EnumerateObject().Select(slot => new SlotEntry(
            $"rnn.{slot.Name}",
            Tensor.From(Shape(slot.Value), Values(slot.Value.GetProperty("values"))),
            "PyTorch"))]);

        return (layer, network);
    }

    private static Tensor Run(Recurrent layer, Tensor x, JsonElement made, ITensorBackend backend)
    {
        var pass = Pass.Evaluation(backend);

        return made.GetProperty("mask").ValueKind == JsonValueKind.Null ? layer.Forward(x, pass) : layer.Forward(x, Mask(made), pass);
    }

    private static Tensor Mask(JsonElement made)
    {
        var rows = made.GetProperty("mask").EnumerateArray().ToArray();

        return Tensor.From(new Shape(rows.Length, rows[0].GetArrayLength()), [.. rows.SelectMany(row => row.EnumerateArray().Select(value => (float)value.GetDouble()))]);
    }

    private static Tensor Series(JsonElement made) => Tensor.From(Shape(made.GetProperty("x")), Values(made.GetProperty("x").GetProperty("values")));

    private static JsonElement Case(string name) => Fixture.GetProperty("cases").EnumerateArray().Single(each => each.GetProperty("name").GetString() == name);

    private static Shape Shape(JsonElement tensor) => new([.. tensor.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]);

    private static float[] Values(JsonElement numbers) => [.. numbers.EnumerateArray().Select(value => (float)value.GetDouble())];

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
