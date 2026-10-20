// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// A table model of the kind Gorishniy et al. describe: each column of a row turned into a token of its own — the column's value times a vector
/// the layer learns, plus a vector it learns — the tokens read by encoder layers, averaged over the columns, normalised, bent and put through a
/// linear layer. The tokens alone and the whole model give PyTorch's values, and the gradient of a loss through them is PyTorch's, to the rows and
/// to every number the model learns, on every engine.
/// </summary>
/// <param name="engine">The engine the arithmetic runs on.</param>
public abstract class ColumnTokensContract(ITensorBackend engine)
{
    private static readonly JsonElement Fixture = JsonDocument
        .Parse(File.ReadAllText(Path.Join(Repository.Root, "Tst", "DeepSharp.Backends.Contract", "Networks", "Fixtures", "table-pytorch.json")))
        .RootElement;

    private readonly ITensorBackend _backend = engine;

    [Fact]
    public void TheTokens_AreEachColumnTimesItsVectorPlusItsVector_AsPyTorchMakesThem_AndSoIsTheGradient()
    {
        var made = Case("tokens");
        var layer = new ColumnTokens(Fixture.GetProperty("columns").GetInt32(), Fixture.GetProperty("embed").GetInt32(), new RandomStream(3).Draw("tokens", 0, 0));
        var network = new LayerStack([new Step(layer, "tokens")]);

        Load(network, made, "tokens.");

        var recording = new RecordingBackend(_backend);
        var x = Rows(made);
        var output = layer.Forward(x, Pass.Evaluation(recording));
        var loss = recording.Mean(recording.Multiply(output, Coefficients(output.Shape)));
        var gradients = recording.GradientsOf(loss, [x, .. layer.Parameters().Select(parameter => parameter.Value)]);

        Assert.Equal(Shape(made.GetProperty("output")), output.Shape);
        AssertClose("the tokens", Values(made.GetProperty("output").GetProperty("values")), output.Values);
        AssertClose("the gradient to the rows", Values(made.GetProperty("gradInput")), gradients[x].Values);

        foreach (var named in network.Slots())
        {
            AssertClose($"the gradient to {named.Path}", Values(made.GetProperty("slots").GetProperty(named.Path["tokens.".Length..]).GetProperty("grad")), gradients[named.Slot.Value].Values);
        }
    }

    [Fact]
    public void ATableModelOfTokensEncoderLayersAndAHead_IsPyTorchs_AndSoIsTheGradientToTheRowsAndToEveryNumber()
    {
        var made = Case("table");
        var network = Table();

        Load(network, made, string.Empty);

        var recording = new RecordingBackend(_backend);
        var x = Rows(made);
        var output = network.Forward(x, Pass.Evaluation(recording));
        var loss = recording.Mean(recording.Multiply(output, Coefficients(output.Shape)));
        var gradients = recording.GradientsOf(loss, [x, .. network.Parameters().Select(parameter => parameter.Value)]);

        Assert.Equal(Shape(made.GetProperty("output")), output.Shape);
        AssertClose("the answers", Values(made.GetProperty("output").GetProperty("values")), output.Values);
        AssertClose("the gradient to the rows", Values(made.GetProperty("gradInput")), gradients[x].Values);

        foreach (var named in network.Slots())
        {
            AssertClose($"the gradient to {named.Path}", Values(made.GetProperty("slots").GetProperty(named.Path).GetProperty("grad")), gradients[named.Slot.Value].Values);
        }
    }

    [Fact]
    public void TheTokensRefuseWhatIsNotARowOfTheirColumns()
    {
        var layer = new ColumnTokens(6, 8, new RandomStream(3).Draw("tokens", 0, 0));
        var pass = Pass.Evaluation(_backend);

        Assert.Throws<ArgumentException>(() => layer.Forward(Tensor.Zeros(new Shape(2, 5)), pass));
        Assert.Throws<ArgumentException>(() => layer.Forward(Tensor.Zeros(new Shape(2, 6, 1)), pass));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ColumnTokens(0, 8, new RandomStream(3).Draw("tokens", 0, 0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ColumnTokens(6, 0, new RandomStream(3).Draw("tokens", 0, 0)));
    }

    // The model the fixture describes, its layers called as the modules of PyTorch's are.
    private static LayerStack Table()
    {
        var columns = Fixture.GetProperty("columns").GetInt32();
        var embed = Fixture.GetProperty("embed").GetInt32();
        var encoder = new EncoderShape(Fixture.GetProperty("heads").GetInt32(), Fixture.GetProperty("feedForward").GetInt32());
        var stream = new RandomStream(7);

        return new LayerStack(
        [
            new Step(new ColumnTokens(columns, embed, stream.Draw("tokens", 0, 0)), "tokens"),
            new Step(Sequential.EncoderWords(encoder, embed).Lower(new Shape(columns, embed), stream), "enc0"),
            new Step(Sequential.EncoderWords(encoder, embed).Lower(new Shape(columns, embed), stream), "enc1"),
            new Step(new GlobalAvgPool1D(), "pool"),
            new Step(new LayerNorm(embed), "norm"),
            new Step(new Relu(), "bend"),
            new Step(new Dense(embed, 1, stream.Draw("head", 0, 0)), "head"),
        ]);
    }

    private static void Load(LayerStack network, JsonElement made, string prefix) =>
        network.Load([.. made.GetProperty("slots").EnumerateObject().Select(slot => new SlotEntry(
            $"{prefix}{slot.Name}",
            Tensor.From(Shape(slot.Value), Values(slot.Value.GetProperty("values"))),
            "PyTorch"))]);

    private static JsonElement Case(string name) => Fixture.GetProperty("cases").EnumerateArray().Single(each => each.GetProperty("name").GetString() == name);

    private static Tensor Rows(JsonElement made) => Tensor.From(Shape(made.GetProperty("x")), Values(made.GetProperty("x").GetProperty("values")));

    private static Shape Shape(JsonElement tensor) => new([.. tensor.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]);

    private static float[] Values(JsonElement numbers) => [.. numbers.EnumerateArray().Select(value => (float)value.GetDouble())];

    private static Tensor Coefficients(Shape shape) => Tensor.From(shape, [.. Enumerable.Range(0, shape.Count).Select(at => ((((3 * at) % 7) - 3) / 10f))]);

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
