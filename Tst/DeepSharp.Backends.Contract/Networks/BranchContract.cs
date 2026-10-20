// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// A network that is not a straight line: a step may ask for what a step other than the one before it made, and two steps' work
/// is added or laid side by side. A residual block, a block where every layer reads everything made before it, a network that takes
/// two blocks of a row's columns and one shaped like a U give PyTorch's values, and the gradient of a loss through them is
/// PyTorch's, to the input and to every number the network learns, on every engine; and a network written so is written down and
/// read back as the very network it was.
/// </summary>
/// <param name="engine">The engine the arithmetic runs on.</param>
public abstract class BranchContract(ITensorBackend engine)
{
    private static readonly JsonElement Fixture = JsonDocument
        .Parse(File.ReadAllText(Path.Join(Repository.Root, "Tst", "DeepSharp.Backends.Contract", "Networks", "Fixtures", "branch-pytorch.json")))
        .RootElement;

    private readonly ITensorBackend _backend = engine;

    public static TheoryData<string> Walks { get; } = [.. Fixture.GetProperty("walks").EnumerateArray().Select(walk => walk.GetProperty("name").GetString()!)];

    [Theory]
    [MemberData(nameof(Walks))]
    public void ABranchingNetwork_GivesPyTorchsValues_AndPyTorchsGradients_ToTheInputAndToEveryNumberItLearns(string name)
    {
        var walk = Walk(name);
        var network = Network(name, walk);
        var recording = new RecordingBackend(_backend);
        var rows = Rows(walk);
        var output = network.Forward(rows, Pass.Evaluation(recording));
        var coefficients = Coefficients(output.Shape);
        var loss = recording.Mean(recording.Multiply(output, coefficients));
        var learned = network.Parameters().ToArray();
        var gradients = recording.GradientsOf(loss, [rows, .. learned.Select(parameter => parameter.Value)]);

        Assert.Equal(new Shape(walk.GetProperty("rows").GetInt32(), walk.GetProperty("outputWidth").GetInt32()), output.Shape);
        AssertClose("the values", Values(walk.GetProperty("output")), output.Values);
        AssertClose("the gradient to the input", Values(walk.GetProperty("gradInput")), gradients[rows].Values);

        foreach (var named in network.Slots())
        {
            var expected = walk.GetProperty("slots").GetProperty(named.Path);

            AssertClose($"the gradient to {named.Path}", Values(expected.GetProperty("grad")), gradients[named.Slot.Value].Values);
        }

        Assert.Equal(walk.GetProperty("slots").EnumerateObject().Select(slot => slot.Name).Order(StringComparer.Ordinal), network.Slots().Select(slot => slot.Path).Order(StringComparer.Ordinal));
    }

    [Theory]
    [MemberData(nameof(Walks))]
    public void ABranchingNetwork_IsWrittenDownAndReadBackAsTheNetworkItWas(string name)
    {
        var walk = Walk(name);
        var network = Network(name, walk);
        var rows = Rows(walk);
        var written = Written(network, new MeanSquaredError());
        var read = NetworkDocument.ReadNetwork(written, "network", NetworkCatalog.BuiltIn());

        Assert.Equal(network.Forward(rows, Pass.Evaluation(_backend)).Values.ToArray(), read.Network.Forward(rows, Pass.Evaluation(_backend)).Values.ToArray());
        Assert.Equal(network.Slots().Select(slot => slot.Path), read.Network.Slots().Select(slot => slot.Path));
        Assert.Equal(written, Written(read.Network, read.Loss));
    }

    [Fact]
    public void TwoStepsThatAddUp_NeedTheSameShape_AndTheRefusalNamesBoth()
    {
        var network = new LayerStack(
            [
                new Step(new Dense(Tensor.Zeros(new Shape(3, 2)), Tensor.Zeros(new Shape(2))), "wide"),
                new Step(new Sum(), "out", ["input", "wide"]),
            ]);

        var refused = Assert.Throws<ArgumentException>(() => network.Forward(Tensor.Zeros(new Shape(2, 3)), Pass.Evaluation(_backend)));

        Assert.Contains("2x3", refused.Message, StringComparison.Ordinal);
        Assert.Contains("2x2", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AJoin_LaysTwoStepsSideBySide_InTheOrderTheyAreTaken_AndColumnsTakeTheBlocksBack()
    {
        var rows = Tensor.From(new Shape(2, 4), [1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f]);
        var network = new LayerStack(
            [
                new Step(new Columns(1, 2), "middle"),
                new Step(new Concatenation(), "out", ["middle", "input", "middle"]),
            ]);

        var joined = network.Forward(rows, Pass.Evaluation(_backend));

        Assert.Equal(new Shape(2, 8), joined.Shape);
        Assert.Equal([2f, 3f, 1f, 2f, 3f, 4f, 2f, 3f, 6f, 7f, 5f, 6f, 7f, 8f, 6f, 7f], joined.Values.ToArray());
    }

    [Fact]
    public void AJoinOfImages_JoinsTheirChannels_WhateverTheRowsTheyAreLaidOutAs()
    {
        var images = Tensor.From(new Shape(1, 2, 2, 1), [1f, 2f, 3f, 4f]);
        var network = new LayerStack(
            [
                new Step(new Relu(), "same"),
                new Step(new Concatenation(), "out", ["input", "same"]),
            ]);

        var joined = network.Forward(images, Pass.Evaluation(_backend));

        Assert.Equal(new Shape(1, 2, 2, 2), joined.Shape);
        Assert.Equal([1f, 1f, 2f, 2f, 3f, 3f, 4f, 4f], joined.Values.ToArray());
    }

    private static string Written(Network network, Loss loss)
    {
        using var stream = new MemoryStream();

        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, network, loss);
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(stream.ToArray());
    }

    private static JsonElement Walk(string name) => Fixture.GetProperty("walks").EnumerateArray().Single(each => each.GetProperty("name").GetString() == name);

    private static double[] Values(JsonElement list) => [.. list.EnumerateArray().Select(each => each.GetDouble())];

    private static Tensor Rows(JsonElement walk)
    {
        var count = walk.GetProperty("rows").GetInt32();
        var width = walk.GetProperty("width").GetInt32();

        return Tensor.From(new Shape(count, width), [.. Enumerable.Range(0, count * width).Select(at => (float)(((((3 * (at / width)) + (5 * (at % width))) % 9 - 4) / 8d) + 0.0137))]);
    }

    private static Tensor Coefficients(Shape shape) => Tensor.From(shape, [.. Enumerable.Range(0, shape.Count).Select(at => (float)((((3 * at) % 7) - 3) / 10d))]);

    private static Dense Dense(JsonElement walk, string name)
    {
        var weight = walk.GetProperty("slots").GetProperty($"{name}.weight");
        var bias = walk.GetProperty("slots").GetProperty($"{name}.bias");

        return new Dense(
            Tensor.From(new Shape([.. weight.GetProperty("shape").EnumerateArray().Select(each => each.GetInt32())]), [.. Values(weight.GetProperty("values")).Select(value => (float)value)]),
            Tensor.From(new Shape([.. bias.GetProperty("shape").EnumerateArray().Select(each => each.GetInt32())]), [.. Values(bias.GetProperty("values")).Select(value => (float)value)]));
    }

    // The network each walk of the fixture is, from the numbers the fixture holds, as steps that name what they take.
    private static LayerStack Network(string name, JsonElement walk) => name switch
    {
        "residual" => new LayerStack(
            [
                new Step(Dense(walk, "first"), "first"),
                new Step(new Relu(), "act"),
                new Step(Dense(walk, "second"), "second"),
                new Step(new Sum(), "out", ["input", "second"]),
            ]),
        "dense block" => new LayerStack(
            [
                new Step(Dense(walk, "a"), "a"),
                new Step(new Relu(), "ra"),
                new Step(new Concatenation(), "c1", ["input", "ra"]),
                new Step(Dense(walk, "b"), "b"),
                new Step(new Relu(), "rb"),
                new Step(new Concatenation(), "c2", ["input", "ra", "rb"]),
                new Step(Dense(walk, "out"), "out"),
            ]),
        "two inputs" => new LayerStack(
            [
                new Step(new Columns(0, 3), "leftColumns", ["input"]),
                new Step(Dense(walk, "left"), "left"),
                new Step(new Columns(3, 3), "rightColumns", ["input"]),
                new Step(Dense(walk, "right"), "right"),
                new Step(new Sum(), "out", ["left", "right"]),
            ]),
        "u shape" => new LayerStack(
            [
                new Step(Dense(walk, "down1"), "down1"),
                new Step(new Relu(), "first"),
                new Step(Dense(walk, "down2"), "down2"),
                new Step(new Relu(), "bottom"),
                new Step(Dense(walk, "up"), "up"),
                new Step(new Relu(), "back"),
                new Step(new Concatenation(), "cat", ["first", "back"]),
                new Step(Dense(walk, "last"), "last"),
            ]),
        _ => throw new ArgumentOutOfRangeException(nameof(name), name, "A walk the fixture does not hold."),
    };

    private static void AssertClose(string what, double[] expected, ReadOnlySpan<float> actual)
    {
        Assert.True(expected.Length == actual.Length, $"{what} holds {actual.Length} numbers, and PyTorch's holds {expected.Length}.");

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.True(Math.Abs(expected[at] - actual[at]) <= Math.Max(1e-5, Math.Abs(expected[at]) * 1e-5), $"{what}, number {at}: PyTorch has {expected[at]}, and this has {actual[at]}.");
        }
    }
}
