// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Export.Onnx;
using DeepSharp.Import.Onnx;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using Onnx;

namespace DeepSharp.Tests.Export.Onnx;

/// <summary>
/// Layers that read a series a step at a time, written as ONNX's LSTM and GRU: the weights turned round and their gates put in ONNX's order
/// (input, output, forget, cell; update, reset, hidden), one node for each layer of a stack and the two directions in the one node, the
/// last state or every step taken out of the node's outputs, and the lengths of the series from the mask a tokenizer's padding makes. What is
/// written is answered through by ONNX Runtime as the network answers.
/// </summary>
public class OnnxRecurrentExportTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private sealed record Reading(Func<LayerStack> Network, Shape Each);

    private static readonly Dictionary<string, Reading> Readings = new()
    {
        ["lstm to its last state"] = new(() => new Sequential().Lstm(4).Dense(2).Lower(new Shape(5, 3), new RandomStream(1)), new Shape(5, 3)),
        ["gru at every step"] = new(() => new Sequential().Gru(4, sequences: true).Flatten().Dense(2).Lower(new Shape(5, 3), new RandomStream(2)), new Shape(5, 3)),
        ["lstm of two layers in both directions"] = new(() => new Sequential().Lstm(3, bothWays: true, layers: 2).Dense(1).Lower(new Shape(5, 3), new RandomStream(3)), new Shape(5, 3)),
        ["gru in both directions at every step"] = new(() => new Sequential().Gru(3, bothWays: true, sequences: true).Dense(2).GlobalMaxPool1D().Dense(1).Lower(new Shape(5, 3), new RandomStream(4)), new Shape(5, 3)),
        ["gru in both directions pooled"] = new(() => new Sequential().Gru(3, bothWays: true, sequences: true).GlobalMaxPool1D().Dense(1).Lower(new Shape(5, 3), new RandomStream(10)), new Shape(5, 3)),
        ["a stack of an lstm and a gru"] = new(() => new Sequential().Lstm(4, sequences: true).Gru(3).Dense(1).Lower(new Shape(6, 2), new RandomStream(5)), new Shape(6, 2)),
        ["a layer run over every step"] = new(() => new Sequential().Gru(4, sequences: true).EveryStep(new Sequential().Dense(3).Relu()).GlobalAvgPool1D().Dense(1).Lower(new Shape(5, 3), new RandomStream(6)), new Shape(5, 3)),
    };

    public static TheoryData<string> EveryReading => [.. Readings.Keys];

    [Theory]
    [MemberData(nameof(EveryReading))]
    public void AWrittenGraph_IsAnsweredThroughByOnnxRuntime_AsTheNetworkAnswers(string name)
    {
        var network = Runtime.Filled(Readings[name].Network());
        var loss = new MeanSquaredError();
        var each = Readings[name].Each;
        var rows = Runtime.Batch(4, each);
        var graph = new OnnxExport(each).ToBytes(network, loss);

        var here = network.Predict(rows, loss, Engine).Values.ToArray();
        var there = Runtime.Answers(graph, Runtime.ChannelsFirst(rows));

        Assert.Equal(here.Length, there.Length);
        Assert.InRange(here.Zip(there, (one, other) => Math.Abs(one - other)).Max(), 0, 1e-5);
    }

    // A dense layer over the steps of a series, and a layer run over every step, are written with turns of the series the reader does not read:
    // it refuses them at the Transpose, by name.
    private static readonly string[] NotReadBack = ["gru in both directions at every step", "a layer run over every step"];

    public static TheoryData<string> EveryReadableReading => [.. Readings.Keys.Except(NotReadBack)];

    [Theory]
    [MemberData(nameof(EveryReadableReading))]
    public void AWrittenGraph_IsReadBackByTheReader_AnsweringAsTheNetworkDoes(string name)
    {
        var network = Runtime.Filled(Readings[name].Network());
        var loss = new MeanSquaredError();
        var each = Readings[name].Each;
        var rows = Runtime.Batch(4, each);
        using var file = new MemoryStream(new OnnxExport(each).ToBytes(network, loss));
        var read = new OnnxFile(loss).Read(file);

        var here = network.Predict(rows, loss, Engine).Values.ToArray();
        var back = read.Network.Predict(rows, read.Loss, Engine).Values.ToArray();

        Assert.Equal(here.Length, back.Length);
        Assert.InRange(here.Zip(back, (one, other) => Math.Abs(one - other)).Max(), 0, 1e-5);
    }

    [Fact]
    public void AnLstmIsOneNodeForEachLayer_ItsGatesInOnnxsOrder_AndAGruSaysItResetsAfter()
    {
        var lstm = ModelProto.Parser.ParseFrom(new OnnxExport(new Shape(5, 3)).ToBytes(Runtime.Filled(Readings["lstm of two layers in both directions"].Network()), new MeanSquaredError()));
        var gru = ModelProto.Parser.ParseFrom(new OnnxExport(new Shape(5, 3)).ToBytes(Runtime.Filled(Readings["gru at every step"].Network()), new MeanSquaredError()));
        var nodes = lstm.Graph.Node.Where(node => node.OpType == "LSTM").ToArray();

        Assert.Equal(2, nodes.Length);
        Assert.All(nodes, node => Assert.Equal("bidirectional", node.Attribute.Single(attribute => attribute.Name == "direction").S.ToStringUtf8()));
        Assert.Equal([2L, 12L, 3L], lstm.Graph.Initializer.Single(each => each.Name == nodes[0].Input[1]).Dims);
        Assert.Equal([2L, 12L, 6L], lstm.Graph.Initializer.Single(each => each.Name == nodes[1].Input[1]).Dims);
        Assert.Equal(1, gru.Graph.Node.Single(node => node.OpType == "GRU").Attribute.Single(attribute => attribute.Name == "linear_before_reset").I);
    }

    [Fact]
    public void ABatchOfTextsOfUnequalLengths_IsGivenItsLengthsFromTheMaskOfThePadding_AndAnswersAsTheNetworkDoes()
    {
        var network = Runtime.Filled(new Sequential().TokenEmbedding(7, 3, keepSteps: true).Named("steps").Unpadded().Named("mask").Takes("input")
            .Lstm(4).Takes("steps", "mask").Dense(1).Lower(new Shape(5), new RandomStream(7)));
        var loss = new MeanSquaredError();
        float[] ids = [3, 4, 0, 0, 0, 1, 2, 5, 6, 3, 2, 0, 0, 0, 0, 5, 5, 5, 5, 5];
        var rows = Tensor.From(new Shape(4, 5), ids);
        var graph = new OnnxExport(new Shape(5)).ToBytes(network, loss);
        var model = ModelProto.Parser.ParseFrom(graph);

        var here = network.Predict(rows, loss, Engine).Values.ToArray();
        var there = Runtime.Answers(graph, rows);

        Assert.Contains(model.Graph.Node, node => node.OpType == "LSTM" && node.Input.Count >= 5 && node.Input[4].Length > 0);
        Assert.InRange(here.Zip(there, (one, other) => Math.Abs(one - other)).Max(), 0, 1e-5);
    }

    [Theory]
    [InlineData("lstm")]
    [InlineData("gru")]
    public void TheStepsBeyondTheEndOfAShorterSeries_AreNothingInTheGraphToo_AsTheNetworkMakesNothingThere(string kind)
    {
        // Measured: ONNX Runtime gives nought for every step beyond the lengths it is handed, as a packed batch in PyTorch does.
        var words = new Sequential().TokenEmbedding(7, 3, keepSteps: true).Named("steps").Unpadded().Named("mask").Takes("input");
        var network = Runtime.Filled((kind == "lstm" ? words.Lstm(4, bothWays: true, sequences: true) : words.Gru(4, bothWays: true, sequences: true)).Takes("steps", "mask").GlobalAvgPool1D().Dense(1)
            .Lower(new Shape(5), new RandomStream(9)));
        var loss = new MeanSquaredError();
        var rows = Tensor.From(new Shape(3, 5), [3, 4, 0, 0, 0, 1, 2, 5, 6, 3, 2, 0, 0, 0, 0]);

        Assert.InRange(
            network.Predict(rows, loss, Engine).Values.ToArray().Zip(Runtime.Answers(new OnnxExport(new Shape(5)).ToBytes(network, loss), rows), (one, other) => Math.Abs(one - other)).Max(),
            0,
            1e-5);
    }

    [Fact]
    public void TheMaskOfARowOfIds_AndTheStepsOfARowOfIds_AreWrittenAndAnsweredThrough()
    {
        var steps = Runtime.Filled(new Sequential().TokenEmbedding(7, 3, keepSteps: true).GlobalAvgPool1D().Dense(1).Lower(new Shape(5), new RandomStream(8)));
        float[] ids = [3, 4, 0, 0, 0, 1, 2, 5, 6, 3];
        var rows = Tensor.From(new Shape(2, 5), ids);
        var loss = new MeanSquaredError();

        Assert.InRange(
            steps.Predict(rows, loss, Engine).Values.ToArray().Zip(Runtime.Answers(new OnnxExport(new Shape(5)).ToBytes(steps, loss), rows), (one, other) => Math.Abs(one - other)).Max(),
            0,
            1e-5);
    }
}
