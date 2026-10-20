// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Export.Onnx;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using Onnx;

namespace DeepSharp.Tests.Export.Onnx;

/// <summary>
/// Attention written as ONNX nodes of the oldest operator sets that hold it: the projections as products of the series by weights cut from
/// the one projection, the heads cut out by a reshape and a transpose, the scores as a batched product, the masks as a sum of what is far
/// below nothing, the softmax over the last axis and the answer taken back through the heads and the output layer. What is written is
/// answered through by ONNX Runtime as the network answers.
/// </summary>
public class OnnxAttentionExportTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private sealed record Reading(Func<LayerStack> Network, Shape Each);

    private static readonly Dictionary<string, Reading> Readings = new()
    {
        ["multi-head attention"] = new(() => new Sequential().MultiHeadAttention(2).GlobalAvgPool1D().Dense(1).Lower(new Shape(5, 8), new RandomStream(1)), new Shape(5, 8)),
        ["multi-head attention, causal"] = new(() => new Sequential().MultiHeadAttention(new Attending(4) { Causal = true }).GlobalAvgPool1D().Dense(1).Lower(new Shape(5, 8), new RandomStream(2)), new Shape(5, 8)),
        ["attention over the series itself"] = new(() => new Sequential().Attention(2).GlobalMaxPool1D().Dense(1).Lower(new Shape(6, 4), new RandomStream(3)), new Shape(6, 4)),
        ["attention over queries, keys and values"] = new(
            () => new Sequential().Dense(4).Named("q").Dense(4).Named("k").Takes("input").Dense(4).Named("v").Takes("input").Attention(2).Takes("q", "k", "v").GlobalAvgPool1D().Dense(1).Lower(new Shape(5, 3), new RandomStream(4)),
            new Shape(5, 3)),
        ["sine positions before attention"] = new(() => new Sequential().SinePositions().MultiHeadAttention(2).GlobalAvgPool1D().Dense(1).Lower(new Shape(7, 8), new RandomStream(5)), new Shape(7, 8)),
        ["an encoder block, norm after"] = new(() => new Sequential().EncoderBlock(new EncoderShape(2, 16)).Named("enc").GlobalAvgPool1D().Dense(1).Lower(new Shape(5, 8), new RandomStream(6)), new Shape(5, 8)),
        ["an encoder block, norm before, gelu"] = new(
            () => new Sequential().EncoderBlock(new EncoderShape(2, 16) { NormFirst = true, Gelu = true }).Named("enc").GlobalAvgPool1D().Dense(1).Lower(new Shape(5, 8), new RandomStream(7)),
            new Shape(5, 8)),
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

    [Fact]
    public void ATextOfUnequalLengths_IsReadByAnEncoderThatLooksPastItsPadding_AndAnswersAsTheNetworkDoes()
    {
        var steps = new Sequential().TokenEmbedding(7, 8, keepSteps: true).Named("steps")
            .Unpadded().Named("mask").Takes("input")
            .EncoderBlock(new EncoderShape(2, 16) { Masked = true }).Named("enc").Takes("steps", "mask")
            .GlobalAvgPool1D().Dense(1);
        var network = Runtime.Filled(steps.Lower(new Shape(5), new RandomStream(8)));
        var loss = new MeanSquaredError();
        float[] ids = [3, 4, 0, 0, 0, 1, 2, 5, 6, 3, 2, 0, 0, 0, 0, 5, 5, 5, 5, 5];
        var rows = Tensor.From(new Shape(4, 5), ids);
        var graph = new OnnxExport(new Shape(5)).ToBytes(network, loss);

        var here = network.Predict(rows, loss, Engine).Values.ToArray();
        var there = Runtime.Answers(graph, rows);

        Assert.InRange(here.Zip(there, (one, other) => Math.Abs(one - other)).Max(), 0, 1e-5);
    }

    [Fact]
    public void ASeriesWithNoStepToLookAt_IsMadeNothingOf_InTheGraphToo()
    {
        var steps = new Sequential().TokenEmbedding(7, 8, keepSteps: true).Named("steps")
            .Unpadded().Named("mask").Takes("input")
            .MultiHeadAttention(2).Named("attended").Takes("steps", "mask")
            .GlobalAvgPool1D().Dense(1);
        var network = Runtime.Filled(steps.Lower(new Shape(5), new RandomStream(9)));
        var loss = new MeanSquaredError();
        var rows = Tensor.From(new Shape(3, 5), [3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 5, 6, 3]);
        var graph = new OnnxExport(new Shape(5)).ToBytes(network, loss);

        Assert.InRange(
            network.Predict(rows, loss, Engine).Values.ToArray().Zip(Runtime.Answers(graph, rows), (one, other) => Math.Abs(one - other)).Max(),
            0,
            1e-5);
    }

    [Fact]
    public void TheHeadsAreCutOutByAReshape_AndTheScoresAreOneBatchedProduct_OfTheOldestOperatorSetsThatHoldThem()
    {
        var model = ModelProto.Parser.ParseFrom(new OnnxExport(new Shape(5, 8)).ToBytes(Runtime.Filled(Readings["multi-head attention"].Network()), new MeanSquaredError()));

        Assert.Equal(7, model.OpsetImport.Single().Version);
        Assert.Contains(model.Graph.Node, node => node.OpType == "Softmax");
        Assert.DoesNotContain(model.Graph.Node, node => node.OpType == "Attention");
        Assert.Equal(["MatMul", "MatMul"], model.Graph.Node.Where(node => node.Name.EndsWith("/MatMul.scores", StringComparison.Ordinal) || node.Name.EndsWith("/MatMul.context", StringComparison.Ordinal)).Select(node => node.OpType));
    }
}
