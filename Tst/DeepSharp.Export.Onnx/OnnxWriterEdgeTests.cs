// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Export.Onnx;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using Onnx;

namespace DeepSharp.Tests.Export.Onnx;

/// <summary>
/// What the writer says no to, by name, and the edges of what it writes: attention given a mask as its second take or as its fourth, and
/// the activation a loss applies written as the graph's last node, a softmax of the old operator sets for a row and of the thirteenth for
/// anything else.
/// </summary>
public class OnnxWriterEdgeTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static readonly Tensor Ids = Tensor.From(new Shape(4, 5), [3, 4, 0, 0, 0, 1, 2, 5, 6, 3, 2, 0, 0, 0, 0, 5, 5, 5, 5, 5]);

    public static TheoryData<string> MaskedAttentions => ["its mask as the second take", "queries, keys, values and the mask"];

    private static Sequential Masked(string name) => name == "its mask as the second take"
        ? new Sequential().TokenEmbedding(7, 8, keepSteps: true).Named("steps").Unpadded().Named("mask").Takes("input").Attention(2).Takes("steps", "mask").GlobalAvgPool1D().Dense(1)
        : new Sequential().TokenEmbedding(7, 8, keepSteps: true).Named("steps").Unpadded().Named("mask").Takes("input")
            .Dense(8).Named("q").Takes("steps").Dense(8).Named("k").Takes("steps").Dense(8).Named("v").Takes("steps")
            .Attention(2).Takes("q", "k", "v", "mask").GlobalAvgPool1D().Dense(1);

    [Theory]
    [MemberData(nameof(MaskedAttentions))]
    public void AttentionGivenAMask_IsAnsweredThroughByOnnxRuntime_LookingAtNoPadding(string name)
    {
        var network = Runtime.Filled(Masked(name).Lower(new Shape(5), new RandomStream(8)));
        var loss = new MeanSquaredError();
        var graph = new OnnxExport(new Shape(5)).ToBytes(network, loss);

        var here = network.Predict(Ids, loss, Engine).Values.ToArray();
        var there = Runtime.Answers(graph, Ids);

        Assert.InRange(here.Zip(there, (one, other) => Math.Abs(one - other)).Max(), 0, 1e-5);
    }

    [Fact]
    public void ARowOfShares_IsWrittenWithTheSoftmaxOfTheOldSets_AndAnswersAsTheNetworkDoes()
    {
        var network = Runtime.Filled(new Sequential().Dense(6).Relu().Dense(3).Lower(new Shape(4), new RandomStream(2)));
        var loss = new CrossEntropy();
        var rows = Runtime.Rows(5, 4);
        var graph = new OnnxExport(new Shape(4)).ToBytes(network, loss);
        var model = ModelProto.Parser.ParseFrom(graph);

        Assert.Equal("Softmax", model.Graph.Node[^1].OpType);
        Assert.Equal(7, model.OpsetImport.Single().Version);
        Assert.InRange(network.Predict(rows, loss, Engine).Values.ToArray().Zip(Runtime.Answers(graph, rows), (one, other) => Math.Abs(one - other)).Max(), 0, 1e-5);
    }

    [Fact]
    public void SharesOfSomethingThatIsNotARow_AreWrittenWithTheSoftmaxOfTheThirteenthSet()
    {
        var network = Runtime.Filled(new Sequential().Conv1D(3, new Window1D(2)).Lower(new Shape(5, 2), new RandomStream(2)));
        var model = ModelProto.Parser.ParseFrom(new OnnxExport(new Shape(5, 2)).ToBytes(network, new CrossEntropy()));

        Assert.Equal("Softmax", model.Graph.Node[^1].OpType);
        Assert.Equal(13, model.OpsetImport.Single().Version);
    }

    [Fact]
    public void ASoftmaxLayerOverARow_IsWrittenWithTheSoftmaxOfTheOldSets_AndAnswersAsTheNetworkDoes()
    {
        var network = Runtime.Filled(new Sequential().Dense(3).Softmax().Lower(new Shape(4), new RandomStream(2)));
        var loss = new MeanSquaredError();
        var rows = Runtime.Rows(5, 4);
        var graph = new OnnxExport(new Shape(4)).ToBytes(network, loss);

        Assert.Equal(1, ModelProto.Parser.ParseFrom(graph).Graph.Node.Count(node => node.OpType == "Softmax"));
        Assert.InRange(network.Predict(rows, loss, Engine).Values.ToArray().Zip(Runtime.Answers(graph, rows), (one, other) => Math.Abs(one - other)).Max(), 0, 1e-5);
    }

    [Fact]
    public void ASingleLayerRunOverEveryStep_NotAStackOfLayers_IsWrittenAndAnswersAsTheNetworkDoes()
    {
        var every = new EveryStep(new Dense(3, 2, new Beginning(new RandomStream(4).Draw("a", 0, 0))));
        var network = Runtime.Filled(new LayerStack([new Step(every, "every"), new Step(new GlobalAvgPool1D(), "pooled")]));
        var loss = new MeanSquaredError();
        var rows = Runtime.Batch(3, new Shape(4, 3));
        var graph = new OnnxExport(new Shape(4, 3)).ToBytes(network, loss);

        Assert.InRange(network.Predict(rows, loss, Engine).Values.ToArray().Zip(Runtime.Answers(graph, Runtime.ChannelsFirst(rows)), (one, other) => Math.Abs(one - other)).Max(), 0, 1e-5);
    }

    [Fact]
    public void AGroupNormalisationOfAFlattenedImage_IsRefusedByName()
    {
        var network = Runtime.Filled(new Sequential().Conv2D(2, new Window(2, 2)).Flatten().GroupNorm(2).Dense(1).Lower(new Shape(4, 4, 2), new RandomStream(3)));

        var refused = Assert.Throws<NotSupportedException>(() => new OnnxExport(new Shape(4, 4, 2)).ToBytes(network, new MeanSquaredError()));

        Assert.Contains("group normalisation of a row a flatten made of an image", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AReshapeIntoMoreThanOneRow_IsRefusedByName()
    {
        var network = Runtime.Filled(new Sequential().Dense(8).Reshape(new Shape(2, 4)).Lower(new Shape(5), new RandomStream(3)));

        var refused = Assert.Throws<NotSupportedException>(() => new OnnxExport(new Shape(5)).ToBytes(network, new MeanSquaredError()));

        Assert.Contains("only a reshape into one row is written", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ALayerRunOverEveryStep_ThatMakesMoreThanARowOfEachStep_IsRefusedByName()
    {
        var network = Runtime.Filled(new Sequential().EveryStep(new Sequential().Reshape(new Shape(1, 3))).Lower(new Shape(4, 3), new RandomStream(3)));

        var refused = Assert.Throws<NotSupportedException>(() => new OnnxExport(new Shape(4, 3)).ToBytes(network, new MeanSquaredError()));

        Assert.Contains("only a layer that makes a row of each step is written", refused.Message, StringComparison.Ordinal);
    }
}
