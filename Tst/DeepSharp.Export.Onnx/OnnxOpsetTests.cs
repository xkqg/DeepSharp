// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Export.Onnx;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using Onnx;

namespace DeepSharp.Tests.Export.Onnx;

/// <summary>
/// The operator set a graph asks for is the greatest of its nodes', and a node written for an older set must be valid in it: ONNX moved the
/// axes of a squeeze, an unsqueeze and a reduction from an attribute to an input in later sets, so the graph states them as its set does.
/// </summary>
public class OnnxOpsetTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static readonly Dictionary<string, (Func<LayerStack> Network, Shape Each)> Readings = new()
    {
        ["a reduction beside a node of the twentieth set"] = (() => new Sequential().Dense(4).Gelu().GlobalAvgPool1D().Dense(1).Lower(new Shape(5, 3), new RandomStream(1)), new Shape(5, 3)),
        ["a maximum beside a node of the twentieth set"] = (() => new Sequential().Dense(4).Gelu().GlobalMaxPool1D().Dense(1).Lower(new Shape(5, 3), new RandomStream(2)), new Shape(5, 3)),
        ["a squeeze beside a node of the twentieth set"] = (() => new Sequential().Lstm(4).Gelu().Dense(1).Lower(new Shape(5, 3), new RandomStream(3)), new Shape(5, 3)),
    };

    public static TheoryData<string> EveryReading => [.. Readings.Keys];

    [Theory]
    [MemberData(nameof(EveryReading))]
    public void AGraphOfANewerSetThanSomeNodeWasWrittenFor_IsStillAnsweredThroughByOnnxRuntime(string name)
    {
        var network = Runtime.Filled(Readings[name].Network());
        var loss = new MeanSquaredError();
        var each = Readings[name].Each;
        var rows = Runtime.Batch(4, each);
        var graph = new OnnxExport(each).ToBytes(network, loss);

        var here = network.Predict(rows, loss, Engine).Values.ToArray();
        var there = Runtime.Answers(graph, Runtime.ChannelsFirst(rows));

        Assert.Equal(20, ModelProto.Parser.ParseFrom(graph).OpsetImport.Single().Version);
        Assert.InRange(here.Zip(there, (one, other) => Math.Abs(one - other)).Max(), 0, 1e-5);
    }

    [Fact]
    public void ANodeThatAsksForAnOlderSet_KeepsItsAxesAsAnAttribute_WhereTheGraphIsOfThatSet()
    {
        var each = new Shape(5, 3);
        var graph = ModelProto.Parser.ParseFrom(new OnnxExport(each).ToBytes(Runtime.Filled(new Sequential().Dense(4).GlobalAvgPool1D().Dense(1).Lower(each, new RandomStream(4))), new MeanSquaredError()));
        var reduce = graph.Graph.Node.Single(node => node.OpType == "ReduceMean");

        Assert.True(graph.OpsetImport.Single().Version < 18);
        Assert.Contains(reduce.Attribute, attribute => attribute.Name == "axes");
        Assert.Single(reduce.Input);
    }
}
