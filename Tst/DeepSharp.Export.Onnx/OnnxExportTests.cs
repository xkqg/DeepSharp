// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Export.Onnx;
using DeepSharp.Import.Onnx;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using Onnx;

namespace DeepSharp.Tests.Export.Onnx;

/// <summary>
/// A network written as an ONNX graph: the oldest operator set that holds its nodes, its numbers as the graph's initializers, its
/// loss's output activation as the last node. What is written is read back by DeepSharp's own reader into the same numbers, and is
/// answered through by ONNX Runtime as the network answers.
/// </summary>
public class OnnxExportTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    // Each network is described in the words that build it, so the graph is the one a network of those words is.
    private static readonly Dictionary<string, Func<LayerStack>> Networks = new()
    {
        ["dense"] = () => new Sequential().Dense(8).Relu().Dense(3).Lower(new Shape(6), new RandomStream(1)),
        ["no bias"] = () => new Sequential().Dense(5, new Beginning { WithoutBias = true }).Tanh().Dense(2, new Beginning { WithoutBias = true }).Lower(new Shape(6), new RandomStream(2)),
        ["activations"] = () => new Sequential().Dense(7).Relu().Dense(7).Tanh().Dense(7).Sigmoid().Dense(7).Gelu().Dense(7).Gelu(approximate: true)
            .Dense(7).Silu().Dense(7).Mish().Dense(7).LeakyRelu(0.2).Dense(7).Elu(0.7).Dense(7).Selu().Dense(7).PRelu(true).Dense(3).Lower(new Shape(6), new RandomStream(3)),
        ["normalisations"] = () => new Sequential().Dense(6).BatchNorm().Relu().Dense(6).LayerNorm().Relu().Dense(6).RmsNorm(1e-6).Dense(2).Lower(new Shape(6), new RandomStream(4)),
        ["normalisations left bare"] = () => new Sequential().Dense(6).LayerNorm(1e-3, new Affine { WithoutScale = true, WithoutShift = true }).Dense(6).RmsNorm(1e-6, scale: false).Dense(6)
            .BatchNorm(0.8, 1e-3, new Affine { WithoutScale = true }).Dense(2).Lower(new Shape(6), new RandomStream(5)),
        ["layer normalisation"] = () => new Sequential().Dense(6).LayerNorm().Dense(2).Lower(new Shape(6), new RandomStream(4)),
        ["dropouts"] = () => new Sequential().Dense(6).Relu().Dropout(0.3).Dense(2).Lower(new Shape(6), new RandomStream(6)),
        ["residual"] = () => new Sequential().Dense(6).Named("first").Relu().Dense(6).Named("second").Add("input", "second").Dense(2).Lower(new Shape(6), new RandomStream(7)),
        ["dense block"] = () => new Sequential()
            .Dense(4).Named("a").Relu().Named("ra").Concatenate("input", "ra").Named("c1")
            .Dense(4).Named("b").Relu().Named("rb").Concatenate("input", "ra", "rb").Dense(2).Lower(new Shape(5), new RandomStream(8)),
        ["two inputs"] = () => new Sequential()
            .Columns(0, 3).Takes("input").Dense(4).Named("left").Columns(3, 3).Takes("input").Dense(4).Named("right").Add("left", "right").Dense(1)
            .Lower(new Shape(6), new RandomStream(9)),
        ["a block"] = () => new Sequential().Block(new Sequential().Dense(6).Named("project").Relu()).Named("encoder").Dense(2).Lower(new Shape(6), new RandomStream(10)),
        ["three added"] = () => new Sequential().Dense(6).Named("a").Dense(6).Named("b").Dense(6).Named("c").Add("a", "b", "c").Dense(1).Lower(new Shape(6), new RandomStream(11)),
    };

    public static TheoryData<string> EveryNetwork => [.. Networks.Keys];

    private static int Width(string name) => name switch { "dense block" => 5, _ => 6 };

    private static ModelProto Parsed(byte[] graph) => ModelProto.Parser.ParseFrom(graph);

    [Theory]
    [MemberData(nameof(EveryNetwork))]
    public void AWrittenGraph_IsReadBackByTheReader_AnsweringAsTheNetworkDoesBitForBit(string name)
    {
        var network = Runtime.Filled(Networks[name]());
        var loss = new MeanSquaredError();
        var graph = new OnnxExport(new Shape(Width(name))).ToBytes(network, loss);
        var rows = Runtime.Rows(5, Width(name));

        var read = new OnnxFile(loss).Read(new MemoryStream(graph));

        Assert.Equal(
            network.Predict(rows, loss, Engine).Values.ToArray(),
            read.Network.Predict(rows, read.Loss, Engine).Values.ToArray());
    }

    [Theory]
    [MemberData(nameof(EveryNetwork))]
    public void AWrittenGraph_IsAnsweredThroughByOnnxRuntime_AsTheNetworkAnswers(string name)
    {
        var network = Runtime.Filled(Networks[name]());
        var loss = new MeanSquaredError();
        var rows = Runtime.Rows(5, Width(name));
        var graph = new OnnxExport(new Shape(Width(name))).ToBytes(network, loss);

        var here = network.Predict(rows, loss, Engine).Values.ToArray();
        var there = Runtime.Answers(graph, rows);

        Assert.Equal(here.Length, there.Length);
        Assert.InRange(here.Zip(there, (one, other) => Math.Abs(one - other)).Max(), 0, 1e-5);
    }

    [Fact]
    public void ADenseNetwork_IsWrittenAsMatMulAndAdd_InTheOldestOperatorSetThatHoldsThem()
    {
        var network = Runtime.Filled(Networks["dense"]());
        var model = Parsed(new OnnxExport(new Shape(6)).ToBytes(network, new MeanSquaredError()));

        Assert.Equal(["MatMul", "Add", "Relu", "MatMul", "Add"], model.Graph.Node.Select(node => node.OpType));
        Assert.Equal(7, Assert.Single(model.OpsetImport).Version);
        Assert.Equal(3, model.IrVersion);
        Assert.Equal(["0.weight", "0.bias", "2.weight", "2.bias"], model.Graph.Initializer.Select(initializer => initializer.Name));
        Assert.Equal("DeepSharp", model.ProducerName);
    }

    [Theory]
    [InlineData("activations", 20, 9)]
    [InlineData("normalisations", 23, 11)]
    [InlineData("layer normalisation", 17, 8)]
    [InlineData("dense", 7, 3)]
    public void TheOperatorSet_IsTheGreatestOneAnyNodeFirstAppearedIn_AndTheIrVersionIsTheOneOnnxPairsWithIt(string name, int opset, long ir)
    {
        var model = Parsed(new OnnxExport(new Shape(6)).ToBytes(Runtime.Filled(Networks[name]()), new MeanSquaredError()));

        Assert.Equal(opset, Assert.Single(model.OpsetImport).Version);
        Assert.Equal(ir, model.IrVersion);
    }

    [Fact]
    public void TheLossThatAppliesAnActivationItself_IsTheLastNodeOfTheGraph()
    {
        var network = Runtime.Filled(new Sequential().Dense(1).Lower(new Shape(6), new RandomStream(1)));
        var rows = Runtime.Rows(4, 6);
        var graph = new OnnxExport(new Shape(6)).ToBytes(network, new BinaryCrossEntropy());
        var model = Parsed(graph);

        Assert.Equal("Sigmoid", model.Graph.Node[^1].OpType);
        Assert.Equal("output", Assert.Single(model.Graph.Output).Name);
        Assert.InRange(
            network.Predict(rows, new BinaryCrossEntropy(), Engine).Values.ToArray().Zip(Runtime.Answers(graph, rows), (one, other) => Math.Abs(one - other)).Max(),
            0,
            1e-6);
    }

    [Fact]
    public void ASoftmaxTheLossAppliesItself_IsWrittenOverTheValuesOfARow()
    {
        var network = Runtime.Filled(new Sequential().Dense(3).Lower(new Shape(6), new RandomStream(1)));
        var rows = Runtime.Rows(4, 6);
        var graph = new OnnxExport(new Shape(6)).ToBytes(network, new CrossEntropy());

        Assert.Equal("Softmax", Parsed(graph).Graph.Node[^1].OpType);
        Assert.InRange(
            network.Predict(rows, new CrossEntropy(), Engine).Values.ToArray().Zip(Runtime.Answers(graph, rows), (one, other) => Math.Abs(one - other)).Max(),
            0,
            1e-6);
    }

    [Fact]
    public void ALowRankPair_IsWrittenAsTheProductsItIs_AndAnsweredThroughAsTheLayerAnswers()
    {
        var network = Runtime.Filled(new Sequential().Dense(5, new Beginning { LowRank = new LowRank(2, 4) }).Relu().Dense(2).Lower(new Shape(6), new RandomStream(1)));
        var loss = new MeanSquaredError();
        var rows = Runtime.Rows(4, 6);
        var graph = new OnnxExport(new Shape(6)).ToBytes(network, loss);

        Assert.Equal(["MatMul", "MatMul", "MatMul", "Mul", "Add", "Add", "Relu", "MatMul", "Add"], Parsed(graph).Graph.Node.Select(node => node.OpType));
        Assert.InRange(
            network.Predict(rows, loss, Engine).Values.ToArray().Zip(Runtime.Answers(graph, rows), (one, other) => Math.Abs(one - other)).Max(),
            0,
            1e-5);
    }

    [Fact]
    public void TheNamesOfTheFeatures_AreWrittenBesideTheGraph()
    {
        var network = Runtime.Filled(Networks["dense"]());
        var model = Parsed(new OnnxExport(new Shape(6)) { Features = ["a", "b", "c", "d", "e", "f"] }.ToBytes(network, new MeanSquaredError()));

        Assert.Equal("a,b,c,d,e,f", Assert.Single(model.MetadataProps, entry => entry.Key == "deepsharp.features").Value);
    }

    [Fact]
    public void AGraphWrittenToAStream_IsTheBytesItIsGiven()
    {
        var network = Runtime.Filled(Networks["dense"]());
        var exporter = new OnnxExport(new Shape(6));
        using var stream = new MemoryStream();

        exporter.Write(network, new MeanSquaredError(), stream);

        Assert.Equal(exporter.ToBytes(network, new MeanSquaredError()), stream.ToArray());
    }

    [Fact]
    public void ANetworkWrittenAsCode_IsRefusedByName_ForItDeclaresNoOrderItsLayersRunIn()
    {
        var code = new TwoLayers();

        var refused = Assert.Throws<NotSupportedException>(() => new OnnxExport(new Shape(2)).ToBytes(code, new MeanSquaredError()));

        Assert.Contains("TwoLayers", refused.Message, StringComparison.Ordinal);
        Assert.Contains("network written as code", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NothingIsWrittenWithoutANetworkALossOrAFile()
    {
        var exporter = new OnnxExport(new Shape(6));

        Assert.Throws<ArgumentNullException>(() => exporter.ToBytes(null!, new MeanSquaredError()));
        Assert.Throws<ArgumentNullException>(() => exporter.ToBytes(Networks["dense"](), null!));
        Assert.Throws<ArgumentNullException>(() => exporter.Write(Networks["dense"](), new MeanSquaredError(), null!));
    }

    // A network written as code: it holds layers and says nothing of the order it runs them in.
    private sealed class TwoLayers : Network
    {
        private readonly Dense _first = new(2, 2, new RandomStream(1).Draw("x", 0, 0));

        public TwoLayers() => AddLayer("first", _first);

        protected override Tensor Compute(Tensor input, Pass pass) => _first.Forward(input, pass);
    }
}
