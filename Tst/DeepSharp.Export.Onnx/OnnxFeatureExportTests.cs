// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Export.Onnx;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Export.Onnx;

/// <summary>
/// The layers that turn the columns of a prepared row into numbers inside the network — a category into a vector, the ids of a text into
/// one, a number into its waves — and the poolings to a number of places, written as the products and operators they are worked out in,
/// and answered through by ONNX Runtime as the network answers. Every layer the library saves is written or refused by name.
/// </summary>
public class OnnxFeatureExportTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static void AnswersAs(LayerStack network, Tensor given, Tensor handed, Loss loss)
    {
        var graph = new OnnxExport(new Shape(given.Shape.Axes[1..].ToArray())).ToBytes(network, loss);
        var here = network.Predict(given, loss, Engine).Values.ToArray();
        var there = Runtime.Answers(graph, handed);

        Assert.Equal(here.Length, there.Length);
        Assert.InRange(here.Zip(there, (one, other) => Math.Abs(one - other)).Max(), 0, 1e-5);
    }

    [Fact]
    public void ACategoryAsAVector_IsWrittenAsTheProductsThatTakeItsBlockOutAndPutTheVectorWhereItStood()
    {
        var network = Runtime.Filled(new Sequential().Embedding([1, 2, 3], 2).Relu().Dense(2).Lower(new Shape(6), new RandomStream(1)));
        var rows = Runtime.Rows(5, 6);

        AnswersAs(network, rows, rows, new MeanSquaredError());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void TheIdsOfATextAsOneVector_AreWrittenAsTheShareEachIdIsOfItsRowTimesTheVectors_PaddingLeftOut(bool lite)
    {
        var network = Runtime.Filled(new Sequential().TokenEmbedding(7, 3).Dense(lite ? 1 : 2).Lower(new Shape(5), new RandomStream(1)));
        float[] ids = [1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 3, 4, 5, 6, 1, 2, 2, 2, 2, 2];
        var rows = Tensor.From(new Shape(4, 5), ids);

        AnswersAs(network, rows, rows, new MeanSquaredError());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ANumberAsItsWaves_IsWrittenAsTheCosinesAndSinesThroughTheLinearLayerEachFeatureHasOrAllShare(bool lite)
    {
        var periodicity = new Periodicity(3, 0.3, 2) { Lite = lite };
        var network = Runtime.Filled(new Sequential().Periodic(periodicity).Relu().Dense(2).Lower(new Shape(4), new RandomStream(1)));
        var rows = Runtime.Rows(5, 4);

        AnswersAs(network, rows, rows, new MeanSquaredError());
    }

    [Theory]
    [InlineData("max")]
    [InlineData("average")]
    public void APoolingToANumberOfPlaces_WhereTheAxisDividesEvenly_IsTheOrdinaryPoolingItIs(string kind)
    {
        var words = new Sequential().Conv2D(2, new Window(1, 1));
        var network = Runtime.Filled((kind == "max" ? words.AdaptiveMaxPool2D(3, 2) : words.AdaptiveAvgPool2D(3, 2)).Flatten().Dense(1).Lower(new Shape(6, 4, 2), new RandomStream(1)));
        var rows = Runtime.Batch(3, new Shape(6, 4, 2));

        AnswersAs(network, rows, Runtime.ChannelsFirst(rows), new MeanSquaredError());
    }

    [Fact]
    public void APoolingToANumberOfPlacesTheAxisDoesNotDivide_IsRefusedByName_ForItsBinsAreOfUnequalSizes()
    {
        var network = Runtime.Filled(new Sequential().Conv2D(2, new Window(1, 1)).AdaptiveAvgPool2D(4, 2).Flatten().Dense(1).Lower(new Shape(6, 4, 2), new RandomStream(1)));

        var refused = Assert.Throws<NotSupportedException>(() => new OnnxExport(new Shape(6, 4, 2)).ToBytes(network, new MeanSquaredError()));

        Assert.Contains("unequal sizes", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryLayerTheLibrarySaves_IsWrittenAsONNXNodes()
    {
        var layers = typeof(Layer).Assembly.GetTypes()
            .Where(type => type is { IsPublic: true, IsAbstract: false } && typeof(Layer).IsAssignableFrom(type))
            .Where(type => type.GetInterfaces().Any(each => each.IsGenericType && each.GetGenericTypeDefinition() == typeof(ISaved<>)));

        Assert.Equal([], layers.Where(type => !ExportKinds.Writes(type)).Select(type => type.Name));
        Assert.True(layers.Count() >= 60);
    }

    [Fact]
    public void ALayerWrittenByNobodyElse_IsRefusedByName()
    {
        var network = new LayerStack(new Unwritten());

        var refused = Assert.Throws<NotSupportedException>(() => new OnnxExport(new Shape(2)).ToBytes(network, new MeanSquaredError()));

        Assert.Contains("Unwritten", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ALayerThatReadsColumnsByTheirPlaces_IsRefusedWhereARowWasAnImageFlattened()
    {
        var network = Runtime.Filled(new Sequential().Conv2D(2, new Window(2, 2)).Flatten().Embedding([0, 1, 2], 2).Lower(new Shape(4, 4, 2), new RandomStream(1)));

        var refused = Assert.Throws<NotSupportedException>(() => new OnnxExport(new Shape(4, 4, 2)).ToBytes(network, new MeanSquaredError()));

        Assert.Contains("by their places", refused.Message, StringComparison.Ordinal);
    }

    // A layer nobody wrote a kind for.
    private sealed class Unwritten : Layer
    {
        protected override Tensor Compute(Tensor input, Pass pass) => input;
    }
}
