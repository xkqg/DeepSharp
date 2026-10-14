// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Import.Onnx;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Import;

/// <summary>
/// An ONNX <c>Pad</c> of mode reflect, edge or wrap, read as a padding that fills the places it adds from the example - mirrored
/// back, repeated from the edge, gone round - and the network the default exporter made of PyTorch's reflection, replication and
/// circular paddings answering as PyTorch answered (Fixtures/onnx-filled-fixtures.py made them). Wrap is ONNX's since operator
/// set 19, so a graph that imports an earlier one and says wrap is refused.
/// </summary>
public class OnnxFilledBorderTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static readonly JsonElement Answers = JsonDocument.Parse(File.ReadAllText(Path.Join(OnnxFixtures.Folder, "onnx-filled.json"))).RootElement;

    private static IEnumerable<T> Layers<T>(SavedNetwork saved)
        where T : Layer =>
        saved.Network.HeldLayers().Select(held => held.Layer).OfType<T>();

    [Theory]
    [InlineData("onnx-filled-series")]
    [InlineData("onnx-filled-image")]
    [InlineData("onnx-filled-volume")]
    public void ANetworkOfReflectionReplicationAndCircularPaddings_AnswersAsPyTorchDid(string network)
    {
        var expected = Answers.GetProperty(network);
        var written = expected.GetProperty("examples");
        var examples = Tensor.From(new Shape([.. written.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]), OnnxFixtures.Values(written.GetProperty("values")));

        using var graph = OnnxFixtures.Open($"{network}.onnx");
        var saved = new OnnxFile(new MeanSquaredError()).Read(graph);
        var here = saved.Network.Predict(examples, saved.Loss, Engine).Values.ToArray();
        var torch = OnnxFixtures.Values(expected.GetProperty("outputs"));

        Assert.Equal(torch.Length, here.Length);
        Assert.InRange(torch.Zip(here, (was, now) => Math.Abs(was - now)).Max(), 0, 1e-5);
    }

    [Fact]
    public void EachModeOfAPad_IsTheFillingOfTheSameName_ReflectMirroredEdgeRepeatedAndWrapGoneRound()
    {
        using var series = OnnxFixtures.Open("onnx-filled-series.onnx");
        using var image = OnnxFixtures.Open("onnx-filled-image.onnx");
        using var volume = OnnxFixtures.Open("onnx-filled-volume.onnx");

        Assert.Equal(
            [Filling.Mirror, Filling.Repeat],
            Layers<Pad1D>(new OnnxFile(new MeanSquaredError()).Read(series)).Select(each => each.Filling));
        Assert.Equal(
            [Filling.Mirror, Filling.Repeat, Filling.Wrap],
            Layers<Pad2D>(new OnnxFile(new MeanSquaredError()).Read(image)).Select(each => each.Filling));
        Assert.Equal(
            [Filling.Wrap, Filling.Mirror, Filling.Repeat],
            Layers<Pad3D>(new OnnxFile(new MeanSquaredError()).Read(volume)).Select(each => each.Filling));
    }

    [Fact]
    public void AGraphThatSaysWrapButImportsAnOperatorSetBeforeNineteen_IsRefusedAtTheNode()
    {
        using var folder = new GraphFolder();
        using var graph = OnnxFixtures.EditedBeside(folder, "onnx-filled-volume.onnx", model => model.OpsetImport[0].Version = 18);

        var refused = Assert.Throws<FormatException>(() => new OnnxFile(new MeanSquaredError()).Read(graph));

        Assert.Contains("wrap", refused.Message, StringComparison.Ordinal);
        Assert.Contains("19", refused.Message, StringComparison.Ordinal);
        Assert.Contains("18", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("ai.onnx", "18")]
    [InlineData("com.example", "0")]
    public void TheOperatorSetOfTheDefaultDomain_IsTheOneWrapIsHeldTo_WhateverNameTheGraphGivesIt_AndNoneIsZero(string domain, string says)
    {
        // "ai.onnx" is the default domain's other name; a graph that imports no set of it imports none, which is before nineteen.
        using var folder = new GraphFolder();
        using var graph = OnnxFixtures.EditedBeside(folder, "onnx-filled-volume.onnx", model =>
        {
            model.OpsetImport[0].Domain = domain;
            model.OpsetImport[0].Version = 18;
        });

        var refused = Assert.Throws<FormatException>(() => new OnnxFile(new MeanSquaredError()).Read(graph));

        Assert.Contains("wrap", refused.Message, StringComparison.Ordinal);
        Assert.Contains($"imports operator set {says}", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AModeThatIsNoneOfONNXs_IsRefusedNamingTheOnesThatAre()
    {
        using var folder = new GraphFolder();
        using var graph = OnnxFixtures.EditedBeside(folder, "onnx-filled-series.onnx", model => model.Graph.Node.First(node => node.OpType == "Pad").SetText("mode", "cubic"));

        var refused = Assert.Throws<FormatException>(() => new OnnxFile(new MeanSquaredError()).Read(graph));

        Assert.Contains("cubic", refused.Message, StringComparison.Ordinal);
        Assert.Contains("reflect", refused.Message, StringComparison.Ordinal);
        Assert.Contains("edge", refused.Message, StringComparison.Ordinal);
        Assert.Contains("wrap", refused.Message, StringComparison.Ordinal);
    }
}
