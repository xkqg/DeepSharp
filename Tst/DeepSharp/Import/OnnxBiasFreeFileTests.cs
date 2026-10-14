// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Import.Onnx;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using Onnx;

namespace DeepSharp.Tests.Import;

/// <summary>
/// An ONNX graph whose layers keep no bias and no shift, read into a network that keeps none: a Gemm or a Conv or a
/// ConvTranspose with no third input, a MatMul with no Add after it — what the TorchScript exporter writes of a linear layer
/// without a bias — and a LayerNormalization with no third input. Each graph PyTorch exported is held to what PyTorch answered
/// through it (Fixtures/onnx-free-fixtures.py made them), and a graph edited to drop a bias is read as the layer without one.
/// </summary>
public class OnnxBiasFreeFileTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static readonly JsonElement Answers = JsonDocument.Parse(File.ReadAllText(Path.Join(OnnxFixtures.Folder, "onnx-free.json"))).RootElement;

    private static IEnumerable<T> Layers<T>(SavedNetwork saved)
        where T : Layer =>
        saved.Network.HeldLayers().Select(held => held.Layer).OfType<T>();

    private static SavedNetwork Read(string file)
    {
        using var graph = OnnxFixtures.Open(file);

        return new OnnxFile(new MeanSquaredError()).Read(graph);
    }

    // The examples PyTorch was handed, laid out as examples are here: the channels of an image, a series or a volume last.
    private static Tensor ExamplesOf(JsonElement written)
    {
        var shape = written.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32()).ToArray();
        var values = OnnxFixtures.Values(written.GetProperty("values"));

        if (shape.Length <= 2)
        {
            return Tensor.From(new Shape(shape), values);
        }

        var batch = shape[0];
        var channels = shape[1];
        var places = values.Length / (batch * channels);
        var turned = new float[values.Length];

        for (var example = 0; example < batch; example++)
        {
            for (var channel = 0; channel < channels; channel++)
            {
                for (var place = 0; place < places; place++)
                {
                    turned[(((example * places) + place) * channels) + channel] = values[(((example * channels) + channel) * places) + place];
                }
            }
        }

        return Tensor.From(new Shape([batch, .. shape[2..], channels]), turned);
    }

    [Theory]
    [InlineData("onnx-free-rows")]
    [InlineData("onnx-free-images")]
    [InlineData("onnx-free-series")]
    [InlineData("onnx-free-volume")]
    public void AGraphOfLayersThatKeepNoBias_AnswersAsPyTorchDid_WhicheverExporterWroteIt(string network)
    {
        var expected = Answers.GetProperty(network);
        var examples = ExamplesOf(expected.GetProperty("examples"));
        var torch = OnnxFixtures.Values(expected.GetProperty("outputs"));

        foreach (var saved in new[] { $"{network}.onnx", $"{network}-torchscript.onnx" }.Select(Read))
        {
            var here = saved.Network.Predict(examples, saved.Loss, Engine).Values.ToArray();

            Assert.Equal(torch.Length, here.Length);
            Assert.InRange(torch.Zip(here, (was, now) => Math.Abs(was - now)).Max(), 0, 1e-5);
        }
    }

    [Fact]
    public void TheDefaultExporterWritesAGemmWithoutAThirdInput_TheTorchScriptOneAMatMulWithoutAnAdd_AndBothAreALayerWithoutABias()
    {
        foreach (var saved in new[] { "onnx-free-rows.onnx", "onnx-free-rows-torchscript.onnx" }.Select(Read))
        {
            Assert.Equal([false, false, true], Layers<Dense>(saved).Select(dense => dense.Bias is not null));
            Assert.DoesNotContain(saved.Network.Slots().Select(named => named.Path), path => path == "0.bias");
        }
    }

    [Fact]
    public void ALayerNormalisationWithoutAThirdInput_KeepsNoShift_WhereOneWhoseShiftIsAConstantOfNoughtsKeepsOneOfNoughts()
    {
        var withoutShift = Assert.Single(Layers<LayerNorm>(Read("onnx-free-rows.onnx")));
        var ofNoughts = Assert.Single(Layers<LayerNorm>(Read("onnx-free-rows-torchscript.onnx")));

        Assert.NotNull(withoutShift.Weight);
        Assert.Null(withoutShift.Bias);
        Assert.NotNull(ofNoughts.Bias);
        Assert.All(ofNoughts.Bias!.Value.Values.ToArray(), value => Assert.Equal(0f, value));
    }

    [Theory]
    [InlineData("onnx-free-images.onnx")]
    [InlineData("onnx-free-images-torchscript.onnx")]
    [InlineData("onnx-free-series.onnx")]
    [InlineData("onnx-free-series-torchscript.onnx")]
    [InlineData("onnx-free-volume.onnx")]
    [InlineData("onnx-free-volume-torchscript.onnx")]
    public void AConvolutionAndATransposedConvolutionWithoutAThirdInput_AreConvolutionsWithoutABias(string file)
    {
        var saved = Read(file);

        Assert.Equal([false, false], Layers<Convolution>(saved).Select(convolution => convolution.Bias is not null));
        Assert.True(Assert.Single(Layers<Dense>(saved)).Bias is not null);
    }

    [Fact]
    public void AGemmWhoseBiasIsTakenOutOrLeftEmpty_IsADenseLayerWithoutABias()
    {
        foreach (var edit in new Action<NodeProto>[] { node => node.Input.RemoveAt(2), node => node.Input[2] = string.Empty })
        {
            using var graph = OnnxFixtures.Edited("onnx-titanic-torchscript.onnx", model => edit(model.Node("/0/Gemm")));
            var saved = new OnnxFile(new BinaryCrossEntropy()).Read(graph);

            Assert.Equal([false, true], Layers<Dense>(saved).Select(dense => dense.Bias is not null));
        }
    }

    [Fact]
    public void AMatMulWhoseAddIsTakenOut_IsADenseLayerWithoutABias_AndTheOtherLayersAreAsTheyWere()
    {
        using var graph = OnnxFixtures.Edited("keras-titanic-export.onnx", model =>
        {
            model.Node("/Cast_4").Input[0] = "/Cast_3_output_0";
            model.Graph.Node.Remove(model.Node("/Add"));
        });
        var saved = new OnnxFile(new BinaryCrossEntropy()).Read(graph);
        var dense = Layers<Dense>(saved).ToArray();

        Assert.Equal(2, dense.Length);
        Assert.Null(dense[0].Bias);
        Assert.NotNull(dense[1].Bias);
    }

    [Fact]
    public void AConvolutionAndATransposedConvolutionWhoseBiasIsTakenOut_AreConvolutionsWithoutABias()
    {
        using var image = OnnxFixtures.Edited("onnx-convolution-torchscript.onnx", model => model.Node("/0/Conv").Input.RemoveAt(2));
        using var transposed = OnnxFixtures.Edited("onnx-transposed.onnx", model => model.Node("node_convolution").Input.RemoveAt(2));

        Assert.Null(Layers<Convolution>(new OnnxFile(new MeanSquaredError()).Read(image)).First().Bias);
        Assert.Null(Layers<Conv2DTranspose>(new OnnxFile(new MeanSquaredError()).Read(transposed)).First().Bias);
    }

    [Fact]
    public void ALayerNormalisationWhoseShiftIsTakenOut_KeepsNoShift()
    {
        using var graph = OnnxFixtures.Edited("onnx-titanic-torchscript.onnx", model =>
        {
            var node = model.Node("/1/Relu");
            node.OpType = "LayerNormalization";
            node.Input.Add("0.bias");
        });
        var saved = new OnnxFile(new BinaryCrossEntropy()).Read(graph);
        var layer = Assert.Single(Layers<LayerNorm>(saved));

        Assert.NotNull(layer.Weight);
        Assert.Null(layer.Bias);
    }
}
