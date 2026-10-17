// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Import.Onnx;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Import;

/// <summary>
/// The ONNX graphs PyTorch's two exporters make of the words 0.11.0 reads (Fixtures/onnx-vocabulary-fixtures.py made them): the
/// single operators of the newer operator sets — Gelu, LeakyRelu, PRelu, Elu, Selu, Mish, Softmax between layers,
/// InstanceNormalization, GroupNormalization as set 21 has it, RMSNormalization — and the spellings an exporter asked for an older set
/// writes the same words in, each recognised as the layer it spells and held to what PyTorch answered through the graph.
/// </summary>
public class OnnxVocabularyFileTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static readonly JsonElement Answers = JsonDocument.Parse(File.ReadAllText(Path.Join(OnnxFixtures.Folder, "onnx-vocabulary.json"))).RootElement;

    public static TheoryData<string, Type> EveryGraph { get; } = new()
    {
        { "ts-gelu", typeof(Gelu) },
        { "ts-gelu-tanh", typeof(Gelu) },
        { "ts-silu", typeof(Silu) },
        { "ts-mish", typeof(Mish) },
        { "ts-groupnorm", typeof(GroupNorm) },
        { "ts-leaky", typeof(LeakyRelu) },
        { "ts-prelu", typeof(PRelu) },
        { "ts-elu", typeof(Elu) },
        { "ts-selu", typeof(Selu) },
        { "ts-softmax", typeof(Softmax) },
        { "dynamo-gelu", typeof(Gelu) },
        { "dynamo-gelu-tanh", typeof(Gelu) },
        { "dynamo-silu", typeof(Silu) },
        { "dynamo-mish", typeof(Mish) },
        { "dynamo-groupnorm", typeof(GroupNorm) },
        { "dynamo-rmsnorm", typeof(RmsNorm) },
        { "dynamo-gelu-21", typeof(Gelu) },
        { "dynamo-gelu-tanh-21", typeof(Gelu) },
        { "dynamo-groupnorm-21", typeof(GroupNorm) },
        { "dynamo-rmsnorm-23", typeof(RmsNorm) },
    };

    private static SavedNetwork Read(string name)
    {
        using var graph = OnnxFixtures.Open($"onnx-vocabulary-{name}.onnx");

        return new OnnxFile(new MeanSquaredError()).Read(graph);
    }

    [Theory]
    [MemberData(nameof(EveryGraph))]
    public void AGraphOfAWord_AsEitherExporterWritesIt_IsReadAsThatWordBetweenItsLayers_AndAnswersAsPyTorchDoes(string name, Type word)
    {
        var read = Read(name);
        var layers = Assert.IsType<LayerStack>(read.Network).Layers;

        Assert.Equal([typeof(Dense), word, typeof(Dense)], layers.Select(layer => layer.GetType()));
        AssertAnswersAsPyTorch(name, read);
    }

    [Fact]
    public void TheWordsSettings_AreTheOnesTheGraphWrites()
    {
        Assert.True(((Gelu)Layer("ts-gelu-tanh")).Approximate);
        Assert.False(((Gelu)Layer("ts-gelu")).Approximate);
        Assert.True(((Gelu)Layer("dynamo-gelu-tanh-21")).Approximate);
        Assert.Equal(0.1, ((LeakyRelu)Layer("ts-leaky")).NegativeSlope, 6);
        Assert.Equal(6, ((PRelu)Layer("ts-prelu")).Count);
        Assert.Equal(0.7, ((Elu)Layer("ts-elu")).Alpha, 6);
        Assert.Equal(3, ((GroupNorm)Layer("ts-groupnorm")).Groups);
        Assert.Equal(3, ((GroupNorm)Layer("dynamo-groupnorm")).Groups);
        Assert.Equal(1e-5, ((GroupNorm)Layer("dynamo-groupnorm-21")).Epsilon, 9);
        Assert.Equal(1.1920928955078125e-07, ((RmsNorm)Layer("dynamo-rmsnorm")).Epsilon, 12);
    }

    [Fact]
    public void AnImageGraphOfAnInstanceNormalisationAParametricRectifierAndASoftmaxOverTheChannels_AnswersAsPyTorchDoes()
    {
        var read = Read("images");

        Assert.Equal(
            [typeof(Conv2D), typeof(InstanceNorm), typeof(PRelu), typeof(Softmax), typeof(Flatten), typeof(Dense)],
            Assert.IsType<LayerStack>(read.Network).Layers.Select(layer => layer.GetType()));
        AssertAnswersAsPyTorch("images", read);
    }

    [Fact]
    public void ASoftmaxWithinAGraph_OrLastBeforeALossThatAppliesNone_IsALayer()
    {
        var within = new OnnxFile(new BinaryCrossEntropy()).Read(OnnxFixtures.Edited("onnx-titanic-torchscript.onnx", model => model.Graph.Node.Single(node => node.Name == "/1/Relu").OpType = "Softmax"));
        var last = new OnnxFile(new MeanSquaredError()).Read(OnnxFixtures.Edited("onnx-vocabulary-ts-softmax.onnx", model =>
        {
            model.Graph.Node.RemoveAt(model.Graph.Node.Count - 1);
            model.Graph.Output[0].Name = model.Graph.Node[^1].Output[0];
        }));

        Assert.IsType<Softmax>(Assert.IsType<LayerStack>(within.Network).Layers[1]);
        Assert.IsType<Softmax>(Assert.IsType<LayerStack>(last.Network).Layers[^1]);
    }

    [Fact]
    public void ALastSoftmaxBeforeACrossEntropy_IsLeftToTheLoss()
    {
        var read = new OnnxFile(new CrossEntropy()).Read(OnnxFixtures.Edited("onnx-vocabulary-ts-softmax.onnx", model =>
        {
            model.Graph.Node.RemoveAt(model.Graph.Node.Count - 1);
            model.Graph.Output[0].Name = model.Graph.Node[^1].Output[0];
        }));

        Assert.IsType<Dense>(Assert.IsType<LayerStack>(read.Network).Layers[^1]);
    }

    [Fact]
    public void AGroupNormalisationOfOperatorSet18_ScalingEachGroup_IsRefusedNamingSet21()
    {
        var refused = Assert.Throws<FormatException>(() => Read("groupnorm18"));

        Assert.Contains("operator set 21", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ASpellingWhoseNumbersAreNotTheWordsOwn_IsNotTakenForIt()
    {
        // The GELU's chain with another number than the root of two: nothing here is that, so the Div is refused as no operator read.
        var edited = OnnxFixtures.Edited("onnx-vocabulary-dynamo-gelu.onnx", model =>
        {
            var root = model.Graph.Initializer.Single(initializer => initializer.Name == model.Graph.Node.Single(node => node.OpType == "Div").Input[1]);
            root.RawData = Google.Protobuf.ByteString.CopyFrom(BitConverter.GetBytes(1.5f));
        });

        var refused = Assert.Throws<FormatException>(() => new OnnxFile(new MeanSquaredError()).Read(edited));

        Assert.Contains("'Div' is no operator a network here is built of", refused.Message, StringComparison.Ordinal);
    }

    private static Layer Layer(string name) => Assert.IsType<LayerStack>(Read(name).Network).Layers[1];

    private static void AssertAnswersAsPyTorch(string name, SavedNetwork read)
    {
        var expected = Answers.GetProperty(name);
        var here = read.Network.Predict(ExamplesOf(expected.GetProperty("examples")), read.Loss, Engine).Values.ToArray();
        var torch = OnnxFixtures.Values(expected.GetProperty("outputs"));

        Assert.Equal(torch.Length, here.Length);
        Assert.InRange(torch.Zip(here, (was, now) => Math.Abs(was - now)).Max(), 0, 1e-5);
    }

    // The examples PyTorch was handed, laid out as examples are here: the channels of an image last.
    private static Tensor ExamplesOf(JsonElement written)
    {
        var shape = written.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32()).ToArray();
        var values = OnnxFixtures.Values(written.GetProperty("values"));

        if (shape.Length <= 2)
        {
            return Tensor.From(new Shape(shape), values);
        }

        var (batch, channels) = (shape[0], shape[1]);
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
}
