// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Import.PyTorch;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using Onnxify.Safetensors;
using static DeepSharp.Tests.Import.HandWrittenPickle;
using static DeepSharp.Tests.Import.PyTorchNetworks;

namespace DeepSharp.Tests.Import;

/// <summary>
/// The state of networks made of the words 0.11.0 adds, as PyTorch saved it as safetensors and as torch.save writes it
/// (Fixtures/pytorch-vocabulary.py made them): the activations, the parametric rectifier's shares, the normalisations over groups
/// of channels, over one example's channels and by the root mean square, and an embedding's vectors each go into the slot the same
/// network written here keeps, and the network answers as PyTorch does — the embedding handed the category's ones and noughts
/// where PyTorch was handed its place.
/// </summary>
public class PyTorchVocabularyTests
{
    private static readonly JsonElement Answers = JsonDocument.Parse(File.ReadAllText(PyTorchFixture.Path("pytorch-vocabulary.json"))).RootElement;

    public static TheoryData<string, string> EveryFile
    {
        get
        {
            var files = new TheoryData<string, string>();

            foreach (var name in new[] { "rows", "images", "embedding" })
            {
                files.Add(name, "safetensors");
                files.Add(name, "pt");
            }

            return files;
        }
    }

    // Each network of pytorch-vocabulary.py written here, and the shape of one example it takes.
    private static SpatialCase Case(string name) => name switch
    {
        "rows" => new(
            () => new Sequential().Dense(6).Gelu().Dense(6).Silu().Mish().LeakyRelu(0.1).PRelu(eachChannel: true).Elu(0.7).Selu().GroupNorm(3).RmsNorm().Dense(4).Softmax()
                .Lower(new Shape(5), new RandomStream(7)),
            new Shape(5)),
        "images" => new(
            () => new Sequential().Conv2D(4, new Window(3, 3)).GroupNorm(2).PRelu(eachChannel: true).InstanceNorm().Flatten().Dense(2).Lower(new Shape(6, 6, 2), new RandomStream(7)),
            new Shape(6, 6, 2)),
        _ => new(
            () => new Sequential().Embedding([0, 1, 2, 3, 4], 3).Flatten().Dense(2).Lower(new Shape(5), new RandomStream(7)),
            new Shape(5)),
    };

    private static PyTorchFile Reader(string kind, Network network, Shape? example) =>
        kind == "pt"
            ? new TorchSaveFile(network, new MeanSquaredError()) { Example = example }
            : new SafetensorsFile(network, new MeanSquaredError()) { Example = example };

    private static Tensor Examples(string name)
    {
        var examples = Answers.GetProperty(name).GetProperty("examples");

        return Tensor.From(new Shape([.. examples.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]), examples.GetProperty("values").Floats());
    }

    [Theory]
    [MemberData(nameof(EveryFile))]
    public void AStateOfTheNewWords_GoesIntoTheSameNetworkWrittenHere_AndAnswersAsPyTorchDoes(string name, string kind)
    {
        var made = Case(name);
        var saved = Reader(kind, made.Network(), made.Example).Read(PyTorchFixture.Open($"vocabulary-{name}.{kind}"));

        var outputs = saved.Network.Predict(Examples(name), saved.Loss, Backend).Values.ToArray();
        var expected = Answers.GetProperty(name).GetProperty("outputs").Floats();

        Assert.Equal(expected.Length, outputs.Length);
        Assert.InRange(Farthest(outputs, expected), 0, 1e-5);
    }

    [Fact]
    public void AnEmbeddingsVectors_AreTheRowsOfPyTorchsWeight_InTheOrderOfItsPlaces()
    {
        var made = Case("embedding");
        var network = made.Network();
        new SafetensorsFile(network, new MeanSquaredError()) { Example = made.Example }.Read(PyTorchFixture.Open("vocabulary-embedding.safetensors"));
        var embedding = Assert.IsType<Embedding>(Assert.IsType<LayerStack>(network).Layers[0]);

        // The second example is the category in place three: its ones and noughts pick the fourth row of the weight.
        var picked = embedding.Forward(Tensor.From(new Shape(1, 5), [0f, 0f, 0f, 1f, 0f]), Pass.Evaluation(Backend)).Values.ToArray();

        Assert.Equal(embedding.Weight.Value.Values[9..12].ToArray(), picked);
    }

    [Fact]
    public void ANetworkWrittenAsCode_TakesItsSharesAndItsVectorsAsPyTorchKeepsThem()
    {
        var network = new Coded(withImages: false);

        new SafetensorsFile(network, new MeanSquaredError()).Read(Written(
        [
            new("embedding.weight", DataType.F32, [3, 2], Floats(1, 2, 3, 4, 5, 6)),
            new("each.weight", DataType.F32, [2], Floats(0.1f, 0.2f)),
            new("one.weight", DataType.F32, [1], Floats(0.3f)),
        ]));

        Assert.Equal([1f, 2f, 3f, 4f, 5f, 6f], network.Embedded.Weight.Value.Values.ToArray());
        Assert.Equal([0.1f, 0.2f], network.Each.Weight.Value.Values.ToArray());
        Assert.Equal([0.3f], network.One.Weight.Value.Values.ToArray());
    }

    [Fact]
    public void ANetworkWrittenAsCodeThatMayFlattenImages_IsRefusedASharePerFeature_ButTakesOneShareForAll()
    {
        var refused = Assert.Throws<SlotLoadException>(() => new SafetensorsFile(new Coded(withImages: true), new MeanSquaredError()).Read(Written(
        [
            new("embedding.weight", DataType.F32, [3, 2], Floats(1, 2, 3, 4, 5, 6)),
            new("each.weight", DataType.F32, [2], Floats(0.1f, 0.2f)),
            new("one.weight", DataType.F32, [1], Floats(0.3f)),
        ])));

        Assert.Contains(refused.Faults, fault => fault.Slot == "each.weight" && fault.Message.Contains("PyTorch orders channel by channel", StringComparison.Ordinal));
        Assert.DoesNotContain(refused.Faults, fault => fault.Slot is "one.weight" or "embedding.weight");
    }

    // A network written as code holding an embedding and two parametric rectifiers; one that may flatten images holds a pooling too.
    // It is never run.
    private sealed class Coded : Network
    {
        public Coded(bool withImages)
        {
            Embedded = AddLayer("embedding", new Embedding(3, [0, 1, 2], 2, new RandomStream(7).Draw("initialise", 0, 0)));
            Each = AddLayer("each", new PRelu(2));
            One = AddLayer("one", new PRelu(1));

            if (withImages)
            {
                AddLayer("pool", new MaxPool2D(new Window(2, 2)));
            }
        }

        public Embedding Embedded { get; }

        public PRelu Each { get; }

        public PRelu One { get; }

        protected override Tensor Compute(Tensor input, Pass pass) => input;
    }
}
