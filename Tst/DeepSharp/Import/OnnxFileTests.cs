// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Import.Keras;
using DeepSharp.Import.Onnx;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;
using DeepSharp.Tests.Learners;
using Google.Protobuf;
using Onnx;

namespace DeepSharp.Tests.Import;

/// <summary>
/// A network PyTorch trained and exported to ONNX — by its default exporter, which keeps the larger numbers in a file
/// beside the graph, or by the TorchScript one before it — read into a network here through the importer seam: each node
/// lowered onto the layer it is, its numbers turned as the node declares them, and answering as PyTorch answered.
/// </summary>
public class OnnxFileTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static readonly JsonElement Titanic = OnnxFixtures.Answers.GetProperty("titanic");

    private static readonly JsonElement Convolution = OnnxFixtures.Answers.GetProperty("convolution");

    private static readonly JsonElement Kinds = OnnxFixtures.Answers.GetProperty("kinds");

    [Theory]
    [InlineData("onnx-titanic.onnx")]
    [InlineData("onnx-titanic-torchscript.onnx")]
    public void TheTitanicNetworkPyTorchTrained_AnswersEveryTestPassengerAsPyTorchDid_WithinEightRoundings(string file)
    {
        var test = WikiTitanic.In(WikiTitanic.DataFolder).Run().Batch(Part.Test, Needs.OneScale);
        var torch = Titanic.GetProperty("test");

        // The rows PyTorch answered are the rows the pipeline hands over, number for number.
        Assert.Equal(135, test.RowCount);
        Assert.Equal(OnnxFixtures.Values(torch.GetProperty("features")), test.Features.SelectMany(row => row.Select(value => (float)value)));

        IImporter importer = new OnnxFile(new BinaryCrossEntropy());
        using var graph = OnnxFixtures.Open(file);
        var saved = importer.Read(graph);

        Assert.InRange(RoundingsApart(OnnxFixtures.Values(torch.GetProperty("chances")), saved.Network.Predict(Batch(test.Features), saved.Loss, Engine)), 0, 8);
    }

    [Theory]
    [InlineData("onnx-titanic.onnx")]
    [InlineData("onnx-titanic-torchscript.onnx")]
    public void TheReadmesPassenger_AndTheWomanOf38_AreGivenTheChancesPyTorchGaveThem(string file)
    {
        var prepared = WikiTitanic.In(WikiTitanic.DataFolder).Run();
        var passengers = new InMemoryRowSource(
            ["pclass", "sex", "age", "sibsp", "parch", "fare"], [["3", "male", "22", "1", "0", "7.25"], ["1", "female", "38", "1", "0", "71.2833"]]);
        var served = prepared.Served(passengers, Needs.OneScale);
        var torch = Titanic.GetProperty("served");

        Assert.Equal(OnnxFixtures.Values(torch.GetProperty("features")), served.Features.SelectMany(row => row.Select(value => (float)value)));

        using var graph = OnnxFixtures.Open(file);
        var saved = new OnnxFile(new BinaryCrossEntropy()).Read(graph);

        Assert.InRange(RoundingsApart(OnnxFixtures.Values(torch.GetProperty("chances")), saved.Network.Predict(Batch(served.Features), saved.Loss, Engine)), 0, 8);
    }

    [Fact]
    public void AKerasModelsOwnExport_AnswersEveryTestPassengerAndTheReadmesPassengerAsKerasDid_WithinEightRoundings()
    {
        var keras = KerasFixtures.Answers.GetProperty("titanic");
        var test = WikiTitanic.In(WikiTitanic.DataFolder).Run().Batch(Part.Test, Needs.OneScale);

        Assert.Equal(keras.GetProperty("test").GetProperty("rows").EnumerateArray().SelectMany(OnnxFixtures.Values), test.Features.SelectMany(row => row.Select(value => (float)value)));

        using var graph = OnnxFixtures.Open("keras-titanic-export.onnx");
        var saved = new OnnxFile(new BinaryCrossEntropy()).Read(graph);
        var served = Tensor.From(new Shape(2, 14), [.. keras.GetProperty("served").GetProperty("rows").EnumerateArray().SelectMany(OnnxFixtures.Values)]);

        Assert.InRange(RoundingsApart(OnnxFixtures.Values(keras.GetProperty("test").GetProperty("chances")), saved.Network.Predict(Batch(test.Features), saved.Loss, Engine)), 0, 8);
        Assert.InRange(RoundingsApart(OnnxFixtures.Values(keras.GetProperty("served").GetProperty("chances")), saved.Network.Predict(served, saved.Loss, Engine)), 0, 8);
    }

    [Fact]
    public void AKerasModelsOwnExport_PutsTheNumbersKerasSavedIntoTheSlotsItsArchiveDoes_ItsCastsNothing_AndEachMatMulWithItsAddADenseLayer()
    {
        using var graph = OnnxFixtures.Open("keras-titanic-export.onnx");
        var exported = new OnnxFile(new BinaryCrossEntropy()).Read(graph).Network;
        var archive = new KerasFile().Read(KerasFixtures.Open("keras-titanic.keras")).Network;

        Assert.Collection(
            Assert.IsType<LayerStack>(exported).Layers,
            layer => Assert.Equal(new Shape(14, 16), Assert.IsType<Dense>(layer).Weight.Value.Shape),
            layer => Assert.IsType<Relu>(layer),
            layer => Assert.Equal(new Shape(16, 1), Assert.IsType<Dense>(layer).Weight.Value.Shape));
        Assert.Equal(archive.Slots().Select(named => named.Path), exported.Slots().Select(named => named.Path));
        Assert.Equal(Bits(archive), Bits(exported));
    }

    [Fact]
    public void AnAddThatWritesItsBiasFirst_IsTheBiasOfTheMatMulBeforeIt_AsOneThatWritesItSecondIs()
    {
        using var graph = OnnxFixtures.Open("keras-titanic-export.onnx");
        var expected = Bits(new OnnxFile(new BinaryCrossEntropy()).Read(graph).Network);

        var read = new OnnxFile(new BinaryCrossEntropy()).Read(OnnxFixtures.Edited("keras-titanic-export.onnx", model =>
        {
            var add = model.Node("/Add");
            var rows = add.Input[0];
            add.Input[0] = add.Input[1];
            add.Input[1] = rows;
        }));

        Assert.Equal(expected, Bits(read.Network));
    }

    [Fact]
    public void TheGraphTfToOnnxMakesOfTheTitanicSavedModel_AnswersAsKerasDid_HoldingTheNumbersItsArchiveHolds()
    {
        var keras = KerasFixtures.Answers.GetProperty("titanic");
        using var graph = OnnxFixtures.Open("tf2onnx-titanic.onnx");
        var saved = new OnnxFile(new BinaryCrossEntropy()).Read(graph);
        var archive = new KerasFile().Read(KerasFixtures.Open("keras-titanic.keras")).Network;

        foreach (var part in new[] { "test", "served" })
        {
            var rows = keras.GetProperty(part).GetProperty("rows");
            var features = Tensor.From(new Shape(rows.GetArrayLength(), 14), [.. rows.EnumerateArray().SelectMany(OnnxFixtures.Values)]);

            Assert.InRange(RoundingsApart(OnnxFixtures.Values(keras.GetProperty(part).GetProperty("chances")), saved.Network.Predict(features, saved.Loss, Engine)), 0, 8);
        }

        Assert.Equal(Bits(archive), Bits(saved.Network));
    }

    [Fact]
    public void TheGraphTfToOnnxMakesOfAConvolutionSavedModel_GivesEachImageTheOutputTensorFlowGaveIt_WithinATenThousandth()
    {
        var tensorFlow = OnnxFixtures.TensorFlow.GetProperty("convolution");
        using var graph = OnnxFixtures.Open("tf2onnx-convolution.onnx");
        var saved = new OnnxFile(new MeanSquaredError()).Read(graph);

        // TensorFlow lays images out with their channels last, as they are here: the graph's two transposes move them after
        // the batch for its convolutions and last again for its flatten, and are nothing here.
        Assert.Collection(
            Assert.IsType<LayerStack>(saved.Network).Layers,
            layer => AssertConvolution(new Window(3, 2) { PaddingMode = PaddingMode.Same }, 2, 4, layer),
            layer => Assert.IsType<Relu>(layer),
            layer => AssertConvolution(new Window(2, 2) { Stride = 2 }, 4, 3, layer),
            layer => Assert.IsType<Flatten>(layer),
            layer => Assert.Equal(new Shape(36, 5), Assert.IsType<Dense>(layer).Weight.Value.Shape),
            layer => Assert.IsType<Tanh>(layer),
            layer => Assert.Equal(new Shape(5, 1), Assert.IsType<Dense>(layer).Weight.Value.Shape));

        var outputs = saved.Network.Predict(OnnxFixtures.Images(tensorFlow.GetProperty("images")), saved.Loss, Engine);

        Assert.InRange(Farthest(OnnxFixtures.Values(tensorFlow.GetProperty("outputs")), outputs), 0, 1e-5);
    }

    [Fact]
    public void ACastIntoSingleNumbersBeforeTheFirstTranspose_LeavesTheLayoutOfTheImagesTheGraphTakesForThatTransposeToDeclare()
    {
        var tensorFlow = OnnxFixtures.TensorFlow.GetProperty("convolution");
        var saved = new OnnxFile(new MeanSquaredError()).Read(OnnxFixtures.Edited("tf2onnx-convolution.onnx", model =>
        {
            var cast = OnnxFixtures.NodeOf("Cast", "cast", ["args_0"], "cast");
            cast.SetWhole("to", 1);
            model.Graph.Node[0].Input[0] = "cast";
            model.Graph.Node.Insert(0, cast);
        }));

        var outputs = saved.Network.Predict(OnnxFixtures.Images(tensorFlow.GetProperty("images")), saved.Loss, Engine);

        Assert.InRange(Farthest(OnnxFixtures.Values(tensorFlow.GetProperty("outputs")), outputs), 0, 1e-5);
    }

    [Fact]
    public void TheTitanicGraph_IsBuiltAsItsNodesSay_ItsLossTheOneHandedIn_AndItsFileLeftOpen()
    {
        var loss = new BinaryCrossEntropy();
        using var graph = OnnxFixtures.Open("onnx-titanic.onnx");
        var saved = new OnnxFile(loss).Read(graph);
        var stack = Assert.IsType<LayerStack>(saved.Network);

        Assert.Collection(
            stack.Layers,
            layer => Assert.Equal(new Shape(14, 16), Assert.IsType<Dense>(layer).Weight.Value.Shape),
            layer => Assert.IsType<Relu>(layer),
            layer => Assert.Equal(new Shape(16, 1), Assert.IsType<Dense>(layer).Weight.Value.Shape));
        Assert.Equal(["0.weight", "0.bias", "2.weight", "2.bias"], stack.Slots().Select(named => named.Path));
        Assert.Same(loss, saved.Loss);
        Assert.Null(saved.TrainedOn);
        Assert.True(graph.CanRead);
        Assert.Same(stack, stack.Compile(new Adam(0.01), saved.Loss).Network);
    }

    [Theory]
    [InlineData("onnx-titanic")]
    [InlineData("onnx-convolution")]
    public void TheDefaultExporter_AndTheTorchScriptOne_PutTheSameNumbersIntoTheSameSlots(string network)
    {
        var loss = new MeanSquaredError();
        using var exported = OnnxFixtures.Open($"{network}.onnx");
        using var torchScript = OnnxFixtures.Open($"{network}-torchscript.onnx");
        var fromDefault = new OnnxFile(loss).Read(exported).Network;
        var fromTorchScript = new OnnxFile(loss).Read(torchScript).Network;

        Assert.Equal(fromDefault.Slots().Select(named => named.Path), fromTorchScript.Slots().Select(named => named.Path));
        Assert.Equal(Bits(fromDefault), Bits(fromTorchScript));
    }

    [Theory]
    [InlineData("onnx-convolution.onnx")]
    [InlineData("onnx-convolution-torchscript.onnx")]
    public void TheNetworkOverImages_GivesEachImageTheOutputPyTorchGaveIt_WithinATenThousandth(string file)
    {
        using var graph = OnnxFixtures.Open(file);
        var saved = new OnnxFile(new MeanSquaredError()).Read(graph);

        // The images go in with their channels last, as every image here does; PyTorch was handed the same ones turned.
        var outputs = saved.Network.Predict(OnnxFixtures.Images(Convolution.GetProperty("images")), saved.Loss, Engine);

        Assert.Equal(new Shape(6, 1), outputs.Shape);
        Assert.InRange(Farthest(OnnxFixtures.Values(Convolution.GetProperty("outputs")), outputs), 0, 1e-5);
    }

    [Fact]
    public void EachConvolution_Normalisation_AndFlatten_IsBuiltAsItsNodeSays()
    {
        using var graph = OnnxFixtures.Open("onnx-convolution.onnx");
        var stack = Assert.IsType<LayerStack>(new OnnxFile(new MeanSquaredError()).Read(graph).Network);

        Assert.Collection(
            stack.Layers,
            layer => AssertConvolution(new Window(3, 2) { Padding = 1 }, 2, 4, layer),
            layer => Assert.IsType<Relu>(layer),
            layer => AssertConvolution(new Window(2, 2) { Stride = 2 }, 4, 3, layer),
            layer => Assert.IsType<Flatten>(layer),
            layer =>
            {
                var norm = Assert.IsType<BatchNorm>(layer);
                Assert.Equal(1 - 0.9, norm.Momentum);
                Assert.Equal(1e-5, norm.Epsilon);
                Assert.Equal(36, norm.Features);
            },
            layer => Assert.Equal(new Shape(36, 5), Assert.IsType<Dense>(layer).Weight.Value.Shape),
            layer => Assert.IsType<Tanh>(layer),
            layer => Assert.Equal(new Shape(5, 1), Assert.IsType<Dense>(layer).Weight.Value.Shape));
    }

    [Fact]
    public void ASigmoidTheGraphEndsIn_IsLiftedIntoABinaryCrossEntropy_AndStaysALayerBeforeALossThatAppliesNone()
    {
        var torch = Titanic.GetProperty("test");
        var chances = OnnxFixtures.Values(torch.GetProperty("chances"));
        var passengers = OnnxFixtures.Passengers(torch.GetProperty("features"));

        using var lifted = OnnxFixtures.Open("onnx-titanic-sigmoid.onnx");
        var intoTheLoss = new OnnxFile(new BinaryCrossEntropy()).Read(lifted);
        using var kept = OnnxFixtures.Open("onnx-titanic-sigmoid.onnx");
        var asALayer = new OnnxFile(new MeanSquaredError()).Read(kept);

        Assert.IsType<Dense>(Assert.IsType<LayerStack>(intoTheLoss.Network).Layers[^1]);
        Assert.IsType<Sigmoid>(Assert.IsType<LayerStack>(asALayer.Network).Layers[^1]);
        Assert.InRange(RoundingsApart(chances, intoTheLoss.Network.Predict(passengers, intoTheLoss.Loss, Engine)), 0, 8);
        Assert.InRange(RoundingsApart(chances, asALayer.Network.Predict(passengers, asALayer.Loss, Engine)), 0, 8);
    }

    [Fact]
    public void ALayerNormalisation_ASigmoidWithin_AndTheSoftmaxACrossEntropyOwns_AnswerAsPyTorchDid()
    {
        using var graph = OnnxFixtures.Open("onnx-kinds.onnx");
        var saved = new OnnxFile(new CrossEntropy()).Read(graph);
        var stack = Assert.IsType<LayerStack>(saved.Network);

        Assert.Collection(
            stack.Layers,
            layer => Assert.Equal(new Shape(14, 8), Assert.IsType<Dense>(layer).Weight.Value.Shape),
            layer =>
            {
                var norm = Assert.IsType<LayerNorm>(layer);
                Assert.Equal(1e-5, norm.Epsilon);
                Assert.Equal(8, norm.Features);
            },
            layer => Assert.IsType<Sigmoid>(layer),
            layer => Assert.Equal(new Shape(8, 3), Assert.IsType<Dense>(layer).Weight.Value.Shape));

        var shares = saved.Network.Predict(OnnxFixtures.Passengers(Kinds.GetProperty("features")), saved.Loss, Engine);

        Assert.InRange(RoundingsApart(OnnxFixtures.Values(Kinds.GetProperty("shares")), shares), 0, 8);
    }

    [Fact]
    public void AWeightMatrixAGemmDeclaresAsInputsByOutputs_IsReadAsWritten_AndOneItDeclaresTurnedIsTurned()
    {
        // The same numbers, the matrices written the other way round and each Gemm saying so: the layout is what the
        // node declares, never what the numbers look like.
        using var written = new MemoryStream(OnnxFixtures.Model("onnx-titanic-torchscript.onnx").ToByteArray());
        var asExported = new OnnxFile(new BinaryCrossEntropy()).Read(written).Network;
        var turned = new OnnxFile(new BinaryCrossEntropy()).Read(OnnxFixtures.Edited("onnx-titanic-torchscript.onnx", model =>
        {
            foreach (var place in new[] { 0, 2 })
            {
                var weights = $"{place}.weight";
                var tensor = model.Initializer(weights);
                var rows = (int)tensor.Dims[0];
                var columns = (int)tensor.Dims[1];
                var values = FloatsOf(tensor);
                var transposed = new float[values.Length];

                for (var row = 0; row < rows; row++)
                {
                    for (var column = 0; column < columns; column++)
                    {
                        transposed[(column * rows) + row] = values[(row * columns) + column];
                    }
                }

                model.Graph.Initializer[model.Graph.Initializer.IndexOf(tensor)] = OnnxFixtures.Floats(weights, [columns, rows], transposed);
                model.Node($"/{place}/Gemm").SetWhole("transB", 0);
            }
        })).Network;

        Assert.Equal(Bits(asExported), Bits(turned));
    }

    [Theory]
    [InlineData("FLOAT16", false)]
    [InlineData("FLOAT16", true)]
    [InlineData("BFLOAT16", false)]
    [InlineData("BFLOAT16", true)]
    [InlineData("DOUBLE", false)]
    [InlineData("DOUBLE", true)]
    [InlineData("FLOAT", true)]
    public void NumbersKeptInAnotherWidth_OrInTheTypedFields_AreWidenedOrRoundedIntoSingleOnes(string width, bool typed)
    {
        var expected = new List<TensorProto>();
        var read = new OnnxFile(new BinaryCrossEntropy()).Read(OnnxFixtures.Edited("onnx-titanic-torchscript.onnx", model =>
        {
            foreach (var tensor in model.Graph.Initializer)
            {
                var values = FloatsOf(tensor);
                var kept = Kept(width, typed, values);
                kept.Name = tensor.Name;
                kept.Dims.Add(tensor.Dims);
                expected.Add(OnnxFixtures.Floats(tensor.Name, [.. tensor.Dims], [.. values.Select(value => Widened(width, value))]));
                model.Graph.Initializer[model.Graph.Initializer.IndexOf(tensor)] = kept;
            }
        })).Network;
        var widened = new OnnxFile(new BinaryCrossEntropy()).Read(OnnxFixtures.Edited("onnx-titanic-torchscript.onnx", model =>
        {
            model.Graph.Initializer.Clear();
            model.Graph.Initializer.Add(expected);
        })).Network;

        Assert.Equal(Bits(widened), Bits(read));
    }

    [Fact]
    public void NumbersKeptBesideTheGraph_AreReadFromWhereTheirRecordSays_ThePartOfTheFileItNamesOrItsEnd()
    {
        using var folder = new GraphFolder();
        var model = OnnxFixtures.Model("onnx-titanic-torchscript.onnx");
        using var beside = new MemoryStream();
        beside.Write(new byte[7]);

        // Every number moved into one file beside the graph: the first ones at an offset with a length, the last one
        // running to the file's end, as a record without a length says.
        for (var at = 0; at < model.Graph.Initializer.Count; at++)
        {
            var tensor = model.Graph.Initializer[at];
            var last = at == model.Graph.Initializer.Count - 1;
            tensor.DataLocation = TensorProto.Types.DataLocation.External;
            tensor.ExternalData.Add(new StringStringEntryProto { Key = "location", Value = "numbers.bin" });
            tensor.ExternalData.Add(new StringStringEntryProto { Key = "offset", Value = beside.Length.ToString(System.Globalization.CultureInfo.InvariantCulture) });

            if (!last)
            {
                tensor.ExternalData.Add(new StringStringEntryProto { Key = "length", Value = tensor.RawData.Length.ToString(System.Globalization.CultureInfo.InvariantCulture) });
            }

            beside.Write(tensor.RawData.Span);
            tensor.RawData = ByteString.Empty;
        }

        folder.Beside("numbers.bin", beside.ToArray());

        using var graph = folder.Written(model);
        using var original = OnnxFixtures.Open("onnx-titanic-torchscript.onnx");

        Assert.Equal(Bits(new OnnxFile(new BinaryCrossEntropy()).Read(original).Network), Bits(new OnnxFile(new BinaryCrossEntropy()).Read(graph).Network));
    }

    [Theory]
    [InlineData(-1L, 36L, 1L)]
    [InlineData(6L, 36L, 1L)]
    [InlineData(0L, 36L, 0L)]
    [InlineData(6L, -1L, 1L)]
    [InlineData(0L, -1L, 0L)]
    public void AReshapeThatFlattensEachExample_IsAFlatten_ItsTargetACountOrAConstant(long batch, long row, long allowZero)
    {
        var flattened = Outputs(OnnxFixtures.Edited("onnx-convolution-torchscript.onnx", model =>
        {
            var node = model.Node("/4/Flatten");
            node.OpType = "Reshape";
            node.Unset("axis");
            node.SetWhole("allowzero", allowZero);
            node.Input.Add("target");

            // The target stands in a constant node, as the TorchScript exporter writes one, handed on by an identity.
            var constant = OnnxFixtures.NodeOf("Constant", "shape", [], "kept");
            constant.Attribute.Add(new AttributeProto { Name = "value", Type = AttributeProto.Types.AttributeType.Tensor, T = OnnxFixtures.Wholes(string.Empty, batch, row) });
            model.Graph.Node.Insert(0, OnnxFixtures.NodeOf("Identity", "handed", ["kept"], "target"));
            model.Graph.Node.Insert(0, constant);
        }));

        Assert.InRange(Farthest(OnnxFixtures.Values(Convolution.GetProperty("outputs")), flattened), 0, 1e-5);
    }

    [Fact]
    public void AnIdentityOnTheValueTheNetworkMakes_HandsItOn_AndOneOnANumber_IsThatNumber()
    {
        using var original = OnnxFixtures.Open("onnx-titanic-torchscript.onnx");
        var expected = Bits(new OnnxFile(new BinaryCrossEntropy()).Read(original).Network);

        var read = new OnnxFile(new BinaryCrossEntropy()).Read(OnnxFixtures.Edited("onnx-titanic-torchscript.onnx", model =>
        {
            model.Node("/1/Relu").Input[0] = "passed";
            model.Node("/2/Gemm").Input[2] = "bias";
            model.Graph.Node.Insert(1, OnnxFixtures.NodeOf("Identity", "pass", ["/0/Gemm_output_0"], "passed"));
            model.Graph.Node.Insert(0, OnnxFixtures.NodeOf("Identity", "alias", ["2.bias"], "bias"));
        })).Network;

        Assert.Equal(expected, Bits(read));
        Assert.Equal(3, Assert.IsType<LayerStack>(read).Layers.Count);
    }

    [Theory]
    [InlineData("/0/Conv", "SAME_UPPER", 0)]
    [InlineData("/0/Conv", "NOTSET", 0)]
    [InlineData("/3/Conv", "VALID", 2)]
    public void AConvolutionsPadding_IsReadAsItsNodeDeclaresIt(string convolution, string autoPad, int layer)
    {
        // Each padding here leaves the image the next layer takes as it was, so the graph's numbers still fit.
        var read = new OnnxFile(new MeanSquaredError()).Read(OnnxFixtures.Edited("onnx-convolution-torchscript.onnx", model =>
        {
            var node = model.Node(convolution);
            node.SetText("auto_pad", autoPad);

            if (autoPad != "NOTSET")
            {
                node.Unset("pads");
            }
        }));
        var window = Assert.IsType<Conv2D>(Assert.IsType<LayerStack>(read.Network).Layers[layer]).Window;

        Assert.Equal(
            autoPad switch
            {
                "SAME_UPPER" => new Window(3, 2) { PaddingMode = PaddingMode.Same },
                "NOTSET" => new Window(3, 2) { Padding = 1 },
                _ => new Window(2, 2) { Stride = 2 },
            },
            window);
    }

    [Fact]
    public void AGraphEndingInImages_GivesThemWithTheirChannelsLast_AsTheLayersOfTheWholeNetworkMakeThem()
    {
        var images = OnnxFixtures.Images(Convolution.GetProperty("images"));
        var pass = Pass.Evaluation(Engine);
        using var whole = OnnxFixtures.Open("onnx-convolution-torchscript.onnx");
        var layers = Assert.IsType<LayerStack>(new OnnxFile(new MeanSquaredError()).Read(whole).Network).Layers;

        var cut = new OnnxFile(new MeanSquaredError()).Read(OnnxFixtures.Edited("onnx-convolution-torchscript.onnx", model =>
        {
            while (model.Graph.Node[^1].Name != "/2/Relu")
            {
                model.Graph.Node.RemoveAt(model.Graph.Node.Count - 1);
            }

            model.Graph.Output[0].Name = "/2/Relu_output_0";
        }));
        var outputs = cut.Network.Predict(images, cut.Loss, Engine);

        // Six images of 8 rows, 7 columns and 4 channels: the graph's batch by channels by rows by columns, channels last.
        Assert.Equal(new Shape(6, 8, 7, 4), outputs.Shape);
        Assert.Equal(layers[1].Forward(layers[0].Forward(images, pass), pass).Values.ToArray(), outputs.Values.ToArray());
    }

    [Fact]
    public void AFlattenOfRows_LeavesThemAsTheyAre_AndTheNumbersAfterItUnturned()
    {
        var passengers = OnnxFixtures.Passengers(Titanic.GetProperty("test").GetProperty("features"));
        using var original = OnnxFixtures.Open("onnx-titanic-torchscript.onnx");
        var expected = new OnnxFile(new BinaryCrossEntropy()).Read(original);

        var flattened = new OnnxFile(new BinaryCrossEntropy()).Read(OnnxFixtures.Edited("onnx-titanic-torchscript.onnx", model =>
        {
            model.Node("/2/Gemm").Input[0] = "flat";
            model.Graph.Node.Insert(2, OnnxFixtures.NodeOf("Flatten", "flatten", ["/1/Relu_output_0"], "flat"));
        }));

        Assert.IsType<Flatten>(Assert.IsType<LayerStack>(flattened.Network).Layers[2]);
        Assert.Equal(
            expected.Network.Predict(passengers, expected.Loss, Engine).Values.ToArray(),
            flattened.Network.Predict(passengers, flattened.Loss, Engine).Values.ToArray());
    }

    [Fact]
    public void AConvolutionsSettingsLeftOut_AreWhatOnnxGivesThem()
    {
        using var original = OnnxFixtures.Open("onnx-convolution-torchscript.onnx");
        var expected = Bits(new OnnxFile(new MeanSquaredError()).Read(original).Network);

        // A stride and a dilation of one, and one group, are what ONNX means when a node says none.
        var read = new OnnxFile(new MeanSquaredError()).Read(OnnxFixtures.Edited("onnx-convolution-torchscript.onnx", model =>
        {
            var node = model.Node("/0/Conv");
            node.Unset("strides");
            node.Unset("dilations");
            node.Unset("group");
        }));

        Assert.Equal(expected, Bits(read.Network));
        Assert.Equal(new Window(3, 2) { Padding = 1 }, Assert.IsType<Conv2D>(Assert.IsType<LayerStack>(read.Network).Layers[0]).Window);
    }

    [Fact]
    public void ALayerNormalisationFromTheAxisAfterTheBatch_IsOneOverTheLastAxisOfARow()
    {
        var read = new OnnxFile(new BinaryCrossEntropy()).Read(OnnxFixtures.Edited("onnx-titanic-torchscript.onnx", model =>
        {
            var node = model.Node("/1/Relu");
            node.OpType = "LayerNormalization";
            node.Input.Add(["0.bias", "0.bias"]);
            node.SetWhole("axis", 1);
        }));

        Assert.Equal(16, Assert.IsType<LayerNorm>(Assert.IsType<LayerStack>(read.Network).Layers[1]).Features);
    }

    [Fact]
    public void AReshapesTargetWrittenInItsTypedField_IsReadAsItsBytesAre()
    {
        var flattened = Outputs(OnnxFixtures.Edited("onnx-convolution-torchscript.onnx", model =>
        {
            var target = OnnxFixtures.Wholes("target", -1, 36);
            target.RawData = ByteString.Empty;
            target.Int64Data.Add([-1, 36]);
            model.Graph.Initializer.Add(target);

            var node = model.Node("/4/Flatten");
            node.OpType = "Reshape";
            node.Unset("axis");
            node.Input.Add("target");
        }));

        Assert.InRange(Farthest(OnnxFixtures.Values(Convolution.GetProperty("outputs")), flattened), 0, 1e-5);
    }

    [Theory]
    [InlineData("")]
    [InlineData("ai.onnx")]
    public void ANodeOfOnnxsOwnDomain_IsReadByEitherNameOfIt(string domain)
    {
        var read = new OnnxFile(new BinaryCrossEntropy()).Read(OnnxFixtures.Edited("onnx-titanic-torchscript.onnx", model => model.Node("/1/Relu").Domain = domain));

        Assert.IsType<Relu>(Assert.IsType<LayerStack>(read.Network).Layers[1]);
    }

    private static void AssertConvolution(Window window, int inChannels, int outChannels, Layer layer)
    {
        var convolution = Assert.IsType<Conv2D>(layer);

        Assert.Equal(window, convolution.Window);
        Assert.Equal(inChannels, convolution.InChannels);
        Assert.Equal(outChannels, convolution.OutChannels);
    }

    // The outputs a graph over the fixture's images gives, read with a loss that applies nothing.
    private static Tensor Outputs(Stream graph)
    {
        var saved = new OnnxFile(new MeanSquaredError()).Read(graph);

        return saved.Network.Predict(OnnxFixtures.Images(Convolution.GetProperty("images")), saved.Loss, Engine);
    }

    // The single-precision numbers an initializer holds as its raw bytes.
    private static float[] FloatsOf(TensorProto tensor) =>
        [.. Enumerable.Range(0, tensor.RawData.Length / 4).Select(at => BitConverter.ToSingle(tensor.RawData.Span[(at * 4)..]))];

    // Numbers kept in another width, in the raw bytes or in the field ONNX keeps that width in.
    private static TensorProto Kept(string width, bool typed, float[] values)
    {
        var tensor = new TensorProto();

        switch (width)
        {
            case "FLOAT16":
                tensor.DataType = (int)TensorProto.Types.DataType.Float16;
                KeepSixteen(tensor, typed, [.. values.Select(value => BitConverter.HalfToUInt16Bits((Half)value))]);
                break;
            case "BFLOAT16":
                tensor.DataType = (int)TensorProto.Types.DataType.Bfloat16;
                KeepSixteen(tensor, typed, [.. values.Select(BrainFloat)]);
                break;
            case "DOUBLE":
                tensor.DataType = (int)TensorProto.Types.DataType.Double;
                var doubles = values.Select(Doubled).ToArray();

                if (typed)
                {
                    tensor.DoubleData.Add(doubles);
                }
                else
                {
                    tensor.RawData = ByteString.CopyFrom([.. doubles.SelectMany(BitConverter.GetBytes)]);
                }

                break;
            default:
                tensor.DataType = (int)TensorProto.Types.DataType.Float;
                tensor.FloatData.Add(values);
                break;
        }

        return tensor;
    }

    private static void KeepSixteen(TensorProto tensor, bool typed, ushort[] bits)
    {
        if (typed)
        {
            tensor.Int32Data.Add(bits.Select(each => (int)each));
        }
        else
        {
            tensor.RawData = ByteString.CopyFrom([.. bits.SelectMany(BitConverter.GetBytes)]);
        }
    }

    // What a number kept in a width becomes as a single one: a half or a brain float exactly, a double rounded to nearest.
    private static float Widened(string width, float value) => width switch
    {
        "FLOAT16" => (float)(Half)value,
        "BFLOAT16" => BitConverter.UInt32BitsToSingle((uint)BrainFloat(value) << 16),
        "DOUBLE" => (float)Doubled(value),
        _ => value,
    };

    // A double just off the single it came from, so rounding it back is a rounding.
    private static double Doubled(float value) => value * (1 + 1e-9);

    // The upper half of a single, rounded to nearest, ties to even: bfloat16 as PyTorch makes it.
    private static ushort BrainFloat(float value)
    {
        var bits = BitConverter.SingleToUInt32Bits(value);

        return (ushort)((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16);
    }

    // The most roundings of a single-precision number any value lies from PyTorch's: the distance between their bit
    // patterns, which for numbers of one sign counts the floats between them.
    private static long RoundingsApart(float[] torch, Tensor here)
    {
        Assert.Equal(torch.Length, here.Values.Length);

        return torch.Zip(here.Values.ToArray(), (expected, actual) => Math.Abs((long)BitConverter.SingleToInt32Bits(expected) - BitConverter.SingleToInt32Bits(actual))).Max();
    }

    private static double Farthest(float[] torch, Tensor here)
    {
        Assert.Equal(torch.Length, here.Values.Length);

        return torch.Zip(here.Values.ToArray(), (expected, actual) => Math.Abs((double)expected - actual)).Max();
    }

    private static Tensor Batch(IReadOnlyList<double[]> rows) =>
        Tensor.From(new Shape(rows.Count, rows[0].Length), [.. rows.SelectMany(row => row.Select(value => (float)value))]);

    private static int[][] Bits(Network network) =>
        [.. network.Slots().Select(named => named.Slot.Value.Values.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray())];
}
