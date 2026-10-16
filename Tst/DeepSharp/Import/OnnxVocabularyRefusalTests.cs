// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Import.Onnx;
using DeepSharp.Networks;
using Onnx;

namespace DeepSharp.Tests.Import;

/// <summary>
/// What the operators 0.11.0 reads cannot say here, each refused where the graph says it: a GELU of another curve, a SELU of other
/// constants, a parametric rectifier whose shares stand along more than one axis or reach flattened images, an instance
/// normalisation of rows, a group normalisation of flattened images, a root-mean-square normalisation of anything but a row's
/// values, and a softmax over another axis than the channels of images laid out channels last.
/// </summary>
public class OnnxVocabularyRefusalTests
{
    private static string Refused(string file, Action<ModelProto> edit) =>
        Assert.Throws<FormatException>(() => new OnnxFile(new MeanSquaredError()).Read(OnnxFixtures.Edited(file, edit))).Message;

    private static NodeProto Named(ModelProto model, string op) => model.Graph.Node.Single(node => node.OpType == op);

    private static void Set(NodeProto node, AttributeProto attribute)
    {
        var was = node.Attribute.SingleOrDefault(each => each.Name == attribute.Name);

        if (was is not null)
        {
            node.Attribute.Remove(was);
        }

        node.Attribute.Add(attribute);
    }

    private static AttributeProto Whole(string name, long value) => new() { Name = name, Type = AttributeProto.Types.AttributeType.Int, I = value };

    // The images graph with its opset raised to 21 and its last Gemm, after the Flatten, made the operator named over the inputs
    // given: what reaches it are images flattened into rows.
    private static Action<ModelProto> AfterTheFlatten(string op, params string[] inputs) => model =>
    {
        model.OpsetImport.Single(import => import.Domain.Length == 0).Version = 21;

        var gemm = Named(model, "Gemm");

        gemm.OpType = op;
        gemm.Attribute.Clear();
        gemm.Input.Clear();
        gemm.Input.Add(["/4/Flatten_output_0", .. inputs]);
    };

    [Fact]
    public void AGeluOfAnotherCurve_IsRefusedNamingIt()
    {
        var refused = Refused("onnx-vocabulary-dynamo-gelu-21.onnx", model =>
            Set(Named(model, "Gelu"), new AttributeProto { Name = "approximate", Type = AttributeProto.Types.AttributeType.String, S = Google.Protobuf.ByteString.CopyFromUtf8("erf") }));

        Assert.Contains("it approximates the curve as 'erf'", refused, StringComparison.Ordinal);
    }

    [Fact]
    public void ASeluOfOtherConstants_IsRefusedNamingThem()
    {
        var refused = Refused("onnx-vocabulary-ts-selu.onnx", model =>
            Set(Named(model, "Selu"), new AttributeProto { Name = "alpha", Type = AttributeProto.Types.AttributeType.Float, F = 2f }));

        Assert.Contains("it bends at an alpha of 2", refused, StringComparison.Ordinal);
    }

    [Fact]
    public void AParametricRectifierWhoseSharesStandAlongTwoAxes_OrReachFlattenedImages_IsRefused()
    {
        var twoAxes = Refused("onnx-vocabulary-ts-prelu.onnx", model =>
        {
            var slope = model.Graph.Initializer.Single(each => each.Name == "1.weight");

            slope.Dims.Clear();
            slope.Dims.Add([2, 3]);
        });
        var flattened = Refused("onnx-vocabulary-images.onnx", AfterTheFlatten("PRelu", "onnx::PRelu_16"));

        Assert.Contains("are written as 2x3", twoAxes, StringComparison.Ordinal);
        Assert.Contains("a parametric rectifier here learns one share, or one for each channel", flattened, StringComparison.Ordinal);
    }

    [Fact]
    public void AnInstanceNormalisationOfRows_IsRefused()
    {
        var refused = Refused("onnx-vocabulary-dynamo-groupnorm-21.onnx", model =>
        {
            var node = Named(model, "GroupNormalization");

            node.OpType = "InstanceNormalization";
            node.Attribute.Clear();
        });

        Assert.Contains("InstanceNormalization", refused, StringComparison.Ordinal);
    }

    [Fact]
    public void AGroupNormalisationOfFlattenedImages_IsRefused()
    {
        var refused = Refused("onnx-vocabulary-images.onnx", model =>
        {
            AfterTheFlatten("GroupNormalization", "1.weight", "1.bias")(model);
            Set(Named(model, "GroupNormalization"), Whole("num_groups", 2));
        });

        Assert.Contains("it normalises along the axis after the batch, and what reaches it does not hold its channels there", refused, StringComparison.Ordinal);
    }

    [Fact]
    public void ARootMeanSquareNormalisationOfImages_OrFromAnotherAxis_IsRefused()
    {
        var images = Refused("onnx-vocabulary-images.onnx", model =>
        {
            model.OpsetImport.Single(import => import.Domain.Length == 0).Version = 23;

            var node = Named(model, "InstanceNormalization");

            node.OpType = "RMSNormalization";
            node.Attribute.Clear();
            node.Input.RemoveAt(2);
        });
        var axis = Refused("onnx-vocabulary-dynamo-rmsnorm-23.onnx", model => Set(Named(model, "RMSNormalization"), Whole("axis", 0)));

        Assert.Contains("it normalises from axis -1 of what reaches it", images, StringComparison.Ordinal);
        Assert.Contains("it normalises from axis 0 of what reaches it", axis, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(3)]
    public void ASoftmaxOverTheChannelsOfImagesLaidOutChannelsLast_IsALayer(long axis)
    {
        var read = new OnnxFile(new MeanSquaredError()).Read(OnnxFixtures.Edited("tf2onnx-convolution.onnx", ChannelsLastSoftmax(axis)));

        Assert.Contains(Assert.IsType<LayerStack>(read.Network).Layers, layer => layer is Softmax);
    }

    [Fact]
    public void ASoftmaxOverAnotherAxisOfImagesLaidOutChannelsLast_IsRefused()
    {
        var refused = Refused("tf2onnx-convolution.onnx", ChannelsLastSoftmax(1));

        Assert.Contains("Softmax", refused, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("the mean drops its axis")]
    [InlineData("the mean is over the batch")]
    [InlineData("the mean is over the batch, as an attribute says")]
    [InlineData("the mean is of the input, not of its squares")]
    [InlineData("the square is by two numbers")]
    [InlineData("a later layer reads a value inside the spelling")]
    public void AChainThatIsNotTheRootMeanSquaresSpelling_IsNotTakenForIt(string change)
    {
        var refused = Refused("onnx-vocabulary-dynamo-rmsnorm.onnx", model => Changed(model, change));

        Assert.Contains("is no operator a network here is built of", refused, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("a scale of other numbers than one")]
    [InlineData("a scale the graph works out rather than holds")]
    [InlineData("a scale of no numbers")]
    public void AnInstanceNormalisationThatIsNotTheGroupNormalisationsSpelling_IsNotTakenForIt(string change)
    {
        var refused = Refused("onnx-vocabulary-dynamo-groupnorm.onnx", model =>
        {
            var normalisation = Named(model, "InstanceNormalization");
            var scale = normalisation.Input[1];

            switch (change)
            {
                case "a scale of other numbers than one":
                    model.Graph.Initializer.Remove(model.Graph.Initializer.Single(each => each.Name == scale));
                    model.Graph.Initializer.Add(OnnxFixtures.Floats(scale, [3], 1f, 2f, 1f));
                    break;
                case "a scale the graph works out rather than holds":
                    normalisation.Input[1] = "linear";
                    break;
                case "a scale of no numbers":
                    model.Graph.Initializer.Remove(model.Graph.Initializer.Single(each => each.Name == scale));
                    model.Graph.Initializer.Add(OnnxFixtures.Floats(scale, [0]));
                    break;
            }
        });

        Assert.Contains("is no operator a network here is built of", refused, StringComparison.Ordinal);
    }

    [Fact]
    public void TheRootMeanSquaresSpelling_IsReadWhereItsMeanSaysItsAxisAsAnAttribute()
    {
        var read = new OnnxFile(new MeanSquaredError()).Read(OnnxFixtures.Edited("onnx-vocabulary-dynamo-rmsnorm.onnx", model => Changed(model, "the axis as an attribute")));

        Assert.IsType<RmsNorm>(Assert.IsType<LayerStack>(read.Network).Layers[1]);
    }

    // The dynamo exporter's spelling of a root-mean-square normalisation for operator set 18, changed as said.
    private static void Changed(ModelProto model, string change)
    {
        var mean = Named(model, "ReduceMean");

        switch (change)
        {
            case "the mean drops its axis":
                Set(mean, Whole("keepdims", 0));
                break;
            case "the mean is over the batch":
                model.Graph.Initializer.Remove(model.Graph.Initializer.Single(each => each.Name == mean.Input[1]));
                model.Graph.Initializer.Add(OnnxFixtures.Wholes(mean.Input[1], 0));
                break;
            case "the mean is over the batch, as an attribute says":
            case "the axis as an attribute":
                mean.Input.RemoveAt(1);
                mean.Attribute.Add(new AttributeProto
                {
                    Name = "axes",
                    Type = AttributeProto.Types.AttributeType.Ints,
                    Ints = { change == "the axis as an attribute" ? -1 : 0 },
                });
                break;
            case "the mean is of the input, not of its squares":
                mean.Input[0] = "linear";
                break;
            case "the square is by two numbers":
                model.Graph.Initializer.Remove(model.Graph.Initializer.Single(each => each.Name == "val_0"));
                model.Graph.Initializer.Add(OnnxFixtures.Floats("val_0", [2], 2f, 2f));
                break;
            case "a later layer reads a value inside the spelling":
                model.Graph.Node.Single(node => node.Name == "node_linear_1").Input[0] = "mul";
                break;
        }
    }

    // A softmax between the Transpose that lays the images out channels last and the Reshape that flattens them.
    private static Action<ModelProto> ChannelsLastSoftmax(long axis) => model =>
    {
        var reshape = Named(model, "Reshape");
        var softmax = OnnxFixtures.NodeOf("Softmax", "softmax", [reshape.Input[0]], "softmax_output");

        softmax.Attribute.Add(Whole("axis", axis));
        reshape.Input[0] = "softmax_output";
        model.Graph.Node.Insert(model.Graph.Node.IndexOf(reshape), softmax);
    };
}
