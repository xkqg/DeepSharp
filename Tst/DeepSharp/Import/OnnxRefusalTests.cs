// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using DeepSharp.Import.Onnx;
using DeepSharp.Networks;
using Google.Protobuf;
using Onnx;

namespace DeepSharp.Tests.Import;

/// <summary>
/// What an ONNX graph says that no network here is built of is refused where the graph says it — each node's fault at that
/// node, every one at once — and numbers that do not fit the layers the graph builds are refused by the initializer that
/// holds each, by the same rule and in the same words as every other file of numbers. A file that is no ONNX model is
/// refused as such.
/// </summary>
public class OnnxRefusalTests
{
    private const string Titanic = "onnx-titanic-torchscript.onnx";

    private const string Images = "onnx-convolution-torchscript.onnx";

    private const string Keras = "keras-titanic-export.onnx";

    private const string TensorFlow = "tf2onnx-convolution.onnx";

    // Where tf2onnx names the nodes of the Keras network it converted.
    private const string Converted = "StatefulPartitionedCall/sequential_1/";

    private const string Batches = "StatefulPartitionedCall/sequential_1_1/";

    // The networks of Fixtures/onnx-spatial-fixtures.py, exported by the TorchScript exporter, and its borders written by hand.
    private const string SeriesNet = "onnx-series-torchscript.onnx";

    private const string SeriesDefault = "onnx-series.onnx";

    private const string ImageNet = "onnx-image-torchscript.onnx";

    private const string VolumeNet = "onnx-volume-torchscript.onnx";

    // What the default exporter writes of a pooling of every place into one: a ReduceMean or a ReduceMax, an Unsqueeze and a Squeeze
    // round it along a series.
    private const string SeriesReduced = "onnx-series-global-average-default.onnx";

    private const string ImageReduced = "onnx-image-global-average-default.onnx";

    private const string VolumeReduced = "onnx-volume-global-average-default.onnx";

    private const string ImageLargestReduced = "onnx-image-global-max-default.onnx";

    private const string SeriesBorders = "onnx-borders-series.onnx";

    private const string VolumeBorders = "onnx-borders-volume.onnx";

    // The nodes, by name, the TorchScript exporter gave the trunk of each of those networks.
    private const string FirstConv = "/0/0.0/Conv";

    private const string MaxPooling = "/0/0.3/MaxPool";

    private const string AveragePooling = "/0/0.5/AveragePool";

    private const string LastConv = "/0/0.7/Conv";

    private const string Operators = "an ONNX graph is read here when each of its nodes is a Gemm, MatMul, Add, Conv, MaxPool, AveragePool, GlobalAveragePool, GlobalMaxPool, ReduceMean, ReduceMax, BatchNormalization, LayerNormalization, Flatten, Reshape, Relu, Tanh, Sigmoid or Softmax, or an Identity, a Cast, a Transpose or a Constant that hands a value on.";

    private const string Pads = "and a window here pads every side alike, or as SAME_UPPER, TensorFlow's 'same'.";

    private const string Takes = "and a network here takes rows (a batch of values), series (a batch of channels by steps), images (a batch of channels by rows by columns) or volumes (a batch of channels by planes by rows by columns).";

    private const string NotAWindow = "and a convolution here slides a window along a series, an image or a volume: channels out by channels in by steps, or by rows by columns, or by planes by rows by columns.";

    private const string Overhang = "it lets its window overhang the end of an axis, ceil_mode 1, and a window here stands only where it fits whole.";

    private const string Neighbours = "and a window here covers neighbouring places.";

    private const string Largest = "and a max pooling here gives the largest values alone.";

    private const string Stack = "and a network here is a stack: each layer takes the value the layer before it made";

    private const string Unsqueezed = $"node 'node_unsqueeze' (Unsqueeze): 'Unsqueeze' is no operator a network here is built of: {Operators}";

    private const string Squeezed = $"node 'node_squeeze' (Squeeze): 'Squeeze' is no operator a network here is built of: {Operators}";

    private const string NoInput = "and a network here takes one, the batch of its examples.";

    private const string NotFlattened = "and a reshape is read here when it flattens each example into one row, as Flatten does.";

    private const string SoftmaxLeft = "it is a softmax, which a network here leaves to its loss: it is read as the last of a graph whose cross-entropy applies it.";

    // What tf2onnx writes of a network with batch normalisations for batches of any length: each batch normalisation a Mul
    // by a number of its own for each channel or value — its shift folded into an Add, or into the next layer's bias — and the
    // flatten's target worked out from the batch's length by a side graph off the images.
    private static readonly string[] BatchesRefusal =
    [
        $"node '{Batches}batch_normalization_1/batchnorm/mul_1' (Mul): 'Mul' is no operator a network here is built of: {Operators}",
        $"node '{Batches}batch_normalization_1/batchnorm/add_1' (Add): it adds 'const_fold_opt__54' to what the node before it made, and an Add is read here as the bias of the MatMul before it.",
        $"node '{Batches}batch_normalization_1_2/batchnorm/mul_1' (Mul): 'Mul' is no operator a network here is built of: {Operators}",
        $"node '{Batches}batch_normalization_1_2/batchnorm/add_1' (Add): it adds 'const_fold_opt__56' to what the node before it made, and an Add is read here as the bias of the MatMul before it.",
        $"node 'Shape__46' (Shape): it takes '{Batches}batch_normalization_1_2/batchnorm/add_1:0', {Stack}, 'Transpose__40:0'.",
        $"node 'Gather__49' (Gather): 'Gather' is no operator a network here is built of: {Operators}",
        $"node '{Batches}flatten_1_1/Shape__14' (Cast): it turns what the node before it made into INT32, and a network here works in single-precision numbers throughout.",
        $"node '{Batches}flatten_1_1/strided_slice' (Slice): 'Slice' is no operator a network here is built of: {Operators}",
        $"node '{Batches}flatten_1_1/Reshape/shape_Concat__24' (Concat): 'Concat' is no operator a network here is built of: {Operators}",
        $"node '{Batches}flatten_1_1/Reshape__25' (Cast): it turns what the node before it made into INT64, and a network here works in single-precision numbers throughout.",
        $"node '{Batches}flatten_1_1/Reshape' (Reshape): it takes 'Transpose__40:0' and '{Batches}flatten_1_1/Reshape__25:0', 2 values the graph works out, {Stack}, '{Batches}flatten_1_1/Reshape__25:0'.",
        $"node '{Batches}batch_normalization_2_1/batchnorm/mul_1' (Mul): 'Mul' is no operator a network here is built of: {Operators}",
        $"node '{Batches}batch_normalization_2_1/batchnorm/add_1' (Add): it adds '{Batches}batch_normalization_2_1/batchnorm/sub:0' to what the node before it made, and an Add is read here as the bias of the MatMul before it.",
    ];

    private static readonly Dictionary<string, Case> Cases = new()
    {
        // What the graph takes and gives.
        ["two inputs"] = new(Titanic, model => model.Graph.Input.Add(new ValueInfoProto { Name = "extra", Type = model.Graph.Input[0].Type }), $"graph 'main_graph': it takes 2 values, {NoInput}"),
        ["no input"] = new(Titanic, model => model.Graph.Input.Clear(), $"graph 'main_graph': it takes 0 values, {NoInput}{Environment.NewLine}node '/0/Gemm' (Gemm): it takes 'passengers', {Stack}, ''."),
        ["an input of doubles"] = new(Titanic, model => model.Graph.Input[0].Type.TensorType.ElemType = 11, "input 'passengers': it takes DOUBLE values, and a network here takes single-precision numbers, FLOAT."),
        ["an input that is no tensor"] = new(Titanic, model => model.Graph.Input[0].Type = new TypeProto { SequenceType = new TypeProto.Types.Sequence() }, "input 'passengers': it takes UNDEFINED values, and a network here takes single-precision numbers, FLOAT."),
        ["an input of no type"] = new(Titanic, model => model.Graph.Input[0].Type = null, "input 'passengers': it takes UNDEFINED values, and a network here takes single-precision numbers, FLOAT."),
        ["an input of no shape"] = new(Titanic, model => model.Graph.Input[0].Type.TensorType.Shape = null, $"input 'passengers': it takes a batch of rank 0, [], {Takes}"),
        ["a length of nothing"] = new(Titanic, model => model.Graph.Input[0].Type.TensorType.Shape.Dim[1].DimValue = 0, "input 'passengers': its shape, [2, 0], does not state every length of an example, and a network here is built for examples of one shape."),
        ["a length too long"] = new(Titanic, model => model.Graph.Input[0].Type.TensorType.Shape.Dim[1].DimValue = 3000000000, "input 'passengers': its shape, [2, 3000000000], does not state every length of an example, and a network here is built for examples of one shape."),
        ["an input of rank six"] = new(Titanic, model => model.Graph.Input[0].Type.TensorType.Shape.Dim.Add(Enumerable.Repeat(new TensorShapeProto.Types.Dimension { DimValue = 1 }, 4)), $"input 'passengers': it takes a batch of rank 6, [2, 14, 1, 1, 1, 1], {Takes}"),
        ["a length named"] = new(Titanic, model => model.Graph.Input[0].Type.TensorType.Shape.Dim[1] = new TensorShapeProto.Types.Dimension { DimParam = "features" }, "input 'passengers': its shape, [2, features], does not state every length of an example, and a network here is built for examples of one shape."),
        ["a length left out"] = new(Titanic, model => model.Graph.Input[0].Type.TensorType.Shape.Dim[1] = new TensorShapeProto.Types.Dimension(), "input 'passengers': its shape, [2, ?], does not state every length of an example, and a network here is built for examples of one shape."),
        ["two outputs"] = new(Titanic, model => model.Graph.Output.Add(new ValueInfoProto { Name = "extra" }), "graph 'main_graph': it gives 2 values, and a network here gives one."),
        ["an output before the end"] = new(Titanic, model => model.Graph.Output[0].Name = "/1/Relu_output_0", "output '/1/Relu_output_0': the graph gives '/1/Relu_output_0', and a network here gives what its last layer makes, 'logits'."),
        ["an image flattened as the output"] = new(Images, model =>
        {
            while (model.Graph.Node[^1].Name != "/4/Flatten")
            {
                model.Graph.Node.RemoveAt(model.Graph.Node.Count - 1);
            }

            model.Graph.Output[0].Name = "/4/Flatten_output_0";
        }, "output '/4/Flatten_output_0': it is an image flattened as ONNX lays an image out, channel by channel, and a network here flattens an image place by place, so its values would come out in another order."),
        ["no layer"] = new(Titanic, model =>
        {
            model.Graph.Node.Clear();
            model.Graph.Output[0].Name = "passengers";
        }, "graph 'main_graph': it holds no layer a network here is built of."),
        ["a value named twice"] = new(Titanic, model => model.Graph.Initializer.Add(OnnxFixtures.Floats("0.bias", [16], new float[16])), "initializer '0.bias': the graph holds a value of that name already."),

        // A branch, and what no network here is built of.
        ["a value taken again"] = new(Titanic, model => model.Node("/2/Gemm").Input[0] = "passengers", $"node '/2/Gemm' (Gemm): it takes 'passengers', {Stack}, '/1/Relu_output_0'."),
        ["no value taken"] = new(Titanic, model => model.Node("/1/Relu").Input[0] = "0.bias", $"node '/1/Relu' (Relu): it takes none of the values the graph works out, {Stack}, '/0/Gemm_output_0'."),
        ["an operator not read"] = new(Titanic, model => model.Node("/1/Relu").OpType = "Elu", $"node '/1/Relu' (Elu): 'Elu' is no operator a network here is built of: {Operators}"),
        ["an operator of another domain"] = new(Titanic, model => model.Node("/1/Relu").Domain = "com.microsoft", $"node '/1/Relu' (com.microsoft.Relu): 'com.microsoft.Relu' is no operator a network here is built of: {Operators}"),
        ["an unnamed node"] = new(Titanic, model =>
        {
            model.Node("/1/Relu").OpType = "Elu";
            model.Graph.Node[1].Name = string.Empty;
        }, $"node 1 (Elu): 'Elu' is no operator a network here is built of: {Operators}"),

        // A dense layer.
        ["a setting of another type"] = new(Titanic, model => model.Node("/0/Gemm").SetNumber("transB", 1), "node '/0/Gemm' (Gemm): its 'transB' is written as FLOAT, which is not what ONNX writes there."),
        ["a whole number too large"] = new(Titanic, model => model.Node("/0/Gemm").SetWhole("transB", 4294967296), "node '/0/Gemm' (Gemm): its 'transB' is written as 4294967296, which is not what ONNX writes there."),
        ["its input turned round"] = new(Titanic, model => model.Node("/0/Gemm").SetWhole("transA", 1), "node '/0/Gemm' (Gemm): it takes its input turned round, transA, and a dense layer here takes each example as a row."),
        ["its product scaled"] = new(Titanic, model => model.Node("/0/Gemm").SetNumber("alpha", 0.5f), "node '/0/Gemm' (Gemm): it scales its product by 0.5 and its bias by 1, and a dense layer here scales neither."),
        ["its bias scaled"] = new(Titanic, model => model.Node("/0/Gemm").SetNumber("beta", 2), "node '/0/Gemm' (Gemm): it scales its product by 1 and its bias by 2, and a dense layer here scales neither."),
        ["no bias"] = new(Titanic, model => model.Node("/0/Gemm").Input.RemoveAt(2), "node '/0/Gemm' (Gemm): it adds no bias, and a dense layer or a convolution here always adds one."),
        ["a bias left empty"] = new(Titanic, model => model.Node("/0/Gemm").Input[2] = string.Empty, "node '/0/Gemm' (Gemm): it adds no bias, and a dense layer or a convolution here always adds one."),
        ["weights left empty"] = new(Titanic, model => model.Node("/0/Gemm").Input[1] = string.Empty, "node '/0/Gemm' (Gemm): its weights, '', are written as scalar, and a dense layer's are a matrix."),
        ["weights that are no matrix"] = new(Titanic, model =>
        {
            model.Initializer("0.weight").Dims.Clear();
            model.Initializer("0.weight").Dims.Add(224);
        }, "node '/0/Gemm' (Gemm): its weights, '0.weight', are written as 224, and a dense layer's are a matrix."),

        // A dense layer as Keras exports one: a MatMul, and an Add of its bias, with casts between.
        ["a cast into another type"] = new(Keras, model => model.Node("/Cast").SetWhole("to", 11), "node '/Cast' (Cast): it turns what the node before it made into DOUBLE, and a network here works in single-precision numbers throughout."),
        ["a matmul of its weights by the rows"] = new(Keras, model =>
        {
            var node = model.Node("/MatMul");
            var rows = node.Input[0];
            node.Input[0] = node.Input[1];
            node.Input[1] = rows;
        }, "node '/MatMul' (MatMul): its right side, '/Cast_2_output_0', is no value the graph holds, and a dense layer here multiplies each row by weights it holds."),
        ["a matmul of two values"] = new(Keras, model => model.Node("/MatMul").Input[1] = "keras_tensor", $"node '/MatMul' (MatMul): it takes '/Cast_2_output_0' and 'keras_tensor', 2 values the graph works out, {Stack}, '/Cast_2_output_0'."),
        ["a matmul by weights that are no matrix"] = new(Keras, model =>
        {
            model.Initializer("onnx::MatMul_24").Dims.Clear();
            model.Initializer("onnx::MatMul_24").Dims.Add(224);
        }, "node '/MatMul' (MatMul): its weights, 'onnx::MatMul_24', are written as 224, and a dense layer's are a matrix."),
        ["a matmul without its add"] = new(Keras, model =>
        {
            model.Node("/Cast_4").Input[0] = "/Cast_3_output_0";
            model.Graph.Node.Remove(model.Node("/Add"));
        }, "node '/MatMul' (MatMul): it adds no bias, and a dense layer or a convolution here always adds one."),
        ["a bias of another width"] = new(Keras, model =>
        {
            var bias = model.Initializer("onnx::Add_25");
            bias.Dims[0] = 8;
            bias.RawData = ByteString.CopyFrom(bias.RawData.Span[..32]);
        }, "node '/Add' (Add): it adds 'onnx::Add_25', written as 8, and a dense layer here adds a bias as long as the layer is wide, 16."),
        ["an add after no matmul"] = new(Keras, model =>
        {
            model.Graph.Node.Insert(model.Graph.Node.IndexOf(model.Node("/Cast_6")), OnnxFixtures.NodeOf("Add", "extra", ["/Relu_output_0", "onnx::Add_25"], "shifted"));
            model.Node("/Cast_6").Input[0] = "shifted";
        }, "node 'extra' (Add): it adds 'onnx::Add_25' to what the node before it made, and an Add is read here as the bias of the MatMul before it."),
        ["an add of two values"] = new(Keras, model => model.Node("/Add").Input[1] = "keras_tensor", $"node '/Add' (Add): it takes '/Cast_3_output_0' and 'keras_tensor', 2 values the graph works out, {Stack}, '/Cast_3_output_0'.{Environment.NewLine}node '/MatMul' (MatMul): it adds no bias, and a dense layer or a convolution here always adds one."),

        // A transpose, which is read only as moving an image's channels after the batch or last.
        ["a transpose of rows"] = new(Keras, model =>
        {
            var turn = OnnxFixtures.NodeOf("Transpose", "turn", ["/Relu_output_0"], "turned");
            turn.SetWholes("perm", 1, 0);
            model.Graph.Node.Insert(model.Graph.Node.IndexOf(model.Node("/Cast_6")), turn);
            model.Node("/Cast_6").Input[0] = "turned";
        }, "node 'turn' (Transpose): it turns its value round by [1, 0], and a Transpose is read here as one that moves an image's channels after the batch, [0, 3, 1, 2], or last, [0, 2, 3, 1]."),
        ["a transpose that writes no turn"] = new(TensorFlow, model => model.Node($"{Converted}conv2d_1_2/BiasAdd__11").Unset("perm"), $"node '{Converted}conv2d_1_2/BiasAdd__11' (Transpose): it turns its value round by [], and a Transpose is read here as one that moves an image's channels after the batch, [0, 3, 1, 2], or last, [0, 2, 3, 1]."),
        ["channels moved after the batch twice"] = new(TensorFlow, model => model.Node($"{Converted}conv2d_1_2/BiasAdd__11").SetWholes("perm", 0, 3, 1, 2), $"node '{Converted}conv2d_1_2/BiasAdd__11' (Transpose): it turns its value round by [0, 3, 1, 2], and a Transpose is read here as one that moves an image's channels after the batch, [0, 3, 1, 2], or last, [0, 2, 3, 1]."),
        ["a convolution of images with their channels last"] = new(TensorFlow, model =>
        {
            var last = OnnxFixtures.NodeOf("Transpose", "last", [$"{Converted}re_lu_1/Relu:0"], "lasted");
            last.SetWholes("perm", 0, 2, 3, 1);
            model.Graph.Node.Insert(model.Graph.Node.IndexOf(model.Node($"{Converted}conv2d_1_2/BiasAdd")), last);
            model.Node($"{Converted}conv2d_1_2/BiasAdd").Input[0] = "lasted";
        }, $"node '{Converted}conv2d_1_2/BiasAdd' (Conv): it takes images with their channels after the batch, as ONNX's Conv does, and the images reaching it have them last."),
        ["a batch normalisation of images with their channels last"] = new(TensorFlow, model => AfterTheLastTranspose(model, "BatchNormalization", 4), "node 'normalised' (BatchNormalization): it normalises along the axis after the batch, the rows of the images reaching it, whose channels come last: a batch normalisation here normalises each channel."),
        ["a layer normalisation of images with their channels last"] = new(TensorFlow, model => AfterTheLastTranspose(model, "LayerNormalization", 2), "node 'normalised' (LayerNormalization): it normalises an image, and a layer normalisation here normalises the values of a row."),

        // A convolution.
        ["a kernel for no window"] = new(Images, model =>
        {
            model.Initializer("onnx::Conv_29").Dims.Clear();
            model.Initializer("onnx::Conv_29").Dims.Add([4, 2]);
        }, $"node '/0/Conv' (Conv): its kernel, 'onnx::Conv_29', is written as 4x2, {NotAWindow}"),
        ["a kernel for a window of four axes"] = new(Images, model =>
        {
            model.Initializer("onnx::Conv_29").Dims.Clear();
            model.Initializer("onnx::Conv_29").Dims.Add([4, 2, 1, 1, 1, 1]);
        }, $"node '/0/Conv' (Conv): its kernel, 'onnx::Conv_29', is written as 4x2x1x1x1x1, {NotAWindow}"),
        ["a kernel along a series with the settings of an image"] = new(Images, model =>
        {
            model.Initializer("onnx::Conv_29").Dims.Clear();
            model.Initializer("onnx::Conv_29").Dims.Add([4, 2, 6]);
        }, string.Join(
            Environment.NewLine,
            "node '/0/Conv' (Conv): its 'strides' is written as [1, 1], and a convolution along a series writes one there.",
            "node '/0/Conv' (Conv): its 'dilations' is written as [1, 1], and a convolution along a series writes one there.",
            $"node '/0/Conv' (Conv): it pads [1, 1, 1, 1] — steps before, then after — {Pads}")),
        ["a kernel along a series for the images reaching it"] = new(Images, model =>
        {
            model.Initializer("onnx::Conv_29").Dims.Clear();
            model.Initializer("onnx::Conv_29").Dims.Add([4, 2, 6]);
            model.Node("/0/Conv").SetWholes("strides", 1);
            model.Node("/0/Conv").SetWholes("dilations", 1);
            model.Node("/0/Conv").SetWholes("pads", 1, 1);
        }, "graph 'main_graph': This network cannot be lowered at 8x6x2: word 1, conv1d, takes each example as a series of steps and channels, and each reaching it is 8x6x2. (Parameter 'input')"),
        ["two strides for a volume"] = new(VolumeNet, model => model.Node(FirstConv).SetWholes("strides", 1, 1), $"node '{FirstConv}' (Conv): its 'strides' is written as [1, 1], and a convolution through a volume writes three there."),
        ["pads through a volume that are not 'same' for the volume"] = new(VolumeBorders, model => model.Node("averagepool_3").SetWholes("pads", 1, 0, 1, 1, 1, 2), "node 'averagepool_3' (AveragePool): it pads [1, 0, 1, 1, 1, 2] — planes, rows and columns before, then after — where TensorFlow's 'same' pads the 5x4x3 volumes reaching it [1, 0, 1, 1, 1, 1], and a window here pads every side alike, or as that 'same'."),
        ["one stride for two axes"] = new(Images, model => model.Node("/0/Conv").SetWholes("strides", 1), "node '/0/Conv' (Conv): its 'strides' is written as [1], and a convolution over an image writes a pair there."),
        ["a stride too long"] = new(Images, model => model.Node("/0/Conv").SetWholes("strides", 4294967296, 1), "node '/0/Conv' (Conv): its 'strides' is written as [4294967296, 1], which is not what ONNX writes there."),
        ["two pads for four sides"] = new(Images, model => model.Node("/0/Conv").SetWholes("pads", 1, 1), $"node '/0/Conv' (Conv): it pads [1, 1] — rows and columns before, then after — {Pads}"),
        ["the odd pixel before"] = new(Images, model => model.Node("/0/Conv").SetText("auto_pad", "SAME_LOWER"), $"node '/0/Conv' (Conv): it pads as SAME_LOWER, {Pads}"),
        ["pads at a stride of two that are not 'same' for the image"] = new(Images, model => model.Node("/3/Conv").SetWholes("pads", 0, 0, 1, 1), "node '/3/Conv' (Conv): it pads [0, 0, 1, 1] — rows and columns before, then after — where TensorFlow's 'same' pads the 8x7 images reaching it [0, 0, 0, 1], and a window here pads every side alike, or as that 'same'."),
        ["pads at a stride of two that are no 'same' at all"] = new(Images, model => model.Node("/3/Conv").SetWholes("pads", 0, 0, 2, 0), $"node '/3/Conv' (Conv): it pads [0, 0, 2, 0] — rows and columns before, then after — {Pads}"),
        ["a convolution without its bias"] = new(Images, model => model.Node("/0/Conv").Input.RemoveAt(2), "node '/0/Conv' (Conv): it adds no bias, and a dense layer or a convolution here always adds one."),
        ["a window that stands nowhere"] = new(Images, model => model.Node("/0/Conv").SetWholes("strides", 0, 0), "node '/0/Conv' (Conv): A window 3x2 (stride 0, padding 1) cannot stand anywhere: its sides and its stride are at least one, and its border at least nothing. (Parameter 'window')"),
        ["a window larger than the image"] = new(Images, model =>
        {
            model.Initializer("onnx::Conv_29").Dims.Clear();
            model.Initializer("onnx::Conv_29").Dims.Add([4, 2, 11, 2]);
        }, "graph 'main_graph': This network cannot be lowered at 8x6x2: word 1, conv2d, slides a window 11x2 (stride 1, padding 1) over each example, and an example 8x6x2 is smaller than it. (Parameter 'input')"),

        // A convolution along a series, and through a volume.
        ["a convolution along a series dilated"] = new(SeriesNet, model => model.Node(FirstConv).SetWholes("dilations", 2), $"node '{FirstConv}' (Conv): its window is dilated by 2, {Neighbours}"),
        ["a convolution along a series in groups"] = new(SeriesNet, model => model.Node(FirstConv).SetWhole("group", 3), $"node '{FirstConv}' (Conv): it splits its channels into 3 groups, and a convolution here takes every channel of a place at once."),
        ["two strides for a series"] = new(SeriesNet, model => model.Node(FirstConv).SetWholes("strides", 2, 2), $"node '{FirstConv}' (Conv): its 'strides' is written as [2, 2], and a convolution along a series writes one there."),
        ["two dilations for a series"] = new(SeriesNet, model => model.Node(FirstConv).SetWholes("dilations", 1, 1), $"node '{FirstConv}' (Conv): its 'dilations' is written as [1, 1], and a convolution along a series writes one there."),
        ["one pad for a series"] = new(SeriesNet, model => model.Node(FirstConv).SetWholes("pads", 1), $"node '{FirstConv}' (Conv): it pads [1] — steps before, then after — {Pads}"),
        ["pads at a stride of two that are no 'same' at all for a series"] = new(SeriesNet, model => model.Node(FirstConv).SetWholes("pads", 0, 2), $"node '{FirstConv}' (Conv): it pads [0, 2] — steps before, then after — {Pads}"),
        ["pads at a stride of two that are not 'same' for the series"] = new(SeriesBorders, model => model.Node("averagepool_3").SetWholes("pads", 1, 2), "node 'averagepool_3' (AveragePool): it pads [1, 2] — steps before, then after — where TensorFlow's 'same' pads the 12 series reaching it [0, 1], and a window here pads every side alike, or as that 'same'."),
        ["a convolution through a volume striding otherwise on each axis"] = new(VolumeNet, model => model.Node(FirstConv).SetWholes("strides", 2, 1, 1), $"node '{FirstConv}' (Conv): it strides 2 deep, 1 down and 1 across, and a window here walks one stride deep, down and across alike."),
        ["a convolution through a volume dilated"] = new(VolumeNet, model => model.Node(FirstConv).SetWholes("dilations", 1, 2, 1), $"node '{FirstConv}' (Conv): its window is dilated by 1 deep, 2 down and 1 across, {Neighbours}"),
        ["a convolution through a volume padded otherwise than alike or as 'same'"] = new(VolumeNet, model => model.Node(FirstConv).SetWholes("pads", 0, 1, 2, 1, 1, 2), $"node '{FirstConv}' (Conv): it pads [0, 1, 2, 1, 1, 2] — planes, rows and columns before, then after — {Pads}"),
        ["pads through a volume at a stride of one that are not 'same'"] = new(VolumeNet, model => model.Node(LastConv).SetWholes("pads", 0, 0, 0, 1, 0, 0), $"node '{LastConv}' (Conv): it pads [0, 0, 0, 1, 0, 0] — planes, rows and columns before, then after — {Pads}"),

        // Poolings.
        ["a max pooling that rounds up"] = new(ImageNet, model => model.Node(MaxPooling).SetWhole("ceil_mode", 1), $"node '{MaxPooling}' (MaxPool): {Overhang}"),
        ["an average pooling that rounds up"] = new(SeriesNet, model => model.Node(AveragePooling).SetWhole("ceil_mode", 1), $"node '{AveragePooling}' (AveragePool): {Overhang}"),
        ["a max pooling through a volume dilated"] = new(VolumeNet, model => model.Node("/0/0.3/MaxPool").SetWholes("dilations", 2, 2, 2), $"node '/0/0.3/MaxPool' (MaxPool): its window is dilated by 2 deep, 2 down and 2 across, {Neighbours}"),
        ["an average pooling over an image dilated"] = new(ImageNet, model => model.Node(AveragePooling).SetWholes("dilations", 1, 2), $"node '{AveragePooling}' (AveragePool): its window is dilated by 1 down and 2 across, {Neighbours}"),
        ["a max pooling striding otherwise down than across"] = new(ImageNet, model => model.Node(MaxPooling).SetWholes("strides", 2, 1), $"node '{MaxPooling}' (MaxPool): it strides 2 down and 1 across, and a window here walks one stride down and across alike."),
        ["an average pooling through a volume striding otherwise on each axis"] = new(VolumeNet, model => model.Node("/0/0.6/AveragePool").SetWholes("strides", 2, 1, 2), "node '/0/0.6/AveragePool' (AveragePool): it strides 2 deep, 1 down and 2 across, and a window here walks one stride deep, down and across alike."),
        ["a max pooling that numbers the places of its largest values by columns"] = new(SeriesDefault, model => model.Node("node_max_pool1d").SetWhole("storage_order", 1), $"node 'node_max_pool1d' (MaxPool): it numbers the places of its largest values column by column, storage_order 1, {Largest}"),
        ["a max pooling whose places of the largest values are used"] = new(SeriesNet, model =>
        {
            model.Node("/0/0.3/MaxPool").Output.Add("indices");
            model.Graph.Output.Add(new ValueInfoProto { Name = "indices" });
        }, $"node '/0/0.3/MaxPool' (MaxPool): it gives the places of its largest values as a second value, 'indices', which the graph uses, {Largest}{Environment.NewLine}graph 'main_graph': it gives 2 values, and a network here gives one."),
        ["a pooling with no window"] = new(ImageNet, model => model.Node(MaxPooling).Unset("kernel_shape"), $"node '{MaxPooling}' (MaxPool): its 'kernel_shape' is written as [], and a pooling here slides a window along one, two or three axes."),
        ["a pooling with a window of four axes"] = new(ImageNet, model => model.Node(MaxPooling).SetWholes("kernel_shape", 2, 2, 2, 2), $"node '{MaxPooling}' (MaxPool): its 'kernel_shape' is written as [2, 2, 2, 2], and a pooling here slides a window along one, two or three axes."),
        ["two strides for a series pooling"] = new(SeriesNet, model => model.Node("/0/0.3/MaxPool").SetWholes("strides", 2, 2), "node '/0/0.3/MaxPool' (MaxPool): its 'strides' is written as [2, 2], and a pooling along a series writes one there."),
        ["a pooling padded from the end at a stride of two over an even length"] = new(ImageNet, model => model.Node(MaxPooling).SetText("auto_pad", "SAME_LOWER"), $"node '{MaxPooling}' (MaxPool): it pads as SAME_LOWER, [1, 1, 0, 0] — rows and columns before, then after — where TensorFlow's 'same' pads the 12x10 images reaching it [0, 0, 1, 1], and a window here pads every side alike, or as that 'same'."),
        ["a pooling padded as no one pads"] = new(ImageNet, model => model.Node(MaxPooling).SetText("auto_pad", "SIDEWAYS"), $"node '{MaxPooling}' (MaxPool): it pads as SIDEWAYS, {Pads}"),
        ["a pooling through a volume padded from the end at a stride of two over an even length"] = new(VolumeBorders, model => model.Node("averagepool_1").SetText("auto_pad", "SAME_LOWER"), "node 'averagepool_1' (AveragePool): it pads as SAME_LOWER, [1, 1, 1, 1, 0, 0] — planes, rows and columns before, then after — where TensorFlow's 'same' pads the 9x8x6 volumes reaching it [1, 0, 0, 1, 1, 1], and a window here pads every side alike, or as that 'same'."),
        ["a pooling window that stands nowhere"] = new(ImageNet, model => model.Node(MaxPooling).SetWholes("strides", 0, 0), $"node '{MaxPooling}' (MaxPool): A window 3x3 (stride 0, padding 1) cannot stand anywhere: its sides and its stride are at least one, and its border at least nothing. (Parameter 'window')"),
        ["a pooling of images with their channels last"] = new(TensorFlow, model => Pooled(model, "MaxPool"), "node 'pooled' (MaxPool): it takes images with their channels after the batch, as ONNX's MaxPool does, and the images reaching it have them last."),
        ["an average pooling of images with their channels last"] = new(TensorFlow, model => Pooled(model, "AveragePool"), "node 'pooled' (AveragePool): it takes images with their channels after the batch, as ONNX's AveragePool does, and the images reaching it have them last."),
        ["a global pooling of images with their channels last"] = new(TensorFlow, model => Pooled(model, "GlobalAveragePool"), "node 'pooled' (GlobalAveragePool): it takes images with their channels after the batch, as ONNX's GlobalAveragePool does, and the images reaching it have them last."),
        ["a global pooling of rows"] = new(Titanic, model => model.Node("/1/Relu").OpType = "GlobalMaxPool", $"node '/1/Relu' (GlobalMaxPool): {TakesImages("GlobalMaxPool")} a row of values."),
        ["a pooling of rows"] = new(Titanic, model =>
        {
            model.Node("/1/Relu").OpType = "MaxPool";
            model.Node("/1/Relu").SetWholes("kernel_shape", 2);
        }, $"node '/1/Relu' (MaxPool): {TakesImages("MaxPool")} a row of values."),
        ["a convolution of rows"] = new(Titanic, model => model.Node("/1/Relu").OpType = "Conv", $"node '/1/Relu' (Conv): {TakesImages("Conv")} a row of values."),
        ["a pooling of images flattened into rows"] = new(Images, model =>
        {
            model.Node("/5/BatchNormalization").OpType = "AveragePool";
            model.Node("/5/BatchNormalization").SetWholes("kernel_shape", 2);
        }, $"node '/5/BatchNormalization' (AveragePool): {TakesImages("AveragePool")} an image flattened into a row."),

        // A ReduceMean or a ReduceMax, which is read as the pooling of every place of each channel into one and as nothing else.
        ["a reduce over the channels"] = new(ImageReduced, model => model.SetAxes("node_mean", 1), $"node 'node_mean' (ReduceMean): {Reduces("[1]", 4, "2, 3")}"),
        ["a reduce over some of the places only"] = new(ImageLargestReduced, model => model.SetAxes("n0", -1), $"node 'n0' (ReduceMax): {Reduces("[-1]", 4, "2, 3")}"),
        ["a reduce over the batch and the places"] = new(ImageReduced, model => model.SetAxes("node_mean", 0, 2, 3), $"node 'node_mean' (ReduceMean): {Reduces("[0, 2, 3]", 4, "2, 3")}"),
        ["a reduce over the rows of a volume only"] = new(VolumeReduced, model => model.SetAxes("node_mean", 3), $"node 'node_mean' (ReduceMean): {Reduces("[3]", 5, "2, 3, 4")}"),
        ["a reduce over an axis twice"] = new(ImageReduced, model => model.SetAxes("node_mean", 2, 2, 3), $"node 'node_mean' (ReduceMean): {Reduces("[2, 2, 3]", 4, "2, 3")}"),
        ["a reduce that names no axes"] = new(ImageReduced, model => model.Node("node_mean").Input.RemoveAt(1), "node 'node_mean' (ReduceMean): it reduces over every axis when it names none, the batch and the channels among them, and a global pooling here reduces over the places of each channel alone."),
        ["a reduce that names no axes and does nothing"] = new(ImageReduced, model =>
        {
            model.Node("node_mean").Input.RemoveAt(1);
            model.Node("node_mean").SetWhole("noop_with_empty_axes", 1);
        }, "node 'node_mean' (ReduceMean): it reduces over nothing when it names no axes, noop_with_empty_axes 1, and a global pooling here reduces over the places of each channel."),
        ["a reduce that names its axes twice"] = new(ImageReduced, model => model.Node("node_mean").SetWholes("axes", 2, 3), "node 'node_mean' (ReduceMean): it says its axes as an input, 'val_12', and as an attribute, and ONNX writes them one way or the other."),
        ["a reduce whose axes are numbers"] = new(ImageReduced, model => model.Graph.Initializer[model.Graph.Initializer.IndexOf(model.Initializer("val_12"))] = OnnxFixtures.Floats("val_12", [2], 2, 3), "node 'node_mean' (ReduceMean): its axes, 'val_12', is written as FLOAT, and ONNX writes one as INT64."),
        ["a reduce whose axis is too long"] = new(ImageReduced, model => model.SetAxes("node_mean", 4294967296, -2), "node 'node_mean' (ReduceMean): its axes, 'val_12', are written as [4294967296, -2], which is not what ONNX writes there."),
        ["a reduce of rows"] = new(Titanic, model => model.Node("/1/Relu").OpType = "ReduceMean", $"node '/1/Relu' (ReduceMean): {TakesImages("ReduceMean")} a row of values."),
        ["a reduce of another kind"] = new(ImageReduced, model => model.Node("node_mean").OpType = "ReduceSum", $"node 'node_mean' (ReduceSum): 'ReduceSum' is no operator a network here is built of: {Operators}"),

        // The Unsqueeze and the Squeeze the default exporter writes round the pooling of a series are read as that pooling and as nothing else.
        ["an unsqueeze of a series that no reduce follows"] = new(SeriesReduced, model =>
        {
            var reduce = model.Node("node_mean");
            model.Graph.Node.Insert(model.Graph.Node.IndexOf(reduce), OnnxFixtures.NodeOf("Relu", "relu", ["unsqueeze"], "relued"));
            reduce.Input[0] = "relued";
        }, string.Join(Environment.NewLine, Unsqueezed, Reduces("node_mean", "ReduceMean", "[-1, -2]", 3, "2"), Squeezed)),
        ["an unsqueeze along another axis"] = new(SeriesReduced, model => model.SetAxes("node_unsqueeze", 3), string.Join(Environment.NewLine, Unsqueezed, Reduces("node_mean", "ReduceMean", "[-1, -2]", 3, "2"), Squeezed)),
        ["a squeeze along another axis"] = new(SeriesReduced, model =>
        {
            model.Graph.Initializer.Add(OnnxFixtures.Wholes("elsewhere", 3));
            model.Node("node_squeeze").Input[1] = "elsewhere";
        }, string.Join(Environment.NewLine, Unsqueezed, Reduces("node_mean", "ReduceMean", "[-1, -2]", 3, "2"), Squeezed)),
        ["a pooling of a series between an unsqueeze and a squeeze that drops its axes"] = new(SeriesReduced, model => model.Node("node_mean").SetWhole("keepdims", 0), string.Join(Environment.NewLine, Unsqueezed, Reduces("node_mean", "ReduceMean", "[-1, -2]", 3, "2"), Squeezed)),
        ["an unsqueezed value the graph gives too"] = new(SeriesReduced, model => model.Graph.Output.Add(new ValueInfoProto { Name = "unsqueeze" }), string.Join(Environment.NewLine, Unsqueezed, Reduces("node_mean", "ReduceMean", "[-1, -2]", 3, "2"), Squeezed, "graph 'main_graph': it gives 2 values, and a network here gives one.")),
        ["a reduce that takes another value than the unsqueeze made"] = new(SeriesReduced, model => model.Node("node_mean").Input[0] = "conv1d_1", string.Join(Environment.NewLine, Unsqueezed, $"node 'node_mean' (ReduceMean): it takes 'conv1d_1', {Stack}, 'unsqueeze'.", Squeezed)),
        ["a squeeze that takes another value than the reduce made"] = new(SeriesReduced, model => model.Node("node_squeeze").Input[0] = "unsqueeze", string.Join(Environment.NewLine, Unsqueezed, Reduces("node_mean", "ReduceMean", "[-1, -2]", 3, "2"), $"node 'node_squeeze' (Squeeze): it takes 'unsqueeze', {Stack}, 'mean'.")),
        ["a pooling of a series with no squeeze after it"] = new(SeriesReduced, model => model.Node("node_squeeze").OpType = "Relu", string.Join(Environment.NewLine, Unsqueezed, Reduces("node_mean", "ReduceMean", "[-1, -2]", 3, "2"))),
        ["a reduced value the graph gives too"] = new(SeriesReduced, model => model.Graph.Output.Add(new ValueInfoProto { Name = "mean" }), string.Join(Environment.NewLine, Unsqueezed, Reduces("node_mean", "ReduceMean", "[-1, -2]", 3, "2"), Squeezed, "graph 'main_graph': it gives 2 values, and a network here gives one.")),
        ["a pooling of a series cut off before its squeeze"] = new(SeriesReduced, model =>
        {
            while (model.Graph.Node[^1].Name != "node_mean")
            {
                model.Graph.Node.RemoveAt(model.Graph.Node.Count - 1);
            }

            model.Graph.Output[0].Name = "mean";
        }, string.Join(Environment.NewLine, Unsqueezed, Reduces("node_mean", "ReduceMean", "[-1, -2]", 3, "2"))),
        ["an unsqueeze and a squeeze round the pooling of an image"] = new(ImageReduced, model =>
        {
            var reduce = model.Node("node_mean");
            var at = model.Graph.Node.IndexOf(reduce);
            model.Graph.Initializer.Add(OnnxFixtures.Wholes("minus_two", -2));
            model.Graph.Node.Insert(at, OnnxFixtures.NodeOf("Unsqueeze", "node_unsqueeze", ["conv2d_1", "minus_two"], "lifted"));
            reduce.Input[0] = "lifted";
            model.Graph.Node.Insert(at + 2, OnnxFixtures.NodeOf("Squeeze", "node_squeeze", ["mean", "minus_two"], "dropped"));
            model.Node("node_Reshape_18").Input[0] = "dropped";
        }, string.Join(Environment.NewLine, Unsqueezed, Squeezed)),

        // A transpose of a series, which is no image.
        ["a transpose of a series"] = new(SeriesNet, model =>
        {
            var turn = OnnxFixtures.NodeOf("Transpose", "turn", ["x"], "turned");
            turn.SetWholes("perm", 0, 3, 1, 2);
            model.Graph.Node.Insert(0, turn);
            model.Node(FirstConv).Input[0] = "turned";
        }, "node 'turn' (Transpose): it turns its value round by [0, 3, 1, 2], and a Transpose is read here as one that moves an image's channels after the batch, [0, 3, 1, 2], or last, [0, 2, 3, 1]."),
        ["a transpose of a series to channels last"] = new(SeriesNet, model =>
        {
            var turn = OnnxFixtures.NodeOf("Transpose", "turn", ["x"], "turned");
            turn.SetWholes("perm", 0, 2, 3, 1);
            model.Graph.Node.Insert(0, turn);
            model.Node(FirstConv).Input[0] = "turned";
        }, "node 'turn' (Transpose): it turns its value round by [0, 2, 3, 1], and a Transpose is read here as one that moves an image's channels after the batch, [0, 3, 1, 2], or last, [0, 2, 3, 1]."),

        // The normalisations.
        ["a batch normalisation in training"] = new(Images, model => model.Node("/5/BatchNormalization").SetWhole("training_mode", 1), "node '/5/BatchNormalization' (BatchNormalization): it measures each batch as it answers, in training mode, and a batch normalisation here answers from its running statistics."),
        ["a batch normalisation of every value"] = new(Images, model => model.Node("/5/BatchNormalization").SetWhole("spatial", 0), "node '/5/BatchNormalization' (BatchNormalization): it normalises every value on its own, spatial 0, and a batch normalisation here normalises each feature, or each channel of an image."),
        ["a momentum that is no share"] = new(Images, model => model.Node("/5/BatchNormalization").SetNumber("momentum", 2), "node '/5/BatchNormalization' (BatchNormalization): Keras's momentum is the share of the running statistics a training batch leaves as they were, between nothing and one. (Parameter 'momentum') Actual value was 2."),
        ["a layer normalisation of an image"] = new(Images, model =>
        {
            model.Graph.Initializer.Add(OnnxFixtures.Floats("scale", [4], 1, 1, 1, 1));
            model.Graph.Initializer.Add(OnnxFixtures.Floats("shift", [4], 0, 0, 0, 0));
            var node = model.Node("/2/Relu");
            node.OpType = "LayerNormalization";
            node.Input.Add(["scale", "shift"]);
        }, "node '/2/Relu' (LayerNormalization): it normalises an image, and a layer normalisation here normalises the values of a row."),
        ["a layer normalisation from another axis"] = new(Titanic, model => Normalised(model, 3).SetWhole("axis", 0), "node '/1/Relu' (LayerNormalization): it normalises from axis 0, and a layer normalisation here normalises over the last axis of a row."),
        ["a layer normalisation without its shift"] = new(Titanic, model => Normalised(model, 2), "node '/1/Relu' (LayerNormalization): it leaves out its shift, and a layer normalisation here learns one."),
        ["an epsilon of nothing"] = new(Titanic, model => Normalised(model, 3).SetNumber("epsilon", 0), "node '/1/Relu' (LayerNormalization): What is added to a variance is a number above nothing. (Parameter 'epsilon') Actual value was 0."),

        // A flatten, and a reshape.
        ["a flatten from another axis"] = new(Images, model => model.Node("/4/Flatten").SetWhole("axis", 2), "node '/4/Flatten' (Flatten): it flattens from axis 2, and a flatten here keeps each example as one row."),
        ["a reshape into an image"] = new(Images, model => Reshaped(model, OnnxFixtures.Wholes("target", 6, 3, 12)), $"node '/4/Flatten' (Reshape): it lays each batch out as [6, 3, 12], {NotFlattened}"),
        ["a reshape into another batch"] = new(Images, model => Reshaped(model, OnnxFixtures.Wholes("target", 7, 36)), $"node '/4/Flatten' (Reshape): it lays each batch out as [7, 36], {NotFlattened}"),
        ["a reshape into no rows"] = new(Images, model => Reshaped(model, OnnxFixtures.Wholes("target", 0, 36)).SetWhole("allowzero", 1), $"node '/4/Flatten' (Reshape): it lays each batch out as [0, 36], {NotFlattened}"),
        ["a reshape leaving both open"] = new(Images, model => Reshaped(model, OnnxFixtures.Wholes("target", -1, -1)), $"node '/4/Flatten' (Reshape): it lays each batch out as [-1, -1], {NotFlattened}"),
        ["a reshape into rows of nothing"] = new(Images, model => Reshaped(model, OnnxFixtures.Wholes("target", -1, 0)), $"node '/4/Flatten' (Reshape): it lays each batch out as [-1, 0], {NotFlattened}"),
        ["a reshape without its target"] = new(Images, model =>
        {
            var node = model.Node("/4/Flatten");
            node.OpType = "Reshape";
            node.Unset("axis");
        }, "node '/4/Flatten' (Reshape): its target, '', is no value the graph holds."),
        ["a target of lengths no tensor holds"] = new(Images, model =>
        {
            var target = OnnxFixtures.Wholes("target", -1, 36);
            target.Dims[0] = -2;
            Reshaped(model, target);
        }, "node '/4/Flatten' (Reshape): its target, 'target', is written with the lengths [-2], which no tensor here holds."),
        ["a target in its typed field cut short"] = new(Images, model =>
        {
            var target = OnnxFixtures.Wholes("target", -1, 36);
            target.RawData = ByteString.Empty;
            target.Int64Data.Add(-1);
            Reshaped(model, target);
        }, "node '/4/Flatten' (Reshape): its target, 'target', is written as a 2 tensor of INT64, and the graph holds 1 numbers for it, where it takes 2."),
        ["a constant of a name the graph holds"] = new(Images, model =>
        {
            var constant = OnnxFixtures.NodeOf("Constant", "again", [], "5.bias");
            constant.Attribute.Add(new AttributeProto { Name = "value", Type = AttributeProto.Types.AttributeType.Tensor, T = OnnxFixtures.Floats(string.Empty, [36], new float[36]) });
            model.Graph.Node.Insert(0, constant);
        }, "node 'again' (Constant): it gives '5.bias', and the graph holds a value of that name already."),
        ["a constant of a value of another kind"] = new(Images, model =>
        {
            var constant = OnnxFixtures.NodeOf("Constant", "shape", [], "target");
            constant.SetWholes("value", -1, 36);
            model.Graph.Node.Insert(0, constant);

            var node = model.Node("/4/Flatten");
            node.OpType = "Reshape";
            node.Unset("axis");
            node.Input.Add("target");
        }, $"node 'shape' (Constant): its 'value' is written as INTS, which is not what ONNX writes there.{Environment.NewLine}node '/4/Flatten' (Reshape): it takes '/3/Conv_output_0' and 'target', 2 values the graph works out, {Stack}, '/3/Conv_output_0'."),
        ["a reshape into rows of another length"] = new(Images, model => Reshaped(model, OnnxFixtures.Wholes("target", -1, 30)), "node '/4/Flatten' (Reshape): it lays each example out as a row of 30 values, and each example reaching it holds 36."),
        ["a target of numbers"] = new(Images, model => Reshaped(model, OnnxFixtures.Floats("target", [2], -1, 36)), "node '/4/Flatten' (Reshape): its target, 'target', is written as FLOAT, and ONNX writes one as INT64."),
        ["a target cut short"] = new(Images, model =>
        {
            var target = OnnxFixtures.Wholes("target", -1, 36);
            target.RawData = ByteString.CopyFrom(target.RawData.Span[..12]);
            Reshaped(model, target);
        }, "node '/4/Flatten' (Reshape): its target, 'target', is written as a 2 tensor of INT64, and the graph holds 12 bytes for it, where it takes 16."),
        ["a constant of whole numbers"] = new(Images, model =>
        {
            var constant = OnnxFixtures.NodeOf("Constant", "shape", [], "target");
            var value = new AttributeProto { Name = "value_ints", Type = AttributeProto.Types.AttributeType.Ints };
            value.Ints.Add([-1, 36]);
            constant.Attribute.Add(value);
            model.Graph.Node.Insert(0, constant);

            var node = model.Node("/4/Flatten");
            node.OpType = "Reshape";
            node.Unset("axis");
            node.Input.Add("target");
        }, $"node 'shape' (Constant): it holds its value as 'value_ints', and a constant is read here when it holds a tensor, 'value'.{Environment.NewLine}node '/4/Flatten' (Reshape): it takes '/3/Conv_output_0' and 'target', 2 values the graph works out, {Stack}, '/3/Conv_output_0'."),

        // A softmax.
        ["a softmax within"] = new(Titanic, model => model.Node("/1/Relu").OpType = "Softmax", $"node '/1/Relu' (Softmax): {SoftmaxLeft}"),
        ["a softmax before a binary cross-entropy"] = new(Titanic, model => Softened(model, 1), $"node 'softmax' (Softmax): {SoftmaxLeft}"),
        ["a softmax along another axis"] = new(Titanic, model => Softened(model, 0), "node 'softmax' (Softmax): it takes shares along axis 0, and a softmax is read here over the values of a row.") { Loss = () => new CrossEntropy() },
        ["a softmax over an image"] = new(Images, model =>
        {
            while (model.Graph.Node[^1].Name != "/0/Conv")
            {
                model.Graph.Node.RemoveAt(model.Graph.Node.Count - 1);
            }

            model.Graph.Node.Add(OnnxFixtures.NodeOf("Softmax", "softmax", ["/0/Conv_output_0"], "shares"));
            model.Graph.Output[0].Name = "shares";
        }, "node 'softmax' (Softmax): it takes shares over an image, and a softmax is read here over the values of a row.") { Loss = () => new CrossEntropy() },
    };

    public static TheoryData<string> Named => new(Cases.Keys);

    [Fact]
    public void AStrideForEachAxis_ADilatedWindow_ChannelsInGroups_APaddingForEachAxis_APoolingThatRoundsUp_AndALeakyRelu_AreEachRefusedAtTheirNode_AllAtOnce()
    {
        using var graph = OnnxFixtures.Open("onnx-refused.onnx");
        var refused = Assert.Throws<FormatException>(() => new OnnxFile(new BinaryCrossEntropy()).Read(graph));

        Assert.Equal(
            [
                "node 'node_conv2d' (Conv): it strides 2 down and 1 across, and a window here walks one stride down and across alike.",
                "node 'node_conv2d_1' (Conv): its window is dilated by 2 down and 2 across, and a window here covers neighbouring places.",
                "node 'node_conv2d_2' (Conv): it splits its channels into 2 groups, and a convolution here takes every channel of a place at once.",
                $"node 'node_conv2d_3' (Conv): it pads [1, 0, 1, 0] — rows and columns before, then after — {Pads}",
                $"node 'node_max_pool2d' (MaxPool): {Overhang}",
                $"node 'node_leaky_relu' (LeakyRelu): 'LeakyRelu' is no operator a network here is built of: {Operators}",
            ],
            refused.Message.Split(Environment.NewLine));
    }

    [Fact]
    public void WhatTfToOnnxWritesOfBatchNormalisations_AndOfAFlattenForBatchesOfAnyLength_IsRefusedAtEachNode_AllAtOnce()
    {
        using var graph = OnnxFixtures.Open("tf2onnx-convolution-batches.onnx");
        var refused = Assert.Throws<FormatException>(() => new OnnxFile(new MeanSquaredError()).Read(graph));

        Assert.Equal(BatchesRefusal, refused.Message.Split(Environment.NewLine));
    }

    [Fact]
    public void PadsAConvolutionWritesOutAtAStrideOfTwo_ThatAreSameForTheImage_AreReadAsSame()
    {
        // 'same' at a stride of two pads the 8x7 images reaching the second convolution one column after them, so it makes
        // images of 4x4 where the graph's numbers were written for 4x3: the pads are read, and the numbers after them refused.
        var refused = Assert.Throws<SlotLoadException>(() => new OnnxFile(new MeanSquaredError()).Read(OnnxFixtures.Edited(Images, model => model.Node("/3/Conv").SetWholes("pads", 0, 0, 0, 1))));

        Assert.Equal(
            [
                "'4.weight' is a 48 slot here, and is written as 36.",
                "'4.bias' is a 48 slot here, and is written as 36.",
                "'4.running_mean' is a 48 slot here, and is written as 36.",
                "'4.running_var' is a 48 slot here, and is written as 36.",
                "'5.weight' is a 48x5 slot here, and is written as 36x5.",
            ],
            refused.Faults.Select(fault => fault.Message));
    }

    [Fact]
    public void ALayerWhoseInputIsAddedBackAfterIt_IsABranch_RefusedAtTheNodeWherePathsMeet()
    {
        using var graph = OnnxFixtures.Open("onnx-branch.onnx");
        var refused = Assert.Throws<FormatException>(() => new OnnxFile(new BinaryCrossEntropy()).Read(graph));

        Assert.Equal($"node 'node_add_6' (Add): it takes 'relu' and 'passengers', 2 values the graph works out, {Stack}, 'relu'.", refused.Message);
    }

    [Theory]
    [MemberData(nameof(Named))]
    public void WhatAGraphSays_ThatNoNetworkHereIsBuiltOf_IsRefusedWhereItSaysIt(string name)
    {
        var said = Cases[name];
        var loss = said.Loss?.Invoke() ?? (said.File is Titanic or Keras ? new BinaryCrossEntropy() : (Loss)new MeanSquaredError());

        var refused = Assert.Throws<FormatException>(() => new OnnxFile(loss).Read(OnnxFixtures.Edited(said.File, said.Edit)));

        Assert.Equal(said.Refusal, refused.Message);
    }

    [Fact]
    public void AForgedOperatorName_HoldingALineBreak_IsShownEscaped_NeverBreakingTheMessageIntoASecondLine()
    {
        var refused = Assert.Throws<FormatException>(() => new OnnxFile(new BinaryCrossEntropy()).Read(
            OnnxFixtures.Edited(Titanic, model => model.Node("/1/Relu").OpType = "Elu\r\nFAKE LINE")));

        Assert.Single(refused.Message.Split(Environment.NewLine));
        Assert.Contains("Elu\\r\\nFAKE LINE", refused.Message, StringComparison.Ordinal);
        Assert.DoesNotContain('\r', refused.Message);
        Assert.DoesNotContain('\n', refused.Message);
    }

    [Fact]
    public void AForgedOperatorName_OfExtremeLength_IsCutShortInTheRefusal_SayingHowLongItWas()
    {
        var huge = new string('m', 10_000);

        var refused = Assert.Throws<FormatException>(() => new OnnxFile(new BinaryCrossEntropy()).Read(
            OnnxFixtures.Edited(Titanic, model => model.Node("/1/Relu").OpType = huge)));

        Assert.True(refused.Message.Length < 1000, $"The message is {refused.Message.Length} characters long.");
        Assert.Contains("(10000 characters)", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryFaultOfAGraph_IsNamedAtOnce()
    {
        var refused = Assert.Throws<FormatException>(() => new OnnxFile(new BinaryCrossEntropy()).Read(OnnxFixtures.Edited(Titanic, model =>
        {
            model.Graph.Input[0].Type.TensorType.ElemType = 11;
            model.Node("/1/Relu").OpType = "Elu";
            model.Graph.Output.Add(new ValueInfoProto { Name = "extra" });
        })));

        Assert.Equal(
            [
                "input 'passengers': it takes DOUBLE values, and a network here takes single-precision numbers, FLOAT.",
                $"node '/1/Relu' (Elu): 'Elu' is no operator a network here is built of: {Operators}",
                "graph 'main_graph': it gives 2 values, and a network here gives one.",
            ],
            refused.Message.Split(Environment.NewLine));
    }

    [Fact]
    public void NumbersThatDoNotFitTheLayersTheGraphBuilds_AreRefusedAtTheirInitializers_InTheLoadsOwnWords()
    {
        var refused = Assert.Throws<SlotLoadException>(() => new OnnxFile(new BinaryCrossEntropy()).Read(OnnxFixtures.Edited(Titanic, model =>
        {
            var bias = model.Initializer("0.bias");
            bias.Dims.Clear();
            bias.Dims.Add([1, 16]);

            var last = model.Initializer("2.bias");
            last.RawData = ByteString.CopyFrom(BitConverter.GetBytes(float.NaN));
        })));

        Assert.Equal(
            [
                new SlotLoadFault("initializer '0.bias'", "0.bias", "'0.bias' is a 16 slot here, and is written as 1x16."),
                new SlotLoadFault("initializer '2.bias'", "2.bias", "'2.bias' holds finite numbers, and its value at 0 is NaN."),
            ],
            refused.Faults);
    }

    [Fact]
    public void NumbersAlongAFlattenedImageOfOtherLengthsThanItsRow_AreRefusedAsTheyAreWritten_NotTurned()
    {
        var refused = Assert.Throws<SlotLoadException>(() => new OnnxFile(new MeanSquaredError()).Read(OnnxFixtures.Edited(Images, model =>
        {
            // The batch normalisation's scale a single number, and the dense layer's weights thirty inputs wide.
            var scale = model.Initializer("5.weight");
            scale.Dims.Clear();
            scale.RawData = ByteString.CopyFrom(scale.RawData.Span[..4]);

            var weights = model.Initializer("6.weight");
            weights.Dims[1] = 30;
            weights.RawData = ByteString.CopyFrom(weights.RawData.Span[..600]);
        })));

        Assert.Equal(
            [
                new SlotLoadFault("initializer '5.weight'", "4.weight", "'4.weight' is a 36 slot here, and is written as scalar."),
                new SlotLoadFault("initializer '6.weight'", "5.weight", "'5.weight' is a 36x5 slot here, and is written as 30x5."),
            ],
            refused.Faults);
    }

    [Theory]
    [InlineData("INT64","'2.bias' holds single-precision numbers here, and is written as INT64.")]
    [InlineData("short", "'2.bias' is written as a 1 tensor of FLOAT, and the graph holds 3 bytes for it, where it takes 4.")]
    [InlineData("none", "'2.bias' is written as a 1 tensor of FLOAT, and the graph holds 0 numbers for it, where it takes 1.")]
    [InlineData("scalar", "'2.bias' is written as a scalar tensor of FLOAT, and the graph holds 3 bytes for it, where it takes 4.")]
    [InlineData("negative", "'2.bias' is written with the lengths [-1], which no tensor here holds.")]
    [InlineData("too many", "'2.bias' is written with the lengths [65536, 65536], which no tensor here holds.")]
    public void NumbersTheGraphHoldsWrongly_AreRefusedAtTheirInitializer_AndNotAlsoAsMissing(string how, string refusal)
    {
        var refused = Assert.Throws<SlotLoadException>(() => new OnnxFile(new BinaryCrossEntropy()).Read(OnnxFixtures.Edited(Titanic, model =>
        {
            var bias = model.Initializer("2.bias");

            switch (how)
            {
                case "INT64":
                    bias.DataType = (int)TensorProto.Types.DataType.Int64;
                    break;
                case "short":
                    bias.RawData = ByteString.CopyFrom(bias.RawData.Span[..3]);
                    break;
                case "scalar":
                    bias.Dims.Clear();
                    bias.RawData = ByteString.CopyFrom(bias.RawData.Span[..3]);
                    break;
                case "none":
                    bias.RawData = ByteString.Empty;
                    break;
                case "negative":
                    bias.Dims[0] = -1;
                    break;
                default:
                    bias.Dims[0] = 65536;
                    bias.Dims.Add(65536);
                    break;
            }
        })));

        Assert.Equal([new SlotLoadFault("initializer '2.bias'", "2.bias", refusal)], refused.Faults);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NumbersKeptBesideAGraphReadFromAStreamThatIsNotItsFile_AreRefused_NamingTheFileAndTheWayOut(bool opened)
    {
        var path = Path.Join(OnnxFixtures.Folder, "onnx-titanic.onnx");

        // A stream of the graph's bytes, or its file opened by a handle the system gave, which knows no path.
        using var handle = File.OpenHandle(path);
        using var graph = opened
            ? new FileStream(new Microsoft.Win32.SafeHandles.SafeFileHandle(handle.DangerousGetHandle(), ownsHandle: false), FileAccess.Read)
            : (Stream)new MemoryStream(File.ReadAllBytes(path));

        Assert.False(Path.IsPathFullyQualified((graph as FileStream)?.Name ?? string.Empty));
        var refused = Assert.Throws<SlotLoadException>(() => new OnnxFile(new BinaryCrossEntropy()).Read(graph));

        Assert.Equal(
            [
                new SlotLoadFault(
                    "initializer '0.weight'",
                    "0.weight",
                    "'0.weight' is kept in 'onnx-titanic.onnx.data', a file beside the graph, and a graph read from a stream that is not its own file has no folder to find it in: read the graph from its file, or export it with external_data=False."),
            ],
            refused.Faults);
    }

    [Theory]
    [InlineData("location=../elsewhere.data", "'0.weight' is said to be kept in '../elsewhere.data', outside the folder the graph stands in, and numbers are read from beside the graph alone.")]
    [InlineData("location=/elsewhere.data", "'0.weight' is said to be kept in '/elsewhere.data', outside the folder the graph stands in, and numbers are read from beside the graph alone.")]
    [InlineData("location=missing.data", "'0.weight' is kept in 'missing.data', and no such file stands beside the graph.")]
    [InlineData("location=short.data|offset=0|length=896", "'0.weight' is kept in 'short.data' from byte 0 for 896 bytes, and the file holds 100.")]
    [InlineData("location=short.data|length=100", "'0.weight' is written as a 16x14 tensor of FLOAT, and the graph holds 100 bytes for it, where it takes 896.")]
    [InlineData("location=short.data|offset=200","'0.weight' is kept in 'short.data' from byte 200, and the file holds 100.")]
    [InlineData("location=short.data|offset=x", "'0.weight' is kept in 'short.data' with its offset written as 'x', which is not a count of bytes.")]
    [InlineData("location=short.data|length=-5", "'0.weight' is kept in 'short.data' with its length written as '-5', which is not a count of bytes.")]
    [InlineData("offset=0", "'0.weight' is kept in another file, and the graph does not say which.")]
    public void NumbersKeptBesideTheGraph_WhereTheirRecordCannotBeFollowed_AreRefusedAtTheirInitializer(string record, string refusal)
    {
        using var folder = new GraphFolder();
        folder.Beside("short.data", new byte[100]);

        var model = OnnxFixtures.Model(Titanic);
        var weights = model.Initializer("0.weight");
        weights.RawData = ByteString.Empty;
        weights.DataLocation = TensorProto.Types.DataLocation.External;

        foreach (var entry in record.Split('|'))
        {
            weights.ExternalData.Add(new StringStringEntryProto { Key = entry[..entry.IndexOf('=', StringComparison.Ordinal)], Value = entry[(entry.IndexOf('=', StringComparison.Ordinal) + 1)..] });
        }

        using var graph = folder.Written(model);
        var refused = Assert.Throws<SlotLoadException>(() => new OnnxFile(new BinaryCrossEntropy()).Read(graph));

        Assert.Equal([new SlotLoadFault("initializer '0.weight'", "0.weight", refusal)], refused.Faults);
    }

    [Fact]
    public void AFileThatIsNoOnnxModel_IsRefusedAsSuch()
    {
        using var words = new MemoryStream(Encoding.UTF8.GetBytes("not a model"));
        using var nothing = new MemoryStream();

        var keras = Assert.Throws<FormatException>(() => new OnnxFile(new BinaryCrossEntropy()).Read(KerasFixtures.Open("keras-titanic.keras")));
        var text = Assert.Throws<FormatException>(() => new OnnxFile(new BinaryCrossEntropy()).Read(words));
        var empty = Assert.Throws<FormatException>(() => new OnnxFile(new BinaryCrossEntropy()).Read(nothing));

        Assert.StartsWith("This is no ONNX model: ", keras.Message, StringComparison.Ordinal);
        Assert.StartsWith("This is no ONNX model: ", text.Message, StringComparison.Ordinal);
        Assert.Equal("This ONNX model holds no graph, and a network is read here from the graph a model holds.", empty.Message);
    }

    [Fact]
    public void TheImporter_IsHandedALoss_AndReadsAFile()
    {
        Assert.Throws<ArgumentNullException>(() => new OnnxFile(null!));
        Assert.Throws<ArgumentNullException>(() => new OnnxFile(new BinaryCrossEntropy()).Read(null!));
    }

    // The Titanic graph's relu turned into a layer normalisation of as many of the numbers the dense layer before it holds
    // as it is handed: the scale, and the shift.
    private static NodeProto Normalised(ModelProto model, int inputs)
    {
        var node = model.Node("/1/Relu");
        node.OpType = "LayerNormalization";
        node.Input.Add(new[] { "0.bias", "0.bias" }.Take(inputs - 1));

        return node;
    }

    // A normalisation of the given operator after the TensorFlow graph's last transpose, which lays its images out with their
    // channels last, taking as many numbers of three channels as it is given.
    private static void AfterTheLastTranspose(ModelProto model, string op, int numbers)
    {
        var names = Enumerable.Range(0, numbers).Select(at => $"numbers{at}").ToArray();
        model.Graph.Initializer.Add(names.Select(name => OnnxFixtures.Floats(name, [3], 1, 1, 1)));

        var reshape = model.Node($"{Converted}flatten_1/Reshape");
        model.Graph.Node.Insert(model.Graph.Node.IndexOf(reshape), OnnxFixtures.NodeOf(op, "normalised", [$"{Converted}conv2d_1_2/BiasAdd__11:0", .. names], "normed"));
        reshape.Input[0] = "normed";
    }

    // A pooling of the given operator after the TensorFlow graph's last transpose, which lays its images out with their
    // channels last, and the convolution after that taking what the pooling made.
    private static void Pooled(ModelProto model, string op)
    {
        var convolution = model.Node($"{Converted}conv2d_1_2/BiasAdd");
        var last = OnnxFixtures.NodeOf("Transpose", "last", [$"{Converted}re_lu_1/Relu:0"], "lasted");
        last.SetWholes("perm", 0, 2, 3, 1);

        var pooling = OnnxFixtures.NodeOf(op, "pooled", ["lasted"], "pooled_output");

        if (op != "GlobalAveragePool")
        {
            pooling.SetWholes("kernel_shape", 2, 2);
        }

        var at = model.Graph.Node.IndexOf(convolution);
        model.Graph.Node.Insert(at, last);
        model.Graph.Node.Insert(at + 1, pooling);
        convolution.Input[0] = "pooled_output";
    }

    // What a ReduceMean or a ReduceMax is refused with for reducing over other axes than every place of each channel.
    private static string Reduces(string written, int rank, string places) =>
        $"it reduces over axes {written} of a value of {rank} axes, and a global pooling here reduces over every place of each channel and nothing else, axes [{places}].";

    private static string Reduces(string node, string op, string written, int rank, string places) => $"node '{node}' ({op}): {Reduces(written, rank, places)}";

    // How an operator over a series, an image or a volume is refused for taking anything else.
    private static string TakesImages(string op) => $"it takes a series, an image or a volume with its channels after the batch, as ONNX's {op} does, and what reaches it is";

    // The flatten of the graph over images turned into a reshape into the target given.
    private static NodeProto Reshaped(ModelProto model, TensorProto target)
    {
        target.Name = "target";
        model.Graph.Initializer.Add(target);

        var node = model.Node("/4/Flatten");
        node.OpType = "Reshape";
        node.Unset("axis");
        node.Input.Add("target");

        return node;
    }

    // The Titanic graph ending in a softmax along the axis given.
    private static void Softened(ModelProto model, long axis)
    {
        var softmax = OnnxFixtures.NodeOf("Softmax", "softmax", ["logits"], "shares");
        softmax.SetWhole("axis", axis);
        model.Graph.Node.Add(softmax);
        model.Graph.Output[0].Name = "shares";
    }

    /// <summary>A graph changed one way, and the words it is refused in.</summary>
    private sealed record Case(string File, Action<ModelProto> Edit, string Refusal)
    {
        public Func<Loss>? Loss { get; init; }
    }
}
