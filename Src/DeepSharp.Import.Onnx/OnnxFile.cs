// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using Google.Protobuf;
using Onnx;

namespace DeepSharp.Import.Onnx;

/// <summary>
/// Reads an ONNX graph into a network here: the one <c>torch.onnx.export</c> writes, by its default exporter or the
/// TorchScript one before it, the one Keras writes with <c>model.export(format="onnx")</c>, or the one tf2onnx converts a
/// TensorFlow SavedModel into — handed the loss the network answers through, since a graph is a forward pass and names none.
/// </summary>
/// <remarks>
/// <para>
/// The graph is read once and lowered onto a <see cref="LayerStack"/>, never kept beside it: each node is written in the
/// words it is — a Gemm, or a MatMul with the Add of its bias after it, a dense layer; a Conv a convolution along a series,
/// over an image or through a volume; a MaxPool or an AveragePool a pooling along as many axes; a GlobalAveragePool or a
/// GlobalMaxPool, or a ReduceMean or ReduceMax over those places alone, a pooling of every place of each channel into one;
/// a BatchNormalization a batch normalisation with ONNX's momentum, which is Keras's, and its epsilon — and the words are lowered as any description is. The graph's numbers go
/// into the slots by path, <c>2.weight</c>, through <see cref="Network.Load"/>, all together or not at all, every one that
/// does not fit named at the initializer that holds it.
/// </para>
/// <para>
/// Layouts are what each node declares, never what the numbers look like. A Gemm says whether its weights are written
/// outputs by inputs, as PyTorch's are, and they are turned round to the inputs by outputs a slot here keeps; a MatMul's
/// weights are inputs by outputs, as its product says; a convolution's kernel is written channels out by channels in by the
/// window's lengths, as ONNX defines it, and is laid out the window's places by channels in by channels out. Series, images
/// and volumes here have their channels last, so one goes into the network, and comes out of it, with its channels last: one
/// the graph takes channels first, as ONNX's Conv does, has its first axis after the batch moved last. An image the graph
/// takes channels last — as TensorFlow lays images out, declared by a Transpose moving the channels after the batch before
/// anything else takes the image — is taken as it is, and the Transpose moving them last again before a flatten is nothing
/// here. A series, an image or a volume flattened into a row with its channels after the batch is flattened channel by
/// channel in ONNX and place by place here, so the numbers along such a row — the next dense layer's weights, and a
/// normalisation's before it — are turned from the one order to the other, the rows, columns and channels taken from an
/// example of nothing sent through the network as it is lowered; a series is taken as an image of one row, and a volume as
/// one whose rows are those of its planes one after another. A Cast into single-precision numbers, which the network works
/// in throughout, is nothing.
/// </para>
/// <para>
/// The numbers the default exporter keeps in a file beside the graph, named in the graph, are read from beside the graph's
/// own file, so the graph is read from its file for them to be found. Numbers of half precision or of PyTorch's bfloat16 are
/// widened, exactly; numbers of double precision are rounded to the nearest single one.
/// </para>
/// <para>
/// What is read: a graph taking one batch — of rows, of series, of images or of volumes — and giving one, each node
/// taking the value the node before it made, of the operators Gemm, MatMul, Add, Conv, MaxPool, AveragePool,
/// GlobalAveragePool, GlobalMaxPool, ReduceMean, ReduceMax, BatchNormalization, LayerNormalization, Flatten, Reshape,
/// Relu, Tanh, Sigmoid and Softmax, and the Identity, Cast, Transpose and Constant nodes that hand values on. The pads
/// of a convolution or a pooling, written out one side at a time, are read as TensorFlow's 'same' when they are the
/// border it gives the images reaching it, and so is auto_pad SAME_UPPER, and SAME_LOWER where the images leave no odd
/// place for it to put before the axis instead of after it; a pooling counts its padding in each average when
/// count_include_pad says so. A dropout of whole channels leaves no node in a graph exported for inference, so nothing
/// is read for it. A ReduceMean or ReduceMax is the global pooling that keeps the axes it pools, or drops them with
/// keepdims 0, when it reduces over every place of each channel, whether the axes are an input of the node (from ONNX
/// 18) or an attribute, counted from either end. Along a series PyTorch's default exporter writes an Unsqueeze before
/// it and a Squeeze after it, and those three nodes are read as that one pooling: an Unsqueeze making the third axis of
/// four, the reduce keeping its axes straight after it, and a Squeeze taking the third axis away straight after that,
/// none of their values taken by anything else. A last Sigmoid before a <see cref="BinaryCrossEntropy"/>, or a last
/// Softmax before a <see cref="CrossEntropy"/>, is lifted into that loss, which applies it itself.
/// </para>
/// <para>
/// What is refused, at the node that says it and every such node at once: a branch — a node taking another value than the
/// one the node before it made, or two — and any other operator, among them every other Reduce, any Unsqueeze and Squeeze
/// but those round the pooling of a series, the Mul tf2onnx writes a batch normalisation as, and the Shape, Gather, Slice and
/// Concat it works a flatten's target out with for batches of any length; a ReduceMean or ReduceMax over any other axes than
/// every place of each channel — the channels, the batch, some of the places only — or over none named, which would be every
/// axis, the channels among them; a Gemm that scales or takes its input turned round; a MatMul by anything but weights the
/// graph holds, or without the Add of its bias; an Add after anything but a MatMul, or of a bias of another width; a Cast
/// into another type; a Transpose other than one moving an image's channels after the batch or last; a convolution or a
/// pooling of anything but a series, an image or a volume with its channels after the batch, striding otherwise on one axis
/// than on another, dilated, rounding the places its window stands at up (ceil_mode), or padded otherwise than alike on
/// every side or as TensorFlow's 'same'; a convolution with its channels in groups; a max pooling that gives the places of
/// its largest values besides, or numbers them by columns; a layer without a bias; a batch normalisation measuring each
/// batch as it answers, or of an image with its channels last; a layer normalisation of an image, from another axis or
/// without its shift; a reshape that does not flatten each example into a row; a softmax anywhere but last before a
/// cross-entropy; and a graph taking other than single-precision numbers, or giving an image flattened channel by channel,
/// whose values would come out in another order.
/// </para>
/// </remarks>
public sealed class OnnxFile : IImporter
{
    private readonly Loss _loss;

    /// <summary>A reader of ONNX graphs, each read as a network answering through the given loss.</summary>
    /// <param name="loss">What the network was trained to bring down: named, never assumed, since a graph names none.</param>
    /// <exception cref="ArgumentNullException">No loss is given.</exception>
    public OnnxFile(Loss loss)
    {
        ArgumentNullException.ThrowIfNull(loss);

        _loss = loss;
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentNullException">No file is given.</exception>
    /// <exception cref="FormatException">
    /// The file is no ONNX model, or its graph says what no network here is built of: every such thing at once, each where
    /// the graph says it.
    /// </exception>
    public SavedNetwork Read(Stream file)
    {
        ArgumentNullException.ThrowIfNull(file);

        ModelProto model;

        try
        {
            model = ModelProto.Parser.ParseFrom(file);
        }
        catch (InvalidProtocolBufferException refused)
        {
            throw new FormatException($"This is no ONNX model: {refused.Message.Quoted()}", refused);
        }

        var graph = model.Graph ?? throw new FormatException("This ONNX model holds no graph, and a network is read here from the graph a model holds.");

        return new OnnxGraph(graph, new GraphNumbers(FolderOf(file)), _loss).Loaded();
    }

    // The folder a graph read from its own file stands in, where its exporter keeps the numbers it keeps beside it; nothing
    // for a graph read from anything else.
    private static string? FolderOf(Stream file) =>
        file is FileStream { Name: var name } && Path.IsPathFullyQualified(name) ? Path.GetDirectoryName(name) : null;
}
