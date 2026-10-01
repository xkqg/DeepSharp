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
/// words it is — a Gemm, or a MatMul with the Add of its bias after it, a dense layer; a Conv a convolution; a
/// BatchNormalization a batch normalisation with ONNX's momentum, which is Keras's, and its epsilon — and the words are
/// lowered as any description is. The graph's numbers go into the slots by path, <c>2.weight</c>, through
/// <see cref="Network.Load"/>, all together or not at all, every one that does not fit named at the initializer that holds it.
/// </para>
/// <para>
/// Layouts are what each node declares, never what the numbers look like. A Gemm says whether its weights are written
/// outputs by inputs, as PyTorch's are, and they are turned round to the inputs by outputs a slot here keeps; a MatMul's
/// weights are inputs by outputs, as its product says; a convolution's kernel is written channels out by channels in by rows
/// by columns, as ONNX defines it, and is laid out rows by columns by channels in by channels out. Images here have their
/// channels last, so an image goes into the network, and comes out of it, with its channels last: an image the graph takes
/// channels first, as ONNX's Conv does, has its first axis after the batch moved last. An image the graph takes channels last
/// — as TensorFlow lays images out, declared by a Transpose moving the channels after the batch before anything else takes
/// the image — is taken as it is, and the Transpose moving them last again before a flatten is nothing here. An image
/// flattened into a row with its channels after the batch is flattened channel by channel in ONNX and place by place here,
/// so the numbers along such a row — the next dense layer's weights, and a normalisation's before it — are turned from the
/// one order to the other, the image's rows, columns and channels taken from an example of nothing sent through the network
/// as it is lowered. A Cast into single-precision numbers, which the network works in throughout, is nothing.
/// </para>
/// <para>
/// The numbers the default exporter keeps in a file beside the graph, named in the graph, are read from beside the graph's
/// own file, so the graph is read from its file for them to be found. Numbers of half precision or of PyTorch's bfloat16 are
/// widened, exactly; numbers of double precision are rounded to the nearest single one.
/// </para>
/// <para>
/// What is read: a graph taking one batch — of rows, or of images — and giving one, each node taking the value the node
/// before it made, of the operators Gemm, MatMul, Add, Conv, BatchNormalization, LayerNormalization, Flatten, Reshape, Relu,
/// Tanh, Sigmoid and Softmax, and the Identity, Cast, Transpose and Constant nodes that hand values on. A convolution's pads
/// written out one side at a time are read as TensorFlow's 'same' when they are the border it gives the image reaching the
/// convolution. A last Sigmoid before a <see cref="BinaryCrossEntropy"/>, or a last Softmax before a
/// <see cref="CrossEntropy"/>, is lifted into that loss, which applies it itself. What is refused, at the node that says it
/// and every such node at once: a branch — a node taking another value than the one the node before it made, or two — and
/// any other operator, pooling among them, and the Mul tf2onnx writes a batch normalisation as, and the Shape, Gather, Slice
/// and Concat it works a flatten's target out with for batches of any length; a Gemm that scales or takes its input turned
/// round; a MatMul by anything but weights the graph holds, or without the Add of its bias; an Add after anything but a
/// MatMul, or of a bias of another width; a Cast into another type; a Transpose other than one moving an image's channels
/// after the batch or last; a convolution other than over an image's rows and columns and its channels after the batch,
/// striding otherwise down than across, dilated, with its channels in groups, or padded otherwise than alike on every side or
/// as TensorFlow's 'same'; a layer without a bias; a batch normalisation measuring each batch as it answers, or of an image
/// with its channels last; a layer normalisation of an image, from another axis or without its shift; a reshape that does
/// not flatten each example into a row; a softmax anywhere but last before a cross-entropy; and a graph taking other than
/// single-precision numbers, or giving an image flattened channel by channel, whose values would come out in another order.
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
