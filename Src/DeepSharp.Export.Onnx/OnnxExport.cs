// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using Google.Protobuf;
using Onnx;

namespace DeepSharp.Export.Onnx;

/// <summary>
/// Writes a network as an ONNX graph: the steps of a <see cref="LayerStack"/> as ONNX nodes, its numbers as the graph's initializers
/// named by the slot's path, and the output activation its loss applies itself as the last node.
/// </summary>
/// <remarks>
/// <para>
/// The graph takes one batch of examples of the shape the exporter is told and gives one batch of what the network, and then its loss's
/// output activation, makes of them. Its operator set is the oldest that holds every node written, and its IR version the one ONNX pairs
/// with that set; the graph is the one <c>DeepSharp.Import.Onnx</c> reads back into the same numbers.
/// </para>
/// <para>
/// A dense layer is a MatMul by weights written inputs by outputs, as the layer keeps them, and the Add of its bias; a low-rank pair beside
/// its held kernel is the products it is. A layer normalisation, a root-mean-square normalisation and a batch normalisation answering
/// from its running statistics are the operators of those names; the activations are the operators of theirs, a sigmoid linear unit
/// the input times its own sigmoid. Steps that take several steps are an Add for a sum, a Concat for what is laid side by side, and a
/// Slice for a block of columns. A dropout, and a variation of the rows made while training, do nothing in a trained network and are left
/// out of the graph.
/// </para>
/// <para>
/// Refused by name: a network written as code, which has no declared order the steps could be written in, and every layer that is
/// written as no ONNX node here.
/// </para>
/// </remarks>
public sealed class OnnxExport
{
    private readonly Shape _each;

    /// <summary>An exporter of networks that take examples of the given shape.</summary>
    /// <param name="each">The shape of one example, without the batch's axis: a row's length.</param>
    public OnnxExport(Shape each) => _each = each;

    /// <summary>The names of the features a row is made of, written beside the graph for whatever serves it to check; nothing, unless said.</summary>
    public IReadOnlyList<string>? Features { get; init; }

    /// <summary>Writes the network as an ONNX graph.</summary>
    /// <param name="network">The network: a stack of steps.</param>
    /// <param name="loss">What it was trained to bring down, whose output activation the graph ends in.</param>
    /// <param name="file">Where the graph is written.</param>
    /// <exception cref="ArgumentNullException">A network, a loss or a file is missing.</exception>
    /// <exception cref="NotSupportedException">The network is written as code, or holds a layer that is written as no ONNX node here.</exception>
    public void Write(Network network, Loss loss, Stream file)
    {
        ArgumentNullException.ThrowIfNull(file);

        Model(network, loss).WriteTo(file);
    }

    /// <summary>Writes the network as an ONNX graph.</summary>
    /// <param name="network">The network: a stack of steps.</param>
    /// <param name="loss">What it was trained to bring down, whose output activation the graph ends in.</param>
    /// <returns>The graph, as the bytes of an ONNX file.</returns>
    /// <exception cref="ArgumentNullException">A network or a loss is missing.</exception>
    /// <exception cref="NotSupportedException">The network is written as code, or holds a layer that is written as no ONNX node here.</exception>
    public byte[] ToBytes(Network network, Loss loss) => Model(network, loss).ToByteArray();

    // The lengths of an example as the graph holds it: a row as it is, a series, an image or a volume with its channels first.
    private static int[] ChannelsFirst(Shape each) => each.Rank == 1 ? each.Axes.ToArray() : [each[each.Rank - 1], .. each.Axes[..^1].ToArray()];

    private ModelProto Model(Network network, Loss loss)
    {
        ArgumentNullException.ThrowIfNull(network);
        ArgumentNullException.ThrowIfNull(loss);

        if (network is not LayerStack stack)
        {
            throw new NotSupportedException(
                $"A {network.GetType().Name} is a network written as code, which declares no order its layers are run in, so there is no graph to write it as: write it as a stack of steps.");
        }

        var graph = new GraphDraft();

        graph.Takes("input", ChannelsFirst(_each));

        var made = new StackWriting(graph).Write(stack, new Value("input", _each), string.Empty);

        if (made.Image is not null)
        {
            throw new NotSupportedException(
                "The network ends in a row a flatten made of an image: ONNX flattens it channel by channel and the network place by place, so the graph would give the values in another order than the network does.");
        }

        var output = loss.Activation switch
        {
            OutputActivation.Sigmoid => graph.Node(new NodeOf("Sigmoid", 6, "output/Sigmoid"), [made.Name], "output"),
            OutputActivation.Softmax => graph.Node(
                new NodeOf("Softmax", made.Each.Rank == 1 ? 1 : 13, "output/Softmax") { Attributes = node => node.State("axis", 1L) },
                [made.Name],
                "output"),
            _ => made.Name,
        };

        graph.Gives(output, ChannelsFirst(made.Each));

        var model = new ModelProto
        {
            IrVersion = graph.IrVersion,
            ProducerName = "DeepSharp",
            ProducerVersion = typeof(OnnxExport).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion?.Split('+')[0] ?? string.Empty,
            Graph = graph.Graph,
        };

        model.OpsetImport.Add(new OperatorSetIdProto { Version = graph.Opset });

        if (Features is { } features)
        {
            model.MetadataProps.Add(new StringStringEntryProto { Key = "deepsharp.features", Value = string.Join(',', features) });
        }

        return model;
    }
}
