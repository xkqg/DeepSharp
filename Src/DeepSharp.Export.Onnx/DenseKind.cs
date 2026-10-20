// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;

namespace DeepSharp.Export.Onnx;

/// <summary>
/// A dense layer: a MatMul of the rows by weights written inputs by outputs, as the layer keeps them, and the Add of its bias. A
/// low-rank pair beside its held kernel is written as the products it is — the rows by the pair's first matrix, that by its second,
/// scaled and added — in the order the layer works them out in. Over a series, an image or a volume, which the graph holds with its
/// channels first, the channels are turned last for the products and back after them: ONNX's MatMul multiplies the last axis.
/// </summary>
internal sealed class DenseKind : ExportKind<Dense>
{
    /// <inheritdoc />
    protected override Value Write(Dense dense, Writing at)
    {
        var graph = at.Graph;
        var spatial = at.Input.Spatial;
        var rank = at.Input.Each.Rank;
        var rows = spatial ? graph.Node(at.Node("Transpose", 1, "Transpose.in") with { Attributes = node => node.States("perm", ChannelsLast(rank)) }, [at.Input.Name], $"{at.Path}/Transpose.in") : at.Input.Name;
        var result = spatial ? $"{at.Path}/Dense" : at.Path;
        var weight = graph.Number(at.Number(dense.Weight.Name), [dense.Inputs, dense.Outputs], at.Input.Ordered(dense.Weight.Value.Values));
        var hasBias = dense.Bias is not null;

        if (dense.LowRank is not { } pair)
        {
            return Returned(at, spatial, rank, Biased(at, dense, graph.Node(at.Node("MatMul", 1), [rows, weight], hasBias ? $"{at.Path}/MatMul" : result), result));
        }

        var first = graph.Number(at.Number(dense.A!.Name), [dense.Inputs, pair.Rank], at.Input.Ordered(dense.A.Value.Values));
        var second = graph.Number(at.Number(dense.B!.Name), [pair.Rank, dense.Outputs], dense.B.Value.Values);
        var scale = graph.Number(at.Number("scale"), [1], [(float)pair.Scale]);
        var kernel = graph.Node(at.Node("MatMul", 1), [rows, weight], $"{at.Path}/MatMul");
        var through = graph.Node(at.Node("MatMul", 1, "MatMul.a"), [rows, first], $"{at.Path}/MatMul.a");
        var back = graph.Node(at.Node("MatMul", 1, "MatMul.b"), [through, second], $"{at.Path}/MatMul.b");
        var scaled = graph.Node(at.Node("Mul", 7), [back, scale], $"{at.Path}/Mul");
        var sum = graph.Node(at.Node("Add", 7, "Add.pair"), [kernel, scaled], hasBias ? $"{at.Path}/Add.pair" : result);

        return Returned(at, spatial, rank, Biased(at, dense, sum, result));
    }

    // The Add of the bias, when the layer has one: the value it gives is the step's.
    private static string Biased(Writing at, Dense dense, string product, string result) =>
        dense.Bias is null
            ? product
            : at.Graph.Node(at.Node("Add", 7), [product, at.Graph.Number(at.Number(dense.Bias.Name), [dense.Outputs], dense.Bias.Value.Values)], result);

    // The value the step makes: the channels turned back first where they were turned last.
    private static Value Returned(Writing at, bool spatial, int rank, string made) =>
        at.Giving(spatial
            ? at.Graph.Node(at.Node("Transpose", 1, "Transpose.out") with { Attributes = node => node.States("perm", ChannelsFirst(rank)) }, [made], at.Path)
            : made);

    // The permutations that turn the channels of a batch of an example of this rank (the channels and the places) last, and first again.
    private static long[] ChannelsLast(int rank) => [0, .. Enumerable.Range(2, rank - 1).Select(axis => (long)axis), 1];

    private static long[] ChannelsFirst(int rank) => [0, rank, .. Enumerable.Range(1, rank - 1).Select(axis => (long)axis)];
}
