// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;

namespace DeepSharp.Export.Onnx;

/// <summary>
/// A dense layer: a MatMul of the rows by weights written inputs by outputs, as the layer keeps them, and the Add of its bias. A
/// low-rank pair beside its held kernel is written as the products it is — the rows by the pair's first matrix, that by its second,
/// scaled and added — in the order the layer works them out in.
/// </summary>
internal sealed class DenseKind : ExportKind<Dense>
{
    /// <inheritdoc />
    protected override Value Write(Dense dense, Writing at)
    {
        var graph = at.Graph;
        var rows = at.Input.Name;
        var weight = graph.Number(at.Number(dense.Weight.Name), [dense.Inputs, dense.Outputs], at.Input.Ordered(dense.Weight.Value.Values));
        var hasBias = dense.Bias is not null;

        if (dense.LowRank is not { } pair)
        {
            return at.Giving(Biased(at, dense, graph.Node(at.Node("MatMul", 1), [rows, weight], hasBias ? $"{at.Path}/MatMul" : at.Path)));
        }

        var first = graph.Number(at.Number(dense.A!.Name), [dense.Inputs, pair.Rank], at.Input.Ordered(dense.A.Value.Values));
        var second = graph.Number(at.Number(dense.B!.Name), [pair.Rank, dense.Outputs], dense.B.Value.Values);
        var scale = graph.Number(at.Number("scale"), [1], [(float)pair.Scale]);
        var kernel = graph.Node(at.Node("MatMul", 1), [rows, weight], $"{at.Path}/MatMul");
        var through = graph.Node(at.Node("MatMul", 1, "MatMul.a"), [rows, first], $"{at.Path}/MatMul.a");
        var back = graph.Node(at.Node("MatMul", 1, "MatMul.b"), [through, second], $"{at.Path}/MatMul.b");
        var scaled = graph.Node(at.Node("Mul", 7), [back, scale], $"{at.Path}/Mul");
        var sum = graph.Node(at.Node("Add", 7, "Add.pair"), [kernel, scaled], hasBias ? $"{at.Path}/Add.pair" : at.Path);

        return at.Giving(Biased(at, dense, sum));
    }

    // The Add of the bias, when the layer has one: the value it gives is the step's.
    private static string Biased(Writing at, Dense dense, string product) =>
        dense.Bias is null
            ? product
            : at.Graph.Node(at.Node("Add", 7), [product, at.Graph.Number(at.Number(dense.Bias.Name), [dense.Outputs], dense.Bias.Value.Values)], at.Path);
}
