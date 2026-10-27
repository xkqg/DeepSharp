// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;

namespace DeepSharp.Export.Onnx;

/// <summary>
/// A BatchEnsemble adapter: a batch of rows given an axis of as many variants as the adapter holds, each row multiplied
/// by a different one of the adapter's own, through ONNX's broadcast — Unsqueeze to open the axis, then Mul against the
/// adapter, whose row for row shape broadcasts against it, exactly as the layer multiplies them.
/// </summary>
/// <remarks>
/// The graph holds a value of more than one axis to an example with its last axis first, right after the batch — as
/// every other multi-axis value here does — so what the layer makes, an example of <see cref="BatchEnsemble.Variants"/>
/// rows of <see cref="BatchEnsemble.Inputs"/>, is a graph tensor of the inputs then the variants: the adapter is written
/// transposed to that same order, so the one Mul broadcasts it correctly without a Transpose of its own.
/// </remarks>
internal sealed class BatchEnsembleKind : ExportKind<BatchEnsemble>
{
    /// <inheritdoc />
    protected override Value Write(BatchEnsemble layer, Writing at)
    {
        var graph = at.Graph;
        var opened = graph.Node(
            at.Node("Unsqueeze", 1, "Unsqueeze") with { Attributes = node => node.States("axes", 2L) },
            [at.Input.Name],
            $"{at.Path}/Unsqueeze");
        var adapter = graph.Number(at.Number(layer.Adapter.Name), [layer.Inputs, layer.Variants], Transposed(layer.Adapter.Value.Values, layer.Variants, layer.Inputs));

        return at.Giving(graph.Node(at.Node("Mul", 7), [opened, adapter], at.Path));
    }

    // The adapter's rows by columns, inputs by variants instead of variants by inputs, row-major.
    private static float[] Transposed(ReadOnlySpan<float> values, int variants, int inputs)
    {
        var transposed = new float[values.Length];

        for (var variant = 0; variant < variants; variant++)
        {
            for (var input = 0; input < inputs; input++)
            {
                transposed[(input * variants) + variant] = values[(variant * inputs) + input];
            }
        }

        return transposed;
    }
}

/// <summary>
/// Variant heads: as many separate projections as a BatchEnsemble opened variants, none sharing the others' weights —
/// ONNX's batched MatMul, each variant its own slice of the rows, turned to the variants' own axis first so the batch
/// axis every variant shares is not the one the weights are laid out by, and turned back after the bias is added.
/// </summary>
/// <remarks>
/// The graph holds the value a BatchEnsemble opened as the inputs then the variants, right after the batch — its own
/// output written that way, see <see cref="BatchEnsembleKind"/> — and gives back the outputs then the variants, the
/// same order, so a <see cref="Dense"/> or an activation standing between the two, which read and write that order
/// unchanged, need nothing written here to agree with them.
/// </remarks>
internal sealed class VariantHeadsKind : ExportKind<VariantHeads>
{
    /// <inheritdoc />
    protected override Value Write(VariantHeads layer, Writing at)
    {
        var graph = at.Graph;
        var turned = graph.Node(
            at.Node("Transpose", 1, "Transpose.in") with { Attributes = node => node.States("perm", 2L, 0L, 1L) },
            [at.Input.Name],
            $"{at.Path}/Transpose.in");
        var weight = graph.Number(at.Number(layer.Weight.Name), [layer.Variants, layer.Inputs, layer.Outputs], layer.Weight.Value.Values);
        var product = graph.Node(at.Node("MatMul", 1), [turned, weight], $"{at.Path}/MatMul");
        var bias = graph.Number(at.Number(layer.Bias.Name), [layer.Variants, 1, layer.Outputs], layer.Bias.Value.Values);
        var added = graph.Node(at.Node("Add", 7), [product, bias], $"{at.Path}/Add");

        return at.Giving(graph.Node(
            at.Node("Transpose", 1, "Transpose.out") with { Attributes = node => node.States("perm", 1L, 2L, 0L) },
            [added],
            at.Path));
    }
}
