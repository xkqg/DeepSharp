// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using Onnx;

namespace DeepSharp.Export.Onnx;

/// <summary>
/// A category as a small vector: the columns of the row that hold the category's ones and noughts, replaced by the vector of the category
/// the row holds. Written as the products the layer works it out in — the block taken out of the row, times the vectors, put where the
/// block stood, beside the row's other columns carried to where they stand after.
/// </summary>
internal sealed class EmbeddingKind : ExportKind<Embedding>
{
    /// <inheritdoc />
    protected override Value Write(Embedding layer, Writing at)
    {
        RequireRow(at, "An embedding");

        var graph = at.Graph;
        var outputs = layer.Outputs;
        var kept = new float[layer.Features * outputs];
        var taken = new float[layer.Features * layer.Block.Count];
        var placed = new float[layer.Width * outputs];
        var first = layer.Block.Min();
        var place = 0;

        for (var column = 0; column < layer.Features; column++)
        {
            if (column == first)
            {
                for (var number = 0; number < layer.Width; number++)
                {
                    placed[(number * outputs) + place + number] = 1f;
                }

                place += layer.Width;
            }

            var category = layer.Block.ToList().IndexOf(column);

            if (category >= 0)
            {
                taken[(column * layer.Block.Count) + category] = 1f;
            }
            else
            {
                kept[(column * outputs) + place] = 1f;
                place++;
            }
        }

        var row = at.Input.Name;
        var vectors = graph.Node(
            at.Node("MatMul", 1, "MatMul.vectors"),
            [graph.Node(at.Node("MatMul", 1, "MatMul.taken"), [row, graph.Number(at.Number("taken"), [layer.Features, layer.Block.Count], taken)], $"{at.Path}/MatMul.taken"), graph.Number(at.Number(layer.Weight.Name), [layer.Block.Count, layer.Width], layer.Weight.Value.Values)],
            $"{at.Path}/MatMul.vectors");
        var put = graph.Node(at.Node("MatMul", 1, "MatMul.placed"), [vectors, graph.Number(at.Number("placed"), [layer.Width, outputs], placed)], $"{at.Path}/MatMul.placed");
        var carried = graph.Node(at.Node("MatMul", 1, "MatMul.kept"), [row, graph.Number(at.Number("kept"), [layer.Features, outputs], kept)], $"{at.Path}/MatMul.kept");

        return at.Giving(graph.Node(at.Node("Add", 7), [carried, put], at.Path));
    }

    // The columns it reads are the row's, by their places: a row a flatten made of an image has them in another order.
    internal static void RequireRow(Writing at, string kind)
    {
        if (at.Input.Image is not null)
        {
            throw new NotSupportedException($"{kind} reads columns of a row by their places, and ONNX flattens an image channel by channel where the network flattens it place by place.");
        }
    }
}

/// <summary>
/// The ids of a text as one small vector: each id looked up in vectors it learns, the vectors of the words averaged and the padding, id
/// nought, left out. Written as the product the layer works it out in — the share each id is of its row, times the vectors — the shares
/// counted by a one-hot of the ids.
/// </summary>
internal sealed class TokenEmbeddingKind : ExportKind<TokenEmbedding>
{
    /// <inheritdoc />
    protected override Value Write(TokenEmbedding layer, Writing at)
    {
        EmbeddingKind.RequireRow(at, "A token embedding");

        var graph = at.Graph;
        var vocabulary = layer.Vocabulary;
        var ids = graph.Node(at.Node("Cast", 6) with { Attributes = node => node.State("to", 7L) }, [at.Input.Name], $"{at.Path}/Cast");

        if (layer.KeepsSteps)
        {
            return Kept(layer, at, ids);
        }

        var hot = graph.Node(
            at.Node("OneHot", 9),
            [ids, graph.Wholes(at.Number("vocabulary"), vocabulary), graph.Number(at.Number("values"), [2], [0f, 1f])],
            $"{at.Path}/OneHot");
        var counts = graph.Node(at.Node("ReduceSum", 1, "ReduceSum.counts") with { Attributes = Over(1, keep: false) }, [hot], $"{at.Path}/ReduceSum.counts");
        var words = graph.Node(at.Node("Mul", 7, "Mul.words"), [counts, graph.Number(at.Number("words"), [vocabulary], [.. Enumerable.Range(0, vocabulary).Select(id => id == 0 ? 0f : 1f)])], $"{at.Path}/Mul.words");
        var total = graph.Node(at.Node("ReduceSum", 1, "ReduceSum.total") with { Attributes = Over(1, keep: true) }, [words], $"{at.Path}/ReduceSum.total");
        var divisor = graph.Node(at.Node("Max", 8), [total, graph.Number(at.Number("one"), [1], [1f])], $"{at.Path}/Max");
        var shares = graph.Node(at.Node("Div", 7), [words, divisor], $"{at.Path}/Div");

        return at.Giving(graph.Node(at.Node("MatMul", 1), [shares, graph.Number(at.Number(layer.Weight.Name), [vocabulary, layer.Width], layer.Weight.Value.Values)], at.Path));
    }

    // Each id's vector at its step, the padding nothing: the vectors looked up by their ids, held where an id is a word, and laid out with the
    // channels first, as the graph holds a series.
    private static Value Kept(TokenEmbedding layer, Writing at, string ids)
    {
        var graph = at.Graph;
        var gathered = graph.Node(
            at.Node("Gather", 1) with { Attributes = node => node.State("axis", 0L) },
            [graph.Number(at.Number(layer.Weight.Name), [layer.Vocabulary, layer.Width], layer.Weight.Value.Values), ids],
            $"{at.Path}/Gather");
        var words = graph.Node(
            at.Node("Unsqueeze", 1) with { Attributes = node => node.States("axes", 2L) },
            [graph.Node(at.Node("Clip", 11), [at.Input.Name, graph.Number(at.Number("least"), [], [0f]), graph.Number(at.Number("most"), [], [1f])], $"{at.Path}/Clip")],
            $"{at.Path}/Unsqueeze");
        var held = graph.Node(at.Node("Mul", 7), [gathered, words], $"{at.Path}/Mul");

        return at.Giving(graph.Node(at.Node("Transpose", 1) with { Attributes = node => node.States("perm", 0L, 2L, 1L) }, [held], at.Path));
    }

    // The axes a reduction runs over, as an attribute (before the thirteenth operator set made them an input), and whether it keeps them.
    private static Action<NodeProto> Over(long axis, bool keep) => node =>
    {
        node.States("axes", axis);
        node.State("keepdims", keep ? 1L : 0L);
    };
}

/// <summary>
/// The periodic embedding of a number: each feature times frequencies it learns, a whole turn round, as its cosines and its sines, and
/// those through a linear layer of each feature's own or one shared by every feature. Written as the products the layer works it out in.
/// </summary>
internal sealed class PeriodicKind : ExportKind<Periodic>
{
    /// <inheritdoc />
    protected override Value Write(Periodic layer, Writing at)
    {
        EmbeddingKind.RequireRow(at, "A periodic embedding");

        var graph = at.Graph;
        var features = layer.Features;
        var count = layer.Periodicity.Frequencies;
        var width = layer.Periodicity.Width;
        var spread = graph.Node(
            at.Node("Mul", 7, "Mul.spread"),
            [graph.Node(at.Node("Unsqueeze", 1) with { Attributes = node => node.States("axes", 2L) }, [at.Input.Name], $"{at.Path}/Unsqueeze"), graph.Number(at.Number(layer.Frequencies.Name), [features, count], layer.Frequencies.Value.Values)],
            $"{at.Path}/Mul.spread");
        var angles = graph.Node(at.Node("Mul", 7, "Mul.turn"), [spread, graph.Number(at.Number("turn"), [1], [(float)(2 * Math.PI)])], $"{at.Path}/Mul.turn");
        var cosines = graph.Node(
            at.Node("Sin", 7, "Sin.cosines"),
            [graph.Node(at.Node("Add", 7, "Add.quarter"), [angles, graph.Number(at.Number("quarter"), [1], [(float)(Math.PI / 2)])], $"{at.Path}/Add.quarter")],
            $"{at.Path}/Sin.cosines");
        var sines = graph.Node(at.Node("Sin", 7, "Sin.sines"), [angles], $"{at.Path}/Sin.sines");
        var made = graph.Node(at.Node("Add", 7, "Add.waves"), [Through(at, layer, cosines, first: true), Through(at, layer, sines, first: false)], $"{at.Path}/Add.waves");
        var bias = graph.Number(at.Number(layer.Bias.Name), [layer.Bias.Value.Shape[0]], layer.Bias.Value.Values);
        var shape = graph.Wholes(at.Number("shape"), 0, features * width);

        // Shared, the one bias is added to every feature's numbers before they are laid in a row; each feature's own, it is added after.
        if (layer.Periodicity.Lite)
        {
            return at.Giving(graph.Node(at.Node("Reshape", 5), [graph.Node(at.Node("Add", 7), [made, bias], $"{at.Path}/Add"), shape], at.Path));
        }

        var turned = graph.Node(at.Node("Transpose", 1, "Transpose.back") with { Attributes = node => node.States("perm", 1L, 0L, 2L) }, [made], $"{at.Path}/Transpose.back");

        return at.Giving(graph.Node(at.Node("Add", 7), [graph.Node(at.Node("Reshape", 5), [turned, shape], $"{at.Path}/Reshape"), bias], at.Path));
    }

    // The waves of every feature through the half of the weight that reads them: each feature's own block of it, or the one shared by all.
    private static string Through(Writing at, Periodic layer, string waves, bool first)
    {
        var graph = at.Graph;
        var count = layer.Periodicity.Frequencies;
        var width = layer.Periodicity.Width;
        var features = layer.Features;
        var weight = layer.Weight.Value;
        var half = first ? 0 : count;
        var name = first ? "cosines" : "sines";

        if (layer.Periodicity.Lite)
        {
            var shared = graph.Number(at.Number($"weight.{name}"), [count, width], weight.Values.Slice(half * width, count * width));

            return graph.Node(at.Node("MatMul", 1, $"MatMul.{name}"), [waves, shared], $"{at.Path}/MatMul.{name}");
        }

        var blocks = new float[features * count * width];

        for (var feature = 0; feature < features; feature++)
        {
            for (var wave = 0; wave < count; wave++)
            {
                weight.Values.Slice(((half + wave) * features * width) + (feature * width), width).CopyTo(blocks.AsSpan(((feature * count) + wave) * width, width));
            }
        }

        var turned = graph.Node(at.Node("Transpose", 1, $"Transpose.{name}") with { Attributes = node => node.States("perm", 1L, 0L, 2L) }, [waves], $"{at.Path}/Transpose.{name}");

        return graph.Node(at.Node("MatMul", 1, $"MatMul.{name}"), [turned, graph.Number(at.Number($"weight.{name}"), [features, count, width], blocks)], $"{at.Path}/MatMul.{name}");
    }
}
