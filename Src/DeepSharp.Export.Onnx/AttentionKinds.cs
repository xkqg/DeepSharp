// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;

namespace DeepSharp.Export.Onnx;

/// <summary>What attention looks at once it is laid out as the graph works it: queries, keys and values as batch by step by channel, and the mask of the keys.</summary>
/// <param name="Query">The queries' name in the graph.</param>
/// <param name="Key">The keys' name.</param>
/// <param name="Value">The values' name.</param>
/// <param name="Mask">The mask's name — a one for a key that takes part, a nought for padding — batch by key; nothing for keys that all take part.</param>
internal readonly record struct Looked(string Query, string Key, string Value, string? Mask);

/// <summary>
/// Scaled dot-product attention written as the nodes of the oldest operator sets that hold it, not as the Attention operator of the
/// twenty-third: the heads cut out by a reshape and a transpose, the scores a batched product scaled by one over the root of a head's width,
/// the keys the mask leaves out far below nothing, a softmax over the last axis, and the values taken in those shares. A step with no key left to
/// look at is made nothing, as the layer makes it.
/// </summary>
/// <typeparam name="TLayer">The layer's class.</typeparam>
internal abstract class AttentiveKind<TLayer> : ExportKind<TLayer>
    where TLayer : Attentive
{
    // Far enough below nothing to share out nothing, near enough to stay finite: the layer's own.
    private const float Far = 1e9f;

    /// <summary>The series a step makes of what attention made, with the channels first again, from the batch by step by channel it was made in.</summary>
    /// <param name="at">The step.</param>
    /// <param name="context">What attention made: batch by step by channel.</param>
    /// <returns>The step's value.</returns>
    protected static Value Series(Writing at, string context) =>
        at.Giving(at.Graph.Node(at.Node("Transpose", 1, "Transpose.out") with { Attributes = node => node.States("perm", 0L, 2L, 1L) }, [context], at.Path));

    /// <summary>A series the graph holds with its channels first, as batch by step by channel.</summary>
    /// <param name="at">The step.</param>
    /// <param name="series">The series' name.</param>
    /// <param name="part">What tells the node from the step's others.</param>
    protected static string StepsFirst(Writing at, string series, string part) =>
        at.Graph.Node(at.Node("Transpose", 1, $"Transpose.{part}") with { Attributes = node => node.States("perm", 0L, 2L, 1L) }, [series], $"{at.Path}/Transpose.{part}");

    /// <summary>Attention over queries, keys and values laid out batch by step by channel.</summary>
    /// <param name="at">The step.</param>
    /// <param name="how">The heads, whether the layer is causal, and its dropout, which a trained network does nothing with.</param>
    /// <param name="looked">What it looks at.</param>
    /// <returns>What it makes, batch by step by channel.</returns>
    protected static string Attend(Writing at, Attending how, Looked looked)
    {
        var graph = at.Graph;
        var steps = at.Input.Each[0];
        var embed = at.Input.Each[1];
        var keys = at.Inputs.Count >= 3 ? at.Inputs[1].Each[0] : steps;
        var each = embed / how.Heads;
        var asked = Heads(at, looked.Query, new Cut("q", how.Heads, each), turned: false);
        var held = Heads(at, looked.Key, new Cut("k", how.Heads, each), turned: true);
        var taken = Heads(at, looked.Value, new Cut("v", how.Heads, each), turned: false);
        var product = graph.Node(at.Node("MatMul", 1, "MatMul.scores"), [asked, held], $"{at.Path}/MatMul.scores");
        var scores = graph.Node(at.Node("Mul", 7, "Mul.scale"), [product, graph.Number(at.Number("scale"), [], [(float)(1 / Math.Sqrt(each))])], $"{at.Path}/Mul.scale");
        var alive = (string?)null;
        var allowed = Allowed(at, how, looked.Mask, new Extent(steps, keys));

        if (allowed is not null)
        {
            var penalty = graph.Node(
                at.Node("Mul", 7, "Mul.far"),
                [graph.Node(at.Node("Sub", 7, "Sub.allowed"), [allowed, graph.Number(at.Number("one"), [], [1f])], $"{at.Path}/Sub.allowed"), graph.Number(at.Number("far"), [], [Far])],
                $"{at.Path}/Mul.far");

            scores = graph.Node(at.Node("Add", 7, "Add.penalty"), [scores, penalty], $"{at.Path}/Add.penalty");

            if (looked.Mask is not null)
            {
                alive = Alive(at, allowed);
            }
        }

        var weights = graph.Node(at.Node("Softmax", 1) with { Attributes = node => node.State("axis", 3L) }, [scores], $"{at.Path}/Softmax");

        if (alive is not null)
        {
            weights = graph.Node(at.Node("Mul", 7, "Mul.alive"), [weights, alive], $"{at.Path}/Mul.alive");
        }

        var context = graph.Node(at.Node("MatMul", 1, "MatMul.context"), [weights, taken], $"{at.Path}/MatMul.context");
        var back = graph.Node(at.Node("Transpose", 1, "Transpose.heads") with { Attributes = node => node.States("perm", 0L, 2L, 1L, 3L) }, [context], $"{at.Path}/Transpose.heads");

        return graph.Node(at.Node("Reshape", 5, "Reshape.heads"), [back, graph.Wholes(at.Number("merged"), 0, 0, embed)], $"{at.Path}/Reshape.heads");
    }

    // The heads of a series laid out batch by step by channel: cut out by a reshape, and turned to batch by head by step by channel — or by
    // channel by step for keys, which the scores are a product with.
    private static string Heads(Writing at, string series, Cut cut, bool turned)
    {
        var graph = at.Graph;
        var split = graph.Node(at.Node("Reshape", 5, $"Reshape.{cut.Part}"), [series, graph.Wholes(at.Number($"heads.{cut.Part}"), 0, 0, cut.Heads, cut.Each)], $"{at.Path}/Reshape.{cut.Part}");

        return graph.Node(
            at.Node("Transpose", 1, $"Transpose.heads.{cut.Part}") with { Attributes = node => node.States("perm", turned ? [0L, 2L, 3L, 1L] : [0L, 2L, 1L, 3L]) },
            [split],
            $"{at.Path}/Transpose.heads.{cut.Part}");
    }

    /// <summary>How a series is cut into heads: which of the three it is, how many heads, and how many channels each asks and answers in.</summary>
    /// <param name="Part">The series' name in the graph: <c>q</c>, <c>k</c> or <c>v</c>.</param>
    /// <param name="Heads">How many heads.</param>
    /// <param name="Each">The channels of one head.</param>
    private readonly record struct Cut(string Part, int Heads, int Each);

    // Which keys a step may look at, batch by one by steps by keys: a one where it may, as the mask of the series and the layer's causality say.
    private static string? Allowed(Writing at, Attending how, string? mask, Extent extent)
    {
        var graph = at.Graph;
        string? allowed = mask is null
            ? null
            : graph.Node(at.Node("Reshape", 5, "Reshape.mask"), [mask, graph.Wholes(at.Number("mask"), 0, 1, 1, extent.Keys)], $"{at.Path}/Reshape.mask");

        if (!how.Causal)
        {
            return allowed;
        }

        var before = graph.Number(
            at.Number("causal"),
            [1, 1, extent.Steps, extent.Keys],
            [.. Enumerable.Range(0, extent.Steps * extent.Keys).Select(place => place % extent.Keys <= place / extent.Keys ? 1f : 0f)]);

        return allowed is null ? before : graph.Node(at.Node("Mul", 7, "Mul.causal"), [allowed, before], $"{at.Path}/Mul.causal");
    }

    // One where a step has a key left to look at and nought where it has none: the keys it may look at added up, held between nought and one.
    private static string Alive(Writing at, string allowed)
    {
        var graph = at.Graph;
        var counted = graph.Node(
            at.Node("ReduceSum", 1, "ReduceSum.alive") with
            {
                Attributes = node =>
                {
                    node.States("axes", 3L);
                    node.State("keepdims", 1L);
                },
            },
            [allowed],
            $"{at.Path}/ReduceSum.alive");

        return graph.Node(
            at.Node("Clip", 11, "Clip.alive"),
            [counted, graph.Number(at.Number("least"), [], [0f]), graph.Number(at.Number("most"), [], [1f])],
            $"{at.Path}/Clip.alive");
    }

    /// <summary>How many steps ask, and how many keys they are held against.</summary>
    /// <param name="Steps">The steps of the queries.</param>
    /// <param name="Keys">The steps of the keys.</param>
    private readonly record struct Extent(int Steps, int Keys);
}

/// <summary>
/// Multi-head attention over a series: the series turned to batch by step by channel, projected into queries, keys and values by the three
/// blocks of the one projection, attended, and the answer put through the output layer and turned back to channels first.
/// </summary>
internal sealed class MultiHeadAttentionKind : AttentiveKind<MultiHeadAttention>
{
    /// <inheritdoc />
    protected override Value Write(MultiHeadAttention layer, Writing at)
    {
        var graph = at.Graph;
        var embed = layer.Embed;
        var series = StepsFirst(at, at.Input.Name, "in");
        var weights = layer.InProjectionWeight.Value.Values.ToArray();
        var biases = layer.InProjectionBias.Value.Values.ToArray();
        var numbers = new Numbers(weights, biases);
        var parts = new[] { "q", "k", "v" }.Select((part, place) => Project(at, series, new Block(part, place, embed), numbers)).ToArray();
        var made = Attend(at, layer.Attending, new Looked(parts[0], parts[1], parts[2], at.Inputs.Count == 2 ? at.Inputs[1].Name : null));
        var through = graph.Node(
            at.Node("MatMul", 1, "MatMul.out"),
            [made, graph.Number(at.Number($"out_proj.{layer.OutProjection.Weight.Name}"), [embed, embed], layer.OutProjection.Weight.Value.Values)],
            $"{at.Path}/MatMul.out");

        return Series(at, graph.Node(at.Node("Add", 7, "Add.out"), [through, graph.Number(at.Number($"out_proj.{layer.OutProjection.Bias!.Name}"), [embed], layer.OutProjection.Bias.Value.Values)], $"{at.Path}/Add.out"));
    }

    // One of the three projections: the series by the block of the projection's columns, and the block of its bias.
    private static string Project(Writing at, string series, Block block, Numbers numbers)
    {
        var graph = at.Graph;
        var embed = block.Width;
        var weight = new float[embed * embed];

        for (var row = 0; row < embed; row++)
        {
            numbers.Weights.AsSpan((row * 3 * embed) + (block.Place * embed), embed).CopyTo(weight.AsSpan(row * embed, embed));
        }

        var product = graph.Node(at.Node("MatMul", 1, $"MatMul.{block.Part}"), [series, graph.Number(at.Number($"in_proj_weight.{block.Part}"), [embed, embed], weight)], $"{at.Path}/MatMul.{block.Part}");

        return graph.Node(
            at.Node("Add", 7, $"Add.{block.Part}"),
            [product, graph.Number(at.Number($"in_proj_bias.{block.Part}"), [embed], numbers.Biases.AsSpan(block.Place * embed, embed))],
            $"{at.Path}/Add.{block.Part}");
    }

    /// <summary>Which of the projection's three blocks, and how wide each is.</summary>
    /// <param name="Part">The block's name in the graph: <c>q</c>, <c>k</c> or <c>v</c>.</param>
    /// <param name="Place">Which block it is, counted from nought: the queries', the keys' or the values'.</param>
    /// <param name="Width">The channels of a step, which is the width of a block.</param>
    private readonly record struct Block(string Part, int Place, int Width);

    /// <summary>The projection's numbers, as the layer keeps them.</summary>
    /// <param name="Weights">The weights: channels by three times the channels.</param>
    /// <param name="Biases">The bias: three times the channels.</param>
    private readonly record struct Numbers(float[] Weights, float[] Biases);
}

/// <summary>Attention with no number of its own: over the series it is handed as queries, keys and values alike, or over the three it is handed apart.</summary>
internal sealed class AttentionKind : AttentiveKind<Attention>
{
    /// <inheritdoc />
    protected override Value Write(Attention layer, Writing at)
    {
        var taken = at.Inputs.Count;
        var first = StepsFirst(at, at.Inputs[0].Name, "q");
        var queries = first;
        var keys = taken >= 3 ? StepsFirst(at, at.Inputs[1].Name, "k") : first;
        var values = taken >= 3 ? StepsFirst(at, at.Inputs[2].Name, "v") : first;
        var mask = taken is 2 or 4 ? at.Inputs[taken - 1].Name : null;

        return Series(at, Attend(at, layer.Attending, new Looked(queries, keys, values, mask)));
    }
}

/// <summary>The sine positions: the constant the formula works out for the series' steps and channels, added to the series with its channels first.</summary>
internal sealed class SinePositionsKind : ExportKind<SinePositions>
{
    /// <inheritdoc />
    protected override Value Write(SinePositions layer, Writing at)
    {
        var steps = at.Input.Each[0];
        var width = at.Input.Each[1];
        var positions = new float[width * steps];

        for (var step = 0; step < steps; step++)
        {
            for (var channel = 0; channel < width; channel++)
            {
                var angle = step / Math.Pow(10000, (channel - (channel % 2)) / (double)width);

                positions[(channel * steps) + step] = (float)(channel % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle));
            }
        }

        return at.Giving(at.Graph.Node(at.Node("Add", 7), [at.Input.Name, at.Graph.Number(at.Number("positions"), [width, steps], positions)], at.Path));
    }
}
