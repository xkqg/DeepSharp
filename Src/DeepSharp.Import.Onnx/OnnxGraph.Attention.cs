// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Networks;

namespace DeepSharp.Import.Onnx;

/// <summary>An input of the Attention operator after the three it looks at, and why a layer here is not given it.</summary>
/// <param name="Name">ONNX's name for the input.</param>
/// <param name="Why">What a layer here does instead.</param>
internal readonly record struct Further(string Name, string Why);

/// <summary>The part of reading a graph that follows the Attention operator: the series it looks across, and what it is not given.</summary>
internal sealed partial class OnnxGraph
{
    // The operator's inputs after the three it looks at, by the names ONNX gives them, and why a layer here is not given one.
    private static readonly Further[] Optional =
    [
        new("attn_mask", "a layer here is given the mask of the padding as a step it takes, and a mask a graph works out is not read"),
        new("past_key", "a layer here looks across the series it is handed alone, not across keys already read"),
        new("past_value", "a layer here looks across the series it is handed alone, not across values already read"),
        new("nonpad_kv_seqlen", "a layer here is given the mask of the padding as a step it takes, not the lengths of the keys"),
    ];

    // A series the graph takes and nothing has turned yet, which a product multiplies along its last axis, has its channels last: the layout
    // the layers here hold a series in. Anything else is as it is.
    private Made Declared(Made made)
    {
        if (!made.Undeclared || made.Reaching is not { Flow: Flow.Images, Axes: 1 })
        {
            return made;
        }

        _channelsLast = true;

        return made with { Reaching = made.Reaching with { Flow = Flow.ImagesLast }, Undeclared = false };
    }

    // An Attention node: the layer that looks across the steps of a series, taking the steps that made its queries, its keys and its values — one
    // step, when they are the one value.
    private void Attended(OnnxNode node)
    {
        var looked = new[] { 0, 1, 2 }.Select(at => _values.TryGetValue(node.Input(at), out var made) ? Declared(made) : default(Made?)).ToArray();

        if (looked.Any(made => made is null))
        {
            node.Refuse("it looks at queries, keys and values that nodes made or the graph takes, and a network here has no weights to hold in their place.");
            _values[node.Output] = _values.GetValueOrDefault(node.Input(0));

            return;
        }

        var queries = looked[0]!.Value;
        var before = _faults.Count;
        var heads = node.Whole("q_num_heads", 0);

        Require(node, heads);

        if (looked.Any(made => made!.Value.Reaching.Flow != Flow.ImagesLast || made.Value.Reaching.Axes != 1))
        {
            node.Refuse("it looks at a series laid out batch by step by channel, 3 axes, with its channels last, and a network here holds a series so.");
        }

        var layer = new OnnxLayer(description => description.Attention(new Attending(Math.Max(heads, 1)) { Causal = node.Whole("is_causal", 0) != 0 }), Flow.ImagesLast);
        var steps = looked.Select(made => made!.Value.Step).ToArray();

        _built.Add(new Built(node, layer, _faults.Count > before) { Sources = steps.Distinct().Count() == 1 ? [steps[0]] : steps });
        _values[node.Output] = new Made(_built.Count - 1, queries.Reaching with { Flow = Flow.ImagesLast }, false, queries.Width);
        _previous = node.Operator;
    }

    // Everything about an Attention node that no layer here does, each noted at the node.
    private static void Require(OnnxNode node, int heads)
    {
        if (heads < 1)
        {
            node.Refuse("it states no q_num_heads, and attention over a series laid out batch by step by channel splits a step into the heads it states.");
        }
        else if (node.Whole("kv_num_heads", heads) != heads)
        {
            node.Refuse(string.Create(CultureInfo.InvariantCulture, $"it keeps {node.Whole("kv_num_heads", heads)} heads of keys and values for {heads} of queries, kv_num_heads, and a layer here keeps as many of them."));
        }

        for (var at = 0; at < Optional.Length; at++)
        {
            if (node.Input(3 + at).Length > 0)
            {
                node.Refuse($"it is given {Optional[at].Name}, '{node.Input(3 + at).Quoted()}', and {Optional[at].Why}.");
            }
        }

        if (node.Says("scale"))
        {
            node.Refuse("it scales its scores by a scale of its own, and a layer here scales them by one over the root of a head's width.");
        }

        if (node.Number("softcap", 0) != 0)
        {
            node.Refuse("it bounds its scores by a softcap, and a layer here does not.");
        }

        if (node.Says("softmax_precision"))
        {
            node.Refuse("it states the precision its softmax works in, softmax_precision, and a layer here works in single precision throughout.");
        }

        if (node.Whole("qk_matmul_output_mode", 0) != 0)
        {
            node.Refuse("it hands on its scores, qk_matmul_output_mode, and a layer here makes what it looks at.");
        }

        if (new[] { 1, 2, 3 }.Any(at => node.Taken(at) is not null))
        {
            node.Refuse("it hands on its keys and values, or its scores, and a layer here makes what it looks at.");
        }
    }
}
