// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Import.Keras;

/// <summary>What a layer of a functional model is called with besides the tensors it takes.</summary>
/// <param name="Causal">Whether it was called with <c>use_causal_mask</c>: each step looks only at the steps up to its own.</param>
internal readonly record struct Called(bool Causal);

/// <summary>
/// Keras's <c>MultiHeadAttention</c> called over one series, as <c>layer(x, x)</c>: the layer of the same name here, whose one projection holds the
/// three that Keras keeps apart, each as a kernel with an axis for the heads, and whose output layer is Keras's <c>attention_output</c>.
/// </summary>
/// <remarks>
/// Attention over two series, one that asks and one that is looked at, is refused where the model is read, since the layer here looks across one.
/// Keras names its four kernels one way in the archive and another in the file it saved before it, and lists them in another order, so they
/// are found by their names.
/// </remarks>
internal sealed class MultiHeadAttentionKind : KerasKind
{
    private static readonly string[] Penalised = ["kernel_regularizer", "bias_regularizer", "activity_regularizer", "kernel_constraint", "bias_constraint"];

    /// <inheritdoc />
    public override string Name => "MultiHeadAttention";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer) => Reading(layer, default);

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer, Called call) => Reading(layer, call);

    private static KerasLayer Reading(KerasSettings layer, Called call)
    {
        var heads = layer.Setting<int>("num_heads");
        var key = layer.Setting<int>("key_dim");
        var dropout = layer.Setting("dropout", 0.0);

        Require(layer, key);

        return new KerasLayer(
            description => description.MultiHeadAttention(new Attending(Math.Max(heads, 1)) { Causal = call.Causal, Dropout = dropout }),
            "linear")
        {
            Remade = numbers => Remake(numbers, new Heads(heads, key)),
        };
    }

    // What no layer here is: every setting an attention layer says that is not the form Keras makes by default.
    private static void Require(KerasSettings layer, int key)
    {
        if (!layer.Setting("use_bias", true))
        {
            layer.Refuse("it keeps no bias, use_bias false, and attention here keeps its biases.");
        }

        if (layer.Setting("value_dim", key) is var value && value != key)
        {
            layer.Refuse(string.Create(CultureInfo.InvariantCulture, $"its value_dim is {value} where its key_dim is {key}, and a head here asks and answers in channels of one width."));
        }

        if (layer.Says("output_shape"))
        {
            layer.Refuse($"it gives an output_shape, {layer.Written("output_shape")!.Quoted()}, and attention here gives a series as wide as the one it was handed.");
        }

        if (layer.Says("attention_axes") && layer.Setting<int[]?>("attention_axes") is not ([1] or [-2]))
        {
            layer.Refuse($"it looks along axes {layer.Written("attention_axes")!.Quoted()}, and attention here looks across the steps of a series, axis 1.");
        }

        if (layer.Setting("use_gate", false))
        {
            layer.Refuse("it gates its heads, use_gate, and attention here does not.");
        }

        if (layer.Says("sliding_window"))
        {
            layer.Refuse($"it looks at a window of steps, sliding_window {layer.Written("sliding_window")!.Quoted()}, and attention here looks at every step, or at every step up to its own.");
        }

        foreach (var penalty in Penalised.Where(layer.Says))
        {
            layer.Refuse($"its {penalty} is {layer.Written(penalty)!.Quoted()}, and attention here takes no penalty or bound on its numbers.");
        }
    }

    // Keras's eight numbers — a kernel and a bias for the query, the key, the value and the output — as the slots of the layer keep them: the
    // three projections laid side by side in one, each head's channels after those of the head before it, and the output's kernel as the
    // linear layer keeps it.
    private static List<RemadeNumber> Remake(IReadOnlyList<HeldNumber> held, Heads heads)
    {
        var projections = new[] { "query", "key", "value", "output" }.Select(part => new Projection(Find(held, part, kernel: true), Find(held, part, kernel: false))).ToArray();

        if (projections.Any(projection => projection.Missing))
        {
            return [];
        }

        var query = projections[0];
        var output = projections[3];

        // Kernels that are not written as the heads and their channels the layer says are kept as they are written, for the slot to refuse by its shape.
        if (projections.Take(3).Any(projection => !projection.Kernel!.Value.Lengths.AsSpan(1).SequenceEqual([(ulong)heads.Count, (ulong)heads.Each]))
            || !output.Kernel!.Value.Lengths.AsSpan(0, 2).SequenceEqual([(ulong)heads.Count, (ulong)heads.Each]))
        {
            return
            [
                new RemadeNumber("in_proj_weight", new Shape([.. query.Kernel!.Value.Lengths.Select(length => checked((int)length))]), query.Kernel.Value.Values!, query.Kernel.Value.Source),
            ];
        }

        var rows = checked((int)query.Kernel!.Value.Lengths[0]);
        var wide = query.Kernel.Value.Values!.Length / Math.Max(rows, 1);
        var joined = new float[rows * 3 * wide];

        for (var part = 0; part < 3; part++)
        {
            var kernel = projections[part].Kernel!.Value;

            for (var row = 0; row < rows; row++)
            {
                kernel.Values.AsSpan(row * wide, wide).CopyTo(joined.AsSpan((row * 3 * wide) + (part * wide), wide));
            }
        }

        var outputs = checked((int)output.Kernel!.Value.Lengths[^1]);

        return
        [
            new RemadeNumber("in_proj_weight", new Shape(rows, 3 * wide), joined, query.Kernel.Value.Source),
            new RemadeNumber("in_proj_bias", new Shape(3 * wide), [.. projections.Take(3).SelectMany(projection => projection.Bias!.Value.Values!)], query.Bias!.Value.Source),
            new RemadeNumber("out_proj.weight", new Shape(output.Kernel.Value.Values!.Length / outputs, outputs), output.Kernel.Value.Values!, output.Kernel.Value.Source),
            new RemadeNumber("out_proj.bias", new Shape(output.Bias!.Value.Values!.Length), output.Bias.Value.Values!, output.Bias.Value.Source),
        ];
    }

    /// <summary>How many heads the layer has, and how many channels each asks and answers in.</summary>
    private readonly record struct Heads(int Count, int Each);

    /// <summary>The kernel and the bias Keras keeps for one of the four projections; nothing for one the file does not hold.</summary>
    private readonly record struct Projection(HeldNumber? Kernel, HeldNumber? Bias)
    {
        public bool Missing => Kernel is null || Bias is null;
    }

    // The kernel or the bias of one projection, by the names the two files give it: query_dense and query, then 0 and kernel, 1 and bias.
    private static HeldNumber? Find(IReadOnlyList<HeldNumber> held, string part, bool kernel)
    {
        foreach (var number in held)
        {
            var names = number.Name.Split('/');
            var last = names[^1];
            var ofPart = names[..^1].Any(name => name.TrimStart('_') == part || name.TrimStart('_') == $"{part}_dense" || (part == "output" && name == "attention_output"));

            if (ofPart && (kernel ? last is "0" or "kernel" : last is "1" or "bias"))
            {
                return number;
            }
        }

        return null;
    }
}
