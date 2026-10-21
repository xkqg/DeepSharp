// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Import.Keras;

/// <summary>
/// Keras's <c>LSTM</c> and <c>GRU</c>, read to the layers of the same names here: one layer, read forwards, to its last state or at every
/// step, with the activations Keras starts them with. A GRU is read as Keras 3 makes it, the reset gate taken after the hidden numbers'
/// own bias, which is how PyTorch's works it out too.
/// </summary>
/// <param name="name">The class name Keras writes: <c>LSTM</c> or <c>GRU</c>.</param>
internal sealed class RecurrentKind(string name) : KerasKind
{
    private static readonly string[] Penalised = ["kernel_regularizer", "recurrent_regularizer", "bias_regularizer", "kernel_constraint", "recurrent_constraint", "bias_constraint"];

    /// <inheritdoc />
    public override string Name => name;

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer) => Reading(name, layer, bothWays: false);

    /// <summary>The layer a recurrent class and its settings are, read forwards or in both directions.</summary>
    /// <param name="kind">The class: <c>LSTM</c> or <c>GRU</c>.</param>
    /// <param name="layer">The layer's settings.</param>
    /// <param name="bothWays">Whether a Bidirectional layer reads it in both directions.</param>
    public static KerasLayer Reading(string kind, KerasSettings layer, bool bothWays)
    {
        var units = layer.Setting<int>("units");
        var sequences = layer.Setting("return_sequences", false);

        Require(layer, kind);

        var held = Held(layer);

        return new KerasLayer(
            description =>
            {
                _ = kind == "LSTM" ? description.Lstm(units, bothWays, 1, sequences) : description.Gru(units, bothWays, 1, sequences);

                if (held)
                {
                    description.Held();
                }
            },
            "linear")
        {
            Remade = numbers => Remake(kind == "GRU", bothWays, numbers),
        };
    }

    // What no layer here is: every setting a recurrent layer says that is not the form Keras makes by default.
    private static void Require(KerasSettings layer, string kind)
    {
        if (layer.Setting("go_backwards", false))
        {
            layer.Refuse("it reads each series backwards, go_backwards, and a recurrent layer here reads forwards, or in both directions through a Bidirectional layer.");
        }

        if (layer.Setting("return_state", false))
        {
            layer.Refuse("it hands back its state as well, return_state, and a recurrent layer here makes what it reads.");
        }

        if (layer.Setting("stateful", false))
        {
            layer.Refuse("it carries its state from one batch to the next, stateful, and a recurrent layer here starts every series from nought.");
        }

        foreach (var key in new[] { "dropout", "recurrent_dropout" }.Where(key => layer.Setting(key, 0.0) is not 0.0))
        {
            layer.Refuse(string.Create(CultureInfo.InvariantCulture, $"it drops numbers out inside the layer, {key} {layer.Setting(key, 0.0)}, and a recurrent layer here has no dropout of its own: put a dropout layer between layers."));
        }

        if (layer.Setting("activation", "tanh") is var activation and not "tanh")
        {
            layer.Refuse($"its activation, '{activation.Quoted()}', and a recurrent layer here works with the hyperbolic tangent and a sigmoid for its gates.");
        }

        if (layer.Setting("recurrent_activation", "sigmoid") is var gates and not "sigmoid")
        {
            layer.Refuse($"its recurrent_activation, '{gates.Quoted()}', and a recurrent layer here works with the hyperbolic tangent and a sigmoid for its gates.");
        }

        if (!layer.Setting("use_bias", true))
        {
            layer.Refuse("it keeps no bias, use_bias false, and a recurrent layer here keeps its biases.");
        }

        if (kind == "GRU" && !layer.Setting("reset_after", true))
        {
            layer.Refuse("its reset_after is false, the form Keras 2 made the default, and a GRU here takes its reset gate after the hidden numbers' own bias, as Keras 3's and PyTorch's do.");
        }

        foreach (var key in Penalised.Where(layer.Says))
        {
            layer.Refuse($"its {key} is {layer.Written(key)!.Quoted()}, and a recurrent layer here takes no penalty or bound on its numbers.");
        }
    }

    // Keras's numbers of one layer — a kernel, a recurrent kernel and a bias for each direction, the forward one first — as the slots
    // of the layer keep them: the kernels as they are, a GRU's gates put in the order PyTorch's stand in, a GRU's two biases apart, and
    // the second bias an LSTM never had nought.
    private static List<RemadeNumber> Remake(bool gru, bool bothWays, IReadOnlyList<HeldNumber> held)
    {
        var made = new List<RemadeNumber>();
        var gates = gru ? 3 : 4;

        for (var direction = 0; direction < (bothWays ? 2 : 1); direction++)
        {
            var own = held.Skip(direction * 3).Take(3).ToArray();

            if (own.Length < 3)
            {
                break;
            }

            var suffix = direction == 0 ? "_l0" : "_l0_reverse";

            made.Add(Matrix($"weight_ih{suffix}", own[0], gru, gates));
            made.Add(Matrix($"weight_hh{suffix}", own[1], gru, gates));

            if (gru && own[2].Lengths is [2, var wide])
            {
                made.Add(Vector($"bias_ih{suffix}", own[2].Values!.AsSpan(0, (int)wide), own[2], new Gating(gru, gates)));
                made.Add(Vector($"bias_hh{suffix}", own[2].Values!.AsSpan((int)wide, (int)wide), own[2], new Gating(gru, gates)));
            }
            else
            {
                made.Add(Vector($"bias_ih{suffix}", own[2].Values!, own[2], new Gating(false, gates)));
                made.Add(Vector($"bias_hh{suffix}", new float[own[2].Values!.Length], own[2], new Gating(false, gates)));
            }
        }

        return made;
    }

    // A kernel or a recurrent kernel, inputs by the gates' outputs: Keras's gate order for a GRU is update, reset, new, and here reset, update, new.
    private static RemadeNumber Matrix(string slot, HeldNumber number, bool gru, int gates)
    {
        var lengths = number.Lengths.Select(length => checked((int)length)).ToArray();
        var values = number.Values!;

        if (gru && lengths is [var rows, var wide] && wide % gates == 0)
        {
            var one = wide / gates;
            var turned = new float[values.Length];

            for (var row = 0; row < rows; row++)
            {
                values.AsSpan(row * wide, one).CopyTo(turned.AsSpan((row * wide) + one, one));
                values.AsSpan((row * wide) + one, one).CopyTo(turned.AsSpan(row * wide, one));
                values.AsSpan((row * wide) + (2 * one), one).CopyTo(turned.AsSpan((row * wide) + (2 * one), one));
            }

            values = turned;
        }

        return new RemadeNumber(slot, new Shape(lengths), values, number.Source);
    }

    // Whether the gates are a GRU's, which Keras orders otherwise than here, and how many gates there are.
    private readonly record struct Gating(bool Gru, int Count);

    // A bias, the gates one after another: put in the order a GRU's gates stand in here.
    private static RemadeNumber Vector(string slot, ReadOnlySpan<float> values, HeldNumber number, Gating gating)
    {
        var one = values.Length / gating.Count;
        var made = values.ToArray();

        if (gating.Gru && values.Length % gating.Count == 0)
        {
            values[..one].CopyTo(made.AsSpan(one, one));
            values[one..(2 * one)].CopyTo(made.AsSpan(0, one));
        }

        return new RemadeNumber(slot, new Shape(made.Length), made, number.Source);
    }
}

/// <summary>
/// Keras's <c>Bidirectional</c> wrapper of an LSTM or a GRU: the layer read in both directions, its two states laid side by side — the
/// wrapper's <c>concat</c> — and, given a layer that returns sequences, at every step.
/// </summary>
internal sealed class BidirectionalKind : KerasKind
{
    /// <inheritdoc />
    public override string Name => "Bidirectional";

    /// <inheritdoc />
    public override KerasLayer Read(KerasSettings layer)
    {
        var inner = layer.Within("layer");
        var kind = inner.Setting("class_name", string.Empty);

        if (layer.Setting("merge_mode", "concat") is var merge and not "concat")
        {
            layer.Refuse($"it merges the two directions by '{merge.Quoted()}', and a bidirectional layer here lays them side by side: concat.");
        }

        // Keras writes the copy it makes of the forward layer to read backwards; a backward layer that is not that copy is the layer's own.
        if (layer.Says("backward_layer") && !IsCopy(layer.Within("backward_layer"), inner, kind))
        {
            layer.Refuse("it is given a backward layer of its own, and a bidirectional layer here reads backwards with a copy of its forward one.");
        }

        if (kind is not ("LSTM" or "GRU"))
        {
            layer.Refuse($"it wraps a {kind.Quoted()}, and a bidirectional layer here wraps an LSTM or a GRU.");

            return new KerasLayer(null, "linear");
        }

        return RecurrentKind.Reading(kind, inner.Within("config"), bothWays: true);
    }

    // Whether a backward layer is a copy of the forward one that reads backwards: the same class, the same width of state, the same sequences.
    private static bool IsCopy(KerasSettings backward, KerasSettings forward, string kind)
    {
        var (copy, own) = (backward.Within("config"), forward.Within("config"));

        return backward.Setting("class_name", string.Empty) == kind
            && copy.Setting("go_backwards", false)
            && copy.Setting<int>("units") == own.Setting<int>("units")
            && copy.Setting("return_sequences", false) == own.Setting("return_sequences", false);
    }
}
