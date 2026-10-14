// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Networks;

public sealed partial class Sequential
{
    /// <summary>
    /// Puts rules on the numbers of the layer the last word makes: a penalty the loss adds, a bound they are held under after a
    /// step — Keras's <c>kernel_regularizer</c>, <c>kernel_constraint</c>, <c>bias_regularizer</c> and <c>bias_constraint</c>.
    /// </summary>
    /// <param name="weights">The rules for its weights — a dense layer's or a convolution's kernel, a normalisation's scale.</param>
    /// <param name="bias">The rules for its bias — a normalisation's shift; none, unless said.</param>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="InvalidOperationException">No word is written yet.</exception>
    /// <remarks>
    /// The rules are put on when the layer is made. A word whose layer holds no number of that name — a rule for a bias on a layer
    /// without one — is refused when the description is lowered.
    /// </remarks>
    public Sequential Ruled(SlotRules weights, SlotRules bias = default)
    {
        RequireWord();
        _words[^1] = new RuledWord(_words[^1], weights, bias, Held: false);

        return this;
    }

    /// <summary>Holds the numbers of the layer the last word makes still: no step moves them. Keras's <c>trainable = False</c>.</summary>
    /// <returns>This description, so the next word can be written after it.</returns>
    /// <exception cref="InvalidOperationException">No word is written yet.</exception>
    /// <remarks>A word whose layer holds no number — an activation, a pooling — has none to hold, and is made as it was.</remarks>
    public Sequential Held()
    {
        RequireWord();
        _words[^1] = new RuledWord(_words[^1], default, default, Held: true);

        return this;
    }

    private void RequireWord()
    {
        if (_words.Count == 0)
        {
            throw new InvalidOperationException("Write the layer the rules are for first: a rule goes on the layer the last word makes.");
        }
    }

    /// <summary>A word whose layer is made as the word makes it, and then has rules put on its numbers.</summary>
    private sealed record RuledWord(IWord Inner, SlotRules Weights, SlotRules Bias, bool Held) : IWord
    {
        public string? Refusal(Shape each) => Inner.Refusal(each);

        public Shape After(Shape each) => Inner.After(each);

        public bool IsAppliedBy(Loss loss) => Inner.IsAppliedBy(loss);

        public Layer Make(Shape each, Draws draws)
        {
            var layer = Inner.Make(each, draws);

            Put(layer, "weight", Weights);
            Put(layer, "bias", Bias);

            if (Held)
            {
                layer.Hold();
            }

            return layer;
        }

        // The rules go on the layer's own number of that name; a layer that holds none is refused where a rule is said for it.
        private static void Put(Layer layer, string name, SlotRules rules)
        {
            if (rules.IsDefault)
            {
                return;
            }

            var parameter = layer.Slots().Select(named => named.Path == name ? named.Slot as Parameter : null).FirstOrDefault(found => found is not null)
                ?? throw new ArgumentException($"A {layer.GetType().Name} holds no '{name}' to put rules on.", nameof(rules));

            parameter.Rules = rules with { Held = parameter.Rules.Held || rules.Held };
        }
    }
}
