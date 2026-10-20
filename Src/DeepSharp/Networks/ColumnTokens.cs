// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// Turns each column of a row into a token of its own: the column's value times a vector the layer learns for that column, plus a vector it
/// learns for it — the numerical tokenizer of Gorishniy et al.'s FT-Transformer. A row of so many columns becomes a series of so many steps,
/// one for each column, of as many channels as the tokens are wide, for attention to look across.
/// </summary>
/// <remarks>
/// It holds a <c>weight</c> and a <c>bias</c> of as many rows as there are columns and as many columns as a token is wide, both starting evenly
/// within one over the root of the token's width, as their package starts them. It is not saved or declared: it is built to be measured, and what
/// the measurement says decides whether it stays.
/// </remarks>
public sealed class ColumnTokens : Layer
{
    /// <summary>A tokenizer of so many columns into tokens of so many channels.</summary>
    /// <param name="columns">How many columns a row has.</param>
    /// <param name="embed">How many channels a token has.</param>
    /// <param name="draws">The draws its start is taken from: the weight first, then the bias.</param>
    /// <exception cref="ArgumentOutOfRangeException">A row has fewer than one column or a token fewer than one channel.</exception>
    public ColumnTokens(int columns, int embed, Draws draws)
    {
        ArgumentNullException.ThrowIfNull(draws);
        ArgumentOutOfRangeException.ThrowIfLessThan(columns, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(embed, 1);

        var bound = 1 / Math.Sqrt(embed);

        Weight = AddParameter("weight", Drawn(new Shape(columns, embed), draws, bound));
        Bias = AddParameter("bias", Drawn(new Shape(columns, embed), draws, bound));
    }

    /// <summary>The vector each column's value is multiplied by: a row for each column.</summary>
    public Parameter Weight { get; }

    /// <summary>The vector added to it: a row for each column.</summary>
    public Parameter Bias { get; }

    /// <summary>How many columns a row has.</summary>
    public int Columns => Weight.Value.Shape[0];

    /// <summary>How many channels a token has.</summary>
    public int Embed => Weight.Value.Shape[1];

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass)
    {
        if (input.Shape.Rank != 2 || input.Shape[1] != Columns)
        {
            throw new ArgumentException($"Tokens are made of rows of {Columns} columns, and this was handed a {input.Shape} batch.", nameof(input));
        }

        var backend = pass.Backend;
        var rows = input.Shape[0];
        var wide = Columns * Embed;
        var spread = backend.MatMul(input, Selection.Of(Columns, wide, (column, place) => place / Embed == column));
        var scaled = backend.Multiply(spread, backend.RowsOf(backend.Reshape(Weight.Value, new Shape(wide)), rows));

        return backend.Reshape(backend.AddRow(scaled, backend.Reshape(Bias.Value, new Shape(wide))), new Shape(rows, Columns, Embed));
    }

    private static Tensor Drawn(Shape shape, Draws draws, double bound) =>
        Tensor.From(shape, [.. Enumerable.Range(0, shape.Count).Select(_ => (float)(bound * ((2 * draws.NextDouble()) - 1)))]);
}
