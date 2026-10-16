// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// A category as a small vector it learns: the columns of a row that hold one category's ones and noughts, replaced by the
/// vector of the category the row holds — the rest of the row as it was.
/// </summary>
/// <remarks>
/// The columns are the block an encoder writes a column of words down as, one for each category, and exactly one of them holds a
/// one; the block times the weight is that category's row of the weight, which is what PyTorch's and Keras's embeddings look up
/// by the category's place — the same numbers, by a product the seam already has. The weight is PyTorch's <c>weight</c>: a row
/// for each column of the block, in the block's order, by the width of a vector, starting from a normal curve of spread one as
/// PyTorch starts it. The vector stands where the block's first column stood, and the row's other columns keep their order.
/// <para>
/// A row a category was missing from is a row whose gap marker is the one, and a category the training rows never held one whose
/// reserved column is: each is a column of the block, so each has a vector of its own.
/// </para>
/// </remarks>
public sealed class Embedding : Layer, ISaved<Embedding>
{
    private const string FeaturesKey = "features";
    private const string BlockKey = "block";
    private const string WidthKey = "width";

    /// <summary>An embedding of the category a block of a row's columns holds, its vectors starting from the given draws.</summary>
    /// <param name="features">How many columns a row holds.</param>
    /// <param name="block">The places of the block's columns in the row, counting from nought, in the order of the categories: two at least, each once.</param>
    /// <param name="width">How many numbers a category's vector holds: one at least.</param>
    /// <param name="draws">The draws the vectors start from.</param>
    /// <exception cref="ArgumentOutOfRangeException">The row or a vector holds fewer than one number.</exception>
    /// <exception cref="ArgumentException">The block holds fewer than two columns, one twice, or one the row does not hold.</exception>
    /// <exception cref="ArgumentNullException">There is no block, or there are no draws.</exception>
    public Embedding(int features, IReadOnlyList<int> block, int width, Draws draws)
    {
        ArgumentNullException.ThrowIfNull(block);
        ArgumentNullException.ThrowIfNull(draws);
        ArgumentOutOfRangeException.ThrowIfLessThan(features, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);

        if (block.Count < 2 || block.Distinct().Count() != block.Count || block.Any(place => place < 0 || place >= features))
        {
            throw new ArgumentException(
                $"An embedding reads a category from two columns of the row at least, each once and each one of the row's {features}, and was told [{string.Join(", ", block)}].",
                nameof(block));
        }

        Features = features;
        Block = [.. block];
        Weight = AddParameter("weight", Tensor.From(new Shape(block.Count, width), [.. Enumerable.Range(0, block.Count * width).Select(_ => (float)draws.NextNormal())]));
    }

    /// <summary>How many columns a row holds.</summary>
    public int Features { get; }

    /// <summary>The places of the block's columns in the row, in the order of the categories.</summary>
    public IReadOnlyList<int> Block { get; }

    /// <summary>The vectors: a row for each column of the block, by the width of a vector.</summary>
    public Parameter Weight { get; }

    /// <summary>How many numbers a category's vector holds.</summary>
    public int Width => Weight.Value.Shape[1];

    /// <summary>How many columns a row comes out with: the block gone, and the vector in its place.</summary>
    public int Outputs => Features - Block.Count + Width;

    /// <inheritdoc />
    public static string Name => "embedding";

    /// <inheritdoc />
    public static Embedding Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        return new(rebuilding.Whole(settings, FeaturesKey), rebuilding.Wholes(settings, BlockKey), rebuilding.Whole(settings, WidthKey), rebuilding.Draws);
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber(FeaturesKey, Features);
        writer.WriteStartArray(BlockKey);

        foreach (var place in Block)
        {
            writer.WriteNumberValue(place);
        }

        writer.WriteEndArray();
        writer.WriteNumber(WidthKey, Width);
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The input is not a batch of rows of as many columns as the embedding was built for.</exception>
    protected override Tensor Compute(Tensor input, Pass pass)
    {
        if (input.Shape.Rank != 2 || input.Shape[1] != Features)
        {
            throw new ArgumentException($"An embedding of a row of {Features} columns takes a batch of such rows, and was handed a {input.Shape} one.", nameof(input));
        }

        var backend = pass.Backend;
        var (kept, taken, placed) = Laid();
        var vectors = backend.MatMul(backend.MatMul(input, taken), Weight.Value);

        return backend.Add(backend.MatMul(input, kept), backend.MatMul(vectors, placed));
    }

    // Three matrices of noughts and ones, worked out from the places alone: the row's other columns carried to where they stand
    // after, the block's columns taken out in the order of the categories, and a vector put where the block's first column stood.
    private Layout Laid()
    {
        var outputs = Outputs;
        var kept = new float[Features * outputs];
        var taken = new float[Features * Block.Count];
        var placed = new float[Width * outputs];
        var first = Block.Min();
        var at = 0;

        for (var column = 0; column < Features; column++)
        {
            if (column == first)
            {
                for (var number = 0; number < Width; number++)
                {
                    placed[(number * outputs) + at + number] = 1f;
                }

                at += Width;
            }

            var category = IndexIn(Block, column);

            if (category >= 0)
            {
                taken[(column * Block.Count) + category] = 1f;
            }
            else
            {
                kept[(column * outputs) + at] = 1f;
                at++;
            }
        }

        return new Layout(
            Tensor.From(new Shape(Features, outputs), kept),
            Tensor.From(new Shape(Features, Block.Count), taken),
            Tensor.From(new Shape(Width, outputs), placed));
    }

    private static int IndexIn(IReadOnlyList<int> block, int column)
    {
        for (var at = 0; at < block.Count; at++)
        {
            if (block[at] == column)
            {
                return at;
            }
        }

        return -1;
    }

    private readonly record struct Layout(Tensor Kept, Tensor Taken, Tensor Placed);
}
