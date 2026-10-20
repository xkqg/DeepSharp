// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// What two steps made, added up value by value: a residual block's way of handing on what it was given beside what it made.
/// </summary>
/// <remarks>
/// A step that takes several others works on them as they stand in a stack that names them; asked alone, in a place that hands
/// it one tensor, it has nothing to add and says so. The sum is the engine's own addition, so every engine gives the same numbers
/// and the gradient goes back to every step added, unchanged. It has no numbers to learn.
/// </remarks>
public sealed class Sum : Layer, ISaved<Sum>
{
    /// <inheritdoc />
    public static string Name => "sum";

    /// <inheritdoc />
    public int FileVersion => NetworkDocument.EighthVersion;

    /// <inheritdoc />
    public static Sum Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">Always: a sum adds up the work of two steps or more, which a stack hands it by their names.</exception>
    protected override Tensor Compute(Tensor input, Pass pass) =>
        throw new InvalidOperationException("A sum adds up what two steps or more made: it takes them by name inside a stack, and is not handed one tensor.");

    internal override bool Accepts(int steps) => steps >= 2;

    internal override Tensor Taking(IReadOnlyList<Tensor> inputs, Pass pass) =>
        inputs.Skip(1).Aggregate(inputs[0], (sum, next) => pass.Backend.Add(sum, next));
}

/// <summary>
/// What two steps made, laid side by side along the last axis, in the order they are taken: the way a block that reads everything
/// made before it hands those on.
/// </summary>
/// <remarks>
/// The steps are alike in every axis but the last, and what comes out is as wide as they are together. It is made of the products
/// the engine already has — each step's values times a matrix of noughts and ones that sets them where they stand — so no engine
/// learned anything for it. As every such product here, a value that is not a finite number reaches the values beside it.
/// </remarks>
public sealed class Concatenation : Layer, ISaved<Concatenation>
{
    /// <inheritdoc />
    public static string Name => "concatenation";

    /// <inheritdoc />
    public int FileVersion => NetworkDocument.EighthVersion;

    /// <inheritdoc />
    public static Concatenation Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">Always: a concatenation lays two steps or more side by side, which a stack hands it by their names.</exception>
    protected override Tensor Compute(Tensor input, Pass pass) =>
        throw new InvalidOperationException("A concatenation lays what two steps or more made side by side: it takes them by name inside a stack, and is not handed one tensor.");

    internal override bool Accepts(int steps) => steps >= 2;

    internal override Tensor Taking(IReadOnlyList<Tensor> inputs, Pass pass)
    {
        var first = inputs[0];

        if (first.Shape.Rank < 2)
        {
            throw new ArgumentException($"Steps are laid side by side along their last axis, and a {first.Shape} tensor has no axis beside the rows.", nameof(inputs));
        }

        var leading = first.Shape.Axes[..^1];

        foreach (var input in inputs.Skip(1))
        {
            if (!input.Shape.Axes[..^1].SequenceEqual(leading) || input.Shape.Rank != first.Shape.Rank)
            {
                throw new ArgumentException(
                    $"Steps are laid side by side only where they are alike in every axis but the last, and these are {string.Join(" and ", inputs.Select(each => each.Shape.ToString()))}.",
                    nameof(inputs));
            }
        }

        var backend = pass.Backend;
        var rows = 1;

        foreach (var length in leading)
        {
            rows *= length;
        }

        var total = inputs.Sum(input => input.Shape[input.Shape.Rank - 1]);
        var joined = default(Tensor);
        var at = 0;

        foreach (var input in inputs)
        {
            var width = input.Shape[input.Shape.Rank - 1];
            var placed = backend.MatMul(backend.Reshape(input, new Shape(rows, width)), Selection.Of(width, total, (place, column) => column == at + place));

            joined = joined is null ? placed : backend.Add(joined, placed);
            at += width;
        }

        return backend.Reshape(joined!, new Shape([.. leading, total]));
    }
}

/// <summary>A block of a row's last columns, as a step of its own: how a network that takes two inputs reads each of them from the one row it is handed.</summary>
/// <remarks>
/// A network is handed one tensor, and a model of two inputs is handed them side by side as the columns of a row; this step takes
/// the columns of one of them, so the layers after it read only that input. It is the product of the values with a matrix of
/// noughts and ones, as a concatenation is.
/// </remarks>
public sealed class Columns : Layer, ISaved<Columns>
{
    private const string FirstKey = "first";
    private const string CountKey = "count";

    /// <summary>Takes the columns from a place on, as many as asked.</summary>
    /// <param name="first">The first column, counted from nought.</param>
    /// <param name="count">How many columns: one at least.</param>
    /// <exception cref="ArgumentOutOfRangeException">The first column is below nought, or no column is taken.</exception>
    public Columns(int first, int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(first);
        ArgumentOutOfRangeException.ThrowIfLessThan(count, 1);

        First = first;
        Count = count;
    }

    /// <summary>The first column taken, counted from nought.</summary>
    public int First { get; }

    /// <summary>How many columns are taken.</summary>
    public int Count { get; }

    /// <inheritdoc />
    public static string Name => "columns";

    /// <inheritdoc />
    public int FileVersion => NetworkDocument.EighthVersion;

    /// <inheritdoc />
    public static Columns Rebuild(JsonElement settings, Rebuilding rebuilding)
    {
        ArgumentNullException.ThrowIfNull(rebuilding);

        var first = rebuilding.Whole(settings, FirstKey);
        var count = rebuilding.Whole(settings, CountKey);

        return first >= 0 && count >= 1 ? new(first, count) : throw rebuilding.Refuse(first < 0 ? FirstKey : CountKey, "a first column from nought, and a count of one or more");
    }

    /// <inheritdoc />
    public void WriteSettings(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber(FirstKey, First);
        writer.WriteNumber(CountKey, Count);
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">The rows have no columns to take from, or fewer than the block reaches.</exception>
    protected override Tensor Compute(Tensor input, Pass pass)
    {
        if (input.Shape.Rank < 2 || input.Shape[input.Shape.Rank - 1] < First + Count)
        {
            throw new ArgumentException(
                string.Create(CultureInfo.InvariantCulture, $"{Count} columns from column {First} are taken from rows of at least {First + Count} columns, and these are {input.Shape}."),
                nameof(input));
        }

        var backend = pass.Backend;
        var width = input.Shape[input.Shape.Rank - 1];
        var rows = input.Shape.Count / width;
        var taken = backend.MatMul(backend.Reshape(input, new Shape(rows, width)), Selection.Of(width, Count, (place, column) => place == First + column));

        return backend.Reshape(taken, new Shape([.. input.Shape.Axes[..^1], Count]));
    }
}

// A matrix of noughts and ones worked out from its shape alone: a one where the rule says, so a product with it sets values where
// they stand.
internal static class Selection
{
    public static Tensor Of(int rows, int columns, Func<int, int, bool> ones)
    {
        var values = new float[rows * columns];

        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                values[(row * columns) + column] = ones(row, column) ? 1f : 0f;
            }
        }

        return Tensor.From(new Shape(rows, columns), values);
    }
}
