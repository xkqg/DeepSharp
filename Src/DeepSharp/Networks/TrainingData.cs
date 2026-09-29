// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// Rows a network learns from or is measured on: the features of each, and the answers it should give.
/// </summary>
/// <remarks>
/// What a loop is handed, whatever prepared the rows: a pipeline's handover, turned into tensors by the package that joins
/// the two, or rows a caller made. The first axis of both is the row. A row is named in a refusal as the rows name it —
/// by where it was read, when whatever prepared them says so — and by its place here otherwise.
/// </remarks>
public sealed class TrainingData
{
    private readonly IReadOnlyList<string>? _rowNames;

    /// <summary>Rows of features and their answers.</summary>
    /// <param name="features">A row for each example, of any shape after it: a row of numbers, an image.</param>
    /// <param name="answers">A row of answers for each example.</param>
    /// <exception cref="ArgumentException">The features have no axis beside the rows, the answers are not a matrix, or the two hold different numbers of rows.</exception>
    public TrainingData(Tensor features, Tensor answers)
    {
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(answers);

        if (features.Shape.Rank < 2 || answers.Shape.Rank != 2 || features.Shape[0] != answers.Shape[0])
        {
            throw new ArgumentException(
                $"Training rows are features with at least one axis beside the rows and a matrix of answers, a row each; these are {features.Shape} and {answers.Shape}.",
                nameof(answers));
        }

        Features = features;
        Answers = answers;
    }

    /// <summary>The features: a row for each example.</summary>
    public Tensor Features { get; }

    /// <summary>The answers: a row for each example.</summary>
    public Tensor Answers { get; }

    /// <summary>How many rows there are.</summary>
    public int Count => Features.Shape[0];

    /// <summary>What each row is called in a refusal — where it was read, say; by its place, from one, unless said.</summary>
    /// <exception cref="ArgumentException">There is not one name for each row.</exception>
    public IReadOnlyList<string>? RowNames
    {
        get => _rowNames;
        init => _rowNames = value is null || value.Count == Count
            ? value
            : throw new ArgumentException($"There are {Count} rows and {value.Count} names for them.", nameof(value));
    }

    /// <summary>What one row is called.</summary>
    internal string NameOf(int row) => _rowNames?[row] ?? $"row {row + 1}";

    /// <summary>The rows at the given places, in that order.</summary>
    internal TrainingData Rows(ReadOnlySpan<int> places) =>
        new(Gathered(Features, places), Gathered(Answers, places));

    /// <summary>So many rows from the given place on.</summary>
    internal TrainingData Slice(int start, int count)
    {
        var places = new int[count];

        for (var at = 0; at < count; at++)
        {
            places[at] = start + at;
        }

        return Rows(places);
    }

    private static Tensor Gathered(Tensor tensor, ReadOnlySpan<int> places)
    {
        var width = tensor.Shape.Count / tensor.Shape[0];
        var values = new float[places.Length * width];
        var source = tensor.Values;

        for (var at = 0; at < places.Length; at++)
        {
            source.Slice(places[at] * width, width).CopyTo(values.AsSpan(at * width, width));
        }

        return Tensor.From(new Shape([places.Length, .. tensor.Shape.Axes[1..]]), values);
    }
}
