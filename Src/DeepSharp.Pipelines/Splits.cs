// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>Which part of the data a row belongs to.</summary>
public enum Split
{
    /// <summary>The rows a model learns from, and the only rows anything is fitted on.</summary>
    Train,

    /// <summary>The rows used while choosing between models.</summary>
    Validation,

    /// <summary>The rows kept back until the end.</summary>
    Test,
}

/// <summary>
/// The three shares a split divides the rows into.
/// </summary>
/// <param name="Train">The share the model learns from.</param>
/// <param name="Validation">The share used while choosing between models.</param>
/// <param name="Test">The share kept back until the end.</param>
/// <remarks>
/// One type rather than three loose numbers, because the three are only meaningful together: each has to be
/// a share, and the three have to make a whole. Written as the negation of what a share is, because every
/// comparison against a not-a-number is false and a range test lets NaN through.
/// </remarks>
public readonly record struct SplitShares(double Train, double Validation, double Test)
{
    /// <summary>Checks that these are three shares that make a whole.</summary>
    /// <exception cref="ArgumentOutOfRangeException">One of them is not a share.</exception>
    /// <exception cref="ArgumentException">Together they are not a whole.</exception>
    public void Validate()
    {
        ThrowIfNotAShare(Train, nameof(Train));
        ThrowIfNotAShare(Validation, nameof(Validation));
        ThrowIfNotAShare(Test, nameof(Test));

        var total = Train + Validation + Test;

        if (Math.Abs(total - 1) > 1e-9)
        {
            throw new ArgumentException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"The three shares add up to {total:0.####} and a split has to use every row."),
                nameof(Train));
        }
    }

    /// <summary>Hands out the splits for a given number of rows, in order.</summary>
    /// <param name="rows">How many rows there are.</param>
    /// <returns>Which split each position belongs to, training first.</returns>
    public Split[] Over(int rows)
    {
        var train = (int)Math.Round(rows * Train, MidpointRounding.AwayFromZero);
        var validation = (int)Math.Round(rows * Validation, MidpointRounding.AwayFromZero);

        var splits = new Split[rows];

        for (var at = 0; at < rows; at++)
        {
            splits[at] = at < train ? Split.Train
                : at < train + validation ? Split.Validation
                : Split.Test;
        }

        return splits;
    }

    private static void ThrowIfNotAShare(double share, string name)
    {
        if (!(share > 0 && share <= 1))
        {
            throw new ArgumentOutOfRangeException(
                name, share, "A share of the data is more than none of it and at most all of it.");
        }
    }
}

/// <summary>
/// A step that says which rows belong to which part of the data.
/// </summary>
public interface IAssignsSplits : ISplitStep
{
    /// <summary>Works out which split every row belongs to.</summary>
    /// <param name="table">The rows to divide.</param>
    /// <returns>One split per row, in row order.</returns>
    Split[] Assign(Table table);
}

/// <summary>
/// Split the rows at random, with a seed so it happens the same way every time.
/// </summary>
/// <remarks>
/// The right split for rows that do not depend on one another. The seed is part of the declaration rather
/// than something the fit discovers, because a split you cannot reproduce makes every number after it
/// unreproducible too.
/// </remarks>
public sealed record SplitAtRandomStep : ISplitStep, IAssignsSplits, IPipelineStep<SplitAtRandomStep>
{
    /// <summary>Declares a split at random.</summary>
    /// <param name="shares">How much goes to training, validation and test.</param>
    /// <param name="seed">The number that makes the shuffle repeatable.</param>
    public SplitAtRandomStep(SplitShares shares, int seed)
    {
        shares.Validate();

        Shares = shares;
        Seed = seed;
    }

    /// <summary>How much goes to training, validation and test.</summary>
    public SplitShares Shares { get; }

    /// <summary>The number that makes the shuffle repeatable.</summary>
    public int Seed { get; }

    /// <inheritdoc />
    public static string Name => "split.atRandom";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public Split[] Assign(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        var order = Enumerable.Range(0, table.RowCount).ToArray();
        var random = new Random(Seed);

        for (var at = order.Length - 1; at > 0; at--)
        {
            var other = random.Next(at + 1);
            (order[at], order[other]) = (order[other], order[at]);
        }

        return Shares.Over(table.RowCount).Placed(order);
    }

    /// <inheritdoc />
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();
        writer.WriteString("step", Verb);
        writer.WriteNumber("train", Shares.Train);
        writer.WriteNumber("validation", Shares.Validation);
        writer.WriteNumber("test", Shares.Test);
        writer.WriteNumber("seed", Seed);
        writer.WriteEndObject();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static SplitAtRandomStep ReadFrom(JsonElement element) =>
        new(new SplitShares(
                element.RequiredNumber("train"),
                element.RequiredNumber("validation"),
                element.RequiredNumber("test")),
            (int)element.RequiredNumber("seed"));
}

/// <summary>
/// Split the rows at random, keeping the mixture of one column the same in every part.
/// </summary>
/// <remarks>
/// The right split when an answer is rare. A plain shuffle can leave a class almost absent from validation,
/// and a model measured there is measured on nothing much; this deals each group out separately so every
/// part looks like the whole.
/// </remarks>
public sealed record SplitStratifiedStep : ISplitStep, IAssignsSplits, IPipelineStep<SplitStratifiedStep>
{
    /// <summary>Declares a split that keeps the mixture of a column.</summary>
    /// <param name="column">The column whose mixture is kept.</param>
    /// <param name="shares">How much goes to training, validation and test.</param>
    /// <param name="seed">The number that makes the shuffle repeatable.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    public SplitStratifiedStep(string column, SplitShares shares, int seed)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(column);
        shares.Validate();

        Column = column;
        Shares = shares;
        Seed = seed;
    }

    /// <summary>The column whose mixture is kept the same in every part.</summary>
    public string Column { get; }

    /// <summary>How much goes to training, validation and test.</summary>
    public SplitShares Shares { get; }

    /// <summary>The number that makes the shuffle repeatable.</summary>
    public int Seed { get; }

    /// <inheritdoc />
    public static string Name => "split.stratified";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public Split[] Assign(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        var column = table[Column];
        var splits = new Split[table.RowCount];
        var random = new Random(Seed);

        var groups = Enumerable.Range(0, table.RowCount)
            .GroupBy(row => column.TextAt(row) ?? "\u0000missing")
            .OrderBy(group => group.Key, StringComparer.Ordinal);

        foreach (var group in groups)
        {
            var rows = group.ToArray();

            for (var at = rows.Length - 1; at > 0; at--)
            {
                var other = random.Next(at + 1);
                (rows[at], rows[other]) = (rows[other], rows[at]);
            }

            var share = Shares.Over(rows.Length);

            for (var at = 0; at < rows.Length; at++)
            {
                splits[rows[at]] = share[at];
            }
        }

        return splits;
    }

    /// <inheritdoc />
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();
        writer.WriteString("step", Verb);
        writer.WriteString("column", Column);
        writer.WriteNumber("train", Shares.Train);
        writer.WriteNumber("validation", Shares.Validation);
        writer.WriteNumber("test", Shares.Test);
        writer.WriteNumber("seed", Seed);
        writer.WriteEndObject();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static SplitStratifiedStep ReadFrom(JsonElement element) =>
        new(element.RequiredString("column"),
            new SplitShares(
                element.RequiredNumber("train"),
                element.RequiredNumber("validation"),
                element.RequiredNumber("test")),
            (int)element.RequiredNumber("seed"));
}

/// <summary>
/// Putting a run of splits back onto the rows they belong to.
/// </summary>
internal static class SplitPlacement
{
    /// <summary>Lays a run of splits out over the rows in a given order.</summary>
    /// <param name="inOrder">The splits, training first.</param>
    /// <param name="rows">The rows, in the order the splits should be handed out.</param>
    /// <returns>One split per row, in row order.</returns>
    internal static Split[] Placed(this Split[] inOrder, int[] rows)
    {
        var splits = new Split[inOrder.Length];

        for (var at = 0; at < rows.Length; at++)
        {
            splits[rows[at]] = inOrder[at];
        }

        return splits;
    }
}
