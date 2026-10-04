// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// Puts the rows in an order drawn from a seed, before they are divided.
/// </summary>
/// <remarks>
/// <para>
/// A file is often written in an order that means something — every survivor first, every month in turn, every machine
/// one after another — and rows divided in that order give parts that are not alike. Shuffling before the split is what
/// gives each part the same mixture. Shuffling again within each epoch is the training loop's own work, and it already
/// does that from its own seed; this one is about which rows end up beside which.
/// </para>
/// <para>
/// Without this verb the rows keep the order they were read in, which is what every file the library has ever written
/// already says. It is the one thing in a pipeline that decides the order, so it is never written beside
/// <see cref="OrderByStep"/>: a series in time is put in order and must stay that way, and a pipeline cannot be told
/// both things at once.
/// </para>
/// <para>
/// The order is drawn the way the splits draw theirs: each row ranked by a digest of what it says together with the
/// seed, rather than by a generator of random numbers. A generator shuffles places, so the same rows arriving in
/// another order would be shuffled differently; a digest ranks a row the same wherever it stands, on every machine.
/// </para>
/// </remarks>
public sealed record ShuffleStep : IPipelineStep<ShuffleStep>, IOrdersRows, IDescribesColumns
{
    private static readonly WholeNumberParameter SeedKey = new(
        "seed", "The number that makes the shuffle repeatable: the same seed puts the same rows in the same order.", 20260929);

    /// <summary>Declares that the rows are shuffled before they are divided.</summary>
    /// <param name="seed">The number the order is drawn from.</param>
    public ShuffleStep(int seed = 20260929) => Seed = SeedKey.Require(seed);

    /// <summary>The number the order is drawn from.</summary>
    public int Seed { get; }

    /// <inheritdoc />
    public static string Name => "shuffle";

    /// <inheritdoc />
    public static string Purpose =>
        "Puts the rows in an order drawn from a seed, before they are divided, so every part holds the same mixture.";

    /// <summary>The first version of the pipeline file that has this verb.</summary>
    public static int Since => 7;

    /// <inheritdoc />
    public static StepParameters<ShuffleStep> Parameters { get; } =
        new StepParameters<ShuffleStep>().With(SeedKey, step => step.Seed);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    /// <remarks>It moves no value and makes no column: only where the rows stand.</remarks>
    public ColumnState After(ColumnState before) => before;

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A parameter is missing, or is not what it should be.</exception>
    public static ShuffleStep ReadFrom(JsonElement element) => new(SeedKey.Read(element));

    /// <inheritdoc />
    public IReadOnlyList<int> RowOrder(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        return table.Ranked(Enumerable.Range(0, table.RowCount), Seed);
    }
}
