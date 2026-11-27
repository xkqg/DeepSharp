// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using Slices = DeepSharp.Pipelines.Fold;
using Places = DeepSharp.Pipelines.Origin;

namespace DeepSharp.Pipelines;

/// <summary>
/// Split the rows at random, with a seed so it happens the same way every time.
/// </summary>
/// <remarks>
/// The right split for rows that do not depend on one another. The seed is part of the declaration rather
/// than something the fit discovers, because a split you cannot reproduce makes every number after it
/// unreproducible too.
/// </remarks>
public sealed record SplitAtRandomStep : ISplitStep, IRanksRows, IPipelineStep<SplitAtRandomStep>, IDescribesColumns
{
    private static readonly SplitSharesParameter SharesKey = new();

    private static readonly WholeNumberParameter SeedKey = new(
        "seed", "The number that makes the shuffle repeatable: the same seed deals the same rows the same way.", 20260923);

    // A seed is a whole number, so what leaving the key out means is a number no seed can be: below nought.
    private static readonly WholeNumberParameter TestSeedKey = new(
        "testSeed",
        "The number that deals the rows to be measured on, none or more. Given, those rows are the same whatever the seed is, and the seed deals the rest. Left out, the seed deals every part.",
        99)
    {
        AtLeast = 0,
        LeftOut = -1,
    };

    /// <summary>Declares a split at random.</summary>
    /// <param name="shares">How much goes to training, validation and test.</param>
    /// <param name="seed">The number that makes the shuffle repeatable.</param>
    public SplitAtRandomStep(SplitShares shares, int seed)
    {
        Shares = SharesKey.Require(shares);
        Seed = SeedKey.Require(seed);
    }

    /// <summary>Declares a split at random whose rows to be measured on are dealt by a number of their own.</summary>
    /// <param name="shares">How much goes to training, validation and test.</param>
    /// <param name="seed">The number that deals the rows the model learns from and chooses with.</param>
    /// <param name="testSeed">The number that deals the rows to be measured on, none or more.</param>
    /// <exception cref="ArgumentOutOfRangeException">The test seed is below nought.</exception>
    public SplitAtRandomStep(SplitShares shares, int seed, int testSeed)
        : this(shares, seed)
    {
        TestSeed = TestSeedKey.Require(testSeed);
    }

    /// <summary>Declares a split at random that cuts the rows that are neither test nor predict into folds, one of them validation.</summary>
    /// <param name="shares">How much goes to training and test; a validation share, which the fold has no use for, only adds to the rows that are cut.</param>
    /// <param name="seed">The number that deals the rows into slices.</param>
    /// <param name="testSeed">The number that deals the rows to be measured on, none or more; nothing, for the seed to deal them.</param>
    /// <param name="fold">Which slice is validation.</param>
    /// <exception cref="ArgumentOutOfRangeException">The test seed is below nought.</exception>
    public SplitAtRandomStep(SplitShares shares, int seed, int? testSeed, Fold fold)
        : this(shares, seed)
    {
        TestSeed = testSeed is { } locked ? TestSeedKey.Require(locked) : null;
        Fold = fold;
    }

    /// <inheritdoc />
    public static StepParameters<SplitAtRandomStep> Parameters { get; } = new StepParameters<SplitAtRandomStep>()
        .With(SharesKey, step => step.Shares)
        .With(SeedKey, step => step.Seed)
        .With(TestSeedKey, step => step.TestSeed ?? TestSeedKey.LeftOut!.Value)
        .With(Slices.FoldsKey, step => step.Fold?.Folds ?? 0)
        .With(Slices.FoldKey, step => step.Fold?.Index ?? -1)
        .With(Ranking.ByIdKey, step => step.ById);

    /// <summary>How much goes to training, validation and test.</summary>
    public SplitShares Shares { get; }

    /// <summary>The number that makes the shuffle repeatable.</summary>
    public int Seed { get; }

    /// <summary>The number that deals the rows to be measured on, or nothing when the seed deals every part.</summary>
    /// <remarks>
    /// Given, the rows to be measured on and the rows kept for predicting are the same whatever <see cref="Seed"/> is, and
    /// the seed deals what is left into training and validation.
    /// </remarks>
    public int? TestSeed { get; }

    /// <summary>Which slice of the rows that are neither test nor predict is validation; nothing when the rows are cut once, as the shares say.</summary>
    public Fold? Fold { get; }

    /// <summary>Whether this step ranks by the declared id rather than the whole record; set by the builder when the schema names one.</summary>
    public bool ById { get; init; }

    /// <inheritdoc />
    IRanksRows IRanksRows.RankedById() => this with { ById = true };

    /// <inheritdoc />
    public static string Name => "split.atRandom";

    /// <inheritdoc />
    /// <remarks>The second version places rows by what they say, not where they stand.</remarks>
    public static int Since => 2;

    /// <inheritdoc />
    public static string Purpose => "Divides the rows at random, the same way every time for the same seed.";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    /// <remarks>
    /// Each row is ranked by a digest of the seed and what the row says, so the same rows are dealt the same
    /// way whatever order they arrive in, and every copy of a repeated row lands where its first copy does. Where
    /// the schema names an id, every row of one id lands together instead, whatever columns are added or dropped.
    /// With a test seed the rows are first dealt by that number alone and only the rows to be measured on and the
    /// rows kept for predicting are taken from it; the rows left are then ranked again by the seed and divided into
    /// training and validation in the numbers the first deal gave them. A digest ranks a row without regard to the
    /// rows around it, so a test seed equal to the seed leaves that second ranking in the order the first gave, and
    /// deals as no test seed does.
    /// </remarks>
    public Part[] Assign(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        // A repeated record, or a repeated id when this step ranks by one, lands where its first copy does, in a part and in a slice.
        return new RowDeal((rows, seed) => table.Ranked(rows, seed, ById), (one, other) => table.Together(one, other, ById))
            .Dealt(table.RowCount, Shares, Seed, new LockedDeal(TestSeed, Fold));
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The file gives a test seed below nought, or a fold it does not have.</exception>
    /// <exception cref="FormatException">The file names one of a fold's two keys without the other.</exception>
    public static SplitAtRandomStep ReadFrom(JsonElement element)
    {
        var step = Slices.ReadFrom(element) is { } folded
            ? new SplitAtRandomStep(SharesKey.Read(element), SeedKey.Read(element), element.TryGetProperty(TestSeedKey.Key, out _) ? TestSeedKey.Read(element) : null, folded)
            : element.TryGetProperty(TestSeedKey.Key, out _)
                ? new SplitAtRandomStep(SharesKey.Read(element), SeedKey.Read(element), TestSeedKey.Read(element))
                : new SplitAtRandomStep(SharesKey.Read(element), SeedKey.Read(element));

        return step with { ById = Ranking.ByIdKey.Read(element) };
    }
}

/// <summary>
/// Split the rows at random, keeping the mixture of one column the same in every part.
/// </summary>
/// <remarks>
/// The right split when an answer is rare. A plain shuffle can leave a class almost absent from validation,
/// and a model measured there is measured on nothing much; this deals each group out separately so every
/// part looks like the whole.
/// </remarks>
public sealed record SplitStratifiedStep : ISplitStep, IRanksRows, IPipelineStep<SplitStratifiedStep>, IDescribesColumns
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column whose mixture of values is kept the same in every part.", "class", ColumnKinds.Any);

    private static readonly SplitSharesParameter SharesKey = new();

    private static readonly WholeNumberParameter SeedKey = new(
        "seed", "The number that makes the shuffle repeatable: the same seed deals the same rows the same way.", 20260923);

    /// <summary>Declares a split that keeps the mixture of a column.</summary>
    /// <param name="column">The column whose mixture is kept.</param>
    /// <param name="shares">How much goes to training, validation and test.</param>
    /// <param name="seed">The number that makes the shuffle repeatable.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    public SplitStratifiedStep(string column, SplitShares shares, int seed)
    {
        Column = ColumnKey.Require(column);
        Shares = SharesKey.Require(shares);
        Seed = SeedKey.Require(seed);
    }

    /// <summary>Declares a split that keeps the mixture of a column in every part and cuts the learning rows into folds.</summary>
    /// <param name="column">The column whose mixture is kept.</param>
    /// <param name="shares">How much goes to training and test; a validation share, which the fold has no use for, only adds to the rows that are cut.</param>
    /// <param name="seed">The number that makes the shuffle repeatable.</param>
    /// <param name="fold">Which slice of each group is validation.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    public SplitStratifiedStep(string column, SplitShares shares, int seed, Fold fold)
        : this(column, shares, seed)
    {
        Fold = fold;
    }

    /// <inheritdoc />
    public static StepParameters<SplitStratifiedStep> Parameters { get; } = new StepParameters<SplitStratifiedStep>()
        .With(ColumnKey, step => step.Column)
        .With(SharesKey, step => step.Shares)
        .With(SeedKey, step => step.Seed)
        .With(Slices.FoldsKey, step => step.Fold?.Folds ?? 0)
        .With(Slices.FoldKey, step => step.Fold?.Index ?? -1)
        .With(Ranking.ByIdKey, step => step.ById);

    /// <summary>The column whose mixture is kept the same in every part.</summary>
    public string Column { get; }

    /// <summary>Which slice of each group of the rows that are neither test nor predict is validation; nothing when the rows are cut once.</summary>
    public Fold? Fold { get; }

    /// <summary>How much goes to training, validation and test.</summary>
    public SplitShares Shares { get; }

    /// <summary>The number that makes the shuffle repeatable.</summary>
    public int Seed { get; }

    /// <summary>Whether this step ranks by the declared id rather than the whole record; set by the builder when the schema names one.</summary>
    public bool ById { get; init; }

    /// <inheritdoc />
    IRanksRows IRanksRows.RankedById() => this with { ById = true };

    /// <inheritdoc />
    public static string Name => "split.stratified";

    /// <inheritdoc />
    /// <remarks>The second version places rows by what they say, not where they stand.</remarks>
    public static int Since => 2;

    /// <inheritdoc />
    public static string Purpose => "Divides the rows at random while keeping the mixture of one column the same in every part.";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    /// <remarks>
    /// Within each group, each row is ranked by a digest of the seed and what the row says, so the same rows
    /// are dealt the same way whatever order they arrive in, and every copy of a repeated row — which is in the
    /// same group, holding the same value — lands where its first copy does. Where the schema names an id, every
    /// row of one id that falls in the same group lands together instead.
    /// </remarks>
    public Part[] Assign(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        var column = table[Column];
        var parts = new Part[table.RowCount];

        var groups = Enumerable.Range(0, table.RowCount)
            .GroupBy(row => column.TextAt(row) ?? "\u0000missing")
            .OrderBy(group => group.Key, StringComparer.Ordinal);

        foreach (var ranked in groups.Select(group => table.Ranked(group, Seed, ById)))
        {
            var share = Shares.Over(ranked.Length);

            for (var at = 0; at < ranked.Length; at++)
            {
                parts[ranked[at]] = share[at];
            }

            parts.KeptTogether(ranked, (one, other) => table.Together(one, other, ById));

            if (Fold is { } slice)
            {
                parts.Sliced(parts.LearningRows(ranked), slice, (one, other) => table.Together(one, other, ById));
            }
        }

        return parts;
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static SplitStratifiedStep ReadFrom(JsonElement element)
    {
        var step = Slices.ReadFrom(element) is { } folded
            ? new SplitStratifiedStep(ColumnKey.Read(element), SharesKey.Read(element), SeedKey.Read(element), folded)
            : new SplitStratifiedStep(ColumnKey.Read(element), SharesKey.Read(element), SeedKey.Read(element));

        return step with { ById = Ranking.ByIdKey.Read(element) };
    }
}

/// <summary>
/// Split the rows into training, validation and test by where they sit in time.
/// </summary>
/// <remarks>
/// This is the line in the chain. Above it nothing may learn from the data; below it the operations that do
/// become available, and each of them is fitted on the training rows alone.
/// <para>
/// A moment is never divided: every row of one moment lands in the part its first row does, so what happened
/// at one time is on one side of the line or the other. Rows are placed in the order of their time and then
/// of their keys, so the same rows are divided the same way whatever order they arrive in.
/// </para>
/// <para>
/// A gap keeps the last moments of every part apart — before each line and at the end — for an answer read
/// from later rows: without one, the last training rows learn their answers from the rows a model is measured
/// on. The rows in the gap are fitted on by nothing and handed to nothing.
/// </para>
/// </remarks>
public sealed record SplitByTimeStep : ISplitStep, IDividesInTime, IPipelineStep<SplitByTimeStep>, IDescribesColumns
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column that says when a row happened.", "when", ColumnKinds.Ordered);

    private static readonly SplitSharesParameter SharesKey = new();

    private static readonly WholeNumberParameter GapKey = new(
        "gap", "How many of the last moments of every part are kept apart, fitted on by nothing and handed to nothing: at least as many as the rows an answer reads ahead. Left out, none.",
        0)
    {
        AtLeast = 0,
        LeftOut = 0,
    };

    /// <summary>Declares a split in time, by shares that together make a whole.</summary>
    /// <param name="column">The column that says when a row happened.</param>
    /// <param name="shares">How much goes to training, validation, test and predicting.</param>
    /// <exception cref="ArgumentException">The column has no name, or the shares do not make a whole.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A share is not a share: nothing, or more than everything.</exception>
    public SplitByTimeStep(string column, SplitShares shares)
        : this(column, shares, 0)
    {
    }

    /// <summary>Declares a split in time that keeps the last moments of every part apart.</summary>
    /// <param name="column">The column that says when a row happened.</param>
    /// <param name="shares">How much goes to training, validation, test and predicting.</param>
    /// <param name="gap">How many of the last moments of every part are kept apart; none for no gap.</param>
    /// <exception cref="ArgumentException">The column has no name, or the shares do not make a whole.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A share is not a share, or the gap is below nothing.</exception>
    public SplitByTimeStep(string column, SplitShares shares, int gap)
    {
        Column = ColumnKey.Require(column);
        Shares = SharesKey.Require(shares);
        Gap = GapKey.Require(gap);
    }

    /// <summary>Declares a split in time whose cut between training and validation is moved to one of several places.</summary>
    /// <param name="column">The column that says when a row happened.</param>
    /// <param name="shares">How much goes to training and test; a validation share, which the origin has no use for, only adds to the rows that are cut.</param>
    /// <param name="gap">How many of the last moments of training and of validation are kept apart; none for no gap.</param>
    /// <param name="origin">Which place the cut stands at.</param>
    /// <exception cref="ArgumentException">The column has no name, or the shares do not make a whole.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A share is not a share, or the gap is below nothing.</exception>
    public SplitByTimeStep(string column, SplitShares shares, int gap, Origin origin)
        : this(column, shares, gap)
    {
        Origin = origin;
    }

    /// <inheritdoc />
    public static StepParameters<SplitByTimeStep> Parameters { get; } = new StepParameters<SplitByTimeStep>()
        .With(ColumnKey, step => step.Column)
        .With(SharesKey, step => step.Shares)
        .With(GapKey, step => step.Gap)
        .With(Places.OriginsKey, step => step.Origin?.Origins ?? 0)
        .With(Places.OriginKey, step => step.Origin?.Index ?? -1);

    /// <summary>The column that says when a row happened.</summary>
    public string Column { get; }

    /// <summary>Which place the cut between training and validation stands at; nothing when the rows are cut once, as the shares say.</summary>
    public Origin? Origin { get; }

    /// <summary>How much goes to training, validation, test and predicting.</summary>
    public SplitShares Shares { get; }

    /// <summary>How many of the last moments of every part are kept apart.</summary>
    public int Gap { get; }

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">A row has no time, or its time is not a number.</exception>
    public Part[] Assign(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        // A row with no time cannot be placed in a split by time, and guessing where it belongs is how a row
        // from next year ends up in the training data.
        var when = table.RowOrderBy(Column);
        var order = Enumerable.Range(0, table.RowCount).ToArray();

        Array.Sort(order, (one, other) =>
        {
            var compared = when(one, other);

            return compared != 0 ? compared : table.Identities[one].Key.CompareTo(table.Identities[other].Key);
        });

        var parts = Shares.Over(table.RowCount).Placed(order);

        bool SameMoment(int one, int other) => when(one, other) == 0;

        parts.KeptTogether(order, SameMoment);

        if (Origin is { } place)
        {
            // The rows that learn in the order of their time, cut into one block more than there are origins: the blocks up to the
            // origin are training, the next is validation, and the rows after it are fitted on by nothing.
            var learning = parts.LearningRows(order);
            var blocks = learning.Pieces(place.Pieces, SameMoment);

            for (var at = 0; at < learning.Length; at++)
            {
                parts[learning[at]] = blocks[at] <= place.Index ? Part.Train : blocks[at] == place.Index + 1 ? Part.Validation : Part.Gap;
            }
        }

        parts.Gapped(order, Gap, SameMoment);

        // A gap wider than a part takes the whole of it, and a model with nothing to learn from, or nothing to be
        // measured on, is a run that should stop rather than succeed.
        ThrowIfTheGapTookAll(parts, Part.Train, "learn from");
        ThrowIfTheGapTookAll(parts, Part.Test, "measure on");

        return parts;
    }

    private void ThrowIfTheGapTookAll(Part[] parts, Part part, string purpose)
    {
        if (Gap > 0 && !parts.Contains(part))
        {
            throw new InvalidOperationException(string.Create(
                CultureInfo.InvariantCulture,
                $"A gap of {Gap} moments takes every row of {part.Word()}, and leaves nothing to {purpose}. Keep fewer moments apart, or divide more rows."));
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// Where each part starts and ends, in the column it was divided by: a model trained on one stretch of
    /// time and measured on the next can say which stretches they were.
    /// </remarks>
    public void Describe(Table table, IReadOnlyList<Part> parts, FittedStepValues seen)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);
        ArgumentNullException.ThrowIfNull(seen);

        var column = table[Column];
        var when = Comparer<int>.Create(table.RowOrderBy(Column));

        foreach (var part in Part.Dealt)
        {
            var rows = Enumerable.Range(0, table.RowCount).Where(row => parts[row] == part).Order(when).ToArray();

            if (rows.Length > 0)
            {
                seen.Learned($"moments.{part.Word()}", [Moment(column, rows[0]), Moment(column, rows[^1])]);
            }
        }
    }

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    // A moment is written the one way that reads the same everywhere; a number as its shortest exact form.
    private static string Moment(IColumn column, int row) =>
        column is Column<DateTime> moments
            ? moments[row]!.Value.ToString("o", CultureInfo.InvariantCulture)
            : column.TextAt(row)!;

    /// <inheritdoc />
    public static string Name => "split.byTime";

    /// <inheritdoc />
    public static string Purpose => "Divides the rows by when they happened: the earliest to learn from, the latest to be measured on.";

    /// <inheritdoc />
    /// <remarks>The second version keeps a moment whole and places rows by what they say, not where they stand.</remarks>
    public static int Since => 2;

    /// <inheritdoc />
    public string Verb => Name;

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A parameter is missing or is of the wrong kind.</exception>
    public static SplitByTimeStep ReadFrom(JsonElement element) =>
        Places.ReadFrom(element) is { } placed
            ? new(ColumnKey.Read(element), SharesKey.Read(element), GapKey.Read(element), placed)
            : new(ColumnKey.Read(element), SharesKey.Read(element), GapKey.Read(element));
}

/// <summary>How a split in blocks of time deals the rows: how many days a block holds, how far the embargo reaches, and the deal's seeds and fold.</summary>
/// <param name="Days">How many days one block holds, counted from the start of the Unix epoch: seven for a week.</param>
/// <param name="EmbargoDays">How many days on either side of a row a model is chosen, calibrated or measured on keep the training rows away: none, unless said.</param>
/// <remarks>
/// The values one deal is made of, said together. A block is counted from the Unix epoch rather than from the first row,
/// so the same day falls in the same block whatever rows are read with it; a block of seven days therefore runs from a
/// Thursday to a Wednesday.
/// </remarks>
public readonly record struct BlockedDeal(int Days, int EmbargoDays)
{
    /// <summary>The number that deals the blocks; the blocks that learn, when a test seed is said.</summary>
    public int Seed { get; init; } = 20260923;

    /// <summary>The number that deals the blocks to be measured on, so they are the same whatever the seed is; nothing, for the seed to deal every part.</summary>
    public int? TestSeed { get; init; }

    /// <summary>Which slice of the blocks that learn is validation; nothing, for the blocks cut once, as the shares say.</summary>
    public Fold? Fold { get; init; }
}

/// <summary>
/// Split the rows in blocks of time, dealing whole blocks at random, and keep the training rows near a measured row out of
/// the training.
/// </summary>
/// <remarks>
/// <para>
/// A split at random flatters a model whenever the rows follow one another in time and the features say something about
/// the time: a row measured on has training rows on both sides of it, and a model interpolates between them, which it
/// cannot do for a row that comes tomorrow. A split in time answers that and leaves one deal. This one keeps the many
/// deals a split at random gives: the rows are grouped into blocks by when they happened, whole blocks are dealt at random
/// to training, validation and test exactly as a split at random deals rows — a test seed locking the blocks to be measured
/// on, a fold cutting the blocks that learn — and every training row within the embargo of a row a model is chosen,
/// calibrated or measured on is left out of the training alone (Racine; Burman, Chow and Nolan; the purge with an
/// embargo of Lopez de Prado).
/// </para>
/// <para>
/// A row left out stands in the gap a split in time already keeps: fitted on by nothing and handed to nothing, and counted
/// in the fit beside every part, so the pipeline knows which rows it left out and every step below respects it. The parts
/// are not runs of time, so a step that reads the rows before a row, or after it, is refused below this split, as it is
/// below a split at random.
/// </para>
/// </remarks>
public sealed record SplitBlockedStep : ISplitStep, IDividesByAColumn, IPipelineStep<SplitBlockedStep>, IDescribesColumns
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column that says when a row happened.", "when", ColumnKinds.Moments);

    private static readonly SplitSharesParameter SharesKey = new();

    private static readonly WholeNumberParameter DaysKey = new(
        "days", "How many days one block holds, counted from the start of the Unix epoch: seven for a week.", 7)
    {
        AtLeast = 1,
    };

    private static readonly WholeNumberParameter EmbargoDaysKey = new(
        "embargoDays", "How many days on either side of a row a model is chosen, calibrated or measured on keep the training rows away. Left out, none.", 14)
    {
        AtLeast = 0,
        LeftOut = 0,
    };

    private static readonly WholeNumberParameter SeedKey = new(
        "seed", "The number that deals the blocks: the same seed deals the same blocks the same way.", 20260923);

    private static readonly WholeNumberParameter TestSeedKey = new(
        "testSeed",
        "The number that deals the blocks to be measured on, none or more. Given, those blocks are the same whatever the seed is, and the seed deals the rest. Left out, the seed deals every part.",
        99)
    {
        AtLeast = 0,
        LeftOut = -1,
    };

    /// <summary>Declares a split in blocks of time.</summary>
    /// <param name="column">The column that says when a row happened.</param>
    /// <param name="shares">How much goes to training, validation and test, counted in rows.</param>
    /// <param name="deal">How many days a block holds, how far the embargo reaches, and the deal's seeds and fold.</param>
    /// <exception cref="ArgumentException">The column has no name, or the shares do not make a whole.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A block holds no day, the embargo is below nothing, or the test seed is below nought.</exception>
    public SplitBlockedStep(string column, SplitShares shares, BlockedDeal deal)
    {
        Column = ColumnKey.Require(column);
        Shares = SharesKey.Require(shares);
        Days = DaysKey.Require(deal.Days);
        EmbargoDays = EmbargoDaysKey.Require(deal.EmbargoDays);
        Seed = SeedKey.Require(deal.Seed);
        TestSeed = deal.TestSeed is { } locked ? TestSeedKey.Require(locked) : null;
        Fold = deal.Fold;
    }

    /// <inheritdoc />
    public static StepParameters<SplitBlockedStep> Parameters { get; } = new StepParameters<SplitBlockedStep>()
        .With(ColumnKey, step => step.Column)
        .With(SharesKey, step => step.Shares)
        .With(DaysKey, step => step.Days)
        .With(EmbargoDaysKey, step => step.EmbargoDays)
        .With(SeedKey, step => step.Seed)
        .With(TestSeedKey, step => step.TestSeed ?? TestSeedKey.LeftOut!.Value)
        .With(Slices.FoldsKey, step => step.Fold?.Folds ?? 0)
        .With(Slices.FoldKey, step => step.Fold?.Index ?? -1);

    /// <summary>The column that says when a row happened.</summary>
    public string Column { get; }

    /// <summary>How much goes to training, validation and test, counted in rows.</summary>
    public SplitShares Shares { get; }

    /// <summary>How many days one block holds.</summary>
    public int Days { get; }

    /// <summary>How many days on either side of a row a model is chosen, calibrated or measured on keep the training rows away.</summary>
    public int EmbargoDays { get; }

    /// <summary>The number that deals the blocks.</summary>
    public int Seed { get; }

    /// <summary>The number that deals the blocks to be measured on, or nothing when the seed deals every part.</summary>
    public int? TestSeed { get; }

    /// <summary>Which slice of the blocks that learn is validation; nothing when they are cut once.</summary>
    public Fold? Fold { get; }

    /// <summary>The deal this split makes, as it was declared.</summary>
    public BlockedDeal Deal => new(Days, EmbargoDays) { Seed = Seed, TestSeed = TestSeed, Fold = Fold };

    /// <inheritdoc />
    public static string Name => "split.blocked";

    /// <inheritdoc />
    public static int Since => 17;

    /// <inheritdoc />
    public static string Purpose =>
        "Divides the rows by dealing whole blocks of time at random, and leaves the training rows near a row a model is measured on out of the training.";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    /// <exception cref="InvalidOperationException">A row has no moment, or the embargo leaves no row to learn from.</exception>
    public Part[] Assign(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        var blocks = BlocksOf(table);
        var block = new RowGroup("block", row => blocks[row].ToString(CultureInfo.InvariantCulture));
        var parts = new RowDeal((rows, seed) => table.RankedByGroup(rows, seed, block), (one, other) => blocks[one] == blocks[other])
            .Dealt(table.RowCount, Shares, Seed, new LockedDeal(TestSeed, Fold));

        Embargoed(table, parts);

        return parts;
    }

    /// <inheritdoc />
    /// <remarks>How many blocks each part holds: a model trained on so many weeks and measured on so many others can say so.</remarks>
    public void Describe(Table table, IReadOnlyList<Part> parts, FittedStepValues seen)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);
        ArgumentNullException.ThrowIfNull(seen);

        var blocks = BlocksOf(table);

        foreach (var part in Part.Dealt)
        {
            if (Enumerable.Range(0, table.RowCount).Where(row => parts[row] == part).Select(row => blocks[row]).Distinct().Count() is > 0 and var held)
            {
                seen.Learned($"blocks.{part.Word()}", held);
            }
        }
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A parameter is missing or is of the wrong kind, or the file names one of a fold's two keys without the other.</exception>
    public static SplitBlockedStep ReadFrom(JsonElement element) =>
        new(ColumnKey.Read(element), SharesKey.Read(element), new BlockedDeal(DaysKey.Read(element), EmbargoDaysKey.Read(element))
        {
            Seed = SeedKey.Read(element),
            TestSeed = element.TryGetProperty(TestSeedKey.Key, out _) ? TestSeedKey.Read(element) : null,
            Fold = Slices.ReadFrom(element),
        });

    // The block every row falls in: whole days from the start of the Unix epoch, divided into blocks of so many, rounded down,
    // so a moment before the epoch falls in a block below nought and the same day always falls in the same block.
    private long[] BlocksOf(Table table)
    {
        var moments = (Column<DateTime>)table[Column];
        var width = Days * TimeSpan.TicksPerDay;

        return
        [
            .. Enumerable.Range(0, table.RowCount).Select(row =>
            {
                var since = (MomentOf(table, moments, row) - DateTime.UnixEpoch).Ticks;

                return since >= 0 ? since / width : -((width - 1 - since) / width);
            }),
        ];
    }

    private DateTime MomentOf(Table table, Column<DateTime> moments, int row) =>
        moments[row] ?? throw new InvalidOperationException(
            string.Create(
                CultureInfo.InvariantCulture,
                $"Row {table.Identities[row].ReadAt + 1} has no '{Column}', and a row with no moment cannot be placed in a block of time. Fill it above the split, or drop the rows without one."));

    // Every training row within the embargo of a row a model is chosen, calibrated or measured on is left out of the training,
    // on both sides of it, after every part is dealt: the rows measured on stay where they were dealt.
    private void Embargoed(Table table, Part[] parts)
    {
        if (EmbargoDays == 0)
        {
            return;
        }

        var moments = (Column<DateTime>)table[Column];
        var reach = EmbargoDays * TimeSpan.TicksPerDay;
        var measured = Enumerable.Range(0, parts.Length)
            .Where(row => parts[row].IsMeasuredOn)
            .Select(row => MomentOf(table, moments, row).Ticks)
            .Order()
            .ToArray();

        for (var row = 0; row < parts.Length && measured.Length > 0; row++)
        {
            if (parts.IsTraining(row) && measured.Nearest(MomentOf(table, moments, row).Ticks) <= reach)
            {
                parts[row] = Part.Gap;
            }
        }

        if (!parts.Any(part => part.IsLearnedFrom))
        {
            throw new InvalidOperationException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"An embargo of {EmbargoDays} days takes every training row, and leaves nothing to learn from. Keep the embargo shorter, or the blocks longer."));
        }
    }
}

/// <summary>How far a moment is from the nearest of a sorted list of moments.</summary>
internal static class NearestMoments
{
    extension(long[] sorted)
    {
        /// <summary>How far a moment is from the nearest moment in this sorted list, which holds at least one.</summary>
        /// <param name="moment">The moment, in ticks.</param>
        /// <returns>The distance, in ticks.</returns>
        internal long Nearest(long moment)
        {
            var at = Array.BinarySearch(sorted, moment);

            if (at >= 0)
            {
                return 0;
            }

            var after = ~at;
            var before = after - 1;

            return Math.Min(
                after < sorted.Length ? sorted[after] - moment : long.MaxValue,
                before >= 0 ? moment - sorted[before] : long.MaxValue);
        }
    }
}