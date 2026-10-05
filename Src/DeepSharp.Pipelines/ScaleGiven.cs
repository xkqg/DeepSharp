// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// The range a column is known to hold, said by whoever knows the data rather than read from it.
/// </summary>
/// <param name="Lowest">The lowest value the column can hold.</param>
/// <param name="Highest">The highest value it can hold.</param>
public readonly record struct KnownRange(double Lowest, double Highest)
{
    /// <summary>The lowest value the column can hold.</summary>
    /// <exception cref="ArgumentException">It is not a finite number.</exception>
    public double Lowest { get; } = Finite(Lowest, nameof(Lowest));

    /// <summary>The highest value it can hold.</summary>
    /// <exception cref="ArgumentException">It is not a finite number, or it is not above the lowest.</exception>
    public double Highest { get; } = Finite(Highest, nameof(Highest)) > Finite(Lowest, nameof(Lowest))
        ? Highest
        : throw new ArgumentException(
            string.Create(CultureInfo.InvariantCulture, $"A range runs from a lower value to a higher one, and this one says {Lowest} to {Highest}."),
            nameof(Highest));

    /// <summary>How wide the range is.</summary>
    public double Width => Highest - Lowest;

    private static double Finite(double value, string name) =>
        double.IsFinite(value)
            ? value
            : throw new ArgumentException($"A range is made of finite numbers, and {name} is not one.", name);
}

/// <summary>
/// Scales a column into a range from bounds you give, where the features are worked out.
/// </summary>
/// <remarks>
/// <para>
/// Every other scaling reads its numbers from the training rows — a mean, a spread, a smallest and a largest — so it
/// stands below the split and is replayed from what it learned. This one is told them: a fare runs from nothing to the
/// most anybody paid, an hour from nought to twenty-three, a share from nought to one. That is knowledge about the
/// column, not about the rows, so no row of validation or test taught it anything and it may stand where the features
/// are worked out — which is what lets a feature be worked out from a column that is already on the scale a model takes.
/// </para>
/// <para>
/// It learns nothing, so it writes nothing into the fitted half of the file, and the way back needs nothing written
/// down: a prediction is stretched by the width you gave and moved back to where you said the column starts, exactly.
/// </para>
/// <para>
/// A value outside the bounds lands outside the range rather than being quietly moved to its edge. The bounds are a
/// statement about the column, and a value beyond them says the statement was wrong, which is worth seeing rather than
/// hiding.
/// </para>
/// </remarks>
public sealed record ScaleGivenStep : IPipelineStep<ScaleGivenStep>, IAddsColumns, IDescribesColumns, IUndoesItself
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column to scale.", "column", ColumnKinds.Fillable);

    private static readonly NumberParameter LowestKey = new("lowest", "The lowest value the column can hold.", 0) { IsABound = true };

    private static readonly NumberParameter HighestKey = new("highest", "The highest value it can hold.", 1) { IsABound = true };

    private static readonly OneOfParameter<Form> LandsKey = new(
        "lands", "Where the scaled values land: between minus one and one, or between nothing and one.", Form.Signed);

    /// <summary>Declares that a column is scaled from bounds you give.</summary>
    /// <param name="column">The column to scale.</param>
    /// <param name="range">The range it is known to hold.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    public ScaleGivenStep(string column, KnownRange range)
    {
        Column = ColumnKey.Require(column);
        Range = range;
    }

    /// <summary>The column this step scales.</summary>
    public string Column { get; }

    /// <summary>The range it is known to hold.</summary>
    public KnownRange Range { get; }

    /// <summary>Where the scaled values land; between minus one and one, unless said.</summary>
    public Form Lands
    {
        get;
        init => field = LandsKey.Require(value);
    } = Form.Signed;

    /// <inheritdoc />
    public static string Name => "scale.given";

    /// <inheritdoc />
    public static string Purpose =>
        "Scales a column into a range from bounds you give, above the split, so a feature worked out after it is worked out from scaled columns.";

    /// <summary>The first version of the pipeline file that has this verb.</summary>
    public static int Since => 7;

    /// <inheritdoc />
    public static StepParameters<ScaleGivenStep> Parameters { get; } = new StepParameters<ScaleGivenStep>()
        .With(ColumnKey, step => step.Column)
        .With(LowestKey, step => step.Range.Lowest)
        .With(HighestKey, step => step.Range.Highest)
        .With(LandsKey, step => step.Lands);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public string Produces => Column;

    /// <inheritdoc />
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return before.With(Column, ColumnKind.Number, Lands);
    }

    /// <inheritdoc />
    public void AddTo(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        if (table[Column] is not Column<double> and not Column<long>)
        {
            throw new InvalidOperationException(
                $"'{Column}' holds {table[Column].Kind.ToString().ToLowerInvariant()}, and a range of numbers does not scale it.");
        }

        var values = table.NumbersOf(Column);

        table.Put(new Column<double>(
            Column, ColumnKind.Number, values.Select(value => value is { } held ? Scaled(held) : (double?)null)));
    }

    /// <inheritdoc />
    /// <remarks>Exact, and with nothing fitted: the bounds were given rather than learned.</remarks>
    public double Undo(double value, FittedStepValues? fitted) =>
        ((Lands is Form.Signed ? (value + 1) / 2 : value) * Range.Width) + Range.Lowest;

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    /// <exception cref="FormatException">A parameter is missing, or is not what it should be.</exception>
    public static ScaleGivenStep ReadFrom(JsonElement element) =>
        new(ColumnKey.Read(element), new KnownRange(LowestKey.Read(element), HighestKey.Read(element)))
        {
            Lands = LandsKey.Read(element),
        };

    // Where a value lands: its share of the range you gave, and for a signed form that share stretched over minus one to
    // one. Nothing is clamped — a value beyond the bounds lands beyond the range, and says so.
    private double Scaled(double value) =>
        Lands is Form.Signed
            ? (((value - Range.Lowest) / Range.Width) * 2) - 1
            : (value - Range.Lowest) / Range.Width;
}

/// <summary>
/// The columns one line scales, each between the bounds it is given.
/// </summary>
/// <remarks>
/// The same line every other per-column verb has, and the same fan-out behind it: what reaches the declaration is one
/// step a column, in the order the columns were named.
/// </remarks>
public sealed class ScaleGivenLine : IDeclaresSteps
{
    private readonly List<IPipelineStep> _steps = [];

    /// <summary>The steps this line declares, in the order the columns were named.</summary>
    IReadOnlyList<IPipelineStep> IDeclaresSteps.Steps => _steps;

    /// <summary>A column, and the range it is known to hold.</summary>
    /// <param name="column">The column.</param>
    /// <param name="lowest">The lowest value it can hold.</param>
    /// <param name="highest">The highest value it can hold.</param>
    /// <param name="lands">Where its scaled values land; between minus one and one, unless said.</param>
    /// <returns>This line, so the next column can be written after it.</returns>
    public ScaleGivenLine Between(string column, double lowest, double highest, Form lands = Form.Signed)
    {
        _steps.Add(new ScaleGivenStep(column, new KnownRange(lowest, highest)) { Lands = lands });

        return this;
    }
}
