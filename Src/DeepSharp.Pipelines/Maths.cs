// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>A shape a column is pulled into, by arithmetic that learns nothing.</summary>
public enum Maths
{
    /// <summary>The natural logarithm. For a column where the ratio matters and the difference does not.</summary>
    Log,

    /// <summary>The logarithm of one plus the value, so that nought stays nought and survives.</summary>
    Log1P,

    /// <summary>One divided by the value.</summary>
    Reciprocal,

    /// <summary>The square root, a gentler version of a logarithm.</summary>
    Sqrt,

    /// <summary>The value times itself, for a relation that bends the other way.</summary>
    Square,

    /// <summary>The arcsine of the square root, for a column of proportions.</summary>
    ArcSin,

    /// <summary>The magnitude, dropping the sign.</summary>
    Abs,

    /// <summary>Minus one, nought or one: only the direction survives.</summary>
    Sign,
}

/// <summary>
/// Pulls a column into another shape, by arithmetic that learns nothing from the data.
/// </summary>
/// <remarks>
/// These are the variance-stabilising transformations, and they belong before the split precisely because
/// they learn nothing: the logarithm of a number does not depend on any other number. A column of money or
/// of volume usually wants one — there the ratio between two values carries the meaning and the difference
/// does not — and the scaling that comes after it then has something symmetric to work with.
/// <para>
/// Each refuses the values it has no answer for rather than handing back a not-a-number: a logarithm of
/// nought or less, a division by nought, a root of a negative. A gap stays a gap.
/// </para>
/// </remarks>
public sealed record MathsStep : IPipelineStep<MathsStep>, IAddsColumns, IUndoesItself, IDescribesColumns
{
    private static readonly ColumnParameter ColumnKey = new(
        "column", "The column to pull into another shape.", "column", ColumnKinds.Numbers);

    private static readonly OneOfParameter<Maths> MathsKey = new(
        "maths", "Which shape: a logarithm, a root, a reciprocal, a square, an arcsine, the magnitude or the sign.", Maths.Log1P);

    private static readonly NewColumnParameter IntoKey = new(
        "into", "What the result is called; the same column, unless a file says otherwise.", "column", optional: true);

    /// <summary>Declares that a column is pulled into another shape.</summary>
    /// <param name="column">The column to reshape.</param>
    /// <param name="maths">Which shape.</param>
    /// <param name="into">What to call the result; the same column, unless you say otherwise.</param>
    /// <exception cref="ArgumentException">The column has no name.</exception>
    public MathsStep(string column, Maths maths, string? into = null)
    {
        Column = ColumnKey.Require(column);
        Maths = MathsKey.Require(maths);
        Into = IntoKey.Require(into) ?? Column;
    }

    /// <inheritdoc />
    public static StepParameters<MathsStep> Parameters { get; } = new StepParameters<MathsStep>()
        .With(ColumnKey, step => step.Column)
        .With(MathsKey, step => step.Maths)
        .With(IntoKey, step => step.Into);

    /// <summary>The column being reshaped.</summary>
    public string Column { get; }

    /// <summary>Which shape it is pulled into.</summary>
    public Maths Maths { get; }

    /// <summary>What the result is called.</summary>
    public string Into { get; }

    /// <inheritdoc />
    public string Produces => Into;

    /// <inheritdoc />
    /// <remarks>
    /// The column it read: a reshaping into a new column leaves the column it was made from behind, and whatever was
    /// done to that one before is undone next.
    /// </remarks>
    public string From(string column) => Column;

    /// <inheritdoc />
    /// <remarks>
    /// Undoing a bend is not free of consequence. A model trained on the logarithm of a price predicts the
    /// logarithm, and the exponent of that is nearer the middle value than the average one — the classic
    /// retransformation bias. What comes back here is the plain inverse and nothing more; correcting it
    /// towards an average needs the model's own residuals, which live on the other side of the handover.
    /// </remarks>
    public double Undo(double value, FittedStepValues? fitted) => Maths switch
    {
        Maths.Log => Math.Exp(value),
        Maths.Log1P => Math.Exp(value) - 1,
        Maths.Reciprocal => value == 0
            ? throw Irreversible("nought has no reciprocal to come back from")
            : 1 / value,
        Maths.Sqrt => value * value,
        Maths.Square => value < 0
            ? throw Irreversible("a square is never negative, so this value cannot have come from one")
            : Math.Sqrt(value),
        Maths.ArcSin => Math.Pow(Math.Sin(value), 2),
        _ => throw Irreversible($"{Maths.ToString().ToLowerInvariant()} throws the sign or the size away"),
    };

    /// <inheritdoc />
    public ColumnState After(ColumnState before)
    {
        ArgumentNullException.ThrowIfNull(before);

        return before.With(Into, ColumnKind.Number);
    }

    /// <inheritdoc />
    public static string Name => "maths";

    /// <inheritdoc />
    public static string Purpose => "Pulls a column into another shape by arithmetic that learns nothing: a logarithm, a root, a reciprocal.";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public void AddTo(Table table)
    {
        ArgumentNullException.ThrowIfNull(table);

        var values = table.NumbersOf(Column);
        var shaped = new double?[values.Length];

        for (var row = 0; row < values.Length; row++)
        {
            if (values[row] is not { } value)
            {
                continue;
            }

            shaped[row] = Shape(value, row);
        }

        table.Put(new Column<double>(Into, ColumnKind.Number, shaped));
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static MathsStep ReadFrom(JsonElement element) =>
        new(ColumnKey.Read(element), MathsKey.Read(element), IntoKey.Read(element));

    private double Shape(double value, int row) => Maths switch
    {
        Maths.Log when value <= 0 => throw Impossible(value, row, "a logarithm of nought or less"),
        Maths.Log => Math.Log(value),
        Maths.Log1P when value <= -1 => throw Impossible(value, row, "a logarithm of nought or less"),
        Maths.Log1P => Math.Log(value + 1),
        Maths.Reciprocal when value == 0 => throw Impossible(value, row, "a division by nought"),
        Maths.Reciprocal => 1 / value,
        Maths.Sqrt when value < 0 => throw Impossible(value, row, "a root of a negative number"),
        Maths.Sqrt => Math.Sqrt(value),
        Maths.Square => value * value,
        Maths.ArcSin when value is < 0 or > 1 => throw Impossible(value, row, "an arcsine outside nought and one"),
        Maths.ArcSin => Math.Asin(Math.Sqrt(value)),
        Maths.Abs => Math.Abs(value),
        _ => Math.Sign(value),
    };

    private InvalidOperationException Irreversible(string why) =>
        new($"'{Into}' cannot be put back into the units it came from: {why}. "
            + "Predict something this pipeline can undo, or undo it yourself.");

    private InvalidOperationException Impossible(double value, int row, string what) =>
        new($"Row {row + 1} of '{Column}' is {value.ToString(CultureInfo.InvariantCulture)}, and "
            + $"{Maths.ToString().ToLowerInvariant()} would be {what}. Shift the column first, or leave this one out.");
}
