// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// How far from what a model says an answer is likely to land, worked out by split conformal prediction on rows the model was
/// neither fitted nor chosen on: one margin for each answer, in the answer's own units.
/// </summary>
/// <param name="Alpha">The share of rows the promise allows to land outside: nine in ten is an alpha of a tenth.</param>
/// <param name="Rows">How many calibration rows the margins were worked out from.</param>
/// <param name="Margins">For each answer, the margin on either side of what the model says, in the answer's own units.</param>
/// <remarks>
/// The margin of an answer is the ⌈(n+1)(1−α)⌉-th smallest of the errors the model made on the n calibration rows, so a row the model
/// has not seen lands within it with a chance of at least 1−α, whatever the model is — a network, an ensemble, a tree — provided the
/// rows are exchangeable with the calibration rows. The width is the same on every row: the guarantee holds on average over rows, and
/// says nothing of any one of them. A series whose later rows differ from its earlier ones breaks the promise it rests on; this
/// holds the exchangeable case only.
/// </remarks>
public sealed record ConformalInterval(double Alpha, int Rows, IReadOnlyList<double> Margins)
{
    private const string AlphaKey = "alpha";
    private const string RowsKey = "rows";
    private const string MarginsKey = "margins";

    /// <summary>Writes the interval as an object: its level, the rows it was worked out from and its margins.</summary>
    /// <param name="writer">Where it is written.</param>
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();
        writer.WriteNumber(AlphaKey, Alpha);
        writer.WriteNumber(RowsKey, Rows);
        writer.WriteStartArray(MarginsKey);

        foreach (var margin in Margins)
        {
            writer.WriteNumberValue(margin);
        }

        writer.WriteEndArray();
        writer.WriteEndObject();
    }

    /// <summary>The ends of an interval around each of some answers: the answers less the margin and plus it.</summary>
    /// <param name="answers">For each row, a number for each answer, in the answer\'s own units.</param>
    /// <returns>The lower and the upper end of each answer of each row.</returns>
    /// <exception cref="ArgumentException">A row has not one number for each margin.</exception>
    public IntervalEnds Around(IReadOnlyList<double[]> answers)
    {
        ArgumentNullException.ThrowIfNull(answers);

        if (answers.Any(row => row.Length != Margins.Count))
        {
            throw new ArgumentException($"An interval has a margin for each of {Margins.Count} answers, and a row has another number of them.", nameof(answers));
        }

        return new IntervalEnds(
            [.. answers.Select(row => row.Select((answer, column) => answer - Margins[column]).ToArray())],
            [.. answers.Select(row => row.Select((answer, column) => answer + Margins[column]).ToArray())]);
    }

    /// <summary>Reads an interval back from the object <see cref="WriteTo"/> writes.</summary>
    /// <param name="element">The object.</param>
    /// <returns>The interval.</returns>
    /// <exception cref="FormatException">It is no object, names a key the interval has not, or its level, rows or margins are not what an interval holds.</exception>
    public static ConformalInterval ReadFrom(JsonElement element)
    {
        if (element.ValueKind != JsonValueKind.Object)
        {
            throw new FormatException("An interval is one object: its level, the rows it was worked out from and its margins.");
        }

        foreach (var property in element.EnumerateObject())
        {
            if (property.Name is not (AlphaKey or RowsKey or MarginsKey))
            {
                throw new FormatException($"An interval has no '{property.Name}': it holds '{AlphaKey}', '{RowsKey}' and '{MarginsKey}'.");
            }
        }

        var alpha = element.TryGetProperty(AlphaKey, out var level) && level.ValueKind == JsonValueKind.Number && level.TryGetDouble(out var shared) && shared > 0 && shared < 1
            ? shared
            : throw new FormatException($"An interval names its level as a number above nothing and below one, under '{AlphaKey}'.");
        var rows = element.TryGetProperty(RowsKey, out var counted) && counted.ValueKind == JsonValueKind.Number && counted.TryGetInt32(out var count) && count > 0
            ? count
            : throw new FormatException($"An interval names the rows it was worked out from as a whole number above nothing, under '{RowsKey}'.");

        if (!element.TryGetProperty(MarginsKey, out var list) || list.ValueKind != JsonValueKind.Array || list.GetArrayLength() == 0)
        {
            throw new FormatException($"An interval names a margin for each answer as a list of numbers, under '{MarginsKey}'.");
        }

        double[] margins =
        [
            .. list.EnumerateArray().Select(value =>
                value.ValueKind == JsonValueKind.Number && value.TryGetDouble(out var margin) && double.IsFinite(margin) && margin >= 0
                    ? margin
                    : throw new FormatException($"A margin is a number that is nought or more, and '{value.GetRawText()}' is none.")),
        ];

        return new ConformalInterval(alpha, rows, margins);
    }
}

/// <summary>The ends of an interval around some answers.</summary>
/// <param name="Lower">For each row, each answer less its margin.</param>
/// <param name="Upper">For each row, each answer plus its margin.</param>
public readonly record struct IntervalEnds(IReadOnlyList<double[]> Lower, IReadOnlyList<double[]> Upper);

/// <summary>Split conformal prediction: the margin the errors on rows set aside to calibrate on allow.</summary>
internal static class Conformal
{
    /// <summary>The interval a model's predictions for the calibration rows allow, at a level.</summary>
    /// <param name="prepared">The run, which set rows aside to calibrate on.</param>
    /// <param name="members">What each model predicted for the calibration rows, in the units the answers were handed over in: one, or several that are combined.</param>
    /// <param name="blend">How several members' answers become one.</param>
    /// <param name="alpha">The share of rows allowed to land outside.</param>
    /// <returns>The interval.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The level is not above nothing and below one.</exception>
    /// <exception cref="InvalidOperationException">
    /// The run sets no rows aside; its answers are classes; or the rows hold no margin that keeps the promise.
    /// </exception>
    /// <exception cref="ArgumentException">The predictions are not for the calibration rows, or not one for each of them, or come back as no number.</exception>
    public static ConformalInterval Of(PreparedData prepared, IReadOnlyList<PartPredictions> members, Blend blend, double alpha)
    {
        ArgumentNullException.ThrowIfNull(prepared);

        if (!(alpha > 0 && alpha < 1))
        {
            throw new ArgumentOutOfRangeException(nameof(alpha), alpha, "A level is the share of rows allowed to land outside the margin: above nothing and below one.");
        }

        prepared.RequireRows();

        var output = prepared.Declaration.Output ?? throw new InvalidOperationException(
            "This pipeline names no answer, so there is nothing to put an interval around: declare its output.");
        var handed = Handover.AnswersOf(prepared, Part.Calibration);

        if (handed.Keys.Count == 0)
        {
            throw new InvalidOperationException(
                "This pipeline sets no rows aside to calibrate an interval on: say Calibrate(…) before its split, so there are rows that nothing was fitted or chosen on.");
        }

        ArgumentNullException.ThrowIfNull(members);

        var calibration = members.Count > 0 ? members[0] : throw new ArgumentException("An interval is worked out from what at least one model predicted for the calibration rows.", nameof(members));

        if (calibration.Batch is not { Part: Part.Calibration, Keys: not null })
        {
            throw new ArgumentException(
                "These predictions were not made for the calibration rows: an interval is worked out from the errors on rows nothing was fitted or chosen on, and no other rows' errors say what those would be.",
                nameof(calibration));
        }

        foreach (var each in members)
        {
            Measurement.ThrowIfOtherRows(Part.Calibration, handed.Keys, each.Batch.Keys ?? []);
        }

        var actual = prepared.BackToOriginal(handed.Answers!, Part.Calibration);
        var predicted = members.Count == 1
            ? prepared.BackToOriginal(calibration.Predictions, Part.Calibration)
            : blend.Combine([.. members.Select(each => prepared.BackToOriginal(each.Predictions, Part.Calibration))], output.Ones);

        if (actual.All(row => row.All(value => value is 0 or 1)))
        {
            throw new InvalidOperationException(
                $"The answers '{string.Join("', '", output.Answers)}' are all nought or one on the calibration rows: they are classes, and a class has a chance, not a width. Give an interval to an amount.");
        }

        var rows = actual.Count;
        var rank = (int)Math.Ceiling(((rows + 1) * (1 - alpha)) - 1e-9);

        if (rank > rows)
        {
            throw new InvalidOperationException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"{rows} rows of calibration hold no margin that keeps a promise of {1 - alpha:0.####} at the level {alpha:0.####}: it takes {Math.Ceiling((1 / alpha) - 1e-9) - 1:0} rows, each as likely to be the one that lands outside. Set more rows aside, or ask for a wider level."));
        }

        double[] margins = new double[output.Answers.Count];

        for (var column = 0; column < margins.Length; column++)
        {
            var errors = new double[rows];

            for (var row = 0; row < rows; row++)
            {
                errors[row] = Math.Abs(actual[row][column] - predicted[row][column]);

                if (!double.IsFinite(errors[row]))
                {
                    throw new ArgumentException(
                        string.Create(CultureInfo.InvariantCulture, $"The prediction for '{output.Answers[column]}' of calibration row {row + 1} comes back as {predicted[row][column]}, which is not a finite number, and no margin is worked out from it."),
                        nameof(calibration));
                }
            }

            Array.Sort(errors);
            margins[column] = errors[rank - 1];
        }

        return new ConformalInterval(alpha, rows, margins);
    }
}
