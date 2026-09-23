// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// Deals with a value that is not a number.
/// </summary>
/// <remarks>
/// Missing and not-a-number are different things and get different verbs. A value is missing when it was
/// never there, which is data; a value is not a number when arithmetic produced no number, which is a fault
/// further upstream. So this refuses by default: quietly replacing it would carry somebody else's broken
/// division into a model and call it a measurement.
/// </remarks>
public sealed record FillNaNStep : IFittedStep, ILearnsFromData, IPipelineStep<FillNaNStep>
{
    /// <summary>Declares what happens to a value in this column that is not a number.</summary>
    /// <param name="column">The column to watch.</param>
    /// <param name="strategy">What to do — refusing is the default and usually the right answer.</param>
    /// <exception cref="ArgumentException">The column has no name, or the strategy is not one of the names.</exception>
    public FillNaNStep(string column, FillStrategy strategy = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(column);

        Strategy = string.IsNullOrWhiteSpace(strategy.Name) ? With.Refuse : strategy;

        if (!With.Knows(Strategy.Name) || Strategy.Name == "previous")
        {
            throw new ArgumentException(
                $"'{Strategy.Name}' is not a way of dealing with a value that is not a number.", nameof(strategy));
        }

        Column = column;
    }

    /// <summary>The column being watched.</summary>
    public string Column { get; }

    /// <summary>What happens to a value that is not a number.</summary>
    public FillStrategy Strategy { get; }

    /// <inheritdoc />
    public static string Name => "fill.nan";

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    public FittedStepValues Fit(Table table, IReadOnlyList<Part> parts)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);

        var values = Numbers.Of(table, Column);
        var learned = new FittedStepValues();

        learned.Learned("notNumbers", values.Count(value => value is { } number && double.IsNaN(number)));

        var training = Enumerable.Range(0, values.Length)
            .Where(row => parts[row] == Part.Train && values[row] is { } number && !double.IsNaN(number))
            .Select(row => values[row]!.Value)
            .Order()
            .ToArray();

        switch (Strategy.Name)
        {
            case "mean":
                learned.Learned("value", Refuse.IfEmpty(training, Column).Average());
                break;

            case "median":
                var sorted = Refuse.IfEmpty(training, Column);
                learned.Learned("value", sorted.Length % 2 == 1
                    ? sorted[sorted.Length / 2]
                    : (sorted[(sorted.Length / 2) - 1] + sorted[sorted.Length / 2]) / 2);
                break;

            case "zero":
                learned.Learned("value", 0);
                break;

            case "constant":
                learned.Learned("value", Strategy.Value!.Value);
                break;

            default:
                break;
        }

        return learned;
    }

    /// <inheritdoc />
    public void ApplyTo(Table table, FittedStepValues fitted)
    {
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(fitted);

        if (table[Column] is not Column<double> numbers)
        {
            throw new InvalidOperationException(
                $"'{Column}' holds {table[Column].Kind.ToString().ToLowerInvariant()}, "
                + "which cannot hold a value that is not a number.");
        }

        for (var row = 0; row < numbers.Count; row++)
        {
            if (numbers[row] is not { } value || !double.IsNaN(value))
            {
                continue;
            }

            numbers[row] = Strategy.Name == "refuse"
                ? throw new InvalidOperationException(
                    $"Row {row + 1} of '{Column}' is not a number. Something upstream produced it, and this "
                    + "pipeline will not carry it into a model as if it were a measurement.")
                : fitted.Number("value");
        }
    }

    /// <inheritdoc />
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();
        writer.WriteString("step", Verb);
        writer.WriteString("column", Column);

        if (Strategy.Value is { } value)
        {
            writer.WriteStartObject("with");
            writer.WriteString("kind", Strategy.Name);
            writer.WriteNumber("value", value);
            writer.WriteEndObject();
        }
        else
        {
            writer.WriteString("with", Strategy.Name);
        }

        writer.WriteEndObject();
    }

    /// <summary>Reads this step back out of a file.</summary>
    /// <param name="element">The JSON object the step was written as.</param>
    /// <returns>The step the file describes.</returns>
    public static FillNaNStep ReadFrom(JsonElement element)
    {
        var column = element.RequiredString("column");

        if (!element.TryGetProperty("with", out var with))
        {
            throw new FormatException("The step is missing a text value for 'with'.");
        }

        return with.ValueKind switch
        {
            JsonValueKind.String => new FillNaNStep(column, new FillStrategy(with.GetString()!)),
            JsonValueKind.Object => new FillNaNStep(
                column, new FillStrategy(with.RequiredString("kind"), with.RequiredNumber("value"))),
            _ => throw new FormatException("The step is missing a text value for 'with'."),
        };
    }
}
