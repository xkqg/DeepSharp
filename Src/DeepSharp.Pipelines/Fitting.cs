// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// What one step learned while it was fitted.
/// </summary>
/// <remarks>
/// A number for a mean or a bound, a list for the categories an encoder found. This is the half of a saved
/// pipeline the fit writes, and it is kept apart from the declaration so the same declaration can be fitted
/// again on fresh data without anybody editing anything.
/// </remarks>
public sealed class FittedStepValues
{
    private readonly Dictionary<string, double> _numbers = [];
    private readonly Dictionary<string, IReadOnlyList<string>> _lists = [];

    /// <summary>What this step learned, as numbers.</summary>
    public IReadOnlyDictionary<string, double> Numbers => _numbers;

    /// <summary>What this step learned, as lists of words.</summary>
    public IReadOnlyDictionary<string, IReadOnlyList<string>> Lists => _lists;

    /// <summary>Records a number this step learned.</summary>
    /// <param name="name">What the number is.</param>
    /// <param name="value">The number.</param>
    public void Learned(string name, double value) => _numbers[name] = value;

    /// <summary>Records a list this step learned.</summary>
    /// <param name="name">What the list is.</param>
    /// <param name="values">The list.</param>
    public void Learned(string name, IReadOnlyList<string> values) => _lists[name] = values;

    /// <summary>The number this step learned under that name.</summary>
    /// <param name="name">What the number is.</param>
    /// <returns>The number.</returns>
    /// <exception cref="InvalidOperationException">The fit never learned it.</exception>
    public double Number(string name) =>
        _numbers.TryGetValue(name, out var value)
            ? value
            : throw new InvalidOperationException(
                $"This pipeline was never fitted for '{name}', so there is nothing to replay.");

    /// <summary>The list this step learned under that name.</summary>
    /// <param name="name">What the list is.</param>
    /// <returns>The list.</returns>
    /// <exception cref="InvalidOperationException">The fit never learned it.</exception>
    public IReadOnlyList<string> List(string name) =>
        _lists.TryGetValue(name, out var values)
            ? values
            : throw new InvalidOperationException(
                $"This pipeline was never fitted for '{name}', so there is nothing to replay.");

    /// <summary>Writes what was learned, as one JSON object.</summary>
    /// <param name="writer">The writer positioned where the object belongs.</param>
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();

        foreach (var (name, value) in _numbers.OrderBy(each => each.Key, StringComparer.Ordinal))
        {
            writer.WriteNumber(name, value);
        }

        foreach (var (name, values) in _lists.OrderBy(each => each.Key, StringComparer.Ordinal))
        {
            writer.WriteStartArray(name);

            foreach (var value in values)
            {
                writer.WriteStringValue(value);
            }

            writer.WriteEndArray();
        }

        writer.WriteEndObject();
    }
}

/// <summary>
/// A step that learns from the training rows and then replays what it learned.
/// </summary>
/// <remarks>
/// Fitting and applying are two separate acts on purpose. The fit sees the training rows and nothing else;
/// applying sees every row and learns nothing, so validation, test and a row arriving in production a year
/// from now are all treated with the same numbers.
/// </remarks>
public interface ILearnsFromData : IFittedStep
{
    /// <summary>Learns whatever this step needs, from the training rows alone.</summary>
    /// <param name="table">The data.</param>
    /// <param name="splits">Which split each row belongs to.</param>
    /// <returns>What was learned.</returns>
    FittedStepValues Fit(Table table, IReadOnlyList<Split> splits);

    /// <summary>Applies what was learned to every row.</summary>
    /// <param name="table">The data, changed in place.</param>
    /// <param name="fitted">What the fit learned.</param>
    void ApplyTo(Table table, FittedStepValues fitted);
}

/// <summary>
/// A pipeline that has been run: the data, where every row landed, and what each step learned.
/// </summary>
public sealed class PreparedData
{
    internal PreparedData(
        PipelineDeclaration declaration,
        Table table,
        IReadOnlyList<Split> splits,
        IReadOnlyDictionary<int, FittedStepValues> fitted)
    {
        Declaration = declaration;
        Table = table;
        Splits = splits;
        Fitted = fitted;
    }

    /// <summary>The steps, exactly as they were declared.</summary>
    public PipelineDeclaration Declaration { get; }

    /// <summary>The data, as the last step left it.</summary>
    public Table Table { get; }

    /// <summary>Which split each row belongs to, in row order.</summary>
    public IReadOnlyList<Split> Splits { get; }

    /// <summary>What each step learned, by its position in the declaration.</summary>
    /// <remarks>
    /// Keyed by position rather than by verb, because a declaration may hold the same step twice and two
    /// identical steps are indistinguishable by value.
    /// </remarks>
    public IReadOnlyDictionary<int, FittedStepValues> Fitted { get; }

    /// <summary>How many rows landed in one split.</summary>
    /// <param name="split">The split to count.</param>
    /// <returns>The number of rows.</returns>
    public int CountIn(Split split) => Splits.Count(each => each == split);

    /// <summary>Writes the whole pipeline: what was declared, and what the fit learned.</summary>
    /// <returns>The pipeline as one JSON document.</returns>
    /// <remarks>
    /// Both halves in one file, because a model without what its pipeline learned cannot be used: the
    /// numbers reaching it would not be the numbers it was trained on.
    /// </remarks>
    public string ToJson()
    {
        var buffer = new MemoryStream();

        using (var writer = new Utf8JsonWriter(buffer, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WriteStartArray("declaration");

            foreach (var step in Declaration.Steps)
            {
                step.WriteTo(writer);
            }

            writer.WriteEndArray();
            writer.WriteStartObject("fitted");

            foreach (var (at, values) in Fitted.OrderBy(each => each.Key))
            {
                writer.WritePropertyName(at.ToString(CultureInfo.InvariantCulture));
                values.WriteTo(writer);
            }

            writer.WriteEndObject();
            writer.WriteEndObject();
        }

        return System.Text.Encoding.UTF8.GetString(buffer.ToArray());
    }
}

/// <summary>
/// Saying no, where a fit has nothing to learn from.
/// </summary>
internal static class Refuse
{
    /// <summary>Hands back the training values, or refuses when there are none.</summary>
    /// <param name="training">The values the training rows offered.</param>
    /// <param name="column">The column being fitted, for the message.</param>
    /// <returns>The values.</returns>
    /// <exception cref="InvalidOperationException">Every training row is a gap.</exception>
    internal static double[] IfEmpty(double[] training, string column) =>
        training.Length > 0
            ? training
            : throw new InvalidOperationException(
                $"Every training row of '{column}' is a gap, so there is nothing to learn a fill value from.");
}
