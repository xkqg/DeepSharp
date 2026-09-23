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
    private readonly Dictionary<string, IReadOnlyList<double>> _curves = [];

    /// <summary>What this step learned, as numbers.</summary>
    public IReadOnlyDictionary<string, double> Numbers => _numbers;

    /// <summary>What this step learned, as lists of words.</summary>
    public IReadOnlyDictionary<string, IReadOnlyList<string>> Lists => _lists;

    /// <summary>What this step learned, as runs of numbers — the shape of a distribution, say.</summary>
    public IReadOnlyDictionary<string, IReadOnlyList<double>> Curves => _curves;

    /// <summary>Records a number this step learned.</summary>
    /// <param name="name">What the number is.</param>
    /// <param name="value">The number.</param>
    public void Learned(string name, double value) => _numbers[name] = value;

    /// <summary>Records a list this step learned.</summary>
    /// <param name="name">What the list is.</param>
    /// <param name="values">The list.</param>
    public void Learned(string name, IReadOnlyList<string> values) => _lists[name] = values;

    /// <summary>Records a run of numbers this step learned.</summary>
    /// <param name="name">What the run is.</param>
    /// <param name="values">The numbers, in order.</param>
    public void Learned(string name, IReadOnlyList<double> values) => _curves[name] = values;

    /// <summary>The run of numbers this step learned under that name.</summary>
    /// <param name="name">What the run is.</param>
    /// <returns>The numbers, in order.</returns>
    /// <exception cref="InvalidOperationException">The fit never learned it.</exception>
    public IReadOnlyList<double> Curve(string name) =>
        _curves.TryGetValue(name, out var values)
            ? values
            : throw new InvalidOperationException(
                $"This pipeline was never fitted for '{name}', so there is nothing to replay.");

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

        foreach (var (name, values) in _curves.OrderBy(each => each.Key, StringComparer.Ordinal))
        {
            writer.WriteStartArray(name);

            foreach (var value in values)
            {
                writer.WriteNumberValue(value);
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
    /// <param name="parts">Which part each row belongs to.</param>
    /// <returns>What was learned.</returns>
    FittedStepValues Fit(Table table, IReadOnlyList<Part> parts);

    /// <summary>Applies what was learned to every row.</summary>
    /// <param name="table">The data, changed in place.</param>
    /// <param name="fitted">What the fit learned.</param>
    void ApplyTo(Table table, FittedStepValues fitted);
}

/// <summary>
/// A step that can put a value back the way it found it.
/// </summary>
/// <remarks>
/// For the target, and only really for the target. Scale what a model predicts and its predictions come
/// back scaled: an error of 0.03 means nothing until it is 0.03 of something, and a report in scaled units
/// flatters every model equally. So the way back is part of the saved pipeline rather than a sum somebody
/// does by hand afterwards.
/// </remarks>
public interface IUndoesItself : IPipelineStep
{
    /// <summary>The column this step leaves behind.</summary>
    string Produces { get; }

    /// <summary>Puts one value back into the units this step found it in.</summary>
    /// <param name="value">The value as this step left it.</param>
    /// <param name="fitted">What this step learned, when it learned anything.</param>
    /// <returns>The value in the units of the column before this step touched it.</returns>
    double Undo(double value, FittedStepValues? fitted);
}

/// <summary>
/// A pipeline that has been run: the data, where every row landed, and what each step learned.
/// </summary>
public sealed class PreparedData
{
    /// <summary>A pipeline that has been run, or one loaded back from the file it was saved as.</summary>
    /// <param name="declaration">The steps, in the order they were written.</param>
    /// <param name="table">The data, as the last step left it.</param>
    /// <param name="parts">Which part each row belongs to.</param>
    /// <param name="fitted">What each step learned, by its position in the declaration.</param>
    /// <remarks>
    /// Public because a saved pipeline has to be usable without the data it was fitted on: a serving host
    /// loads the declaration and what it learned, hands in a row, and gets it prepared the way the training
    /// rows were. See <see cref="FromJson(string)"/>.
    /// </remarks>
    public PreparedData(
        PipelineDeclaration declaration,
        Table table,
        IReadOnlyList<Part> parts,
        IReadOnlyDictionary<int, FittedStepValues> fitted)
    {
        ArgumentNullException.ThrowIfNull(declaration);
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);
        ArgumentNullException.ThrowIfNull(fitted);

        Declaration = declaration;
        Table = table;
        Parts = parts;
        Fitted = fitted;
    }

    /// <summary>Loads a saved pipeline: what was declared, and what the fit learned.</summary>
    /// <param name="json">The document the pipeline was written as.</param>
    /// <returns>The pipeline, with no data and everything it learned.</returns>
    /// <exception cref="FormatException">The document is not a saved pipeline.</exception>
    /// <remarks>
    /// The table that comes back is empty, because a saved pipeline carries no data — that is the point of
    /// saving it. Hand rows to <see cref="Replay"/> and they are prepared exactly as the training rows were.
    /// </remarks>
    public static PreparedData FromJson(string json)
    {
        var declaration = PipelineDeclaration.FromJson(json);
        var fitted = new Dictionary<int, FittedStepValues>();

        using var document = JsonDocument.Parse(json);

        if (document.RootElement.TryGetProperty("fitted", out var saved)
            && saved.ValueKind == JsonValueKind.Object)
        {
            foreach (var step in saved.EnumerateObject())
            {
                if (!int.TryParse(step.Name, CultureInfo.InvariantCulture, out var at))
                {
                    throw new FormatException(
                        $"The fitted half is written by a step's position, and '{step.Name}' is not one.");
                }

                fitted[at] = Read(step.Value);
            }
        }

        return new PreparedData(declaration, new Table([]), [], fitted);
    }

    private static FittedStepValues Read(JsonElement element)
    {
        var values = new FittedStepValues();

        foreach (var learned in element.EnumerateObject())
        {
            switch (learned.Value.ValueKind)
            {
                case JsonValueKind.Number:
                    values.Learned(learned.Name, learned.Value.GetDouble());
                    break;

                case JsonValueKind.Array when learned.Value.EnumerateArray()
                        .All(each => each.ValueKind == JsonValueKind.Number):
                    values.Learned(learned.Name, learned.Value.EnumerateArray().Select(each => each.GetDouble()).ToArray());
                    break;

                case JsonValueKind.Array:
                    values.Learned(
                        learned.Name,
                        learned.Value.EnumerateArray().Select(each => each.GetString() ?? string.Empty).ToArray());
                    break;

                default:
                    throw new FormatException(
                        $"A fit learns numbers and lists, and '{learned.Name}' is neither.");
            }
        }

        return values;
    }

    /// <summary>The steps, exactly as they were declared.</summary>
    public PipelineDeclaration Declaration { get; }

    /// <summary>The data, as the last step left it.</summary>
    public Table Table { get; }

    /// <summary>Which part of the data each row belongs to, in row order.</summary>
    public IReadOnlyList<Part> Parts { get; }

    /// <summary>What each step learned, by its position in the declaration.</summary>
    /// <remarks>
    /// Keyed by position rather than by verb, because a declaration may hold the same step twice and two
    /// identical steps are indistinguishable by value.
    /// </remarks>
    public IReadOnlyDictionary<int, FittedStepValues> Fitted { get; }

    /// <summary>How many rows landed in one split.</summary>
    /// <param name="part">The part to count.</param>
    /// <returns>The number of rows.</returns>
    public int CountIn(Part part) => Parts.Count(each => each == part);

    /// <summary>Runs the same declaration over new rows, with the same numbers it learned before.</summary>
    /// <param name="rows">The rows to prepare — one of them, or a million.</param>
    /// <returns>The new rows, prepared exactly as the training data was.</returns>
    /// <exception cref="InvalidOperationException">The declaration names no columns.</exception>
    /// <remarks>
    /// This is what serving is. Nothing is fitted again and nothing is learned: the means, the fill values
    /// and the category lists are the ones the training rows produced, so a row arriving a year from now
    /// meets exactly the numbers the model was trained on. It is also why a model without its pipeline
    /// cannot be used at all.
    /// </remarks>
    public Table Replay(IRowSource rows)
    {
        ArgumentNullException.ThrowIfNull(rows);

        var schema = Declaration.Steps.OfType<IBindsColumns>().FirstOrDefault()
            ?? throw new InvalidOperationException(
                "This pipeline never says which columns take part, so there is nothing to replay.");

        var table = schema.Bind(rows);

        for (var at = 0; at < Declaration.Steps.Count; at++)
        {
            switch (Declaration.Steps[at])
            {
                case ILearnsFromData learns when Fitted.TryGetValue(at, out var learned):
                    learns.ApplyTo(table, learned);
                    break;

                case IAddsColumns adds:
                    adds.AddTo(table);
                    break;
            }
        }

        return table;
    }

    /// <summary>Puts predictions back into the units the target was read in.</summary>
    /// <param name="predictions">What a model said, in the units it was trained on.</param>
    /// <returns>The same numbers, in the units of the column the pipeline started from.</returns>
    /// <exception cref="InvalidOperationException">
    /// The pipeline names no target, or a step on the way to it cannot be undone.
    /// </exception>
    /// <remarks>
    /// The steps that touched the target are walked backwards, each undoing what it did. A step that
    /// cannot be undone — one that clipped, or blanked, or threw information away — says so rather than
    /// quietly handing back a number in the wrong units, which is the failure this exists to prevent.
    /// </remarks>
    public IReadOnlyList<double> BackToOriginal(IEnumerable<double> predictions)
    {
        ArgumentNullException.ThrowIfNull(predictions);

        var target = Declaration.Steps.OfType<TargetStep>().LastOrDefault()?.Column
            ?? throw new InvalidOperationException(
                "This pipeline names no target, so there is nothing to put back into any units.");

        var undoing = new List<(IUndoesItself Step, FittedStepValues? Fitted)>();

        for (var at = Declaration.Steps.Count - 1; at >= 0; at--)
        {
            if (Declaration.Steps[at] is IUndoesItself step && step.Produces == target)
            {
                undoing.Add((step, Fitted.GetValueOrDefault(at)));
            }
        }

        return [.. predictions.Select(value => undoing.Aggregate(value, (each, undo) => undo.Step.Undo(each, undo.Fitted)))];
    }

    /// <summary>Puts one prediction back into the units the target was read in.</summary>
    /// <param name="prediction">What a model said.</param>
    /// <returns>The number in the units of the column the pipeline started from.</returns>
    public double BackToOriginal(double prediction) => BackToOriginal([prediction])[0];

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
