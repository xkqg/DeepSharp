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
    private readonly Dictionary<string, string> _texts = [];

    /// <summary>What this step learned, as numbers.</summary>
    public IReadOnlyDictionary<string, double> Numbers => _numbers;

    /// <summary>What this step learned, as lists of words.</summary>
    public IReadOnlyDictionary<string, IReadOnlyList<string>> Lists => _lists;

    /// <summary>What this step learned, as runs of numbers — the shape of a distribution, say.</summary>
    public IReadOnlyDictionary<string, IReadOnlyList<double>> Curves => _curves;

    /// <summary>What this step wrote down in words — the digest of the rows a split divided, say.</summary>
    public IReadOnlyDictionary<string, string> Texts => _texts;

    /// <summary>Records a number this step learned.</summary>
    /// <param name="name">What the number is.</param>
    /// <param name="value">The number.</param>
    /// <exception cref="ArgumentException">The name already holds another kind of thing.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// The number is not finite: no file can hold it, and no replay could use it, so it is refused by the fit
    /// that learned it rather than by whatever writes it down later.
    /// </exception>
    public void Learned(string name, double value)
    {
        ThrowIfHeldAsAnotherKind(name, Kind.Number);
        ThrowIfNotFinite(name, value);

        _numbers[name] = value;
    }

    /// <summary>Records a list this step learned.</summary>
    /// <param name="name">What the list is.</param>
    /// <param name="values">The list.</param>
    /// <exception cref="ArgumentException">The name already holds another kind of thing.</exception>
    public void Learned(string name, IReadOnlyList<string> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        ThrowIfHeldAsAnotherKind(name, Kind.List);

        _lists[name] = values;
    }

    /// <summary>Records a run of numbers this step learned.</summary>
    /// <param name="name">What the run is.</param>
    /// <param name="values">The numbers, in order.</param>
    /// <exception cref="ArgumentException">The name already holds another kind of thing.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A number in the run is not finite.</exception>
    public void Learned(string name, IReadOnlyList<double> values)
    {
        ArgumentNullException.ThrowIfNull(values);
        ThrowIfHeldAsAnotherKind(name, Kind.Run);

        foreach (var value in values)
        {
            ThrowIfNotFinite(name, value);
        }

        _curves[name] = values;
    }

    /// <summary>Records something this step wrote down in words.</summary>
    /// <param name="name">What it is.</param>
    /// <param name="value">The words.</param>
    /// <exception cref="ArgumentException">The name already holds another kind of thing.</exception>
    public void Learned(string name, string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        ThrowIfHeldAsAnotherKind(name, Kind.Words);

        _texts[name] = value;
    }

    // A name holds one kind of thing: the file writes every name once, whatever it holds, and a name written
    // twice is a file that says two things and is refused when it is read back.
    private void ThrowIfHeldAsAnotherKind(string name, Kind kind)
    {
        ArgumentNullException.ThrowIfNull(name);

        Kind? held = _numbers.ContainsKey(name) ? Kind.Number
            : _lists.ContainsKey(name) ? Kind.List
            : _curves.ContainsKey(name) ? Kind.Run
            : _texts.ContainsKey(name) ? Kind.Words
            : null;

        if (held is { } other && other != kind)
        {
            throw new ArgumentException(
                $"'{name}' was learned as {Described(other)} already, and a name holds one kind of thing.", nameof(name));
        }
    }

    private static void ThrowIfNotFinite(string name, double value)
    {
        if (!double.IsFinite(value))
        {
            throw new ArgumentOutOfRangeException(
                nameof(value), value, $"What was learned for '{name}' is not a finite number, and nothing can be replayed from it.");
        }
    }

    private static string Described(Kind kind) => kind switch
    {
        Kind.Number => "a number",
        Kind.List => "a list of words",
        Kind.Run => "a run of numbers",
        _ => "words",
    };

    /// <summary>The words this step wrote down under that name.</summary>
    /// <param name="name">What they are.</param>
    /// <returns>The words.</returns>
    /// <exception cref="InvalidOperationException">The fit never wrote them.</exception>
    public string Text(string name) =>
        _texts.TryGetValue(name, out var value)
            ? value
            : throw new InvalidOperationException(
                $"This pipeline was never fitted for '{name}', so there is nothing to replay.");

    /// <summary>The run of numbers this step learned under that name.</summary>
    /// <param name="name">What the run is.</param>
    /// <returns>The numbers, in order.</returns>
    /// <exception cref="InvalidOperationException">The fit never learned it.</exception>
    /// <remarks>An empty list of words reads as an empty run too: a file writes both as <c>[]</c>.</remarks>
    public IReadOnlyList<double> Curve(string name) =>
        _curves.TryGetValue(name, out var values)
            ? values
            : _lists.TryGetValue(name, out var words) && words.Count == 0
                ? Array.Empty<double>()
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
    /// <remarks>An empty run of numbers reads as an empty list too: a file writes both as <c>[]</c>.</remarks>
    public IReadOnlyList<string> List(string name) =>
        _lists.TryGetValue(name, out var values)
            ? values
            : _curves.TryGetValue(name, out var run) && run.Count == 0
                ? Array.Empty<string>()
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

        foreach (var (name, value) in _texts.OrderBy(each => each.Key, StringComparer.Ordinal))
        {
            writer.WriteString(name, value);
        }

        writer.WriteEndObject();
    }

    /// <summary>The kinds of thing a name can hold.</summary>
    private enum Kind
    {
        Number,
        List,
        Run,
        Words,
    }
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
    /// rows were. See <see cref="FromJson(string, StepCatalog)"/>.
    /// </remarks>
    /// <exception cref="ArgumentException">
    /// A step that learns has nothing it learned here, or something is here for a step that learns nothing.
    /// </exception>
    public PreparedData(
        PipelineDeclaration declaration,
        Table table,
        IReadOnlyList<Part> parts,
        IReadOnlyDictionary<int, FittedStepValues> fitted)
        : this(declaration, table, parts, fitted, new Dictionary<int, Evidence>())
    {
    }

    /// <summary>A pipeline that has been run, with the evidence its run produced.</summary>
    /// <param name="declaration">The steps, in the order they were written.</param>
    /// <param name="table">The data, as the last step left it.</param>
    /// <param name="parts">Which part each row belongs to.</param>
    /// <param name="fitted">What each step learned, by its position in the declaration.</param>
    /// <param name="evidence">What each step that produces evidence produced, by its position.</param>
    /// <exception cref="ArgumentException">
    /// A step that learns has nothing it learned here, or something is here for a step that learns nothing.
    /// </exception>
    public PreparedData(
        PipelineDeclaration declaration,
        Table table,
        IReadOnlyList<Part> parts,
        IReadOnlyDictionary<int, FittedStepValues> fitted,
        IReadOnlyDictionary<int, Evidence> evidence)
    {
        ArgumentNullException.ThrowIfNull(declaration);
        ArgumentNullException.ThrowIfNull(table);
        ArgumentNullException.ThrowIfNull(parts);
        ArgumentNullException.ThrowIfNull(fitted);
        ArgumentNullException.ThrowIfNull(evidence);

        ThrowIfTheFitsDoNotMatch(declaration, fitted);

        Declaration = declaration;
        Table = table;
        Parts = parts;
        Fitted = fitted;
        Evidence = evidence;
    }

    /// <summary>Refuses fits that are not exactly one per step that learns, and one for the split.</summary>
    /// <remarks>
    /// A step that learns and has nothing it learned used to be skipped when the pipeline was replayed, so a
    /// saved declaration without its fitted half served every value unscaled and said nothing about it. The
    /// split's entry is what the fit saw, and a fitted half without it is one nobody can trace to its data.
    /// </remarks>
    private static void ThrowIfTheFitsDoNotMatch(
        PipelineDeclaration declaration, IReadOnlyDictionary<int, FittedStepValues> fitted)
    {
        foreach (var at in fitted.Keys)
        {
            if (at < 0 || at >= declaration.Steps.Count || !WritesAnEntry(declaration.Steps[at]))
            {
                throw new ArgumentException(
                    at < 0 || at >= declaration.Steps.Count
                        ? $"There is no step {at + 1} for anything to have been learned at."
                        : $"Step {at + 1}, '{declaration.Steps[at].Verb}', learns nothing, so nothing it learned can be here.",
                    nameof(fitted));
            }
        }

        if (FitsMissing(declaration, fitted).ToArray() is [var missing, ..])
        {
            throw new ArgumentException(missing.ToString(), nameof(fitted));
        }
    }

    /// <summary>Every step that writes an entry and has none among these fits.</summary>
    /// <param name="declaration">The steps.</param>
    /// <param name="fitted">What was learned, by position.</param>
    /// <returns>A fault for each step whose entry is missing, in the order of the steps.</returns>
    /// <remarks>One rule for this constructor and for the file a pipeline is read from, which says where each is.</remarks>
    internal static IEnumerable<DeclarationFault> FitsMissing(
        PipelineDeclaration declaration, IReadOnlyDictionary<int, FittedStepValues> fitted)
    {
        for (var at = 0; at < declaration.Steps.Count; at++)
        {
            var step = declaration.Steps[at];

            if (WritesAnEntry(step) && !fitted.ContainsKey(at))
            {
                yield return new DeclarationFault(
                    at,
                    step.Verb,
                    step is ISplitStep
                        ? "divides the rows, and what it saw of them is not here. Fit the pipeline again."
                        : "learns from the data, and nothing it learned is here. Fit the pipeline again.");
            }
        }
    }

    /// <summary>Whether a fit writes an entry for this step: each step that learns, and the split it learned behind.</summary>
    /// <param name="step">The step.</param>
    /// <returns><see langword="true"/> for a step that learns and for the split.</returns>
    internal static bool WritesAnEntry(IPipelineStep step) => step is IFittedStep or ISplitStep;

    /// <summary>Loads a saved pipeline, reading its steps with the verbs a given catalog knows.</summary>
    /// <param name="json">The file the pipeline was written as.</param>
    /// <param name="catalog">The verbs that may appear in it, another package's included.</param>
    /// <returns>The pipeline, with no data and everything it learned.</returns>
    /// <exception cref="PipelineFileException">
    /// Anything in the file is wrong, every fault named at its line and column: a step that cannot be read, a
    /// rule the steps break, a step that learned and has no entry, or an entry learned behind other steps than
    /// these — which is what a declaration edited after its fit looks like.
    /// </exception>
    /// <remarks>
    /// The table that comes back is empty, because a saved pipeline carries no data — that is the point of
    /// saving it. Hand rows to <see cref="Replay"/> and they are prepared exactly as the training rows were.
    /// The catalog is handed over rather than assumed, because a verb another package brings has to be as
    /// readable here as anywhere, or a pipeline could be saved and never served.
    /// </remarks>
    public static PreparedData FromJson(string json, StepCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(json);
        ArgumentNullException.ThrowIfNull(catalog);

        var saved = PipelineDocument.ReadPipeline(json, catalog);

        return new PreparedData(saved.Declaration, new Table([]), [], saved.Fitted);
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

    /// <summary>What each step that produces evidence produced when the pipeline was run, by its position.</summary>
    /// <remarks>
    /// Output, not the pipeline: it is kept with the run and never written into the pipeline's file, so a
    /// pipeline loaded from one has none, and a replay produces none.
    /// </remarks>
    public IReadOnlyDictionary<int, Evidence> Evidence { get; }

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

        return new Walk(Declaration, new ReplayWhatWasFitted(Fitted), SourceFolder.WorkingDirectory).Through(rows).Table;
    }

    /// <summary>Puts predictions back into the units the target was read in.</summary>
    /// <param name="predictions">What a model said, in the units it was trained on.</param>
    /// <returns>The same numbers, in the units of the column the pipeline started from.</returns>
    /// <exception cref="InvalidOperationException">
    /// The pipeline names no answer, or several, or a step on the way to it cannot be undone.
    /// </exception>
    /// <remarks>
    /// The steps that touched the target are walked backwards, each undoing what it did. A step that
    /// cannot be undone — one that clipped, or blanked, or threw information away — says so rather than
    /// quietly handing back a number in the wrong units, which is the failure this exists to prevent.
    /// </remarks>
    public IReadOnlyList<double> BackToOriginal(IEnumerable<double> predictions)
    {
        ArgumentNullException.ThrowIfNull(predictions);

        var target = Declaration.Output switch
        {
            { Answers: [var only] } => only,
            null => throw new InvalidOperationException(
                "This pipeline names no answer, so there is nothing to put back into any units."),
            var output => throw new InvalidOperationException(string.Create(
                CultureInfo.InvariantCulture,
                $"This pipeline's output names {output.Answers.Count} answers, and one number goes back into the units of one of them.")),
        };

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

    /// <summary>Writes the whole pipeline: the version, what was declared, and what the fit learned.</summary>
    /// <returns>The pipeline as one JSON document.</returns>
    /// <remarks>
    /// Both halves in one file, because a model without what its pipeline learned cannot be used: the
    /// numbers reaching it would not be the numbers it was trained on. Each entry of the fitted half names
    /// the step that learned it and the key of the steps it was learned behind
    /// (<see cref="PipelineDeclaration.KeyAt"/>), which is what keeps a fit from being served under steps
    /// that changed after it.
    /// </remarks>
    public string ToJson() => PipelineDocument.Write(Declaration, Fitted);
}
