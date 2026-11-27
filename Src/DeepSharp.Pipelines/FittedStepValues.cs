// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// What one step wrote down while it was fitted.
/// </summary>
/// <remarks>
/// A number for a mean or a bound, a list for the categories an encoder found. This is the half of a saved
/// pipeline the fit writes, and it is kept apart from the declaration so the same declaration can be fitted
/// again on fresh data without anybody editing anything.
/// <para>
/// Three kinds of thing go in, and they are said apart because they are not alike. A step <em>learns</em> a value, and a
/// replay reads it. A step <em>decides</em> something from what it saw — whether a column with that many gaps is filled at
/// all — and a replay reads that too, so the rows a model is served keep meeting the pipeline it was trained with. And a
/// step <em>sees</em> how many gaps there were, how many values were no number, how many lay outside its bounds: that is
/// evidence, written down so a person can see what the fit was looking at, and no replay may depend on it. The file holds
/// one object of names and values and marks none of them, so the three words are the fit's own and a bag read back from a
/// file counts nothing.
/// </para>
/// </remarks>
public sealed class FittedStepValues
{
    private readonly NamedValues<double> _numbers = new();
    private readonly NamedValues<IReadOnlyList<string>> _lists = new();
    private readonly NamedValues<IReadOnlyList<double>> _curves = new();
    private readonly NamedValues<string> _texts = new();
    private readonly HashSet<string> _counts = new(StringComparer.Ordinal);
    private readonly HashSet<string> _decisions = new(StringComparer.Ordinal);

    /// <summary>What this step learned, as numbers.</summary>
    public IReadOnlyDictionary<string, double> Numbers => _numbers.Held;

    /// <summary>The names of what this step only saw — a count, or a run or a list it kept to be looked at — which no replay reads.</summary>
    public IReadOnlyCollection<string> Counts => _counts;

    /// <summary>The names of what this step decided from what it saw, which a replay reads.</summary>
    public IReadOnlyCollection<string> Decisions => _decisions;

    /// <summary>What this step learned, as lists of words.</summary>
    public IReadOnlyDictionary<string, IReadOnlyList<string>> Lists => _lists.Held;

    /// <summary>What this step learned, as runs of numbers — the shape of a distribution, say.</summary>
    public IReadOnlyDictionary<string, IReadOnlyList<double>> Curves => _curves.Held;

    /// <summary>What this step wrote down in words — the digest of the rows a split divided, say.</summary>
    public IReadOnlyDictionary<string, string> Texts => _texts.Held;

    /// <summary>Records a number this step learned.</summary>
    /// <param name="name">What the number is.</param>
    /// <param name="value">The number.</param>
    /// <exception cref="ArgumentException">The name already holds another kind of thing.</exception>
    /// <exception cref="ArgumentOutOfRangeException">
    /// The number is not finite: no file can hold it, and no replay could use it, so it is refused by the fit
    /// that learned it rather than by whatever writes it down later.
    /// </exception>
    public void Learned(string name, double value) => Number(name, value, Said.Learned);

    /// <summary>Records a count of what this step saw while it was fitted: evidence, which no replay reads.</summary>
    /// <param name="name">What was counted.</param>
    /// <param name="count">How many there were.</param>
    /// <exception cref="ArgumentException">The name already holds another kind of thing.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The count is not a finite number.</exception>
    public void Saw(string name, double count) => Number(name, count, Said.Seen);

    /// <summary>Records what this step decided from what it saw, which a replay reads.</summary>
    /// <param name="name">What was decided.</param>
    /// <param name="choice">The decision, as a number.</param>
    /// <exception cref="ArgumentException">The name already holds another kind of thing.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The decision is not a finite number.</exception>
    public void Decided(string name, double choice) => Number(name, choice, Said.Decided);

    // One number under one name, and the word it was written down under: the last writing of a name is the one that stands,
    // so a name is never in two of the three at once.
    private void Number(string name, double value, Said said)
    {
        ThrowIfHeldAsAnotherKind(name, Kind.Number);
        ThrowIfNotFinite(name, value);

        _numbers.Put(name, value);
        Seen(name, said);

        if (said == Said.Decided)
        {
            _decisions.Add(name);
        }
        else
        {
            _decisions.Remove(name);
        }
    }

    /// <summary>Records a list this step learned.</summary>
    /// <param name="name">What the list is.</param>
    /// <param name="values">The list.</param>
    /// <exception cref="ArgumentException">The name already holds another kind of thing.</exception>
    public void Learned(string name, IReadOnlyList<string> values) => List(name, values, Said.Learned);

    /// <summary>Records a list of words this step saw while it was fitted: evidence, which no replay reads.</summary>
    /// <param name="name">What the list is.</param>
    /// <param name="values">The words.</param>
    /// <exception cref="ArgumentException">The name already holds another kind of thing.</exception>
    public void Saw(string name, IReadOnlyList<string> values) => List(name, values, Said.Seen);

    private void List(string name, IReadOnlyList<string> values, Said said)
    {
        ArgumentNullException.ThrowIfNull(values);
        ThrowIfHeldAsAnotherKind(name, Kind.List);

        _lists.Put(name, values);
        Seen(name, said);
    }

    /// <summary>Records a run of numbers this step learned.</summary>
    /// <param name="name">What the run is.</param>
    /// <param name="values">The numbers, in order.</param>
    /// <exception cref="ArgumentException">The name already holds another kind of thing.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A number in the run is not finite.</exception>
    public void Learned(string name, IReadOnlyList<double> values) => Run(name, values, Said.Learned);

    /// <summary>Records a run of numbers this step saw while it was fitted: evidence, which no replay reads.</summary>
    /// <param name="name">What the run is.</param>
    /// <param name="values">The numbers, in order.</param>
    /// <exception cref="ArgumentException">The name already holds another kind of thing.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A number in the run is not finite.</exception>
    public void Saw(string name, IReadOnlyList<double> values) => Run(name, values, Said.Seen);

    private void Run(string name, IReadOnlyList<double> values, Said said)
    {
        ArgumentNullException.ThrowIfNull(values);
        ThrowIfHeldAsAnotherKind(name, Kind.Run);

        foreach (var value in values)
        {
            ThrowIfNotFinite(name, value);
        }

        _curves.Put(name, values);
        Seen(name, said);
    }

    // A name is among what was only seen exactly when its last writing said so.
    private void Seen(string name, Said said)
    {
        if (said == Said.Seen)
        {
            _counts.Add(name);
        }
        else
        {
            _counts.Remove(name);
        }
    }

    /// <summary>Records something this step wrote down in words.</summary>
    /// <param name="name">What it is.</param>
    /// <param name="value">The words.</param>
    /// <exception cref="ArgumentException">The name already holds another kind of thing.</exception>
    public void Learned(string name, string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        ThrowIfHeldAsAnotherKind(name, Kind.Words);

        _texts.Put(name, value);
    }

    // A name holds one kind of thing: the file writes every name once, whatever it holds, and a name written
    // twice is a file that says two things and is refused when it is read back.
    private void ThrowIfHeldAsAnotherKind(string name, Kind kind)
    {
        ArgumentNullException.ThrowIfNull(name);

        if (Held(name) is { } other && other != kind)
        {
            throw new ArgumentException(
                $"'{name}' was learned as {Described(other)} already, and a name holds one kind of thing.", nameof(name));
        }
    }

    // The kind of thing a name holds, when it holds one.
    private Kind? Held(string name) =>
        _numbers.Holds(name) ? Kind.Number
        : _lists.Holds(name) ? Kind.List
        : _curves.Holds(name) ? Kind.Run
        : _texts.Holds(name) ? Kind.Words
        : null;

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
    public string Text(string name) => _texts.Require(name);

    /// <summary>The run of numbers this step learned under that name.</summary>
    /// <param name="name">What the run is.</param>
    /// <returns>The numbers, in order.</returns>
    /// <exception cref="InvalidOperationException">The fit never learned it.</exception>
    /// <remarks>An empty list of words reads as an empty run too: a file writes both as <c>[]</c>.</remarks>
    public IReadOnlyList<double> Curve(string name) =>
        _curves.TryGet(name) ?? (_lists.TryGet(name) is { Count: 0 } ? Array.Empty<double>() : throw NeverFitted(name));

    /// <summary>The number this step learned under that name.</summary>
    /// <param name="name">What the number is.</param>
    /// <returns>The number.</returns>
    /// <exception cref="InvalidOperationException">The fit never learned it.</exception>
    public double Number(string name) => _numbers.Require(name);

    /// <summary>What this step chose under that name, as the whole number it is.</summary>
    /// <param name="name">What was chosen.</param>
    /// <returns>The choice, or nothing when the fit wrote none under that name.</returns>
    /// <remarks>
    /// A decision is written as a number, because the file holds one object of numbers — but a decision is a nought or a
    /// one and never a measurement, so it is read back as the whole number it is rather than compared as a fraction. A
    /// file is text and text can be edited, so a value a hair wide of a whole number is read as the nearest one. A bag
    /// read back from a file marks none of its names, so this answers for any number written under the name, and it is a
    /// replay's question: a step asks it where it would otherwise ask whether a number equals nought.
    /// </remarks>
    public long? Choice(string name) =>
        _numbers.Holds(name)
            ? (long)Math.Round(_numbers.Require(name), MidpointRounding.AwayFromZero)
            : null;

    /// <summary>The list this step learned under that name.</summary>
    /// <param name="name">What the list is.</param>
    /// <returns>The list.</returns>
    /// <exception cref="InvalidOperationException">The fit never learned it.</exception>
    /// <remarks>An empty run of numbers reads as an empty list too: a file writes both as <c>[]</c>.</remarks>
    public IReadOnlyList<string> List(string name) =>
        _lists.TryGet(name) ?? (_curves.TryGet(name) is { Count: 0 } ? Array.Empty<string>() : throw NeverFitted(name));

    /// <summary>Writes what was learned, as one JSON object.</summary>
    /// <param name="writer">The writer positioned where the object belongs.</param>
    public void WriteTo(Utf8JsonWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteStartObject();

        _numbers.WriteTo(writer, (to, name, value) => to.WriteNumber(name, value));
        _lists.WriteTo(writer, (to, name, values) => WriteRun(to, name, values, (each, word) => each.WriteStringValue(word)));
        _curves.WriteTo(writer, (to, name, values) => WriteRun(to, name, values, (each, number) => each.WriteNumberValue(number)));
        _texts.WriteTo(writer, (to, name, value) => to.WriteString(name, value));

        writer.WriteEndObject();
    }

    private static void WriteRun<T>(Utf8JsonWriter writer, string name, IReadOnlyList<T> values, Action<Utf8JsonWriter, T> write)
    {
        writer.WriteStartArray(name);

        foreach (var value in values)
        {
            write(writer, value);
        }

        writer.WriteEndArray();
    }

    // The one sentence for what a replay asks of a fit that never wrote it.
    private static InvalidOperationException NeverFitted(string name) =>
        new($"This pipeline was never fitted for '{name}', so there is nothing to replay.");

    /// <summary>The kinds of thing a name can hold.</summary>
    private enum Kind
    {
        Number,
        List,
        Run,
        Words,
    }

    /// <summary>How a number came to be written down.</summary>
    private enum Said
    {
        /// <summary>A value the step learned, which a replay reads.</summary>
        Learned,

        /// <summary>Something the step decided from what it saw, which a replay reads.</summary>
        Decided,

        /// <summary>A count of what the step saw, which no replay reads.</summary>
        Seen,
    }

    /// <summary>One kind of thing a fit wrote down, each under a name: what a name holds, and what is said of one that holds nothing.</summary>
    /// <typeparam name="T">What this kind holds.</typeparam>
    private sealed class NamedValues<T>
    {
        private readonly Dictionary<string, T> _values = [];

        public IReadOnlyDictionary<string, T> Held => _values;

        public bool Holds(string name) => _values.ContainsKey(name);

        public void Put(string name, T value) => _values[name] = value;

        // The value under that name, or nothing where this kind holds none under it.
        public T? TryGet(string name) => _values.TryGetValue(name, out var value) ? value : default;

        public T Require(string name) => _values.TryGetValue(name, out var value) ? value : throw NeverFitted(name);

        // Every name in the order of the names, each written as the kind writes it.
        public void WriteTo(Utf8JsonWriter writer, Action<Utf8JsonWriter, string, T> write)
        {
            foreach (var (name, value) in _values.OrderBy(each => each.Key, StringComparer.Ordinal))
            {
                write(writer, name, value);
            }
        }
    }
}
