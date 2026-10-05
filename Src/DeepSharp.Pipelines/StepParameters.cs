// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// One parameter of a step: the key it is written under, what it means, and what it may hold.
/// </summary>
/// <remarks>
/// The kinds of value a parameter can hold are a closed set — text, a path, a number, a whole number, true or
/// false, one of a set of words, a column, and the few shapes that belong together — while the verbs that
/// use them stay open. Every place a step's parameters are needed reads them from here: writing the step,
/// reading it back, refusing a key nobody defined, the starter template of a new block, the JSON Schema an
/// editor checks a file against, the reference page, and the form a notebook shows. Before this each of
/// those typed the keys by hand, and they drifted.
/// </remarks>
public abstract class StepParameter
{
    private protected StepParameter(string key, string description)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(key);
        ArgumentException.ThrowIfNullOrWhiteSpace(description);

        Key = key;
        Description = description;
    }

    /// <summary>The key the parameter is written under.</summary>
    public string Key { get; }

    /// <summary>What the parameter means, in the words a person reads it in.</summary>
    public string Description { get; }

    /// <summary>Every key the parameter is written under; one, except for values that belong together.</summary>
    public virtual IReadOnlyList<string> Keys => [Key];

    /// <summary>The keys a file has to hold for this parameter; every key, unless leaving one out means something.</summary>
    public virtual IReadOnlyList<string> RequiredKeys => Keys;

    /// <summary>
    /// Whether the value a course starts a step with is left to the person: what names something of theirs — a column, a
    /// file, the columns a schema declares — or a bound nothing could suggest.
    /// </summary>
    /// <remarks>
    /// An example of such a value is not a default. A column called "column" reads as well as any other name, so a step that
    /// kept it would pass for one somebody decided, and would be found wrong only by the rule that notices the column is not
    /// there. What only settles how — a seed, a share, a word from a set — starts as the verb starts it, which is a value
    /// worth having.
    /// </remarks>
    internal virtual bool IsLeftToThePerson => false;

    /// <summary>The keys a course leaves waiting for the person: those this parameter has to hold, when it is left to them.</summary>
    internal IReadOnlyList<string> WaitingKeys => IsLeftToThePerson ? RequiredKeys : [];

    /// <summary>Hands this parameter to whatever is built from the kinds, as the kind it is.</summary>
    /// <typeparam name="TResult">What the visitor builds.</typeparam>
    /// <param name="visitor">The visitor.</param>
    /// <returns>What the visitor built for this parameter.</returns>
    /// <remarks>
    /// The kinds are a closed set and what is built from them — a form, a schema, a reference page — is open,
    /// so each of those visits the kinds rather than asking a parameter what type it is.
    /// </remarks>
    public abstract TResult Accept<TResult>(IStepParameterVisitor<TResult> visitor);

    /// <summary>Writes an example of the value: the default where absence means something, an example otherwise.</summary>
    /// <param name="writer">The writer positioned inside the step's object.</param>
    internal abstract void WriteExample(Utf8JsonWriter writer);

    /// <summary>Whether a step read from a file writes this parameter back as the value the file held.</summary>
    /// <param name="written">The step as the file wrote it.</param>
    /// <param name="rewritten">The same step, written back after it was read.</param>
    /// <returns><see langword="true"/> when the two read as the same value.</returns>
    /// <remarks>
    /// Compared as values of the kind, not as text: <c>MinMax</c> and <c>minmax</c> are the same word, and
    /// <c>3.0</c> and <c>3</c> the same whole number.
    /// </remarks>
    internal abstract bool KeptAsWritten(JsonElement written, JsonElement rewritten);

    /// <summary>Whether a file wrote any of this parameter's keys at all.</summary>
    /// <param name="step">The step's JSON object.</param>
    /// <returns><see langword="true"/> when at least one key is present.</returns>
    internal bool IsWrittenIn(JsonElement step) => Keys.Any(key => step.TryGetProperty(key, out _));

    /// <summary>This parameter's keys and values as a step holds them, for a message.</summary>
    /// <param name="step">The step's JSON object.</param>
    /// <returns>The keys that are present, each with its value as it was written.</returns>
    internal string AsWrittenIn(JsonElement step) =>
        string.Join(", ", Keys
            .Where(key => step.TryGetProperty(key, out _))
            .Select(key => $"\"{key}\": {step.GetProperty(key).GetRawText().Quoted()}"));

    /// <summary>The one rule a name is held to: it is there, and it is more than spaces.</summary>
    /// <param name="value">The name.</param>
    /// <returns>The name.</returns>
    /// <exception cref="ArgumentException">The name is absent or nothing but spaces.</exception>
    private protected string Required(string? value) =>
        string.IsNullOrWhiteSpace(value)
            ? throw new ArgumentException($"'{Key}' needs a value: {Description}", Key)
            : value;
}

/// <summary>
/// A parameter holding one kind of value.
/// </summary>
/// <typeparam name="T">What the parameter holds once it is read.</typeparam>
public abstract class StepParameter<T> : StepParameter
{
    private protected StepParameter(string key, string description, T example)
        : base(key, description) =>
        Example = example;

    /// <summary>An example of the value.</summary>
    public T Example { get; }

    /// <summary>Reads the value out of the object a step was written as.</summary>
    /// <param name="step">The step's JSON object.</param>
    /// <returns>The value.</returns>
    /// <exception cref="FormatException">The value is absent where it may not be, or is not of this kind.</exception>
    public abstract T Read(JsonElement step);

    /// <summary>Writes the value under its key.</summary>
    /// <param name="writer">The writer positioned inside the step's object.</param>
    /// <param name="value">The value.</param>
    public abstract void Write(Utf8JsonWriter writer, T value);

    /// <summary>The columns a value of this kind names as ones the step reads.</summary>
    /// <param name="value">The value.</param>
    /// <returns>None, unless this is a kind that names columns to read.</returns>
    internal virtual IEnumerable<ColumnRead> ReadsIn(T value) => [];

    /// <summary>Checks a value handed in at a call site, by the same rule a file is held to.</summary>
    /// <param name="value">The value.</param>
    /// <returns>The value, when it keeps the rule.</returns>
    /// <exception cref="ArgumentException">The value breaks the rule.</exception>
    public virtual T Require(T value) => value;

    /// <inheritdoc />
    internal override void WriteExample(Utf8JsonWriter writer) => Write(writer, Example);

    /// <inheritdoc />
    internal override bool KeptAsWritten(JsonElement written, JsonElement rewritten) =>
        Same(Read(written), Read(rewritten));

    /// <summary>Whether two values of this kind say the same thing.</summary>
    /// <param name="one">One value.</param>
    /// <param name="other">The other.</param>
    /// <returns><see langword="true"/> when they are equal as values of the kind.</returns>
    private protected virtual bool Same(T one, T other) => EqualityComparer<T>.Default.Equals(one, other);
}

/// <summary>
/// Something built from the kinds of parameter, one kind at a time.
/// </summary>
/// <typeparam name="TResult">What it builds for each parameter.</typeparam>
/// <remarks>
/// One method per kind, because the kinds are a closed set: a form, a schema and a reference page each
/// implement this once, and a kind nobody handled is a compile error rather than a branch that was forgotten.
/// </remarks>
public interface IStepParameterVisitor<out TResult>
{
    /// <summary>Words: a description, a name that is not a column.</summary>
    /// <param name="parameter">The parameter.</param>
    /// <returns>What the visitor builds for it.</returns>
    TResult Visit(TextParameter parameter);

    /// <summary>The path of a file the pipeline reads.</summary>
    /// <param name="parameter">The parameter.</param>
    /// <returns>What the visitor builds for it.</returns>
    TResult Visit(FilePathParameter parameter);

    /// <summary>A column the step reads.</summary>
    /// <param name="parameter">The parameter.</param>
    /// <returns>What the visitor builds for it.</returns>
    TResult Visit(ColumnParameter parameter);

    /// <summary>A column the step makes.</summary>
    /// <param name="parameter">The parameter.</param>
    /// <returns>What the visitor builds for it.</returns>
    TResult Visit(NewColumnParameter parameter);

    /// <summary>Several columns the step reads.</summary>
    /// <param name="parameter">The parameter.</param>
    /// <returns>What the visitor builds for it.</returns>
    TResult Visit(ColumnsParameter parameter);

    /// <summary>A number that may have a fraction.</summary>
    /// <param name="parameter">The parameter.</param>
    /// <returns>What the visitor builds for it.</returns>
    TResult Visit(NumberParameter parameter);

    /// <summary>A whole number.</summary>
    /// <param name="parameter">The parameter.</param>
    /// <returns>What the visitor builds for it.</returns>
    TResult Visit(WholeNumberParameter parameter);

    /// <summary>True or false.</summary>
    /// <param name="parameter">The parameter.</param>
    /// <returns>What the visitor builds for it.</returns>
    TResult Visit(TrueOrFalseParameter parameter);

    /// <summary>A share, from nought to one, that a file may leave out.</summary>
    /// <param name="parameter">The parameter.</param>
    /// <returns>What the visitor builds for it.</returns>
    TResult Visit(ShareParameter parameter);

    /// <summary>One of a named set of words.</summary>
    /// <typeparam name="TEnum">The set of words.</typeparam>
    /// <param name="parameter">The parameter.</param>
    /// <returns>What the visitor builds for it.</returns>
    TResult Visit<TEnum>(OneOfParameter<TEnum> parameter)
        where TEnum : struct, Enum;

    /// <summary>Several of a named set of words.</summary>
    /// <typeparam name="TEnum">The set of words.</typeparam>
    /// <param name="parameter">The parameter.</param>
    /// <returns>What the visitor builds for it.</returns>
    TResult Visit<TEnum>(SeveralOfParameter<TEnum> parameter)
        where TEnum : struct, Enum;

    /// <summary>How a value is filled.</summary>
    /// <param name="parameter">The parameter.</param>
    /// <returns>What the visitor builds for it.</returns>
    TResult Visit(FillStrategyParameter parameter);

    /// <summary>The shares a split divides the rows into.</summary>
    /// <param name="parameter">The parameter.</param>
    /// <returns>What the visitor builds for it.</returns>
    TResult Visit(SplitSharesParameter parameter);

    /// <summary>The columns a schema declares.</summary>
    /// <param name="parameter">The parameter.</param>
    /// <returns>What the visitor builds for it.</returns>
    TResult Visit(ColumnDeclarationsParameter parameter);

    /// <summary>The parts of a model a step declares by name: the layers, the optimizer, the loss.</summary>
    /// <param name="parameter">The parameter.</param>
    /// <returns>What the visitor builds for it.</returns>
    TResult Visit(PartsParameter parameter);
}

/// <summary>
/// The parameters of one kind of step, in the order they are written, each bound to where its value lives.
/// </summary>
/// <typeparam name="TStep">The step they belong to.</typeparam>
/// <remarks>
/// Said once per step type, beside its name and its reader. The step is then written from this and nothing
/// else, so a key is typed in exactly one place: the parameter it belongs to.
/// </remarks>
public sealed class StepParameters<TStep>
    where TStep : IPipelineStep<TStep>
{
    private readonly Bound[] _bound;

    /// <summary>A step with no parameters, to add them to one at a time.</summary>
    public StepParameters()
        : this([])
    {
    }

    private StepParameters(Bound[] bound) => _bound = bound;

    /// <summary>Every parameter, in the order they are written.</summary>
    public IReadOnlyList<StepParameter> All => [.. _bound.Select(bound => bound.Parameter)];

    /// <summary>The same parameters with one more, and where its value is found on a step.</summary>
    /// <typeparam name="T">What the parameter holds.</typeparam>
    /// <param name="parameter">The parameter.</param>
    /// <param name="value">How to take its value from a step.</param>
    /// <returns>A description with the parameter added at the end.</returns>
    public StepParameters<TStep> With<T>(StepParameter<T> parameter, Func<TStep, T> value)
    {
        ArgumentNullException.ThrowIfNull(parameter);
        ArgumentNullException.ThrowIfNull(value);

        return new([.. _bound, new Bound<T>(parameter, value)]);
    }

    /// <summary>The columns a step reads, as its parameters that name columns say.</summary>
    /// <param name="step">The step.</param>
    /// <returns>Each column it reads, with the kinds it can work on, in the order of its parameters.</returns>
    public IReadOnlyList<ColumnRead> ReadBy(TStep step)
    {
        ArgumentNullException.ThrowIfNull(step);

        return [.. _bound.SelectMany(bound => bound.ReadBy(step))];
    }

    /// <summary>Writes a step as one JSON object: its verb, then each parameter in order.</summary>
    /// <param name="writer">The writer positioned where the object belongs.</param>
    /// <param name="step">The step.</param>
    public void Write(Utf8JsonWriter writer, TStep step)
    {
        ArgumentNullException.ThrowIfNull(writer);
        ArgumentNullException.ThrowIfNull(step);

        writer.WriteStartObject();
        writer.WriteString(StepCatalog.StepKey, TStep.Name);

        foreach (var bound in _bound)
        {
            bound.Write(writer, step);
        }

        writer.WriteEndObject();
    }

    private abstract class Bound
    {
        public abstract StepParameter Parameter { get; }

        public abstract void Write(Utf8JsonWriter writer, TStep step);

        public abstract IEnumerable<ColumnRead> ReadBy(TStep step);
    }

    private sealed class Bound<T>(StepParameter<T> parameter, Func<TStep, T> value) : Bound
    {
        public override StepParameter Parameter => parameter;

        public override void Write(Utf8JsonWriter writer, TStep step) => parameter.Write(writer, value(step));

        public override IEnumerable<ColumnRead> ReadBy(TStep step) => parameter.ReadsIn(value(step));
    }
}

/// <summary>
/// What a verb is: its name, what it does, its parameters, and an example of the step.
/// </summary>
/// <remarks>
/// The catalog's view of a step type, for everything that has to know a verb without having a step of it:
/// the file door refusing a key nobody defined, a notebook's form, the starter template, the schema and the
/// reference page.
/// </remarks>
public sealed class StepDescription
{
    internal StepDescription(string verb, string purpose, int since, IReadOnlyList<StepParameter> parameters)
    {
        Verb = verb;
        Purpose = purpose;
        Since = since;
        Parameters = parameters;
        Keys = new HashSet<string>(parameters.SelectMany(parameter => parameter.Keys), StringComparer.Ordinal);
        Template = WriteTemplate(verb, parameters);
        Skeleton = WriteSkeleton(verb, parameters);
        Waiting = [.. parameters.SelectMany(parameter => parameter.WaitingKeys)];
    }

    /// <summary>The name the step is written under.</summary>
    public string Verb { get; }

    /// <summary>What the step does, in one sentence.</summary>
    public string Purpose { get; }

    /// <summary>The version of the pipeline file from which the verb means what it says now.</summary>
    public int Since { get; }

    /// <summary>The step's parameters, in the order they are written.</summary>
    public IReadOnlyList<StepParameter> Parameters { get; }

    /// <summary>Every key a file may use for this step, beside its <c>step</c>.</summary>
    public IReadOnlySet<string> Keys { get; }

    /// <summary>The step with every key at an example of its value, as the JSON object it is written as.</summary>
    /// <remarks>
    /// What a gesture starts a step from where it makes one. A course starts from <see cref="Skeleton"/> instead: an example of
    /// a column is not a column, and a step that kept it would pass for one somebody chose.
    /// </remarks>
    public string Template { get; }

    /// <summary>
    /// The step a course starts with, as the JSON object it is written as: the template, but for the keys that name something
    /// of the person's — a key the step needs waits as nothing, and one it may leave out is left out.
    /// </summary>
    internal string Skeleton { get; }

    /// <summary>The keys the skeleton leaves waiting for the person, in the order they are written.</summary>
    internal IReadOnlyList<string> Waiting { get; }

    private static string WriteSkeleton(string verb, IReadOnlyList<StepParameter> parameters)
    {
        var buffer = new MemoryStream();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WriteString(StepCatalog.StepKey, verb);

            foreach (var parameter in parameters)
            {
                if (!parameter.IsLeftToThePerson)
                {
                    parameter.WriteExample(writer);

                    continue;
                }

                // A key the step needs waits as nothing; one it may leave out stays out, which is how the step decides it itself.
                foreach (var key in parameter.WaitingKeys)
                {
                    writer.WriteNull(key);
                }
            }

            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.ToArray());
    }

    private static string WriteTemplate(string verb, IReadOnlyList<StepParameter> parameters)
    {
        var buffer = new MemoryStream();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WriteString(StepCatalog.StepKey, verb);

            foreach (var parameter in parameters)
            {
                parameter.WriteExample(writer);
            }

            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.ToArray());
    }
}
