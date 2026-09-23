// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>A parameter holding words: a description, a name that is not a column.</summary>
/// <param name="key">The key it is written under.</param>
/// <param name="description">What it means.</param>
/// <param name="example">The value a new block starts with.</param>
public sealed class TextParameter(string key, string description, string example)
    : StepParameter<string>(key, description, example)
{
    /// <inheritdoc />
    public override string Read(JsonElement step) => step.RequiredString(Key);

    /// <inheritdoc />
    public override void Write(Utf8JsonWriter writer, string value)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteString(Key, value);
    }

    /// <inheritdoc />
    public override string Require(string value) => Required(value);

    /// <inheritdoc />
    public override TResult Accept<TResult>(IStepParameterVisitor<TResult> visitor)
    {
        ArgumentNullException.ThrowIfNull(visitor);

        return visitor.Visit(this);
    }
}

/// <summary>A parameter holding the path of a file the pipeline reads.</summary>
/// <param name="key">The key it is written under.</param>
/// <param name="description">What it means.</param>
/// <param name="example">The value a new block starts with.</param>
/// <remarks>
/// Words, said to be a path so that whatever resolves a path — against the folder a notebook sits in, say —
/// knows which of a step's words to resolve.
/// </remarks>
public sealed class FilePathParameter(string key, string description, string example)
    : StepParameter<string>(key, description, example)
{
    /// <inheritdoc />
    public override string Read(JsonElement step) => step.RequiredString(Key);

    /// <inheritdoc />
    public override void Write(Utf8JsonWriter writer, string value)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteString(Key, value);
    }

    /// <inheritdoc />
    public override string Require(string value) => Required(value);

    /// <inheritdoc />
    public override TResult Accept<TResult>(IStepParameterVisitor<TResult> visitor)
    {
        ArgumentNullException.ThrowIfNull(visitor);

        return visitor.Visit(this);
    }
}

/// <summary>A parameter naming a column the step reads.</summary>
/// <param name="key">The key it is written under.</param>
/// <param name="description">What it means.</param>
/// <param name="example">The value a new block starts with.</param>
/// <param name="accepts">The kinds of column the step can work on.</param>
/// <remarks>
/// A name, said to be a column so that a form can offer the columns there are rather than an empty box,
/// and so that a column the source does not have is found before anything runs.
/// </remarks>
public sealed class ColumnParameter(string key, string description, string example, IReadOnlyList<ColumnKind> accepts)
    : StepParameter<string>(key, description, example)
{
    /// <summary>The kinds of column the step can work on.</summary>
    public IReadOnlyList<ColumnKind> Accepts { get; } = accepts;

    /// <inheritdoc />
    public override string Read(JsonElement step) => step.RequiredString(Key);

    /// <inheritdoc />
    public override void Write(Utf8JsonWriter writer, string value)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteString(Key, value);
    }

    /// <inheritdoc />
    public override string Require(string value) => Required(value);

    /// <inheritdoc />
    public override TResult Accept<TResult>(IStepParameterVisitor<TResult> visitor)
    {
        ArgumentNullException.ThrowIfNull(visitor);

        return visitor.Visit(this);
    }

    /// <inheritdoc />
    internal override IEnumerable<ColumnRead> ReadsIn(string value) => [new ColumnRead(value, Accepts)];
}

/// <summary>A parameter naming a column the step makes.</summary>
/// <param name="key">The key it is written under.</param>
/// <param name="description">What it means.</param>
/// <param name="example">The value a new block starts with.</param>
/// <param name="optional">Whether a file may leave it out, in which case the step decides the name itself.</param>
public sealed class NewColumnParameter(string key, string description, string example, bool optional = false)
    : StepParameter<string?>(key, description, example)
{
    /// <summary>Whether a file may leave it out.</summary>
    public bool Optional { get; } = optional;

    /// <inheritdoc />
    public override IReadOnlyList<string> RequiredKeys => Optional ? [] : Keys;

    /// <inheritdoc />
    public override string? Read(JsonElement step) =>
        Optional && !step.TryGetProperty(Key, out _) ? null : step.RequiredString(Key);

    /// <inheritdoc />
    public override void Write(Utf8JsonWriter writer, string? value)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteString(Key, value);
    }

    /// <inheritdoc />
    /// <remarks>An optional name left empty is left to the step, which is what leaving it out of a file means too.</remarks>
    public override string? Require(string? value) =>
        Optional && string.IsNullOrWhiteSpace(value) ? null : Required(value);

    /// <inheritdoc />
    public override TResult Accept<TResult>(IStepParameterVisitor<TResult> visitor)
    {
        ArgumentNullException.ThrowIfNull(visitor);

        return visitor.Visit(this);
    }
}

/// <summary>A parameter naming several columns.</summary>
/// <param name="key">The key it is written under.</param>
/// <param name="description">What it means.</param>
/// <param name="example">The value a new block starts with.</param>
/// <param name="accepts">The kinds of column the step can work on.</param>
/// <param name="optional">
/// Whether none may be named, in which case the step decides — every column, say. None named is written as
/// the key left out.
/// </param>
/// <param name="repeatable">
/// Whether one column may stand in more than one place: when each place is a role of its own, as an indicator's
/// high, low and close are. Otherwise the list is a set of columns, and a column is named in it once.
/// </param>
public sealed class ColumnsParameter(
    string key, string description, IReadOnlyList<string> example, IReadOnlyList<ColumnKind> accepts, bool optional = false, bool repeatable = false)
    : StepParameter<IReadOnlyList<string>>(key, description, example)
{
    /// <summary>The kinds of column the step can work on.</summary>
    public IReadOnlyList<ColumnKind> Accepts { get; } = accepts;

    /// <summary>Whether none may be named.</summary>
    public bool Optional { get; } = optional;

    /// <summary>Whether one column may stand in more than one place, each place being a role of its own.</summary>
    public bool Repeatable { get; } = repeatable;

    /// <inheritdoc />
    public override IReadOnlyList<string> RequiredKeys => Optional ? [] : Keys;

    /// <inheritdoc />
    public override IReadOnlyList<string> Read(JsonElement step)
    {
        if (Optional && !step.TryGetProperty(Key, out _))
        {
            return [];
        }

        if (!step.TryGetProperty(Key, out var list) || list.ValueKind != JsonValueKind.Array)
        {
            throw new FormatException($"The step holds a '{Key}' list.");
        }

        return [.. list.EnumerateArray().Select(item => item.ValueKind == JsonValueKind.String
            ? item.GetString()!
            : throw new FormatException($"Every entry in '{Key}' is the name of a column."))];
    }

    /// <inheritdoc />
    public override void Write(Utf8JsonWriter writer, IReadOnlyList<string> value)
    {
        ArgumentNullException.ThrowIfNull(writer);
        ArgumentNullException.ThrowIfNull(value);

        if (Optional && value.Count == 0)
        {
            return;
        }

        writer.WriteStartArray(Key);

        foreach (var column in value)
        {
            writer.WriteStringValue(column);
        }

        writer.WriteEndArray();
    }

    /// <inheritdoc />
    public override IReadOnlyList<string> Require(IReadOnlyList<string> value)
    {
        ArgumentNullException.ThrowIfNull(value);

        if (value.Count == 0 && !Optional)
        {
            throw new ArgumentException($"'{Key}' names at least one column: {Description}", Key);
        }

        var named = new HashSet<string>(StringComparer.Ordinal);

        foreach (var column in value)
        {
            Required(column);

            // A set that names a column twice says nothing more the second time, and a step leaving it out would
            // find it gone by then.
            if (!named.Add(column) && !Repeatable)
            {
                throw new ArgumentException($"'{Key}' names '{column}' twice; name each column once.", Key);
            }
        }

        return value;
    }

    /// <inheritdoc />
    public override TResult Accept<TResult>(IStepParameterVisitor<TResult> visitor)
    {
        ArgumentNullException.ThrowIfNull(visitor);

        return visitor.Visit(this);
    }

    /// <inheritdoc />
    private protected override bool Same(IReadOnlyList<string> one, IReadOnlyList<string> other) =>
        one.SequenceEqual(other, StringComparer.Ordinal);

    /// <inheritdoc />
    /// <remarks>A column named in several places — the roles of an indicator — is one column read.</remarks>
    internal override IEnumerable<ColumnRead> ReadsIn(IReadOnlyList<string> value) =>
        value.Distinct(StringComparer.Ordinal).Select(column => new ColumnRead(column, Accepts));
}

/// <summary>A parameter holding a number that may have a fraction.</summary>
/// <param name="key">The key it is written under.</param>
/// <param name="description">What it means.</param>
/// <param name="example">The value a new block starts with.</param>
/// <param name="above">A bound the number has to be strictly above, when there is one.</param>
public sealed class NumberParameter(string key, string description, double example, double? above = null)
    : StepParameter<double>(key, description, example)
{
    /// <summary>The bound the number has to be strictly above, when there is one.</summary>
    public double? Above { get; } = above;

    /// <inheritdoc />
    public override double Read(JsonElement step) => step.RequiredNumber(Key);

    /// <inheritdoc />
    public override void Write(Utf8JsonWriter writer, double value)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber(Key, value);
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentOutOfRangeException">The number is not finite, or is not above its bound.</exception>
    public override double Require(double value) =>
        double.IsFinite(value) && (Above is not { } bound || value > bound)
            ? value
            : throw new ArgumentOutOfRangeException(Key, value, $"'{Key}' is {(Above is { } floor ? $"a number above {floor}" : "a finite number")}: {Description}");

    /// <inheritdoc />
    public override TResult Accept<TResult>(IStepParameterVisitor<TResult> visitor)
    {
        ArgumentNullException.ThrowIfNull(visitor);

        return visitor.Visit(this);
    }
}

/// <summary>A parameter holding a whole number: a seed, a limit, a period.</summary>
/// <param name="key">The key it is written under.</param>
/// <param name="description">What it means.</param>
/// <param name="example">The value a new block starts with.</param>
/// <param name="atLeast">The smallest value it may hold, when there is one.</param>
public sealed class WholeNumberParameter(string key, string description, int example, int? atLeast = null)
    : StepParameter<int>(key, description, example)
{
    /// <summary>The smallest value it may hold, when there is one.</summary>
    public int? AtLeast { get; } = atLeast;

    /// <inheritdoc />
    public override int Read(JsonElement step) => step.RequiredWholeNumber(Key);

    /// <inheritdoc />
    public override void Write(Utf8JsonWriter writer, int value)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber(Key, value);
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentOutOfRangeException">The number is below its smallest value.</exception>
    public override int Require(int value) =>
        AtLeast is { } least && value < least
            ? throw new ArgumentOutOfRangeException(Key, value, $"'{Key}' is at least {least}: {Description}")
            : value;

    /// <inheritdoc />
    public override TResult Accept<TResult>(IStepParameterVisitor<TResult> visitor)
    {
        ArgumentNullException.ThrowIfNull(visitor);

        return visitor.Visit(this);
    }
}

/// <summary>A parameter holding a share, from nought to one, that a file may leave out.</summary>
/// <param name="key">The key it is written under.</param>
/// <param name="description">What it means, and what leaving it out means.</param>
/// <remarks>
/// Left out, there is none, and no number is invented to stand for it: a limit nobody set is no limit, not a
/// guess at a sensible one.
/// </remarks>
public sealed class ShareParameter(string key, string description)
    : StepParameter<double?>(key, description, null)
{
    /// <inheritdoc />
    public override IReadOnlyList<string> RequiredKeys => [];

    /// <inheritdoc />
    public override double? Read(JsonElement step) =>
        step.TryGetProperty(Key, out _) ? step.RequiredNumber(Key) : null;

    /// <inheritdoc />
    /// <remarks>Nothing is written for a share that was left out.</remarks>
    public override void Write(Utf8JsonWriter writer, double? value)
    {
        ArgumentNullException.ThrowIfNull(writer);

        if (value is { } share)
        {
            writer.WriteNumber(Key, share);
        }
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentOutOfRangeException">The value is not a share: below nought, above one, or not a number.</exception>
    public override double? Require(double? value) =>
        value is not { } share || (share >= 0 && share <= 1)
            ? value
            : throw new ArgumentOutOfRangeException(Key, value, $"'{Key}' is a share, from nought to one: {Description}");

    /// <inheritdoc />
    public override TResult Accept<TResult>(IStepParameterVisitor<TResult> visitor)
    {
        ArgumentNullException.ThrowIfNull(visitor);

        return visitor.Visit(this);
    }
}

/// <summary>A parameter holding true or false.</summary>
/// <param name="key">The key it is written under.</param>
/// <param name="description">What it means.</param>
/// <param name="example">The value a new block starts with.</param>
public sealed class TrueOrFalseParameter(string key, string description, bool example)
    : StepParameter<bool>(key, description, example)
{
    /// <inheritdoc />
    public override bool Read(JsonElement step) => step.RequiredBoolean(Key);

    /// <inheritdoc />
    public override void Write(Utf8JsonWriter writer, bool value)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteBoolean(Key, value);
    }

    /// <inheritdoc />
    public override TResult Accept<TResult>(IStepParameterVisitor<TResult> visitor)
    {
        ArgumentNullException.ThrowIfNull(visitor);

        return visitor.Visit(this);
    }
}

/// <summary>A parameter holding one of a named set of words.</summary>
/// <typeparam name="TEnum">The set of words.</typeparam>
/// <param name="key">The key it is written under.</param>
/// <param name="description">What it means.</param>
/// <param name="example">The value a new block starts with.</param>
public sealed class OneOfParameter<TEnum>(string key, string description, TEnum example)
    : StepParameter<TEnum>(key, description, example)
    where TEnum : struct, Enum
{
    /// <summary>The words it may hold, as they are written.</summary>
    public IReadOnlyList<string> Choices => Vocabulary<TEnum>.Words;

    /// <inheritdoc />
    /// <remarks>One of the words, in whatever case it was typed, and nothing else.</remarks>
    public override TEnum Read(JsonElement step) => Vocabulary<TEnum>.Read(step.RequiredString(Key), Key);

    /// <inheritdoc />
    public override void Write(Utf8JsonWriter writer, TEnum value)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteString(Key, Vocabulary<TEnum>.WordFor(value, Key));
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentOutOfRangeException">The value is none of the words: a number cast to the set.</exception>
    public override TEnum Require(TEnum value)
    {
        Vocabulary<TEnum>.WordFor(value, Key);

        return value;
    }

    /// <inheritdoc />
    public override TResult Accept<TResult>(IStepParameterVisitor<TResult> visitor)
    {
        ArgumentNullException.ThrowIfNull(visitor);

        return visitor.Visit(this);
    }
}

/// <summary>A parameter holding several of a named set of words.</summary>
/// <typeparam name="TEnum">The set of words.</typeparam>
/// <param name="key">The key it is written under.</param>
/// <param name="description">What it means.</param>
/// <param name="example">The value a new block starts with.</param>
public sealed class SeveralOfParameter<TEnum>(string key, string description, IReadOnlyList<TEnum> example)
    : StepParameter<IReadOnlyList<TEnum>>(key, description, example)
    where TEnum : struct, Enum
{
    /// <summary>The words it may hold, as they are written.</summary>
    public IReadOnlyList<string> Choices => Vocabulary<TEnum>.Words;

    /// <inheritdoc />
    public override IReadOnlyList<TEnum> Read(JsonElement step)
    {
        if (!step.TryGetProperty(Key, out var list) || list.ValueKind != JsonValueKind.Array)
        {
            throw new FormatException($"The step holds a '{Key}' list.");
        }

        return [.. list.EnumerateArray().Select(item => item.ValueKind == JsonValueKind.String
            ? Vocabulary<TEnum>.Read(item.GetString()!, Key)
            : throw new FormatException(
                $"'{item}' is not one of the things '{Key}' can hold: {string.Join(", ", Choices)}."))];
    }

    /// <inheritdoc />
    public override void Write(Utf8JsonWriter writer, IReadOnlyList<TEnum> value)
    {
        ArgumentNullException.ThrowIfNull(writer);
        ArgumentNullException.ThrowIfNull(value);

        writer.WriteStartArray(Key);

        foreach (var each in value)
        {
            writer.WriteStringValue(Vocabulary<TEnum>.WordFor(each, Key));
        }

        writer.WriteEndArray();
    }

    /// <inheritdoc />
    public override IReadOnlyList<TEnum> Require(IReadOnlyList<TEnum> value)
    {
        ArgumentNullException.ThrowIfNull(value);

        if (value.Count == 0)
        {
            throw new ArgumentException($"'{Key}' holds at least one: {Description}", Key);
        }

        foreach (var each in value)
        {
            Vocabulary<TEnum>.WordFor(each, Key);
        }

        return value;
    }

    /// <inheritdoc />
    public override TResult Accept<TResult>(IStepParameterVisitor<TResult> visitor)
    {
        ArgumentNullException.ThrowIfNull(visitor);

        return visitor.Visit(this);
    }

    /// <inheritdoc />
    private protected override bool Same(IReadOnlyList<TEnum> one, IReadOnlyList<TEnum> other) => one.SequenceEqual(other);
}

/// <summary>A parameter holding how a value is filled: a strategy by name, and its number when it takes one.</summary>
/// <param name="key">The key it is written under.</param>
/// <param name="description">What it means.</param>
/// <param name="example">The value a new block starts with.</param>
/// <param name="allowed">The strategies this step may use.</param>
/// <param name="what">What the strategy does, for the message: "filling a gap".</param>
public sealed class FillStrategyParameter(
    string key, string description, FillStrategy example, IReadOnlyList<string> allowed, string what)
    : StepParameter<FillStrategy>(key, description, example)
{
    /// <summary>The key a strategy carrying a number writes its name under.</summary>
    public const string KindKey = "kind";

    /// <summary>The key a strategy carrying a number writes the number under.</summary>
    public const string ValueKey = "value";

    /// <summary>The strategies this step may use, as they are written.</summary>
    public IReadOnlyList<string> Allowed { get; } = allowed;

    /// <summary>The keys a strategy carrying a number is written with.</summary>
    public IReadOnlyList<string> StrategyKeys { get; } = [KindKey, ValueKey];

    /// <inheritdoc />
    /// <remarks>
    /// A strategy with no number is the word alone, which is what a person reads best; one that carries a
    /// number is an object, so the number has somewhere to live — and nothing else may live there.
    /// </remarks>
    public override FillStrategy Read(JsonElement step)
    {
        if (!step.TryGetProperty(Key, out var with))
        {
            throw new FormatException($"The step is missing a text value for '{Key}'.");
        }

        switch (with.ValueKind)
        {
            case JsonValueKind.String:
                return new FillStrategy(with.GetString()!);

            case JsonValueKind.Object:
                foreach (var property in with.EnumerateObject().Where(property => !StrategyKeys.Contains(property.Name)))
                {
                    throw new FormatException(
                        $"A strategy written with its number has no '{property.Name}'. It takes: {string.Join(", ", StrategyKeys)}.");
                }

                return new FillStrategy(with.RequiredString(KindKey), with.RequiredNumber(ValueKey));

            default:
                throw new FormatException($"The step is missing a text value for '{Key}'.");
        }
    }

    /// <inheritdoc />
    public override void Write(Utf8JsonWriter writer, FillStrategy value)
    {
        ArgumentNullException.ThrowIfNull(writer);

        if (value.Value is { } number)
        {
            writer.WriteStartObject(Key);
            writer.WriteString(KindKey, value.Name);
            writer.WriteNumber(ValueKey, number);
            writer.WriteEndObject();
        }
        else
        {
            writer.WriteString(Key, value.Name);
        }
    }

    /// <inheritdoc />
    /// <remarks>
    /// A strategy is a name, so a default one carries no name at all and a hand-written file can carry any
    /// word. Both are caught here, at the one point a strategy enters a step.
    /// </remarks>
    public override FillStrategy Require(FillStrategy value)
    {
        if (string.IsNullOrWhiteSpace(value.Name))
        {
            throw new ArgumentException($"{char.ToUpperInvariant(what[0])}{what[1..]} needs a strategy; With has the names.", Key);
        }

        if (!With.Knows(value.Name) || !Allowed.Contains(value.Name, StringComparer.Ordinal))
        {
            throw new ArgumentException($"'{value.Name}' is not a way of {what}. With has the names.", Key);
        }

        if (With.TakesAValue(value.Name) != value.Value.HasValue)
        {
            throw new ArgumentException(
                $"The strategy '{value.Name}' is written {(value.Value.HasValue ? "without" : "with")} a number.", Key);
        }

        return value;
    }

    /// <inheritdoc />
    public override TResult Accept<TResult>(IStepParameterVisitor<TResult> visitor)
    {
        ArgumentNullException.ThrowIfNull(visitor);

        return visitor.Visit(this);
    }
}

/// <summary>A parameter holding the shares a split divides the rows into.</summary>
/// <remarks>
/// Four numbers written under four keys, because they are only meaningful together: each is a share and
/// together they make a whole. The share to predict on may be left out of a file, because having none is
/// the ordinary case.
/// </remarks>
public sealed class SplitSharesParameter() : StepParameter<SplitShares>(
    "train",
    "How the rows are shared out: training, validation, test and a part to predict on, which together make the whole.",
    new SplitShares(0.70, 0.15, 0.15))
{
    /// <summary>
    /// The key of the share measured on: the one a person never has to choose, since it is whatever the others
    /// leave — worked out wherever the shares are set one at a time.
    /// </summary>
    public const string TestKey = "test";

    /// <inheritdoc />
    public override IReadOnlyList<string> Keys { get; } = ["train", "validation", TestKey, "predict"];

    /// <inheritdoc />
    public override IReadOnlyList<string> RequiredKeys { get; } = ["train", "validation", TestKey];

    /// <inheritdoc />
    public override SplitShares Read(JsonElement step) =>
        new(step.RequiredNumber("train"),
            step.RequiredNumber("validation"),
            step.RequiredNumber(TestKey),
            step.OptionalNumber("predict"));

    /// <inheritdoc />
    public override void Write(Utf8JsonWriter writer, SplitShares value)
    {
        ArgumentNullException.ThrowIfNull(writer);

        writer.WriteNumber("train", value.Train);
        writer.WriteNumber("validation", value.Validation);
        writer.WriteNumber(TestKey, value.Test);
        writer.WriteNumber("predict", value.Predict);
    }

    /// <inheritdoc />
    public override SplitShares Require(SplitShares value)
    {
        value.Validate();

        return value;
    }

    /// <inheritdoc />
    public override TResult Accept<TResult>(IStepParameterVisitor<TResult> visitor)
    {
        ArgumentNullException.ThrowIfNull(visitor);

        return visitor.Visit(this);
    }
}

/// <summary>A parameter holding the columns a schema declares: each one's name, kind, and whether it may be absent.</summary>
/// <param name="key">The key it is written under.</param>
/// <param name="description">What it means.</param>
/// <param name="example">The value a new block starts with.</param>
public sealed class ColumnDeclarationsParameter(string key, string description, IReadOnlyList<ColumnDeclaration> example)
    : StepParameter<IReadOnlyList<ColumnDeclaration>>(key, description, example)
{
    /// <summary>The name of one declared column, as it is written inside the list.</summary>
    public TextParameter Name { get; } = new("name", "The column's name in the source.", "column");

    /// <summary>What one declared column holds, as it is written inside the list.</summary>
    public OneOfParameter<ColumnKind> Kind { get; } = new("kind", "What the column holds.", ColumnKind.Number);

    /// <summary>Whether one declared column may be absent, as it is written inside the list.</summary>
    public TrueOrFalseParameter Optional { get; } = new(
        "optional", "Whether the source is allowed not to have the column at all.", false);

    /// <summary>The keys each declared column is written with.</summary>
    public IReadOnlyList<string> ColumnKeys => [Name.Key, Kind.Key, Optional.Key];

    /// <inheritdoc />
    public override IReadOnlyList<ColumnDeclaration> Read(JsonElement step)
    {
        if (!step.TryGetProperty(Key, out var columns) || columns.ValueKind != JsonValueKind.Array)
        {
            throw new FormatException($"A declare step holds a '{Key}' list.");
        }

        return [.. columns.EnumerateArray().Select(column =>
        {
            if (column.ValueKind != JsonValueKind.Object)
            {
                throw new FormatException($"Every entry in '{Key}' is a column: {string.Join(", ", ColumnKeys)}.");
            }

            foreach (var property in column.EnumerateObject().Where(property => !ColumnKeys.Contains(property.Name)))
            {
                throw new FormatException(
                    $"A declared column has no '{property.Name}'. It takes: {string.Join(", ", ColumnKeys)}.");
            }

            return new ColumnDeclaration(Name.Read(column), Kind.Read(column), Optional.Read(column));
        })];
    }

    /// <inheritdoc />
    public override void Write(Utf8JsonWriter writer, IReadOnlyList<ColumnDeclaration> value)
    {
        ArgumentNullException.ThrowIfNull(writer);
        ArgumentNullException.ThrowIfNull(value);

        writer.WriteStartArray(Key);

        foreach (var column in value)
        {
            writer.WriteStartObject();
            Name.Write(writer, column.Name);
            Kind.Write(writer, column.Kind);
            Optional.Write(writer, column.Optional);
            writer.WriteEndObject();
        }

        writer.WriteEndArray();
    }

    /// <inheritdoc />
    /// <remarks>
    /// Two declarations of one column cannot both be right, and the second silently winning is how a column
    /// ends up typed one way in the schema and another way in everybody's head.
    /// </remarks>
    public override IReadOnlyList<ColumnDeclaration> Require(IReadOnlyList<ColumnDeclaration> value)
    {
        ArgumentNullException.ThrowIfNull(value);

        if (value.Count == 0)
        {
            throw new ArgumentException("A schema names at least one column.", Key);
        }

        foreach (var column in value)
        {
            Name.Require(column.Name);
            Kind.Require(column.Kind);
        }

        var duplicate = value.GroupBy(column => column.Name).FirstOrDefault(group => group.Count() > 1);

        return duplicate is null
            ? value
            : throw new ArgumentException($"The column '{duplicate.Key}' is declared twice.", Key);
    }

    /// <inheritdoc />
    public override TResult Accept<TResult>(IStepParameterVisitor<TResult> visitor)
    {
        ArgumentNullException.ThrowIfNull(visitor);

        return visitor.Visit(this);
    }

    /// <inheritdoc />
    private protected override bool Same(IReadOnlyList<ColumnDeclaration> one, IReadOnlyList<ColumnDeclaration> other) =>
        one.SequenceEqual(other);
}

/// <summary>The kinds of column a step can work on, in the groups the steps share.</summary>
public static class ColumnKinds
{
    /// <summary>Anything a column can hold.</summary>
    public static IReadOnlyList<ColumnKind> Any { get; } = Enum.GetValues<ColumnKind>();

    /// <summary>What reads as a number: a number, a whole number, true or false.</summary>
    public static IReadOnlyList<ColumnKind> Numbers { get; } = [ColumnKind.Number, ColumnKind.Integer, ColumnKind.Boolean];

    /// <summary>What a gap can be filled with a number in: a number or a whole number.</summary>
    public static IReadOnlyList<ColumnKind> Fillable { get; } = [ColumnKind.Number, ColumnKind.Integer];

    /// <summary>What can hold a value that is not a number: a number with a fraction.</summary>
    public static IReadOnlyList<ColumnKind> Fractions { get; } = [ColumnKind.Number];

    /// <summary>A moment in time.</summary>
    public static IReadOnlyList<ColumnKind> Moments { get; } = [ColumnKind.Timestamp];

    /// <summary>What rows can be put in order by: a moment, a whole number, a number.</summary>
    public static IReadOnlyList<ColumnKind> Ordered { get; } = [ColumnKind.Timestamp, ColumnKind.Integer, ColumnKind.Number];
}

/// <summary>
/// The words a set of values is written as in a file, and the one rule they are read back by.
/// </summary>
/// <typeparam name="TEnum">The set of values.</typeparam>
/// <remarks>
/// Each value is its name in lower case, and a word is read back when it is one of those names in whatever
/// case it was typed — and nothing else. The runtime's own parser takes more: a number standing for a place
/// in the list, and several names joined by commas, which it adds up into a different one.
/// </remarks>
internal static class Vocabulary<TEnum>
    where TEnum : struct, Enum
{
    private static readonly TEnum[] Values = Enum.GetValues<TEnum>();

    /// <summary>The words, in the order of the values they stand for.</summary>
    public static IReadOnlyList<string> Words { get; } = [.. Enum.GetNames<TEnum>().Select(name => name.ToLowerInvariant())];

    /// <summary>The value a written word stands for.</summary>
    /// <param name="written">The word as a file holds it.</param>
    /// <param name="key">The key it was written under, for the message.</param>
    /// <returns>The value.</returns>
    /// <exception cref="FormatException">The word is none of the words.</exception>
    public static TEnum Read(string written, string key)
    {
        for (var at = 0; at < Words.Count; at++)
        {
            if (string.Equals(Words[at], written, StringComparison.OrdinalIgnoreCase))
            {
                return Values[at];
            }
        }

        throw new FormatException($"'{written}' is not one of the things '{key}' can be: {string.Join(", ", Words)}.");
    }

    /// <summary>The word a value is written as.</summary>
    /// <param name="value">The value.</param>
    /// <param name="key">The key it is written under, for the message.</param>
    /// <returns>The word.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The value is none of the words: a number cast to the set.</exception>
    public static string WordFor(TEnum value, string key)
    {
        var at = Array.IndexOf(Values, value);

        return at >= 0
            ? Words[at]
            : throw new ArgumentOutOfRangeException(
                key, value, $"'{key}' is one of: {string.Join(", ", Words)}.");
    }
}
