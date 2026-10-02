// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>What one setting of a declared part holds: a number, a word, or yes or no.</summary>
/// <remarks>The three a parameter of a part can hold, so a part's settings compare and are written as themselves.</remarks>
public enum PartValues
{
    /// <summary>A number, whole or not.</summary>
    Number,

    /// <summary>A word.</summary>
    Text,

    /// <summary>Yes or no.</summary>
    YesOrNo,
}

/// <summary>One value a setting of a declared part holds.</summary>
/// <param name="Holds">Which of the three it is.</param>
/// <param name="Number">The number, when it holds one.</param>
/// <param name="Text">The word, when it holds one.</param>
/// <param name="YesOrNo">Whether it is so, when it holds that.</param>
/// <remarks>
/// One shape for the three, so a part compares as a whole and nothing is boxed into a value of no kind. It is made
/// through <see cref="Of(double)"/>, <see cref="Of(string)"/> and <see cref="Of(bool)"/>, which is how a value and what
/// it holds are said in one breath.
/// </remarks>
public readonly record struct PartValue(PartValues Holds, double Number, string Text, bool YesOrNo)
{
    /// <summary>A setting that holds a number.</summary>
    /// <param name="number">The number.</param>
    /// <returns>The value.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It is not a number.</exception>
    public static PartValue Of(double number) => double.IsFinite(number)
        ? new(PartValues.Number, number, string.Empty, false)
        : throw new ArgumentOutOfRangeException(nameof(number), number, "A setting holds a number, and this is none.");

    /// <summary>A setting that holds a word.</summary>
    /// <param name="text">The word.</param>
    /// <returns>The value.</returns>
    /// <exception cref="ArgumentNullException">There is no word.</exception>
    public static PartValue Of(string text) => new(PartValues.Text, 0, text ?? throw new ArgumentNullException(nameof(text)), false);

    /// <summary>A setting that holds whether something is so.</summary>
    /// <param name="yesOrNo">Whether it is so.</param>
    /// <returns>The value.</returns>
    public static PartValue Of(bool yesOrNo) => new(PartValues.YesOrNo, 0, string.Empty, yesOrNo);
}

/// <summary>One setting of a declared part: the key it is written under, and what it holds.</summary>
/// <param name="Key">The key, as the part's own parameter names it.</param>
/// <param name="Value">What it holds.</param>
public readonly record struct PartSetting(string Key, PartValue Value);

/// <summary>
/// A part of a model as a declaration names it: a name, and the settings that name takes.
/// </summary>
/// <param name="Kind">The name — a layer's, an optimizer's, a loss's — as the step that reads it knows them.</param>
/// <param name="Settings">Its settings, in the order the name's own parameters stand.</param>
/// <remarks>
/// The pipeline knows no layer and no optimizer: a part is a name and its settings, and the package that brings the step
/// says which names there are, what each takes and what each becomes. Two parts that say the same thing are the same
/// part, settings and all, because a declaration is compared step by step.
/// </remarks>
public readonly record struct PartDeclaration(string Kind, IReadOnlyList<PartSetting> Settings)
{
    /// <summary>The number a setting holds.</summary>
    /// <param name="key">The setting's key.</param>
    /// <returns>The number.</returns>
    /// <exception cref="KeyNotFoundException">This part holds no such setting.</exception>
    public double Number(string key) => Setting(key).Number;

    /// <summary>The whole number a setting holds.</summary>
    /// <param name="key">The setting's key.</param>
    /// <returns>The number, as a whole one.</returns>
    /// <exception cref="KeyNotFoundException">This part holds no such setting.</exception>
    public int Whole(string key) => (int)Setting(key).Number;

    /// <summary>The word a setting holds.</summary>
    /// <param name="key">The setting's key.</param>
    /// <returns>The word.</returns>
    /// <exception cref="KeyNotFoundException">This part holds no such setting.</exception>
    public string Text(string key) => Setting(key).Text;

    /// <summary>Whether a setting says so.</summary>
    /// <param name="key">The setting's key.</param>
    /// <returns>Whether it is so.</returns>
    /// <exception cref="KeyNotFoundException">This part holds no such setting.</exception>
    public bool YesOrNo(string key) => Setting(key).YesOrNo;

    /// <summary>Whether this part holds a setting under that key.</summary>
    /// <param name="key">The setting's key.</param>
    /// <returns>Whether it holds one.</returns>
    public bool Holds(string key) => (Settings ?? []).Any(setting => setting.Key == key);

    /// <inheritdoc />
    public bool Equals(PartDeclaration other) =>
        Kind == other.Kind && (Settings ?? []).SequenceEqual(other.Settings ?? []);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        hash.Add(Kind);

        foreach (var setting in Settings ?? [])
        {
            hash.Add(setting);
        }

        return hash.ToHashCode();
    }

    private PartValue Setting(string key) =>
        (Settings ?? []).FirstOrDefault(setting => setting.Key == key, new PartSetting(string.Empty, default)) is { Key.Length: > 0 } found
            ? found.Value
            : throw new KeyNotFoundException(string.Create(CultureInfo.InvariantCulture, $"This '{Kind}' holds no '{key}'."));
}

/// <summary>
/// One name a declared part may give, and the settings that name takes.
/// </summary>
/// <remarks>
/// The step that brings a parts parameter says which names there are — a dense layer, an optimizer, a loss — and gives
/// each the parameters its settings are written and read through, so the file, the schema, the reference page and a
/// notebook's form all describe a part through the same kinds every other parameter is described through.
/// </remarks>
public sealed class PartKind
{
    /// <summary>A name a part may give, and what that name takes.</summary>
    /// <param name="name">The name, as it is written under 'kind'.</param>
    /// <param name="purpose">What a part of this name is, for whoever writes one.</param>
    /// <param name="settings">The settings the name takes, in the order they are written; none for a name that takes none.</param>
    /// <exception cref="ArgumentException">The name or the purpose is empty, or two settings share a key.</exception>
    /// <exception cref="ArgumentNullException">There are no settings at all, not even none of them.</exception>
    public PartKind(string name, string purpose, IReadOnlyList<StepParameter> settings)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentException.ThrowIfNullOrWhiteSpace(purpose);
        ArgumentNullException.ThrowIfNull(settings);

        if (settings.Select(setting => setting.Key).Distinct(StringComparer.Ordinal).Count() != settings.Count)
        {
            throw new ArgumentException($"The settings of '{name}' are written under one key each.", nameof(settings));
        }

        Name = name;
        Purpose = purpose;
        Settings = [.. settings];
        Starts = [.. settings.Select(setting => new PartSetting(setting.Key, PartsParameter.Holds(name, setting)))];
    }

    /// <summary>The name, as it is written under 'kind'.</summary>
    public string Name { get; }

    /// <summary>What a part of this name is.</summary>
    public string Purpose { get; }

    /// <summary>The settings this name takes, in the order they are written.</summary>
    public IReadOnlyList<StepParameter> Settings { get; }

    /// <summary>This name as a part, with every setting at the value a new one starts with.</summary>
    /// <returns>The part.</returns>
    public PartDeclaration Declared() => new(Name, Starts);

    // Every setting at the value a new part of this name starts with: read once, where the name was given its settings.
    private IReadOnlyList<PartSetting> Starts { get; }

    /// <summary>The setting written under a key, or nothing when this name takes none.</summary>
    /// <param name="key">The key.</param>
    /// <returns>The setting's parameter, or nothing.</returns>
    public StepParameter? Setting(string key) => Settings.FirstOrDefault(setting => setting.Key == key);
}

/// <summary>
/// The parts of a model a step declares by name: the layers of a network, the optimizer that moves it, the loss it is
/// judged by — each a name and the settings that name takes.
/// </summary>
/// <remarks>
/// The step that brings the parameter says which names there are and what each takes (<see cref="PartKind"/>), so the
/// pipeline holds a declared model to names it never has to know. Every setting a name takes is written: a file that
/// leaves one out is refused where it is read, rather than filled in behind the writer's back, because a pipeline is
/// replayed from what it says and a number nobody wrote down would make two runs of one declaration differ. One part is
/// written as itself (<see cref="Single"/>); several are written as a list, in the order they are read.
/// </remarks>
public sealed class PartsParameter : StepParameter<IReadOnlyList<PartDeclaration>>
{
    private const string KindKey = "kind";

    /// <summary>A parameter whose value is parts of these names.</summary>
    /// <param name="key">The key the parts are written under.</param>
    /// <param name="description">What the parts are, for whoever writes them.</param>
    /// <param name="example">The parts a new block starts with.</param>
    /// <param name="kinds">The names a part may give, and what each takes.</param>
    /// <exception cref="ArgumentException">There are no names at all, or two share one.</exception>
    /// <exception cref="ArgumentNullException">There are no names, or no parts to start with.</exception>
    public PartsParameter(string key, string description, IReadOnlyList<PartDeclaration> example, IReadOnlyList<PartKind> kinds)
        : base(key, description, example)
    {
        ArgumentNullException.ThrowIfNull(kinds);

        if (kinds.Count == 0)
        {
            throw new ArgumentException($"'{key}' says which names a part may give, and names none.", nameof(kinds));
        }

        if (kinds.Select(kind => kind.Name).Distinct(StringComparer.Ordinal).Count() != kinds.Count)
        {
            throw new ArgumentException($"The names '{key}' takes are one each.", nameof(kinds));
        }

        Kinds = [.. kinds];
    }

    /// <summary>The names a part may give, and what each takes.</summary>
    public IReadOnlyList<PartKind> Kinds { get; }

    /// <summary>Whether one part is written as itself rather than as a list; a list, unless said.</summary>
    public bool Single { get; init; }

    /// <summary>The names a part may give, as the file writes them.</summary>
    public IReadOnlyList<string> Names => [.. Kinds.Select(kind => kind.Name)];

    /// <inheritdoc />
    public override IReadOnlyList<PartDeclaration> Read(JsonElement step)
    {
        if (!step.TryGetProperty(Key, out var parts))
        {
            throw new FormatException($"This step is written with '{Key}': {Told()}.");
        }

        if (Single)
        {
            return parts.ValueKind == JsonValueKind.Object
                ? [Part(parts)]
                : throw new FormatException($"'{Key}' is one part, written as itself: {Told()}.");
        }

        return parts.ValueKind == JsonValueKind.Array
            ? [.. parts.EnumerateArray().Select(Part)]
            : throw new FormatException($"'{Key}' is a list of parts: {Told()}.");
    }

    /// <inheritdoc />
    public override void Write(Utf8JsonWriter writer, IReadOnlyList<PartDeclaration> value)
    {
        ArgumentNullException.ThrowIfNull(writer);
        ArgumentNullException.ThrowIfNull(value);

        if (Single)
        {
            writer.WritePropertyName(Key);
            Written(writer, value[0]);

            return;
        }

        writer.WriteStartArray(Key);

        foreach (var part in value)
        {
            Written(writer, part);
        }

        writer.WriteEndArray();
    }

    /// <inheritdoc />
    /// <exception cref="ArgumentException">
    /// There are no parts, or more than one where one is written; a part gives a name this parameter was never given, or
    /// holds a setting that name does not take, or leaves out one it does.
    /// </exception>
    public override IReadOnlyList<PartDeclaration> Require(IReadOnlyList<PartDeclaration> value)
    {
        ArgumentNullException.ThrowIfNull(value);

        if (value.Count == 0)
        {
            throw new ArgumentException($"'{Key}' is written with at least one part: {Told()}.", Key);
        }

        if (Single && value.Count > 1)
        {
            throw new ArgumentException($"'{Key}' is one part, and these are {value.Count}.", Key);
        }

        foreach (var part in value)
        {
            var kind = Named(part.Kind, fault => new ArgumentException(fault, Key));

            foreach (var setting in part.Settings.Where(setting => kind.Setting(setting.Key) is null))
            {
                throw new ArgumentException($"A '{kind.Name}' takes no '{setting.Key}': it takes {Takes(kind)}.", Key);
            }

            foreach (var setting in kind.Settings.Where(setting => !part.Holds(setting.Key)))
            {
                throw new ArgumentException($"This '{kind.Name}' leaves out '{setting.Key}': a part is written with everything its name takes.", Key);
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
    private protected override bool Same(IReadOnlyList<PartDeclaration> one, IReadOnlyList<PartDeclaration> other) =>
        one.SequenceEqual(other);

    private static string Takes(PartKind kind) =>
        kind.Settings.Count == 0 ? "nothing" : string.Join(", ", kind.Settings.Select(setting => $"'{setting.Key}'"));

    // The value a setting of this name starts at, read from the parameter's own example through the one reader every part
    // is read through: a kind that holds several values is refused here, where the name was given its settings.
    internal static PartValue Holds(string name, StepParameter setting)
    {
        ArgumentNullException.ThrowIfNull(setting);

        var buffer = new System.Buffers.ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            setting.WriteExample(writer);
            writer.WriteEndObject();
        }

        using var example = JsonDocument.Parse(buffer.WrittenMemory);

        try
        {
            return setting.Accept(new Held(example.RootElement));
        }
        catch (NotSupportedException fault)
        {
            throw new ArgumentException($"'{name}' takes a setting that cannot be written as one value. {fault.Message}", nameof(setting), fault);
        }
    }

    // One part as the file holds it: its name, then every setting that name takes, read through the setting's own parameter.
    private PartDeclaration Part(JsonElement part)
    {
        if (part.ValueKind != JsonValueKind.Object)
        {
            throw new FormatException($"Every part under '{Key}' is written as a name and its settings: {Told()}.");
        }

        if (!part.TryGetProperty(KindKey, out var named) || named.ValueKind != JsonValueKind.String)
        {
            throw new FormatException($"Every part under '{Key}' says which it is, under '{KindKey}': {Told()}.");
        }

        var kind = Named(named.GetString()!, fault => new FormatException(fault));

        foreach (var property in part.EnumerateObject().Where(property => property.Name != KindKey && kind.Setting(property.Name) is null))
        {
            throw new FormatException($"A '{kind.Name}' takes no '{property.Name}': it takes {Takes(kind)}.");
        }

        return new PartDeclaration(kind.Name, [.. kind.Settings.Select(setting => Setting(part, kind, setting))]);
    }

    // What one setting holds, read through the kind of value its own parameter holds.
    private PartSetting Setting(JsonElement part, PartKind kind, StepParameter setting)
    {
        if (!part.TryGetProperty(setting.Key, out _))
        {
            throw new FormatException(
                $"This '{kind.Name}' leaves out '{setting.Key}': a part is written with everything its name takes, and a '{kind.Name}' takes {Takes(kind)}.");
        }

        return new PartSetting(setting.Key, setting.Accept(new Held(part)));
    }

    private PartKind Named(string name, Func<string, Exception> fault) =>
        Kinds.FirstOrDefault(kind => kind.Name == name)
        ?? throw fault($"'{name}' is not a part this step knows: {Told()}.");

    private string Told() => string.Join(", ", Kinds.Select(kind => $"'{kind.Name}'"));

    // One part as this parameter writes it: its name first, then its settings in the order the name takes them.
    private void Written(Utf8JsonWriter writer, PartDeclaration part)
    {
        var kind = Named(part.Kind, fault => new ArgumentException(fault, Key));

        writer.WriteStartObject();
        writer.WriteString(KindKey, kind.Name);

        foreach (var setting in kind.Settings)
        {
            var value = part.Settings.First(held => held.Key == setting.Key).Value;

            switch (value.Holds)
            {
                case PartValues.Number:
                    writer.WriteNumber(setting.Key, value.Number);
                    break;

                case PartValues.YesOrNo:
                    writer.WriteBoolean(setting.Key, value.YesOrNo);
                    break;

                default:
                    writer.WriteString(setting.Key, value.Text);
                    break;
            }
        }

        writer.WriteEndObject();
    }

    // What a setting holds, read by the kind of value its parameter holds: a number, a word, or yes or no.
    private sealed class Held(JsonElement part) : IStepParameterVisitor<PartValue>
    {
        // A setting of a part holds one value — a number, a word, or yes or no — and is always written, so that a part
        // compares, and is written and read back, as itself. A kind that holds several, or that a file may leave out, is
        // refused where the name was given its settings, which is where whoever wrote the step can see it.
        private static NotSupportedException Several(StepParameter parameter) =>
            new($"A setting holds one value, and '{parameter.Key}' holds several: a part takes words, numbers, and whether something is so.");

        private static NotSupportedException Absent(StepParameter parameter) =>
            new($"A setting is always written, and '{parameter.Key}' may be left out: a part is written with everything its name takes.");

        public PartValue Visit(TextParameter parameter) => PartValue.Of(parameter.Read(part));

        public PartValue Visit(FilePathParameter parameter) => PartValue.Of(parameter.Read(part));

        public PartValue Visit(ColumnParameter parameter) => PartValue.Of(parameter.Read(part));

        public PartValue Visit(NewColumnParameter parameter) => parameter.Optional ? throw Absent(parameter) : PartValue.Of(parameter.Read(part)!);

        public PartValue Visit(ColumnsParameter parameter) => throw Several(parameter);

        public PartValue Visit(NumberParameter parameter) => PartValue.Of(parameter.Read(part));

        public PartValue Visit(WholeNumberParameter parameter) => PartValue.Of(parameter.Read(part));

        public PartValue Visit(TrueOrFalseParameter parameter) => PartValue.Of(parameter.Read(part));

        public PartValue Visit(ShareParameter parameter) => throw Absent(parameter);

        public PartValue Visit<TEnum>(OneOfParameter<TEnum> parameter)
            where TEnum : struct, Enum => PartValue.Of(parameter.Read(part).ToString()!);

        public PartValue Visit<TEnum>(SeveralOfParameter<TEnum> parameter)
            where TEnum : struct, Enum => throw Several(parameter);

        public PartValue Visit(FillStrategyParameter parameter) => PartValue.Of(parameter.Read(part).ToString());

        public PartValue Visit(SplitSharesParameter parameter) => throw Several(parameter);

        public PartValue Visit(ColumnDeclarationsParameter parameter) => throw Several(parameter);

        public PartValue Visit(PartsParameter parameter) => throw Several(parameter);
    }

}
