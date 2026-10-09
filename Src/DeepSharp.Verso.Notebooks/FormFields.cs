// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>What the form knows of the columns around a block.</summary>
/// <param name="Columns">The columns there before the block, as the last gesture assembled the notebook; nothing when unknown.</param>
/// <param name="Source">The columns the source has, in its order, when its rows were read; nothing when they were not.</param>
internal readonly record struct FormScope(IReadOnlyList<KnownColumn>? Columns, IReadOnlyList<string>? Source)
{
    /// <summary>Nothing known: no gesture has shown the notebook's pipeline yet.</summary>
    public static FormScope Unknown { get; } = new(null, null);

    /// <summary>The columns of a kind a parameter works on, in the order they come; none when nothing is known.</summary>
    /// <param name="accepts">The kinds.</param>
    /// <returns>Their names.</returns>
    public IReadOnlyList<string> Of(IReadOnlyList<ColumnKind> accepts) => (Columns ?? []).NamesOf(accepts);
}

/// <summary>
/// The fields of the form for each kind of parameter, read from the step as it writes itself.
/// </summary>
/// <param name="step">The step's JSON, as the step writes it.</param>
/// <param name="scope">The columns known around the block.</param>
/// <remarks>
/// A projection of the kinds like the schema and the completions. Every number is a text field in its invariant
/// spelling: a remote front end hands a number field back through the interface's culture, so 0.7 would come back
/// as 0,7. A column is picked from the columns in scope of a kind the parameter works on, and written as text when
/// none is known — never offered as an empty list. A list of columns is one switch per column, since Verso joins
/// the picks of a several-choice field with commas, which a column's name may hold; a list whose places are roles
/// is one pick per place.
/// </remarks>
internal sealed class FormFields(JsonElement step, FormScope scope) : IStepParameterVisitor<IEnumerable<PropertyField>>
{
    public IEnumerable<PropertyField> Visit(TextParameter parameter) => [Text(parameter, parameter.Read(step))];

    public IEnumerable<PropertyField> Visit(FilePathParameter parameter) => [Text(parameter, parameter.Read(step))];

    public IEnumerable<PropertyField> Visit(ColumnParameter parameter)
    {
        var current = parameter.Read(step);
        var names = scope.Of(parameter.Accepts);

        return [names.Count == 0 ? Text(parameter, current) : Field.Of(parameter).Pick(current, names)];
    }

    public IEnumerable<PropertyField> Visit(NewColumnParameter parameter) => [Text(parameter, parameter.Read(step))];

    public IEnumerable<PropertyField> Visit(ColumnsParameter parameter)
    {
        var current = parameter.Read(step);
        var names = scope.Of(parameter.Accepts);

        if (names.Count == 0)
        {
            return [Text(parameter, current.AsListText())];
        }

        if (parameter.Repeatable)
        {
            return current.Select((name, place) =>
                new Field(parameter.Place(place), $"{parameter.Key} {place + 1}", parameter.Description).Pick(name, names));
        }

        return names.Concat(current.Except(names, StringComparer.Ordinal)).Select(name => new PropertyField(
            parameter.Member(name), name, PropertyFieldType.Toggle, current.Contains(name, StringComparer.Ordinal), parameter.Description));
    }

    // A number a file may leave out shows the value leaving it out means.
    public IEnumerable<PropertyField> Visit(NumberParameter parameter) =>
        [Text(parameter, step.TryGetProperty(parameter.Key, out var written) ? written.GetRawText() : parameter.LeftOut?.ToString(CultureInfo.InvariantCulture))];

    // A whole number a file may leave out shows the value leaving it out means.
    public IEnumerable<PropertyField> Visit(WholeNumberParameter parameter) =>
        [Text(parameter, step.TryGetProperty(parameter.Key, out var written) ? written.GetRawText() : parameter.LeftOut?.ToString(CultureInfo.InvariantCulture))];

    public IEnumerable<PropertyField> Visit(TrueOrFalseParameter parameter) =>
        [new(parameter.Key, parameter.Key, PropertyFieldType.Toggle, parameter.Read(step), parameter.Description)];

    public IEnumerable<PropertyField> Visit(ShareParameter parameter) =>
        [Text(parameter, step.TryGetProperty(parameter.Key, out _) ? Written(parameter.Key) : string.Empty)];

    // A word a file may leave out shows the word leaving it out means.
    public IEnumerable<PropertyField> Visit<TEnum>(OneOfParameter<TEnum> parameter)
        where TEnum : struct, Enum =>
        [Field.Of(parameter).Choose(
            step.TryGetProperty(parameter.Key, out var written) ? written.GetString() : parameter.LeftOut?.Word(),
            parameter.Choices)];

    public IEnumerable<PropertyField> Visit<TEnum>(SeveralOfParameter<TEnum> parameter)
        where TEnum : struct, Enum =>
        [new(
            parameter.Key, parameter.Key, PropertyFieldType.MultiSelect,
            string.Join(',', step.GetProperty(parameter.Key).EnumerateArray().Select(item => item.GetString())),
            parameter.Description,
            Options(parameter.Choices))];

    public IEnumerable<PropertyField> Visit(FillStrategyParameter parameter)
    {
        var strategy = parameter.Read(step);
        var name = Field.Of(parameter).Choose(strategy.Name, parameter.Allowed);

        return strategy.Value is null
            ? [name]
            : [name, new(
                parameter.Number(), FillStrategyParameter.ValueKey, PropertyFieldType.Text,
                Written(parameter.Key, FillStrategyParameter.ValueKey), parameter.Description)];
    }

    // Train, validation and predict are set; test is what they leave, worked out rather than typed.
    public IEnumerable<PropertyField> Visit(SplitSharesParameter parameter) =>
        parameter.Keys.Select(key => new PropertyField(
            key, key, PropertyFieldType.Text, Written(key), parameter.Description, IsReadOnly: key == SplitSharesParameter.TestKey));

    // One pick per column the source has — the kind the schema gives it, or not taken — whether each column taken may
    // be absent from the rows, for each taken timestamp how its moments are written, empty for ISO 8601, for each
    // taken column the value that stands for a gap, empty for none, and for each taken column that can name a row —
    // whole numbers, a category, words — whether it is the id.
    public IEnumerable<PropertyField> Visit(ColumnDeclarationsParameter parameter)
    {
        // The columns as the schema reads them: one it excludes is not taken, and keeps its kind for when it is again.
        // Each taken column's kind is as the step wrote it, so the words are the reader's own.
        var declared = parameter.Read(step);
        var taking = step.GetProperty(parameter.Key).EnumerateArray().Zip(declared)
            .Where(column => !column.Second.Excluded)
            .ToDictionary(
                column => column.Second.Name,
                column => new Taken(column.First.GetProperty(parameter.Kind.Key).GetString()!, column.Second),
                StringComparer.Ordinal);
        var names = (scope.Source ?? []).Concat(declared.Select(column => column.Name)).Distinct(StringComparer.Ordinal);
        var fields = new List<PropertyField>();

        foreach (var name in names)
        {
            var taken = taking.GetValueOrDefault(name);

            fields.Add(new Field(parameter.Kind(name), name, parameter.Kind.Description).Choose(
                taken?.Kind ?? FormVocabulary.NotTaken, [FormVocabulary.NotTaken, .. parameter.Kind.Choices]));

            if (taken is not null)
            {
                fields.Add(new(
                    parameter.Absent(name), $"{name} may be absent", PropertyFieldType.Toggle, taken.Declared.Optional,
                    parameter.Optional.Description));
            }

            if (taken?.Declared.Kind == ColumnKind.Timestamp)
            {
                fields.Add(new(
                    parameter.Format(name), $"{name} is written as", PropertyFieldType.Text, taken.Declared.Format,
                    parameter.Format.Description));
            }

            if (taken is not null)
            {
                fields.Add(new(
                    parameter.Missing(name), $"{name} is a gap when it holds", PropertyFieldType.Text, taken.Declared.Missing,
                    parameter.Missing.Description));
            }

            if (taken?.Declared.Kind is ColumnKind.Integer or ColumnKind.Category or ColumnKind.Text)
            {
                fields.Add(new(
                    parameter.Id(name), $"{name} names each row", PropertyFieldType.Toggle, taken.Declared.Id, parameter.Id.Description));
            }
        }

        return fields;
    }

    // The parts of a model, one group of fields each: which name the part gives, and a field for every setting that
    // name takes, drawn by the setting's own kind. The fields are named by the part's place, so a block of any step
    // that declares parts is drawn the same way and nothing here knows which step it is. A key the step leaves out shows
    // the parts leaving it out means.
    public IEnumerable<PropertyField> Visit(PartsParameter parameter)
    {
        var declared = parameter.Read(step);
        var written = step.TryGetProperty(parameter.Key, out var held) ? held : Shown(parameter.AsJson(declared));
        var fields = new List<PropertyField>();

        for (var place = 0; place < declared.Count; place++)
        {
            var part = declared[place];
            var kind = parameter.Kinds.First(each => each.Name == part.Kind);
            var inside = parameter.Single ? written : written[place];

            fields.Add(new Field(
                parameter.Place(place),
                parameter.Single ? parameter.Key : $"{parameter.Key} {place + 1}",
                kind.Purpose).Choose(part.Kind, parameter.Names));

            foreach (var setting in kind.Settings)
            {
                foreach (var field in setting.Accept(new FormFields(inside, scope)))
                {
                    fields.Add(new PropertyField(
                        parameter.Setting(place, field.Name),
                        parameter.Single ? field.DisplayName : $"{field.DisplayName} {place + 1}",
                        field.FieldType,
                        field.CurrentValue,
                        field.Description,
                        field.Options,
                        field.IsReadOnly));
                }
            }
        }

        return fields;
    }

    // What the step would hold had it written what it leaves out, to draw the fields of.
    private static JsonElement Shown(System.Text.Json.Nodes.JsonNode node)
    {
        using var document = JsonDocument.Parse(node.ToJsonString());

        return document.RootElement.Clone();
    }

    private string Written(string key) => step.GetProperty(key).GetRawText();

    private string Written(string key, string inner) => step.GetProperty(key).GetProperty(inner).GetRawText();

    // A value the step leaves out shows as an empty field.
    private static PropertyField Text(StepParameter parameter, string? value) =>
        new(parameter.Key, parameter.Key, PropertyFieldType.Text, value, parameter.Description);

    private static IReadOnlyList<PropertyFieldOption> Options(IEnumerable<string> choices) =>
        [.. choices.Select(choice => new PropertyFieldOption(choice, choice))];

    /// <summary>One field of the form: the name a change to it comes back under, the label it shows, and what it means.</summary>
    /// <param name="Name">The name a change to it comes back under.</param>
    /// <param name="Label">The label it shows.</param>
    /// <param name="Description">What it means.</param>
    private readonly record struct Field(string Name, string Label, string Description)
    {
        /// <summary>The field of a parameter: named and labelled by its key.</summary>
        public static Field Of(StepParameter parameter) => new(parameter.Key, parameter.Key, parameter.Description);

        /// <summary>The field as a pick among some choices, at the one it holds.</summary>
        public PropertyField Choose(string? current, IReadOnlyList<string> choices) =>
            new(Name, Label, PropertyFieldType.Select, current, Description, Options(choices));

        /// <summary>The field as a pick among columns: the current one is always among them, so the field shows what the step holds.</summary>
        public PropertyField Pick(string current, IReadOnlyList<string> columns) =>
            Choose(current, columns.Contains(current, StringComparer.Ordinal) ? columns : [.. columns, current]);
    }

    /// <summary>One column a schema takes, as the step wrote it and as the schema reads it.</summary>
    /// <param name="Kind">The kind, as written.</param>
    /// <param name="Declared">The column as the schema reads it: whether it may be absent, how its moments are written, and which value stands for a gap.</param>
    private sealed record Taken(string Kind, ColumnDeclaration Declared);
}
