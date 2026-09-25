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

        return [names.Count == 0 ? Text(parameter, current) : Pick(parameter.Key, parameter.Key, parameter.Description, current, names)];
    }

    public IEnumerable<PropertyField> Visit(NewColumnParameter parameter) => [Text(parameter, parameter.Read(step))];

    public IEnumerable<PropertyField> Visit(ColumnsParameter parameter)
    {
        var current = parameter.Read(step);
        var names = scope.Of(parameter.Accepts);

        if (names.Count == 0)
        {
            return [Text(parameter, FormVocabulary.ListText(current))];
        }

        if (parameter.Repeatable)
        {
            return current.Select((name, place) =>
                Pick(FormVocabulary.Place(parameter.Key, place), $"{parameter.Key} {place + 1}", parameter.Description, name, names));
        }

        return names.Concat(current.Except(names, StringComparer.Ordinal)).Select(name => new PropertyField(
            FormVocabulary.Member(parameter.Key, name), name, PropertyFieldType.Toggle, current.Contains(name, StringComparer.Ordinal), parameter.Description));
    }

    public IEnumerable<PropertyField> Visit(NumberParameter parameter) => [Text(parameter, Written(parameter.Key))];

    // A whole number a file may leave out shows the value leaving it out means.
    public IEnumerable<PropertyField> Visit(WholeNumberParameter parameter) =>
        [Text(parameter, step.TryGetProperty(parameter.Key, out var written) ? written.GetRawText() : parameter.LeftOut?.ToString(CultureInfo.InvariantCulture))];

    public IEnumerable<PropertyField> Visit(TrueOrFalseParameter parameter) =>
        [new(parameter.Key, parameter.Key, PropertyFieldType.Toggle, parameter.Read(step), parameter.Description)];

    public IEnumerable<PropertyField> Visit(ShareParameter parameter) =>
        [Text(parameter, step.TryGetProperty(parameter.Key, out _) ? Written(parameter.Key) : string.Empty)];

    public IEnumerable<PropertyField> Visit<TEnum>(OneOfParameter<TEnum> parameter)
        where TEnum : struct, Enum =>
        [Choose(parameter.Key, parameter.Key, parameter.Description, step.GetProperty(parameter.Key).GetString(), parameter.Choices)];

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
        var name = Choose(parameter.Key, parameter.Key, parameter.Description, strategy.Name, parameter.Allowed);

        return strategy.Value is null
            ? [name]
            : [name, new(
                FormVocabulary.StrategyValue(parameter.Key), FillStrategyParameter.ValueKey, PropertyFieldType.Text,
                Written(parameter.Key, FillStrategyParameter.ValueKey), parameter.Description)];
    }

    // Train, validation and predict are set; test is what they leave, worked out rather than typed.
    public IEnumerable<PropertyField> Visit(SplitSharesParameter parameter) =>
        parameter.Keys.Select(key => new PropertyField(
            key, key, PropertyFieldType.Text, Written(key), parameter.Description, IsReadOnly: key == SplitSharesParameter.TestKey));

    // One pick per column the source has — the kind the schema gives it, or not taken — and whether each column
    // taken may be absent from the rows.
    public IEnumerable<PropertyField> Visit(ColumnDeclarationsParameter parameter)
    {
        // The kind of each taken column as the step wrote it, so the words are the reader's own.
        var declared = step.GetProperty(parameter.Key).EnumerateArray()
            .ToDictionary(
                column => column.GetProperty(parameter.Name.Key).GetString()!,
                column => new Taken(column.GetProperty(parameter.Kind.Key).GetString()!, column.GetProperty(parameter.Optional.Key).GetBoolean()),
                StringComparer.Ordinal);
        var names = (scope.Source ?? []).Concat(declared.Keys).Distinct(StringComparer.Ordinal);
        var fields = new List<PropertyField>();

        foreach (var name in names)
        {
            var taken = declared.GetValueOrDefault(name);

            fields.Add(Choose(
                FormVocabulary.Kind(parameter.Key, name), name, parameter.Kind.Description, taken?.Kind ?? FormVocabulary.NotTaken,
                [FormVocabulary.NotTaken, .. parameter.Kind.Choices]));

            if (taken is not null)
            {
                fields.Add(new(
                    FormVocabulary.Absent(parameter.Key, name), $"{name} may be absent", PropertyFieldType.Toggle, taken.Optional,
                    parameter.Optional.Description));
            }
        }

        return fields;
    }

    private string Written(string key) => step.GetProperty(key).GetRawText();

    private string Written(string key, string inner) => step.GetProperty(key).GetProperty(inner).GetRawText();

    // A value the step leaves out shows as an empty field.
    private static PropertyField Text(StepParameter parameter, string? value) =>
        new(parameter.Key, parameter.Key, PropertyFieldType.Text, value, parameter.Description);

    // A pick among columns: the current one is always among them, so the field shows what the step holds.
    private static PropertyField Pick(string name, string label, string description, string current, IReadOnlyList<string> columns) =>
        Choose(name, label, description, current, columns.Contains(current, StringComparer.Ordinal) ? columns : [.. columns, current]);

    private static PropertyField Choose(string name, string label, string description, string? current, IReadOnlyList<string> choices) =>
        new(name, label, PropertyFieldType.Select, current, description, Options(choices));

    private static IReadOnlyList<PropertyFieldOption> Options(IEnumerable<string> choices) =>
        [.. choices.Select(choice => new PropertyFieldOption(choice, choice))];

    /// <summary>One column a schema takes, as the step wrote it.</summary>
    /// <param name="Kind">The kind, as written.</param>
    /// <param name="Optional">Whether it may be absent.</param>
    private sealed record Taken(string Kind, bool Optional);
}
