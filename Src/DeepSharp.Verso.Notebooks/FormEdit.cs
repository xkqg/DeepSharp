// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using DeepSharp.Pipelines;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// One change made in the form, written into the step's JSON by the kind of the parameter the field belongs to.
/// </summary>
/// <param name="field">The field that changed.</param>
/// <param name="value">What it was set to.</param>
/// <param name="step">The step's JSON, changed in place.</param>
/// <param name="scope">The columns known around the block.</param>
/// <param name="read">The step as it reads, before the change.</param>
/// <remarks>
/// Each kind says whether the field is one of its own and writes the value the way a pipeline file holds it; the
/// step is then read back through the catalog, which holds it to every rule the text is held to. A value that
/// cannot be one of the kind — a word where a number goes, a number with a decimal comma — is refused here, in the
/// words the form shows, and nothing is written.
/// </remarks>
internal sealed class FormEdit(string field, FieldValue value, JsonObject step, FormScope scope, IPipelineStep read) : IStepParameterVisitor<bool>
{
    public bool Visit(TextParameter parameter) => Set(parameter.Key, () => Words());

    public bool Visit(FilePathParameter parameter) => Set(parameter.Key, () => Words());

    public bool Visit(ColumnParameter parameter) =>
        Set(parameter.Key, () => parameter.Optional && string.IsNullOrWhiteSpace(value.Text) ? null : Words());

    public bool Visit(NewColumnParameter parameter) =>
        Set(parameter.Key, () => parameter.Optional && string.IsNullOrWhiteSpace(value.Text) ? null : Words());

    public bool Visit(ColumnsParameter parameter)
    {
        if (field == parameter.Key)
        {
            return Written(parameter.Key, parameter.Optional, List());
        }

        List<string> names = [.. step[parameter.Key]?.AsArray().Select(item => item!.GetValue<string>()) ?? []];

        if (parameter.Repeatable)
        {
            if (!FormVocabulary.IsPlace(field, parameter.Key, out var place) || place >= names.Count)
            {
                return false;
            }

            names[place] = Words();
        }
        else if (!FormVocabulary.IsMember(field, parameter.Key, out var column))
        {
            return false;
        }
        else if (!Switch())
        {
            names.Remove(column);
        }
        else if (!names.Contains(column, StringComparer.Ordinal))
        {
            names.Add(column);
        }

        return Written(parameter.Key, parameter.Optional, names);
    }

    public bool Visit(NumberParameter parameter) => Set(parameter.Key, Number);

    // A whole number a file may leave out is left out when its field is emptied.
    public bool Visit(WholeNumberParameter parameter) =>
        Set(parameter.Key, () => parameter.LeftOut is not null && string.IsNullOrWhiteSpace(value.Text) ? null : Number());

    public bool Visit(TrueOrFalseParameter parameter) => Set(parameter.Key, () => Switch());

    public bool Visit(ShareParameter parameter) =>
        Set(parameter.Key, () => string.IsNullOrWhiteSpace(value.Text) ? null : Number());

    public bool Visit<TEnum>(OneOfParameter<TEnum> parameter)
        where TEnum : struct, Enum => Set(parameter.Key, () => Words());

    public bool Visit<TEnum>(SeveralOfParameter<TEnum> parameter)
        where TEnum : struct, Enum => Set(parameter.Key, () => new JsonArray([.. value.Picked.Select(item => (JsonNode?)item)]));

    // A way of filling that takes a number is written with one, starting at nought; its number is set on its own.
    public bool Visit(FillStrategyParameter parameter)
    {
        if (field == FormVocabulary.StrategyValue(parameter.Key))
        {
            if (step[parameter.Key] is not JsonObject strategy)
            {
                return false;
            }

            strategy[FillStrategyParameter.ValueKey] = Number();

            return true;
        }

        return Set(parameter.Key, () =>
        {
            var name = Words();
            var number = (step[parameter.Key] as JsonObject)?[FillStrategyParameter.ValueKey]?.DeepClone() ?? 0;

            return With.TakesAValue(name)
                ? (JsonNode)new JsonObject { [FillStrategyParameter.KindKey] = name, [FillStrategyParameter.ValueKey] = number }
                : name;
        });
    }

    // The share set is taken as written, and test is what the others leave: worked out in decimals, so the four
    // make a whole exactly as a person reads them.
    public bool Visit(SplitSharesParameter parameter)
    {
        if (field == SplitSharesParameter.TestKey || !parameter.Keys.Contains(field, StringComparer.Ordinal))
        {
            return false;
        }

        step[field] = Share();

        // A split writes all four shares, predict too, so each is there to add up.
        var others = parameter.Keys.Where(key => key != SplitSharesParameter.TestKey).Sum(key => step[key]!.GetValue<decimal>());

        step[SplitSharesParameter.TestKey] = 1m - others;

        return true;
    }

    // A column's kind, or "not taken", is the schema's own operation on the step as it reads, and the columns are
    // written back as the schema writes itself: a column not taken stays in it, excluded with its kind, and a step
    // below that reads it says so at its own block. Whether a taken column may be absent is written into it.
    public bool Visit(ColumnDeclarationsParameter parameter)
    {
        if (read is not DeclareStep declare)
        {
            return false;
        }

        if (FormVocabulary.IsKind(field, parameter.Key, out var name))
        {
            var kind = Words();

            Declared(parameter, kind == FormVocabulary.NotTaken
                ? () => declare.WithColumnExcluded(name)
                : () => Kinded(declare, name, KindOf(parameter, kind)));

            return true;
        }

        if (FormVocabulary.IsAbsent(field, parameter.Key, out var absent))
        {
            if (declare.Taking.All(column => column.Name != absent))
            {
                throw new FormatException($"'{absent}' is not taken, so whether it may be absent says nothing.");
            }

            var taken = step[parameter.Key]!.AsArray().OfType<JsonObject>().First(column => Named(parameter, column) == absent);

            taken[parameter.Optional.Key] = Switch();

            return true;
        }

        return false;
    }

    // The schema as one of its own operations changes it, written back into the step as the schema writes itself; what
    // the schema refuses is said in the form, and nothing is written.
    private void Declared(ColumnDeclarationsParameter parameter, Func<DeclareStep> change)
    {
        DeclareStep changed;

        try
        {
            changed = change();
        }
        catch (ArgumentException refused)
        {
            throw new FormatException(refused.Message, refused);
        }

        step[parameter.Key] = JsonNode.Parse(changed.AsBlockText())![parameter.Key]!.DeepClone();
    }

    // A kind picked for a column: given to it when the schema takes it; otherwise the column is taken in with that kind,
    // back where the schema names it or where the source has it.
    private DeclareStep Kinded(DeclareStep declare, string name, ColumnKind kind) =>
        (declare.Taking.Any(column => column.Name == name) ? declare : declare.WithColumn(name, kind, scope.Source ?? [])).WithColumnKind(name, kind);

    // A kind's word read by the kind's own reader, so a word it does not know is refused in its words.
    private static ColumnKind KindOf(ColumnDeclarationsParameter parameter, string word)
    {
        using var written = JsonDocument.Parse(new JsonObject { [parameter.Kind.Key] = word }.ToJsonString());

        return parameter.Kind.Read(written.RootElement);
    }

    private static string Named(ColumnDeclarationsParameter parameter, JsonObject column) =>
        column[parameter.Name.Key]!.GetValue<string>();

    // Sets a key when the field is the key's own; a value of nothing leaves the key out.
    private bool Set(string key, Func<JsonNode?> written)
    {
        if (field != key)
        {
            return false;
        }

        if (written() is { } node)
        {
            step[key] = node;
        }
        else
        {
            step.Remove(key);
        }

        return true;
    }

    private bool Written(string key, bool optional, IReadOnlyList<string> names)
    {
        if (optional && names.Count == 0)
        {
            step.Remove(key);
        }
        else
        {
            step[key] = new JsonArray([.. names.Select(name => (JsonNode?)name)]);
        }

        return true;
    }

    private string Words() => value.Text ?? throw new FormatException($"'{field}' needs a value.");

    private bool Switch() => value.Flag ?? throw new FormatException($"'{value.Text}' is neither true nor false.");

    // A list written as JSON, so a name may hold a comma.
    private List<string> List()
    {
        try
        {
            return JsonSerializer.Deserialize<List<string>>(Words()) ?? [];
        }
        catch (JsonException)
        {
            throw new FormatException($"'{value.Text}' is not a list of names written as JSON, as [\"a\", \"b\"].");
        }
    }

    // A number as a pipeline file writes one: the invariant spelling, with a full stop. A decimal comma is not
    // guessed at, since 1,5 could as well be fifteen; the form says how to write it instead.
    private JsonNode Number()
    {
        var text = Spelled();

        if (decimal.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var exact))
        {
            return exact;
        }

        return double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var number) && double.IsFinite(number)
            ? number
            : throw new FormatException($"'{text}' is not a number.");
    }

    // A share is at most one, so it is always a decimal: summed as one, the shares make a whole exactly.
    private decimal Share()
    {
        var text = Spelled();

        return decimal.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var share)
            ? share
            : throw new FormatException($"'{text}' is not a share: a number from nought to one.");
    }

    private string Spelled()
    {
        var text = Words().Trim();

        return text.Contains(',', StringComparison.Ordinal)
            ? throw new FormatException($"'{text}' is written with a comma; a pipeline writes a number with a full stop, as {text.Replace(',', '.')}.")
            : text;
    }
}
