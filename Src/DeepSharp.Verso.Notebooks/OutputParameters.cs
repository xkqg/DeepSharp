// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>One of the output's own values beside its answer, as the list offers it.</summary>
/// <param name="Key">The value's key, as a file writes it.</param>
/// <param name="Current">The value the output holds, as a file writes it; empty when it is not said.</param>
/// <param name="Options">
/// Every value the rules keep, the one held among them, an empty one meaning not said; nothing when the list cannot
/// offer them.
/// </param>
internal readonly record struct OutputParameter(string Key, string Current, IReadOnlyList<string>? Options);

/// <summary>
/// The output's own values beside its answer — how many rows ahead, whether a return, how many ones a row holds, what
/// the shares are shares of — offered as the values the rules keep, and set as the output block's form sets them.
/// </summary>
/// <remarks>
/// One rule reads a value into the output: the form's, so the list and the form never differ in what a value means. The
/// values offered are the form's own choices — a column of a kind the value takes, a word from its set — or, for a whole
/// number, every one from the least it may be up to the last the rules keep. Each is tried on the output, placed as the
/// column rules place one, and offered only when the pipeline keeps every rule. A value a file may leave out is offered
/// as not said. A value neither the form's choices nor the rules bound is set in the output block's form.
/// </remarks>
internal static class OutputParameters
{
    // How far a whole number is counted before no rule is taken to bound it.
    private const int Counted = 1000;

    /// <summary>The standing output's own values, each with what the list offers for it.</summary>
    /// <param name="catalog">The verbs the notebook knows.</param>
    /// <param name="declaration">The declaration the blocks make.</param>
    /// <param name="output">Its output.</param>
    /// <param name="header">The source's columns, when they are known.</param>
    /// <returns>Every value the output's kind takes but its answer, in the order the kind writes them.</returns>
    public static IReadOnlyList<OutputParameter> Of(StepCatalog catalog, PipelineDeclaration declaration, INamesTheAnswer output, IReadOnlyList<string>? header)
    {
        var scope = ScopeOf(declaration, output, header);
        using var written = JsonDocument.Parse(output.AsBlockText());
        var fields = new FormFields(written.RootElement, scope);
        var answer = OutputBox.AnswerOf(catalog, output.Verb);

        return
        [
            .. catalog.Describe(output.Verb).Parameters.Where(parameter => parameter != answer).Select(parameter =>
            {
                var current = written.RootElement.TryGetProperty(parameter.Key, out var value)
                    ? value.ValueKind == JsonValueKind.String ? value.GetString()! : value.GetRawText()
                    : string.Empty;

                bool Keeps(string text) => Kept(catalog, declaration, output, parameter.Key, text, scope);

                IReadOnlyList<string>? said = parameter is WholeNumberParameter whole ? Whole(whole, Keeps)
                    : parameter.Accept(fields).ToArray() is [{ FieldType: PropertyFieldType.Select, Options: { } options }]
                        ? [.. options.Select(option => option.Value).Where(Keeps)]
                        : null;

                return new OutputParameter(
                    parameter.Key, current,
                    said is null
                        ? null
                        : [.. (parameter.RequiredKeys.Count == 0 ? [string.Empty] : Array.Empty<string>()).Concat(said).Append(current).Distinct(StringComparer.Ordinal)]);
            }),
        ];
    }

    /// <summary>What a value picked for one of the output's own values asks for.</summary>
    /// <param name="catalog">The verbs the notebook knows.</param>
    /// <param name="drawn">The declaration the pick is made from.</param>
    /// <param name="key">The value's key.</param>
    /// <param name="text">The value, as the select sends it; empty for not said.</param>
    /// <param name="header">The source's columns, when they are known.</param>
    /// <returns>
    /// The steps with the output holding that value, placed as the column rules place one; why not, in the form's words,
    /// for a value the output cannot hold; nothing for a key that is none of the output's values beside its answer.
    /// </returns>
    public static ListOutputChange Picked(StepCatalog catalog, PipelineDeclaration drawn, string key, string? text, IReadOnlyList<string>? header)
    {
        if (drawn.Output is not { } output
            || !catalog.Describe(output.Verb).Parameters.Any(parameter => parameter.Key == key && parameter != OutputBox.AnswerOf(catalog, output.Verb)))
        {
            return new ListOutputChange(null, []);
        }

        try
        {
            var edited = (INamesTheAnswer)StepForm.Edited(output, key, FieldValue.Of(text), ScopeOf(drawn, output, header), catalog)!;

            return new ListOutputChange(drawn.WithOutput(edited), []);
        }
        catch (FormatException refused)
        {
            return new ListOutputChange(null, [refused.Message]);
        }
    }

    // Every whole number from the least it may be — the one that leaves it out is not said, so it is not counted — up to
    // the last one the rules keep; nothing when no rule stops it.
    private static IReadOnlyList<string>? Whole(WholeNumberParameter parameter, Func<string, bool> keeps)
    {
        List<string> kept = [];

        for (var value = Math.Max(parameter.AtLeast ?? 0, (parameter.LeftOut ?? -1) + 1); value < Counted; value++)
        {
            var text = value.ToString(CultureInfo.InvariantCulture);

            if (!keeps(text))
            {
                return kept;
            }

            kept.Add(text);
        }

        return null;
    }

    // Whether a value, read into the output as the form reads it, leaves a pipeline that keeps every rule.
    private static bool Kept(StepCatalog catalog, PipelineDeclaration declaration, INamesTheAnswer output, string key, string text, FormScope scope)
    {
        try
        {
            var edited = (INamesTheAnswer)StepForm.Edited(output, key, FieldValue.Of(text), scope, catalog)!;

            return PipelineDeclaration.FaultsIn(declaration.WithOutput(edited)).Count == 0;
        }
        catch (FormatException)
        {
            return false;
        }
    }

    // What the form knows around the output: the columns before it, and the source's.
    private static FormScope ScopeOf(PipelineDeclaration declaration, INamesTheAnswer output, IReadOnlyList<string>? header) =>
        new(declaration.ColumnsBefore(declaration.Steps.ToList().IndexOf(output)).Columns, header);
}
