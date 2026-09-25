// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// The page a person reads to write a pipeline by hand: every verb, what it does, and what it takes.
/// </summary>
/// <remarks>
/// Written from the descriptions of the verbs rather than typed beside them, so it says exactly what the
/// reader reads. Anything hand-maintained drifts from the code it describes, and a reference that drifts is
/// worse than none.
/// </remarks>
internal static class VerbReference
{
    /// <summary>Writes the page for these verbs.</summary>
    /// <param name="verbs">What each verb is.</param>
    /// <returns>The page, in Markdown, ending in a line break.</returns>
    public static string Write(IReadOnlyList<StepDescription> verbs)
    {
        var page = new StringBuilder();

        page.Append("# The verbs\n\n")
            .Append("Every step a pipeline file may hold: what it does, and what it takes. This page is written by the\n")
            .Append("steps themselves, so it is exactly what the library reads, and a step that changes changes it too.\n\n")
            .Append("A step is one JSON object in the file's `declaration`, named by its `step`, with its parameters\n")
            .Append("beside it. A key a step does not take is refused, and so is a word it does not know.\n\n")
            .Append("| verb | what it does |\n")
            .Append("|---|---|\n");

        foreach (var verb in verbs)
        {
            page.Append(CultureInfo.InvariantCulture, $"| [`{verb.Verb}`](#{Anchor(verb.Verb)}) | {verb.Purpose} |\n");
        }

        foreach (var verb in verbs)
        {
            page.Append(CultureInfo.InvariantCulture, $"\n## `{verb.Verb}`\n\n{verb.Purpose}\n\n");

            using var template = JsonDocument.Parse(verb.Template);

            page.Append("```json\n").Append(verb.Template).Append("\n```\n\n");

            if (verb.Parameters.Count > 0)
            {
                page.Append("| key | holds | a new block starts with |\n")
                    .Append("|---|---|---|\n");

                foreach (var row in verb.Parameters.SelectMany(parameter => parameter.Accept(new WhatAParameterHolds())))
                {
                    var startsWith = template.RootElement.TryGetProperty(row.Key, out var value)
                        ? $"`{value.GetRawText()}`"
                        : "left out";

                    page.Append(CultureInfo.InvariantCulture, $"| `{row.Key}` | {row.Holds} | {startsWith} |\n");
                }

                page.Append('\n');

                foreach (var parameter in verb.Parameters)
                {
                    page.Append(CultureInfo.InvariantCulture, $"- **`{parameter.Key}`**: {parameter.Description}\n");
                }

                page.Append('\n');
            }

            page.Append(CultureInfo.InvariantCulture, $"Means what it says from version {verb.Since} of the file.\n");
        }

        return page.ToString();
    }

    // The anchor a Markdown renderer gives a heading: lower case, with everything but letters, digits,
    // hyphens and underscores taken out.
    private static string Anchor(string verb) =>
        new([.. verb.ToLowerInvariant().Where(character => char.IsLetterOrDigit(character) || character is '-' or '_')]);

    /// <summary>One key of a step and what it holds, in words.</summary>
    /// <param name="Key">The key.</param>
    /// <param name="Holds">What a file may write under it.</param>
    private readonly record struct Row(string Key, string Holds);

    /// <summary>What each kind holds, in the words of the page.</summary>
    private sealed class WhatAParameterHolds : IStepParameterVisitor<IReadOnlyList<Row>>
    {
        // What a column of each kind holds, in the words of the page.
        private static readonly Dictionary<ColumnKind, string> KindWords = new()
        {
            [ColumnKind.Text] = "words",
            [ColumnKind.Number] = "a number",
            [ColumnKind.Integer] = "a whole number",
            [ColumnKind.Boolean] = "true or false",
            [ColumnKind.Timestamp] = "a moment in time",
            [ColumnKind.Category] = "a category",
        };

        public IReadOnlyList<Row> Visit(TextParameter parameter) => [new(parameter.Key, "words")];

        public IReadOnlyList<Row> Visit(FilePathParameter parameter) =>
            [new(parameter.Key, "the path of a file; a relative one is read from the pipeline's folder")];

        public IReadOnlyList<Row> Visit(ColumnParameter parameter) =>
            [new(parameter.Key, $"the name of a column holding {Kinds(parameter.Accepts)}{(parameter.Optional ? "; may be left out" : string.Empty)}")];

        public IReadOnlyList<Row> Visit(NewColumnParameter parameter) =>
            [new(parameter.Key, parameter.Optional
                ? "the name of the column it makes; left out, the step decides"
                : "the name of the column it makes")];

        public IReadOnlyList<Row> Visit(ColumnsParameter parameter) =>
            [new(parameter.Key, $"a list of the names of columns holding {Kinds(parameter.Accepts)}{(parameter.Repeatable ? string.Empty : ", each named once")}{(parameter.Optional ? "; may be left out" : string.Empty)}")];

        public IReadOnlyList<Row> Visit(NumberParameter parameter) =>
            [new(parameter.Key, parameter.Above is { } above
                ? string.Create(CultureInfo.InvariantCulture, $"a number above {above}")
                : "a number")];

        public IReadOnlyList<Row> Visit(WholeNumberParameter parameter) =>
            [new(parameter.Key, string.Concat(
                "a whole number",
                parameter.AtLeast is { } least ? string.Create(CultureInfo.InvariantCulture, $", at least {least}") : string.Empty,
                parameter.LeftOut is { } left ? string.Create(CultureInfo.InvariantCulture, $"; left out, {left}") : string.Empty))];

        public IReadOnlyList<Row> Visit(TrueOrFalseParameter parameter) => [new(parameter.Key, "`true` or `false`")];

        public IReadOnlyList<Row> Visit(ShareParameter parameter) =>
            [new(parameter.Key, "a share, from nought to one; left out, there is none")];

        public IReadOnlyList<Row> Visit<TEnum>(OneOfParameter<TEnum> parameter)
            where TEnum : struct, Enum =>
            [new(parameter.Key, $"one of {Choices(parameter.Choices)}")];

        public IReadOnlyList<Row> Visit<TEnum>(SeveralOfParameter<TEnum> parameter)
            where TEnum : struct, Enum =>
            [new(parameter.Key, $"a list of one or more of {Choices(parameter.Choices)}")];

        public IReadOnlyList<Row> Visit(FillStrategyParameter parameter)
        {
            var named = parameter.Allowed.Where(name => !With.TakesAValue(name)).Select(name => $"`\"{name}\"`");
            var numbered = parameter.Allowed.Where(With.TakesAValue).Select(name => $"`{{\"kind\": \"{name}\", \"value\": a number}}`");

            return [new(parameter.Key, $"one of {Joined([.. named, .. numbered])}")];
        }

        public IReadOnlyList<Row> Visit(SplitSharesParameter parameter) =>
        [
            new("train", "the share the model learns from: above nought, at most one"),
            new("validation", "the share used while choosing between models: nought to one"),
            new("test", "the share kept back until the end: above nought, at most one"),
            new("predict", "the share held back to predict on: nought to one, and none when it is left out"),
        ];

        public IReadOnlyList<Row> Visit(ColumnDeclarationsParameter parameter) =>
        [
            new(parameter.Key, $"a list of columns, each with a `name`, a `kind` that is one of {Choices(parameter.Kind.Choices)}, "
                               + "and whether it is `optional`"),
        ];

        private static string Choices(IReadOnlyList<string> words) => Joined([.. words.Select(word => $"`{word}`")]);

        private static string Kinds(IReadOnlyList<ColumnKind> kinds) =>
            kinds.Count == KindWords.Count ? "anything" : Joined([.. kinds.Select(kind => KindWords[kind])]);

        private static string Joined(IReadOnlyList<string> words) =>
            words.Count == 1 ? words[0] : $"{string.Join(", ", words.Take(words.Count - 1))} or {words[^1]}";
    }
}
