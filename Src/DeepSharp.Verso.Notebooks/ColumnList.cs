// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// Every column of the source as one row — its name, its first values, whether it is in, and its kind — so nothing a
/// person leaves out ever disappears from sight.
/// </summary>
/// <remarks>
/// The rows are the source's columns in its order, then every column the schema names that the source lacks. A row's
/// box is the grid's box, asked of the same column rules; it carries its column, the kind the row shows — the one the
/// schema declares, else the one the saved file gives it, else text as the source holds it — and what the list was drawn
/// from: the key of the blocks and the fingerprint of the source's bytes. Everything that comes from a person or a file
/// is encoded before it reaches the page.
/// </remarks>
internal static class ColumnList
{
    /// <summary>How many of a column's first values a row shows.</summary>
    public const int ValuesShown = 10;

    private const string Style =
        "<style>.deepsharp-list{margin-top:.4em}"
        + ".deepsharp-list table{border-collapse:collapse}"
        + ".deepsharp-list th,.deepsharp-list td{padding:1px 6px;border:1px solid rgba(128,128,128,.25);white-space:nowrap}"
        + ".deepsharp-list td.deepsharp-values{opacity:.8;max-width:32em;overflow:hidden;text-overflow:ellipsis}</style>";

    // Written for a value the source left blank, so a gap reads as one.
    private const string Blank = "∅";

    /// <summary>The list of a source's columns under a declaration.</summary>
    /// <param name="declaration">The declaration the blocks make.</param>
    /// <param name="source">The rows as the source reads them, and the fingerprint of their bytes.</param>
    /// <param name="fresh">The source's columns the saved file never showed, marked new.</param>
    /// <param name="stored">What the file beside the notebook holds, when it holds anything.</param>
    /// <param name="drawn">The key of the whole declaration the list is drawn for.</param>
    /// <returns>The list.</returns>
    public static CellOutput Of(
        PipelineDeclaration declaration, SourceRows source, IReadOnlyList<string> fresh, PipelinePreset? stored, string drawn)
    {
        var header = source.Rows.ColumnNames;
        var declared = declaration.Steps.OfType<DeclareStep>().FirstOrDefault()?.Columns ?? [];
        IReadOnlyList<string> columns = [.. header.Concat(declared.Select(column => column.Name).Where(name => !header.Contains(name, StringComparer.Ordinal)))];
        var first = source.Rows.Rows.Take(ValuesShown).ToArray();
        var html = new StringBuilder(Style).Append("<div class=\"deepsharp-list\">")
            .Append("<div class=\"deepsharp-head\">The source's columns, each as the pipeline takes it:</div>")
            .Append("<table><thead><tr><th>column</th><th>first values</th><th>in</th><th>kind</th><th></th></tr></thead><tbody>");

        foreach (var choice in declaration.ChoicesFor(columns).Rows)
        {
            var at = IndexIn(header, choice.Name);
            var kind = choice.Kind ?? StoredKind(stored, choice.Name) ?? ColumnKind.Text;
            var mark = at < 0 ? "not in the source" : fresh.Contains(choice.Name, StringComparer.Ordinal) ? "new" : string.Empty;

            html.Append("<tr data-column=\"").Append(Encoded(choice.Name)).Append("\">")
                .Append("<td class=\"deepsharp-name\">").Append(Encoded(choice.Name)).Append("</td>")
                .Append("<td class=\"deepsharp-values\">").Append(Encoded(at < 0 ? string.Empty : Values(first, at))).Append("</td>")
                .Append("<td class=\"deepsharp-in\">");
            Box(html, choice, kind, drawn, source.Fingerprint);
            html.Append("</td><td class=\"deepsharp-kind\">");
            Select(html, declaration, choice, declared.FirstOrDefault(column => column.Name == choice.Name)?.Kind, header, drawn, source.Fingerprint);
            html.Append("</td><td class=\"deepsharp-mark\">").Append(Encoded(mark)).Append("</td></tr>");
        }

        return CellOutput.Html(html.Append("</tbody></table></div>").ToString());
    }

    // A row's kind select: the kind the schema declares, or none for a column it does not name, then every kind the rules
    // let the column take. It carries its column and what the list was drawn from, and the router sends its value.
    private static void Select(
        StringBuilder html, PipelineDeclaration declaration, ColumnChoice choice, ColumnKind? declared, IReadOnlyList<string> header, string drawn, string fingerprint)
    {
        var kinds = declaration.KindsFor(choice.Name, header);
        var action = ControlAction.Of(StepRenderer.ListKind, new JsonObject
        {
            [StepRenderer.ColumnKey] = choice.Name,
            [StepRenderer.DrawnKey] = drawn,
            [StepRenderer.SourceKey] = fingerprint,
        });

        html.Append("<select data-action=\"").Append(Encoded(action))
            .Append("\" data-extension-id=\"").Append(StepRenderer.Id).Append('"')
            .Append(kinds.Count == 0 ? " disabled" : string.Empty).Append('>');

        // A column the schema does not name picks no kind yet: sent again, as a click or a key sends it, it changes nothing.
        if (declared is null)
        {
            html.Append("<option value=\"\" selected>")
                .Append(choice.Standing == ColumnStanding.Kept ? "kept as text" : "not taken").Append("</option>");
        }

        foreach (var kind in kinds)
        {
            html.Append("<option value=\"").Append(Word(kind)).Append('"')
                .Append(kind == declared ? " selected" : string.Empty).Append('>').Append(Word(kind)).Append("</option>");
        }

        html.Append("</select>");
    }

    // A row's box: the gesture, the column, the kind a tick takes it in with, and what the list was drawn from.
    private static void Box(StringBuilder html, ColumnChoice choice, ColumnKind kind, string drawn, string fingerprint)
    {
        var box = choice.IncludedBox();
        var action = ControlAction.Of(StepRenderer.ListInclude, new JsonObject
        {
            [StepRenderer.ColumnKey] = choice.Name,
            [StepRenderer.KindKey] = Word(kind),
            [StepRenderer.DrawnKey] = drawn,
            [StepRenderer.SourceKey] = fingerprint,
        });

        html.Append("<label><input type=\"checkbox\" data-action=\"").Append(Encoded(action))
            .Append("\" data-extension-id=\"").Append(StepRenderer.Id).Append('"')
            .Append(box.Ticked ? " checked" : string.Empty)
            .Append(box.Enabled ? string.Empty : " disabled")
            .Append("> in</label>");
    }

    // The kind the saved file gives a column its schema names — an excluded one too — when it names it.
    private static ColumnKind? StoredKind(PipelinePreset? stored, string column) =>
        stored?.Declare.Columns.FirstOrDefault(declared => declared.Name == column)?.Kind;

    private static string Values(IReadOnlyList<IReadOnlyList<string?>> rows, int at) =>
        string.Join(", ", rows.Select(row => row[at] is { Length: > 0 } value ? value : Blank));

    private static int IndexIn(IReadOnlyList<string> header, string column)
    {
        for (var at = 0; at < header.Count; at++)
        {
            if (header[at] == column)
            {
                return at;
            }
        }

        return -1;
    }

    private static string Word(ColumnKind kind) => kind.ToString().ToLowerInvariant();

    private static string Encoded(string text) => WebUtility.HtmlEncode(text);
}
