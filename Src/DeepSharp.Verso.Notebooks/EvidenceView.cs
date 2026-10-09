// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using DeepSharp.Charts;
using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// Evidence a block declared, drawn under its grid.
/// </summary>
/// <remarks>
/// The core measures and this only draws: a profile as a table with its alerts and the rows that are there more
/// than once, a correlation as a heatmap or as its coefficients, with how many rows it was drawn from, out of how
/// many, and by which rule — a correlation needs a value in every column of a row, so the rows with a gap in any
/// of them are left out, and how many is part of what it shows. The coefficients are the pipeline's, worked out once
/// from those rows, so the picture, the table and the profile's alerts read the same numbers; a pair with no coefficient — a
/// column that never changes — is written as not available, never as nought. A
/// trained model's measures are shown as DeepSharp.Charts renders its pipeline's report — the one rendering, which a C#
/// cell that ends with the report shows too. Every drawing is DeepSharp.Charts', so a figure is drawn one way wherever it
/// is shown. Every number is written in the invariant culture.
/// </remarks>
internal sealed class EvidenceView : IEvidenceVisitor<CellOutput>
{
    private const string Style =
        "<style>.deepsharp-evidence{margin-top:.4em}"
        + ".deepsharp-evidence table{border-collapse:collapse;font-variant-numeric:tabular-nums}"
        + ".deepsharp-evidence th,.deepsharp-evidence td{padding:1px 6px;border:1px solid rgba(128,128,128,.25)}"
        + ".deepsharp-evidence td.deepsharp-number{text-align:right}"
        + ".deepsharp-alerts{margin:.3em 0;padding-left:1.2em}</style>";

    /// <inheritdoc />
    public CellOutput Visit(DataProfile profile)
    {
        ArgumentNullException.ThrowIfNull(profile);

        var html = new StringBuilder(Style).Append("<div class=\"deepsharp-evidence\">")
            .Append("<div class=\"deepsharp-summary\">Profile of the ").Append(Invariant(profile.Rows)).Append(' ')
            .Append(RowsWord(profile.Over)).Append("</div>")
            .Append("<table><thead><tr><th>column</th><th>kind</th><th>gaps</th><th>not finite</th><th>distinct</th>")
            .Append("<th>min</th><th>max</th><th>mean</th><th>median</th></tr></thead><tbody>");

        foreach (var column in profile.Columns)
        {
            html.Append("<tr><td>").Append(Encoded(column.Name)).Append("</td><td>")
                .Append(column.Kind.ToString().ToLowerInvariant()).Append(column.Constant ? ", constant" : string.Empty).Append("</td>");

            foreach (var count in new[] { column.Gaps, column.NotFinite, column.Distinct })
            {
                html.Append("<td class=\"deepsharp-number\">").Append(Invariant(count)).Append("</td>");
            }

            foreach (var value in new[] { column.Min, column.Max, column.Mean, column.Median })
            {
                html.Append("<td class=\"deepsharp-number\">").Append(Number(value)).Append("</td>");
            }

            html.Append("</tr>");
        }

        html.Append("</tbody></table><ul class=\"deepsharp-alerts\">");

        foreach (var alert in profile.Alerts)
        {
            html.Append("<li><code>").Append(Encoded(alert.Column)).Append("</code> ").Append(Encoded(alert.Says))
                .Append(" — ").Append(Answered(alert.Answer));
            Control(html, alert.Answer);
            html.Append("</li>");
        }

        var duplicates = profile.Duplicates;

        return CellOutput.Html(html.Append("</ul><div>")
            .Append(Invariant(duplicates.Groups)).Append(" groups of rows are there more than once: ")
            .Append(Invariant(duplicates.Rows)).Append(" rows, ").Append(Invariant(duplicates.ExtraCopies)).Append(" extra copies; ")
            .Append(Invariant(duplicates.GroupsAcrossParts)).Append(" of the groups have copies in more than one part.</div></div>")
            .ToString());
    }

    /// <inheritdoc />
    public CellOutput Visit(CorrelationInput correlation)
    {
        ArgumentNullException.ThrowIfNull(correlation);

        var html = new StringBuilder(Style).Append("<div class=\"deepsharp-evidence\"><div class=\"deepsharp-summary\">");
        var names = correlation.Columns.ToArray();

        html.Append(correlation.Coefficient == Coefficient.Spearman ? "Spearman correlation, on the ranks, of " : "Correlation of ")
            .Append(Encoded(string.Join(", ", names))).Append(" over ")
            .Append(Invariant(correlation.Kept)).Append(" of the ").Append(Invariant(correlation.Total)).Append(' ')
            .Append(RowsWord(correlation.Over)).Append(", kept by the rule '")
            .Append(Encoded(correlation.Policy)).Append("': a row with a gap, or a value that is not a finite number, in any of these columns is left out.</div>");

        if (correlation.Kept < 2)
        {
            return CellOutput.Html(html.Append("<div>That is too few rows to correlate anything.</div></div>").ToString());
        }

        var coefficients = correlation.Correlate().Of(correlation.Coefficient);

        if (correlation.Shown != Shown.Numbers)
        {
            return CellOutput.Html(Undefined(html.Append(correlation.Heatmap()), coefficients).Append("</div>").ToString());
        }

        return CellOutput.Html(Undefined(Numbers(html, coefficients), coefficients).Append("</div>").ToString());
    }

    /// <inheritdoc />
    /// <remarks>
    /// As DeepSharp.Charts renders the report (<c>measures.Report().ToHtml()</c>): the numbers, the charts, or both, as the
    /// report says they are shown — reached from a report's block once a C# cell has handed the predictions it measured back.
    /// </remarks>
    public CellOutput Visit(Measures measures)
    {
        ArgumentNullException.ThrowIfNull(measures);

        return CellOutput.Html(measures.Report().ToHtml());
    }

    private static StringBuilder Numbers(StringBuilder html, CorrelationCoefficients coefficients)
    {
        html.Append("<table><thead><tr><th></th>");

        foreach (var name in coefficients.Columns)
        {
            html.Append("<th>").Append(Encoded(name)).Append("</th>");
        }

        html.Append("</tr></thead><tbody>");

        for (var row = 0; row < coefficients.Size; row++)
        {
            html.Append("<tr><th>").Append(Encoded(coefficients.Columns[row])).Append("</th>");

            for (var column = 0; column < coefficients.Size; column++)
            {
                html.Append("<td class=\"deepsharp-number\">").Append(CoefficientText(coefficients[row, column])).Append("</td>");
            }

            html.Append("</tr>");
        }

        return html.Append("</tbody></table>");
    }

    // The columns that have no coefficient with any other, said under the picture or the table: their cells are blank or
    // not available, and a reader should not have to guess why.
    private static StringBuilder Undefined(StringBuilder html, CorrelationCoefficients coefficients)
    {
        foreach (var name in coefficients.Columns.Where((_, at) => double.IsNaN(coefficients[at, at])))
        {
            html.Append("<div><code>").Append(Encoded(name)).Append("</code> never changes over these rows, so it has no coefficient with any column.</div>");
        }

        return html;
    }

    private static string CoefficientText(double value) =>
        double.IsNaN(value) ? "n/a" : value.ToString("0.000", CultureInfo.InvariantCulture);

    // An alert whose answer changes the columns has a box that gives it, drawn unticked: it carries its column, what
    // answering does and the value it says, and no data-payload, so Verso's router sends whether it is ticked. One answered
    // by a step to be written has none, since where a step belongs is a person's to say.
    private static void Control(StringBuilder html, AlertAnswer answer)
    {
        if (answer.Action == AlertAction.Step)
        {
            return;
        }

        var carried = new JsonObject
        {
            [StepRenderer.ColumnKey] = answer.Column,
            [StepRenderer.AnswerKey] = answer.Action.ToString().ToLowerInvariant(),
        };

        if (answer.Value is { } value)
        {
            carried[StepRenderer.ValueKey] = value;
        }

        html.Append(" <label><input type=\"checkbox\" data-action=\"").Append(Encoded(ControlAction.Of(StepRenderer.Answer, carried)))
            .Append("\" data-extension-id=\"").Append(StepRenderer.Id).Append("\"> ")
            .Append(answer.Action == AlertAction.LeaveOut ? "leave it out" : "say so").Append("</label>");
    }

    // How an alert is answered, in the words the profile shows.
    private static string Answered(AlertAnswer answer) => answer.Action switch
    {
        AlertAction.LeaveOut => "answered by leaving it out",
        AlertAction.SayMissing => $"answered by saying in the schema that <code>{Encoded(answer.Value!)}</code> stands for a gap",
        _ => $"answered by <code>{Encoded(answer.Verb!)}</code>",
    };

    private static string RowsWord(Standing over) => over == Standing.Train ? "training rows" : "rows";

    private static string Number(double? value) =>
        value is { } number ? number.ToString("G6", CultureInfo.InvariantCulture) : string.Empty;

    private static string Invariant(int number) => number.ToString(CultureInfo.InvariantCulture);

    private static string Encoded(string text) => WebUtility.HtmlEncode(text);
}
