// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using DeepSharp.Charts;
using DeepSharp.Pipelines;
using MatPlotLibNet.Numerics;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// Evidence a block declared, drawn under its grid.
/// </summary>
/// <remarks>
/// The core measures and this only draws: a profile as a table with its alerts and the rows that are there more
/// than once, a correlation as a heatmap or as its coefficients, with how many rows it was drawn from, out of how
/// many, and by which rule — a correlation needs a value in every column of a row, so the rows with a gap in any
/// of them are left out, and how many is part of what it shows. The correlation itself is MatPlotLibNet's. A
/// trained model's measures are written as their numbers, each beside the training rows' average, or drawn. Every
/// drawing is DeepSharp.Charts', so a figure is drawn one way wherever it is shown. Every number is written in the
/// invariant culture.
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

        html.Append("Correlation of ").Append(Encoded(string.Join(", ", names))).Append(" over ")
            .Append(Invariant(correlation.Kept)).Append(" of the ").Append(Invariant(correlation.Total)).Append(' ')
            .Append(RowsWord(correlation.Over)).Append(", kept by the rule '")
            .Append(Encoded(correlation.Policy)).Append("': a row with a gap, or a value that is not a finite number, in any of these columns is left out.</div>");

        if (correlation.Kept < 2)
        {
            return CellOutput.Html(html.Append("<div>That is too few rows to correlate anything.</div></div>").ToString());
        }

        if (correlation.Shown != Shown.Numbers)
        {
            return CellOutput.Html(html.Append(correlation.Heatmap()).Append("</div>").ToString());
        }

        var matrix = NpStats.Corrcoef([.. Enumerable.Range(0, names.Length).Select(column => correlation.Rows.Select(row => row[column]).ToArray())]);

        return CellOutput.Html(Numbers(html, names, matrix).Append("</div>").ToString());
    }

    /// <inheritdoc />
    /// <remarks>
    /// As the report says they are shown. As numbers: a part to a row, each measure beside predicting the training rows'
    /// average, then each confusion matrix, a class held to a row and a class predicted to a column, with the average's count
    /// in brackets. Drawn: every measure as bars beside the average, each confusion matrix as a heatmap of counts, and — where
    /// an amount is measured — what was predicted against what was there, and what was left over.
    /// </remarks>
    public CellOutput Visit(Measures measures)
    {
        ArgumentNullException.ThrowIfNull(measures);

        var html = new StringBuilder(Style).Append("<div class=\"deepsharp-evidence\"><div class=\"deepsharp-summary\">Measures of ")
            .Append(Encoded(string.Join(", ", measures.Answers)))
            .Append(" in their own units, each beside predicting the training rows' average.</div>");

        if (measures.Shown.Contains(Shown.Numbers))
        {
            Table(html, measures);
        }

        if (measures.Shown.Contains(Shown.Drawn))
        {
            Drawn(html, measures);
        }

        return CellOutput.Html(html.Append("</div>").ToString());
    }

    // The measures as numbers: a part to a row, then every confusion matrix, the average's count in brackets.
    private static void Table(StringBuilder html, Measures measures)
    {
        html.Append("<table><thead><tr><th>part</th><th>rows</th>");

        foreach (var metric in measures.Metrics.Where(metric => metric != Metric.ConfusionMatrix))
        {
            html.Append("<th>").Append(metric.Word()).Append("</th><th>baseline</th>");
        }

        html.Append("</tr></thead><tbody>");

        foreach (var part in measures.Parts)
        {
            html.Append("<tr><td>").Append(part.Part.Word()).Append("</td><td class=\"deepsharp-number\">").Append(Invariant(part.Rows)).Append("</td>");

            foreach (var measured in part.Values)
            {
                html.Append("<td class=\"deepsharp-number\">").Append(Number(measured.Value)).Append("</td>")
                    .Append("<td class=\"deepsharp-number\">").Append(Number(measured.Baseline)).Append("</td>");
            }

            html.Append("</tr>");
        }

        html.Append("</tbody></table>");

        if (measures.Parts[0].Confusions.Count > 0)
        {
            html.Append("<div>A confusion matrix counts the rows, a class held to a row and a class predicted to a column, the average's count in brackets.</div>");
        }

        foreach (var part in measures.Parts)
        {
            foreach (var confusion in part.Confusions)
            {
                Matrix(html, part.Part, confusion);
            }
        }
    }

    // The measures drawn: the bars of every number, the heatmap of every confusion matrix, and — where an amount is
    // measured — what was predicted against what was there, and what was left over.
    private static void Drawn(StringBuilder html, Measures measures)
    {
        if (measures.Parts[0].Values.Count > 0)
        {
            html.Append(measures.Bars());
        }

        if (measures.Parts[0].Confusions.Count > 0)
        {
            html.Append(measures.ConfusionMatrices());
        }

        if (measures.Metrics.Any(metric => metric is Metric.Rmse or Metric.Mae or Metric.R2))
        {
            html.Append(measures.PredictedAgainstActual()).Append(measures.Residuals());
        }
    }

    // One confusion matrix: a class held to a row, a class predicted to a column, each count beside the average's.
    private static void Matrix(StringBuilder html, Part part, Confusion confusion)
    {
        html.Append("<div>").Append(part.Word()).Append(confusion.Answer is { } answer ? $": {Encoded(answer)}" : string.Empty)
            .Append("</div><table><thead><tr><th>held \\ predicted</th>");

        foreach (var name in confusion.Classes)
        {
            html.Append("<th>").Append(Encoded(name)).Append("</th>");
        }

        html.Append("</tr></thead><tbody>");

        for (var held = 0; held < confusion.Classes.Count; held++)
        {
            html.Append("<tr><th>").Append(Encoded(confusion.Classes[held])).Append("</th>");

            for (var predicted = 0; predicted < confusion.Classes.Count; predicted++)
            {
                html.Append("<td class=\"deepsharp-number\">").Append(Invariant(confusion.Counts[held][predicted]))
                    .Append(" (").Append(Invariant(confusion.Baseline[held][predicted])).Append(")</td>");
            }

            html.Append("</tr>");
        }

        html.Append("</tbody></table>");
    }

    private static StringBuilder Numbers(StringBuilder html, string[] names, Mat matrix)
    {
        html.Append("<table><thead><tr><th></th>");

        foreach (var name in names)
        {
            html.Append("<th>").Append(Encoded(name)).Append("</th>");
        }

        html.Append("</tr></thead><tbody>");

        for (var row = 0; row < names.Length; row++)
        {
            html.Append("<tr><th>").Append(Encoded(names[row])).Append("</th>");

            for (var column = 0; column < names.Length; column++)
            {
                html.Append("<td class=\"deepsharp-number\">").Append(matrix[row, column].ToString("0.000", CultureInfo.InvariantCulture)).Append("</td>");
            }

            html.Append("</tr>");
        }

        return html.Append("</tbody></table>");
    }

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
