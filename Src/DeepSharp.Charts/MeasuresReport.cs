// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Net;
using System.Text;
using DeepSharp.Pipelines;

namespace DeepSharp.Charts;

/// <summary>
/// A trained model's measures as its pipeline's report says they are shown — the numbers, the charts, or both — as one
/// piece of HTML: the one rendering of a report, wherever it is shown.
/// </summary>
/// <remarks>
/// A notebook shows a value with a public <see cref="ToHtml"/> as the HTML it gives, whichever assembly the value's type was
/// loaded from, so a C# cell that ends with <c>trained.Measures!.Report()</c> shows the report; and a notebook's report
/// block, handed back the predictions such a cell measured, draws the same rendering under its grid. As numbers: a part to
/// a row, each measure beside predicting the training rows' average — and beside its rows, when the predictions said it, how
/// many of them the model learned nothing about — then each confusion matrix, a class held to a row and a class predicted to
/// a column, with the average's count in brackets. Drawn: every measure as bars beside the average, each confusion matrix as
/// a heatmap of counts, and — where an amount is measured — what was predicted against what was there, and what was left
/// over, each the chart <see cref="MeasureCharts"/> draws. A divergence without end is written as ∞ and drawn as no bar. Every
/// number is written in the invariant culture, every name a
/// pipeline gives is written as text, and the HTML carries its own look, so it stands on its own wherever it lands.
/// </remarks>
public sealed class MeasuresReport
{
    private const string Style =
        "<style>.deepsharp-evidence{margin-top:.4em}"
        + ".deepsharp-evidence table{border-collapse:collapse;font-variant-numeric:tabular-nums}"
        + ".deepsharp-evidence th,.deepsharp-evidence td{padding:1px 6px;border:1px solid rgba(128,128,128,.25)}"
        + ".deepsharp-evidence td.deepsharp-number{text-align:right}</style>";

    private readonly Measures _measures;

    internal MeasuresReport(Measures measures) => _measures = measures;

    /// <summary>The report as HTML: the numbers, the charts, or both, as the pipeline's report says they are shown.</summary>
    /// <returns>The HTML, whole: its look, the line that says what was measured, and what the report shows.</returns>
    public string ToHtml()
    {
        var html = new StringBuilder(Style).Append("<div class=\"deepsharp-evidence\"><div class=\"deepsharp-summary\">Measures of ")
            .Append(Encoded(string.Join(", ", _measures.Answers)))
            .Append(" in their own units, each beside predicting the training rows' average.</div>");

        if (_measures.Shown.Contains(Shown.Numbers))
        {
            Table(html, _measures);
        }

        if (_measures.Shown.Contains(Shown.Drawn))
        {
            Drawn(html, _measures);
        }

        return html.Append("</div>").ToString();
    }

    // The measures as numbers: a part to a row — beside its rows, how many of them the model said it learned nothing about,
    // when its predictions said — then every confusion matrix, the average's count in brackets.
    private static void Table(StringBuilder html, Measures measures)
    {
        var unfamiliar = measures.Parts.Any(part => part.UnfamiliarRows is not null);

        html.Append("<table><thead><tr><th>part</th><th>rows</th>").Append(unfamiliar ? "<th>unfamiliar</th>" : string.Empty);

        foreach (var metric in measures.Metrics.Where(metric => metric != Metric.ConfusionMatrix))
        {
            html.Append("<th>").Append(metric.Word()).Append("</th><th>baseline</th>");
        }

        html.Append("</tr></thead><tbody>");

        foreach (var part in measures.Parts)
        {
            html.Append("<tr><td>").Append(part.Part.Word()).Append("</td><td class=\"deepsharp-number\">").Append(Invariant(part.Rows)).Append("</td>");

            if (unfamiliar)
            {
                html.Append("<td class=\"deepsharp-number\">").Append(part.UnfamiliarRows is { } count ? Invariant(count) : string.Empty).Append("</td>");
            }

            foreach (var measured in part.Values)
            {
                html.Append("<td class=\"deepsharp-number\">").Append(Number(measured.Value)).Append("</td>")
                    .Append("<td class=\"deepsharp-number\">").Append(Number(measured.Baseline)).Append("</td>");
            }

            html.Append("</tr>");
        }

        html.Append("</tbody></table>");

        if (unfamiliar)
        {
            html.Append("<div>Unfamiliar counts the rows the model said it learned nothing about — for a network, a row that moves a feature ")
                .Append("away from the one value every training row held it at — so their answers were never learned.</div>");
        }

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

        if (measures.Metrics.Any(metric => metric.Family() == MetricFamily.Amounts))
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

    // A divergence without end is written as the infinity it is, in the sign a person reads.
    private static string Number(double value) => double.IsPositiveInfinity(value) ? "∞" : value.ToString("G6", CultureInfo.InvariantCulture);

    private static string Invariant(int number) => number.ToString(CultureInfo.InvariantCulture);

    private static string Encoded(string text) => WebUtility.HtmlEncode(text);
}
