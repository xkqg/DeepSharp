// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Net;
using System.Text;
using DeepSharp.Pipelines;
using MatPlotLibNet;
using MatPlotLibNet.Numerics;
using MatPlotLibNet.Rendering.TickFormatters;
using MatPlotLibNet.Rendering.TickLocators;
using MatPlotLibNet.Styling.ColorMaps;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// Evidence a block declared, drawn under its grid.
/// </summary>
/// <remarks>
/// The core measures and this only draws: a profile as a table with its alerts and the rows that are there more
/// than once, a correlation as a heatmap or as its coefficients, with how many rows it was drawn from, out of how
/// many, and by which rule — a correlation needs a value in every column of a row, so the rows with a gap in any
/// of them are left out, and how many is part of what it shows. The correlation itself is MatPlotLibNet's, as is
/// the drawing. Every number is written in the invariant culture.
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
                .Append(" — answered by <code>").Append(Encoded(alert.Verb)).Append("</code></li>");
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

        var matrix = NpStats.Corrcoef([.. Enumerable.Range(0, names.Length).Select(column => correlation.Rows.Select(row => row[column]).ToArray())]);

        return CellOutput.Html((correlation.Shown == Shown.Numbers ? Numbers(html, names, matrix) : Drawn(html, names, matrix))
            .Append("</div>").ToString());
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

    // A heatmap of the coefficients from -1 to 1, blue to red, each cell labelled with its value.
    private static StringBuilder Drawn(StringBuilder html, string[] names, Mat matrix)
    {
        var data = new double[names.Length, names.Length];
        var positions = Enumerable.Range(0, names.Length).Select(position => (double)position).ToArray();

        for (var row = 0; row < names.Length; row++)
        {
            for (var column = 0; column < names.Length; column++)
            {
                data[row, column] = matrix[row, column];
            }
        }

        var size = 160 + (60 * names.Length);
        var svg = new FigureBuilder()
            .WithSize(size + 120, size)
            .AddSubPlot(1, 1, 1, axes => axes
                .Heatmap(data, series =>
                {
                    series.ColorMap = ColorMaps.Coolwarm;
                    series.Normalizer = FromMinusOneToOne.Instance;
                    series.ShowLabels = true;
                    series.LabelFormat = "0.00";
                })
                .SetXTickLocator(new FixedLocator(positions))
                .SetXTickFormatter(new CategoryFormatter(names))
                .SetYTickLocator(new FixedLocator(positions))
                .SetYTickFormatter(new CategoryFormatter(names, reversed: true))
                .WithColorBar())
            .ToSvg();

        return html.Append(svg);
    }

    private static string RowsWord(Standing over) => over == Standing.Train ? "training rows" : "rows";

    private static string Number(double? value) =>
        value is { } number ? number.ToString("G6", CultureInfo.InvariantCulture) : string.Empty;

    private static string Invariant(int number) => number.ToString(CultureInfo.InvariantCulture);

    private static string Encoded(string text) => WebUtility.HtmlEncode(text);
}
