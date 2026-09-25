// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Net;
using System.Text;
using DeepSharp.Pipelines;
using MatPlotLibNet.Styling;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>What a grid's header offered: the columns it showed, and for each what could be done from it.</summary>
/// <param name="Columns">The columns, in the order the grid showed them.</param>
/// <param name="Offers">For each column in that order, whether it could be excluded and whether it could be marked a category.</param>
internal sealed record GridHeader(IReadOnlyList<string> Columns, string Offers)
{
    /// <summary>Whether a grid of these columns offers the same under the declaration as it is now.</summary>
    /// <param name="now">The declaration now.</param>
    /// <returns><see langword="true"/> while every button it drew still does what it says.</returns>
    public bool StillHolds(PipelineDeclaration now) => DataGrid.HeaderOf(now, Columns).Offers == Offers;
}

/// <summary>
/// The rows at a block, a page at a time, each number coloured by where it lies in its column.
/// </summary>
/// <remarks>
/// A column's range is learned the way a fit learns: over the rows the pipeline measures on — the training rows of
/// the split, wherever it is declared, or every row when nothing divides them — and over their finite values alone,
/// so one infinity cannot squash every other value to one end. Blue is the smallest training value and red the
/// largest; a value outside that range, as a validation row may well be, has a darker colour of its own, and a gap
/// has none. Everything is written in the invariant culture, so the page is the same whichever language the
/// notebook's interface speaks.
/// </remarks>
internal static class DataGrid
{
    /// <summary>How many rows a page shows.</summary>
    public const int PageSize = 50;

    private const string Style =
        "<style>.deepsharp-grid{margin-top:.4em}"
        + ".deepsharp-grid .deepsharp-scroll{overflow:auto;max-height:28em}"
        + ".deepsharp-grid table{border-collapse:collapse;font-variant-numeric:tabular-nums}"
        + ".deepsharp-grid th,.deepsharp-grid td{padding:1px 6px;border:1px solid rgba(128,128,128,.25);white-space:nowrap}"
        + ".deepsharp-grid th{position:sticky;top:0;background:inherit}"
        + ".deepsharp-grid td.deepsharp-number{text-align:right}</style>";

    /// <summary>The page of the rows at a block.</summary>
    /// <param name="view">The rows there, and where each stands.</param>
    /// <param name="page">Which page, counting from nought; one beyond the last shows the last.</param>
    /// <param name="declaration">The declaration the rows come from, whose schema says which columns may be marked a category.</param>
    /// <returns>The grid.</returns>
    /// <remarks>
    /// Every column that reaches the end of the declaration can be excluded from its header — one the schema leaves
    /// out, or a step below takes away, is excluded already — and every column the schema declares, and does not
    /// already read as a category, can be marked one. The gestures write the declaration, never a value.
    /// </remarks>
    public static CellOutput Of(PipelineView view, int page, PipelineDeclaration declaration)
    {
        var offers = Offered.By(declaration);
        var table = view.Table;
        var pages = Math.Max(1, (table.RowCount + PageSize - 1) / PageSize);
        var shown = Math.Clamp(page, 0, pages - 1);
        var first = shown * PageSize;
        var last = Math.Min(first + PageSize, table.RowCount);
        var columns = table.Columns.Select(column => Colouring.Of(view, column)).ToArray();
        var html = new StringBuilder(Style).Append("<div class=\"deepsharp-grid\">");

        Summary(html, view);

        html.Append("<div class=\"deepsharp-scroll\"><table><thead><tr><th>#</th><th>part</th>");

        foreach (var column in table.Columns)
        {
            html.Append("<th>").Append(Encoded(column.Name));

            if (offers.Excludes(column.Name))
            {
                Action(html, StepRenderer.Drop, column.Name, "exclude");
            }

            if (offers.Marks(column.Name))
            {
                Action(html, StepRenderer.Category, column.Name, "category");
            }

            html.Append("</th>");
        }

        html.Append("</tr></thead><tbody>");

        for (var row = first; row < last; row++)
        {
            html.Append("<tr><td>").Append(Invariant(row + 1)).Append("</td><td>")
                .Append(Word(view.Standings[row])).Append("</td>");

            foreach (var column in columns)
            {
                column.Cell(html, row);
            }

            html.Append("</tr>");
        }

        html.Append("</tbody></table></div>");
        Pager(html, shown, pages, first, last, table.RowCount);

        return CellOutput.Html(html.Append("</div>").ToString());
    }

    /// <summary>What a grid of these columns offers in its header under a declaration: the rule the grid draws by.</summary>
    /// <param name="declaration">The declaration.</param>
    /// <param name="columns">The columns the grid shows.</param>
    /// <returns>The header.</returns>
    public static GridHeader HeaderOf(PipelineDeclaration declaration, IReadOnlyList<string> columns)
    {
        var offers = Offered.By(declaration);

        return new GridHeader(
            columns,
            string.Join('|', columns.Select(column => $"{(offers.Excludes(column) ? 'x' : '-')}{(offers.Marks(column) ? 'c' : '-')}")));
    }

    private static void Summary(StringBuilder html, PipelineView view)
    {
        var counts = Enum.GetValues<Standing>()
            .Select(standing => new StandingCount(standing, view.Standings.Count(each => each == standing)))
            .Where(each => each.Rows > 0)
            .Select(each => $"{Invariant(each.Rows)} {Word(each.Standing)}");

        html.Append("<div class=\"deepsharp-summary\">").Append(Invariant(view.Table.RowCount)).Append(" rows")
            .Append(string.Concat(counts.Select(count => $" · {count}")))
            .Append(" — each number coloured by where it lies among the ")
            .Append(view.Measured == Standing.Train ? "training rows" : "rows")
            .Append(" of its column</div>");
    }

    private static void Pager(StringBuilder html, int shown, int pages, int first, int last, int rows)
    {
        if (rows == 0)
        {
            html.Append("<div class=\"deepsharp-pages\">no rows</div>");

            return;
        }

        html.Append("<div class=\"deepsharp-pages\">rows ").Append(Invariant(first + 1)).Append('–').Append(Invariant(last))
            .Append(" of ").Append(Invariant(rows));

        if (shown > 0)
        {
            Button(html, shown - 1, "previous");
        }

        if (shown < pages - 1)
        {
            Button(html, shown + 1, "next");
        }

        html.Append("</div>");
    }

    // Verso's router reads what a button carries from its data-payload; a value it reads only on a field.
    private static void Action(StringBuilder html, string gesture, string column, string label) =>
        html.Append(" <button type=\"button\" data-action=\"").Append(gesture)
            .Append("\" data-extension-id=\"").Append(StepRenderer.Id)
            .Append("\" data-payload=\"").Append(Encoded(column)).Append("\">").Append(label).Append("</button>");

    private static void Button(StringBuilder html, int page, string label) =>
        html.Append(" <button type=\"button\" data-action=\"").Append(StepRenderer.Page)
            .Append("\" data-extension-id=\"").Append(StepRenderer.Id)
            .Append("\" data-payload=\"").Append(Invariant(page)).Append("\">").Append(label).Append("</button>");

    private static string Word(Standing standing) => standing.ToString().ToLowerInvariant();

    private static string Invariant(int number) => number.ToString(CultureInfo.InvariantCulture);

    private static string Encoded(string text) => WebUtility.HtmlEncode(text);

    /// <summary>How many rows stand one way.</summary>
    private readonly record struct StandingCount(Standing Standing, int Rows);

    /// <summary>What a grid's header offers under a declaration, column by column.</summary>
    /// <param name="reaching">The columns there at the end of the declaration.</param>
    /// <param name="markable">The columns the schema takes, and not as categories.</param>
    private sealed class Offered(ColumnState reaching, HashSet<string> markable)
    {
        public static Offered By(PipelineDeclaration declaration) =>
            new(
                declaration.ColumnsBefore(declaration.Steps.Count),
                declaration.Steps.OfType<DeclareStep>()
                    .SelectMany(declare => declare.Taking)
                    .Where(column => column.Kind != ColumnKind.Category)
                    .Select(column => column.Name)
                    .ToHashSet(StringComparer.Ordinal));

        // A column that reaches the end can be excluded; one the schema leaves out, or a step below takes away, is already.
        public bool Excludes(string column) => reaching.Allows(column);

        // A column the schema takes, and not as a category, can be marked one; one it excludes is not there to mark.
        public bool Marks(string column) => markable.Contains(column);
    }

    /// <summary>One column as the grid writes it: its values as text, and a colour for each number.</summary>
    private sealed class Colouring
    {
        private readonly IColumn _column;
        private readonly double?[]? _numbers;
        private readonly double _low;
        private readonly double _high;

        private Colouring(IColumn column, double?[]? numbers, double low, double high)
        {
            _column = column;
            _numbers = numbers;
            _low = low;
            _high = high;
        }

        public static Colouring Of(PipelineView view, IColumn column)
        {
            if (column.Kind is not (ColumnKind.Number or ColumnKind.Integer))
            {
                return new Colouring(column, null, 0, 0);
            }

            var measured = view.MeasuredValues(column.Name).Finite;

            return measured.Count == 0
                ? new Colouring(column, null, 0, 0)
                : new Colouring(column, view.Table.NumbersOf(column.Name), measured[0], measured[^1]);
        }

        public void Cell(StringBuilder html, int row)
        {
            var text = Text(row);

            if (_numbers is null)
            {
                html.Append("<td>").Append(Encoded(text)).Append("</td>");

                return;
            }

            var fill = Fill(_numbers[row]);

            html.Append("<td class=\"deepsharp-number\" style=\"background:").Append(fill.ToHex())
                .Append(";color:").Append(fill.ContrastingTextColor().ToHex()).Append("\">")
                .Append(Encoded(text)).Append("</td>");
        }

        // An infinity lies beyond any finite range, so it takes the colour past that end; a gap and a value that
        // is not a number take none.
        private Color Fill(double? value)
        {
            var map = EdgesColorMap.Coolwarm;

            if (value is not { } number || double.IsNaN(number))
            {
                return map.GetBadColor()!.Value;
            }

            if (number < _low)
            {
                return map.GetUnderColor()!.Value;
            }

            return number > _high
                ? map.GetOverColor()!.Value
                : map.GetColor(_high > _low ? (number - _low) / (_high - _low) : 0.5);
        }

        private string Text(int row) => _column switch
        {
            _ when _column.IsMissing(row) => string.Empty,
            Column<double> numbers => numbers[row]!.Value.ToString("G6", CultureInfo.InvariantCulture),
            Column<DateTime> moments => Moment(moments[row]!.Value),
            _ => _column.TextAt(row)!,
        };

        private static string Moment(DateTime moment) =>
            moment.ToString(moment.TimeOfDay == TimeSpan.Zero ? "yyyy-MM-dd" : "yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);
    }
}
