// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using DeepSharp.Pipelines;
using MatPlotLibNet.Styling;
using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>A box in a grid's header, as it is drawn.</summary>
/// <param name="Ticked">Whether it is ticked.</param>
/// <param name="Enabled">Whether it can be clicked: whether the rules allow what unticking it, or ticking it, asks for.</param>
internal readonly record struct HeaderBox(bool Ticked, bool Enabled);

/// <summary>What a grid's header draws for one column.</summary>
/// <param name="Name">The column.</param>
/// <param name="Included">Its box for whether it is in.</param>
/// <param name="Category">Its box for whether it is a category, for a column the schema takes; nothing for any other.</param>
internal readonly record struct HeaderColumn(string Name, HeaderBox Included, HeaderBox? Category);

/// <summary>What a grid's header drew: each column's boxes, in the order the grid showed the columns.</summary>
/// <param name="Columns">The columns.</param>
internal sealed record GridHeader(IReadOnlyList<HeaderColumn> Columns)
{
    /// <summary>Whether a grid of these columns draws the same under the declaration as it is now.</summary>
    /// <param name="now">The declaration now.</param>
    /// <returns><see langword="true"/> while every box it drew still says what holds, and can be clicked as it could.</returns>
    public bool StillHolds(PipelineDeclaration now) =>
        DataGrid.HeaderOf(now, [.. Columns.Select(column => column.Name)]).Columns.SequenceEqual(Columns);
}

/// <summary>A grid as drawn: the page, and the header it drew.</summary>
/// <param name="Output">The page.</param>
/// <param name="Header">What its header drew, which is what the grid is later checked against.</param>
internal readonly record struct DrawnGrid(CellOutput Output, GridHeader Header);

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
    /// <param name="declaration">The declaration the rows come from, which says what each column's boxes say.</param>
    /// <returns>The grid, and the header it drew.</returns>
    /// <remarks>
    /// Every column has a box saying whether it is in, and every column the schema takes one saying whether it is a
    /// category; a box that the rules would not let change is drawn but cannot be clicked. The boxes write the
    /// declaration, never a value.
    /// </remarks>
    public static DrawnGrid Of(PipelineView view, int page, PipelineDeclaration declaration)
    {
        var table = view.Table;
        var header = HeaderOf(declaration, [.. table.Columns.Select(column => column.Name)]);
        var pages = Math.Max(1, (table.RowCount + PageSize - 1) / PageSize);
        var shown = Math.Clamp(page, 0, pages - 1);
        var first = shown * PageSize;
        var last = Math.Min(first + PageSize, table.RowCount);
        var columns = table.Columns.Select(column => Colouring.Of(view, column)).ToArray();
        var html = new StringBuilder(Style).Append("<div class=\"deepsharp-grid\">");

        Summary(html, view);

        html.Append("<div class=\"deepsharp-scroll\"><table><thead><tr><th>#</th><th>part</th>");

        foreach (var column in header.Columns)
        {
            html.Append("<th>").Append(Encoded(column.Name));
            Box(html, StepRenderer.Include, column.Name, column.Included, "included");

            if (column.Category is { } category)
            {
                Box(html, StepRenderer.Category, column.Name, category, "category");
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

        return new DrawnGrid(CellOutput.Html(html.Append("</div>").ToString()), header);
    }

    /// <summary>What a grid of these columns draws in its header under a declaration: the rule the grid draws by.</summary>
    /// <param name="declaration">The declaration.</param>
    /// <param name="columns">The columns the grid shows.</param>
    /// <returns>The header.</returns>
    /// <remarks>
    /// Asked of the column rules, the one set every door that changes the columns goes through: a column is ticked in
    /// when it takes part, is kept or is made, and a box can be clicked when the rules allow what the click asks for. A
    /// category's box can be unticked only when the category says which kind it was.
    /// </remarks>
    public static GridHeader HeaderOf(PipelineDeclaration declaration, IReadOnlyList<string> columns)
    {
        HashSet<string> taken = [.. declaration.Steps.OfType<DeclareStep>().FirstOrDefault()?.Taking.Select(column => column.Name) ?? []];

        return new([.. declaration.ChoicesFor(columns).Rows.Select(choice => new HeaderColumn(
            choice.Name,
            Box(choice.Standing is ColumnStanding.Taking or ColumnStanding.Kept or ColumnStanding.Made, choice.Offers, ColumnOffers.Exclude, ColumnOffers.Include),
            taken.Contains(choice.Name) ? Box(choice.Kind == ColumnKind.Category, choice.Offers, ColumnOffers.BackToWas, ColumnOffers.MakeCategory) : null))]);
    }

    // A box, ticked or not, that can be clicked when the rules offer what the click asks for: unticking it when it is
    // ticked, ticking it when it is not.
    private static HeaderBox Box(bool ticked, ColumnOffers offers, ColumnOffers untick, ColumnOffers tick) =>
        new(ticked, offers.HasFlag(ticked ? untick : tick));

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

    // A box carries its gesture and its column in its data-action and no data-payload, so Verso's router sends the
    // state it is in with them: whether it is ticked.
    private static void Box(StringBuilder html, string gesture, string column, HeaderBox box, string label) =>
        html.Append(" <label><input type=\"checkbox\" data-action=\"")
            .Append(Encoded(ControlAction.Of(gesture, new JsonObject { [StepRenderer.ColumnKey] = column })))
            .Append("\" data-extension-id=\"").Append(StepRenderer.Id).Append('"')
            .Append(box.Ticked ? " checked" : string.Empty)
            .Append(box.Enabled ? string.Empty : " disabled")
            .Append("> ").Append(label).Append("</label>");

    // Verso's router reads what a button carries from its data-payload.
    private static void Button(StringBuilder html, int page, string label) =>
        html.Append(" <button type=\"button\" data-action=\"").Append(StepRenderer.Page)
            .Append("\" data-extension-id=\"").Append(StepRenderer.Id)
            .Append("\" data-payload=\"").Append(Invariant(page)).Append("\">").Append(label).Append("</button>");

    private static string Word(Standing standing) => standing.ToString().ToLowerInvariant();

    private static string Invariant(int number) => number.ToString(CultureInfo.InvariantCulture);

    private static string Encoded(string text) => WebUtility.HtmlEncode(text);

    /// <summary>How many rows stand one way.</summary>
    private readonly record struct StandingCount(Standing Standing, int Rows);

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
