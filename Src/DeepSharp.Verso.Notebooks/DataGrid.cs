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
/// The rows at a block, a page at a time, each number coloured by where it lies in its column, each category by whether
/// the training rows hold it, and a column that is not in black.
/// </summary>
/// <remarks>
/// A column's range is learned the way a fit learns: over the rows the pipeline measures on — the training rows of
/// the split, wherever it is declared, or every row when nothing divides them — and over their finite values alone,
/// so one infinity cannot squash every other value to one end. Blue is the smallest training value and red the
/// largest; a value outside that range, as a validation row may well be, has a darker colour of its own, and a gap
/// has none. A category is one colour, and darker for a value those rows never held, read by the rule an encoder
/// learns its categories by. A column whose box says it is not in is black, and its values are not written at all.
/// Everything is written in the invariant culture, so the page is the same whichever language the notebook's
/// interface speaks.
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
        + ".deepsharp-grid td.deepsharp-number{text-align:right}"
        + ".deepsharp-grid td.deepsharp-out{background:#000}</style>";

    /// <summary>The page of the rows at a block.</summary>
    /// <param name="view">The rows there, and where each stands.</param>
    /// <param name="page">Which page, counting from nought; one beyond the last shows the last.</param>
    /// <param name="declaration">The declaration the rows come from, which says what each column's boxes say.</param>
    /// <returns>The grid, and the header it drew.</returns>
    /// <remarks>
    /// Every column has a box saying whether it is in, and every column the schema takes one saying whether it is a
    /// category; a box that the rules would not let change is drawn but cannot be clicked. The boxes write the
    /// declaration, never a value. A column whose box says it is not in is drawn black, with its box, so ticking it
    /// brings its values back.
    /// </remarks>
    public static DrawnGrid Of(PipelineView view, int page, PipelineDeclaration declaration)
    {
        var table = view.Table;
        var header = HeaderOf(declaration, [.. table.Columns.Select(column => column.Name)]);
        var pages = Math.Max(1, (table.RowCount + PageSize - 1) / PageSize);
        var shown = Math.Clamp(page, 0, pages - 1);
        var first = shown * PageSize;
        var last = Math.Min(first + PageSize, table.RowCount);

        // The header asks about the table's columns in their order, and says in that order how each stands.
        Colouring[] columns = [.. table.Columns.Select((column, at) => Colouring.Of(view, column, header.Columns[at].Included.Ticked))];
        var html = new StringBuilder(Style).Append("<div class=\"deepsharp-grid\">");

        Summary(html, view, columns);

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
            choice.Name, choice.IncludedBox(), taken.Contains(choice.Name) ? choice.CategoryBox() : null))]);
    }

    // How many rows stand where, and what the colours mean: numbers always, and each other way of colouring the page
    // uses, once, in the order its first column stands.
    private static void Summary(StringBuilder html, PipelineView view, IReadOnlyList<Colouring> columns)
    {
        var counts = Enum.GetValues<Standing>()
            .Select(standing => new StandingCount(standing, view.Standings.Count(each => each == standing)))
            .Where(each => each.Rows > 0)
            .Select(each => $"{Invariant(each.Rows)} {Word(each.Standing)}");

        html.Append("<div class=\"deepsharp-summary\">").Append(Invariant(view.Table.RowCount)).Append(" rows")
            .Append(string.Concat(counts.Select(count => $" · {count}")))
            .Append(" — each number coloured by where it lies among the ")
            .Append(view.Measured == Standing.Train ? "training rows" : "rows")
            .Append(" of its column")
            .Append(string.Concat(columns.Select(column => column.Legend).OfType<string>().Distinct().Select(legend => $"; {legend}")))
            .Append("</div>");
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

    /// <summary>One column as the grid writes it: its values as text, coloured the way its kind is, or black when it is not in.</summary>
    /// <param name="column">The column.</param>
    private abstract class Colouring(IColumn column)
    {
        /// <summary>The column.</summary>
        protected IColumn Column { get; } = column;

        /// <summary>What the page says its colours mean, for a column coloured this way; nothing when it says so already.</summary>
        public virtual string? Legend => null;

        /// <summary>How a column is written: black when it is not in, else by its kind.</summary>
        /// <param name="view">The rows there, and which of them are measured.</param>
        /// <param name="column">The column.</param>
        /// <param name="included">Whether its box says it is in.</param>
        /// <returns>The way it is written.</returns>
        public static Colouring Of(PipelineView view, IColumn column, bool included)
        {
            if (!included)
            {
                return new Out(column);
            }

            if (column.Kind == ColumnKind.Category)
            {
                return new Categories(column, view.MeasuredCategories(column.Name).ToHashSet(StringComparer.Ordinal));
            }

            if (column.Kind is not (ColumnKind.Number or ColumnKind.Integer))
            {
                return new Plain(column);
            }

            var measured = view.MeasuredValues(column.Name).Finite;

            return measured.Count == 0 ? new Plain(column) : new Numbers(column, view.Table.NumbersOf(column.Name), measured[0], measured[^1]);
        }

        /// <summary>Writes the column's cell on one row.</summary>
        /// <param name="html">The page.</param>
        /// <param name="row">The row.</param>
        public abstract void Cell(StringBuilder html, int row);

        /// <summary>A cell filled with a colour, its text in the colour that reads on it.</summary>
        /// <param name="html">The page.</param>
        /// <param name="kind">The cell's class.</param>
        /// <param name="fill">The colour.</param>
        /// <param name="row">The row.</param>
        protected void Filled(StringBuilder html, string kind, Color fill, int row) =>
            html.Append("<td class=\"").Append(kind).Append("\" style=\"background:").Append(fill.ToHex())
                .Append(";color:").Append(fill.ContrastingTextColor().ToHex()).Append("\">")
                .Append(Encoded(Text(row))).Append("</td>");

        /// <summary>The value on a row as text: nothing for a gap.</summary>
        /// <param name="row">The row.</param>
        /// <returns>The text.</returns>
        protected string Text(int row) => Column switch
        {
            _ when Column.IsMissing(row) => string.Empty,
            Column<double> numbers => numbers[row]!.Value.ToString("G6", CultureInfo.InvariantCulture),
            Column<DateTime> moments => Moment(moments[row]!.Value),
            _ => Column.TextAt(row)!,
        };

        private static string Moment(DateTime moment) =>
            moment.ToString(moment.TimeOfDay == TimeSpan.Zero ? "yyyy-MM-dd" : "yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);
    }

    /// <summary>A column that is not in: black, and none of its values written.</summary>
    /// <param name="column">The column.</param>
    private sealed class Out(IColumn column) : Colouring(column)
    {
        /// <inheritdoc />
        public override string Legend => "a column that is not in is black";

        /// <inheritdoc />
        public override void Cell(StringBuilder html, int row) => html.Append("<td class=\"deepsharp-out\"></td>");
    }

    /// <summary>A column its kind gives no colour to: words, moments, a number with no value among the measured rows.</summary>
    /// <param name="column">The column.</param>
    private sealed class Plain(IColumn column) : Colouring(column)
    {
        /// <inheritdoc />
        public override void Cell(StringBuilder html, int row) => html.Append("<td>").Append(Encoded(Text(row))).Append("</td>");
    }

    /// <summary>A column of numbers, coloured by where each lies in the range the measured rows hold.</summary>
    /// <param name="column">The column.</param>
    /// <param name="numbers">Its values as numbers.</param>
    /// <param name="low">The smallest finite value the measured rows hold.</param>
    /// <param name="high">The largest.</param>
    private sealed class Numbers(IColumn column, double?[] numbers, double low, double high) : Colouring(column)
    {
        /// <inheritdoc />
        public override void Cell(StringBuilder html, int row) => Filled(html, "deepsharp-number", Fill(numbers[row]), row);

        // An infinity lies beyond any finite range, so it takes the colour past that end; a gap and a value that
        // is not a number take none.
        private Color Fill(double? value)
        {
            var map = EdgesColorMap.Coolwarm;

            if (value is not { } number || double.IsNaN(number))
            {
                return map.GetBadColor()!.Value;
            }

            if (number < low)
            {
                return map.GetUnderColor()!.Value;
            }

            return number > high
                ? map.GetOverColor()!.Value
                : map.GetColor(high > low ? (number - low) / (high - low) : 0.5);
        }
    }

    /// <summary>A category: one colour, darker for a value the measured rows never held, grey for a gap.</summary>
    /// <param name="column">The column.</param>
    /// <param name="held">The values the measured rows hold.</param>
    private sealed class Categories(IColumn column, IReadOnlySet<string> held) : Colouring(column)
    {
        /// <inheritdoc />
        public override string Legend => "a category is one colour, darker for a value those rows never held";

        /// <inheritdoc />
        public override void Cell(StringBuilder html, int row) => Filled(html, "deepsharp-category", Fill(row), row);

        private Color Fill(int row)
        {
            var map = EdgesColorMap.Category;

            return Column.IsMissing(row) ? map.GetBadColor()!.Value
                : held.Contains(Column.TextAt(row)!) ? map.GetColor(0)
                : map.GetOverColor()!.Value;
        }
    }
}
