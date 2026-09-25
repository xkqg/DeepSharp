// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Verso.Notebooks;
using DeepSharp.Pipelines;
using MatPlotLibNet.Styling.ColorMaps;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// The grid under a block: the rows there a page at a time, each number coloured by where it lies in its column
/// — blue at the smallest training value, red at the largest, a darker colour past either end, none where there is
/// no value. The range is learned the way a fit learns, over the training rows and their finite values alone, and
/// the page is written the same way whatever language the notebook's interface speaks.
/// </summary>
public class GridTests
{
    // The grid at a place in a small pipeline over rows handed in.
    private static string Grid(string csv, Action<SchemaBuilder> schema, int steps = 2, Func<PipelineBuilder, PipelineDeclaration>? more = null, int page = 0)
    {
        var builder = Pdd.Create().Read(CsvRowSource.FromText(csv), "rows").Declare(schema);
        var declaration = more?.Invoke(builder) ?? builder.Declaration;

        return DataGrid.Of(new Pipeline(declaration, CsvRowSource.FromText(csv)).ViewAt(steps), page, declaration).Output.Content;
    }

    private static string Fill(double fraction) => ColorMaps.Coolwarm.GetColor(fraction).ToHex();

    // The coloured cell holding a number written as the given text; the row numbers are cells too, and uncoloured.
    private static string Cell(string html, string text) =>
        html.Split("<td").First(cell =>
            cell.Contains("deepsharp-number", StringComparison.Ordinal) && cell.Contains($">{text}</td>", StringComparison.Ordinal));

    // A black cell: a column that is not in, whose value is not written.
    private const string Black = "<td class=\"deepsharp-out\"></td>";

    private static int Count(string html, string piece) => html.Split(piece).Length - 1;

    // A header's box for a column, as its input tag is written.
    private static string Box(string html, string gesture, string column) =>
        html.Split("<input").Skip(1).Select(tag => tag[..tag.IndexOf('>', StringComparison.Ordinal)])
            .First(tag => tag.Contains($"data-action=\"{gesture} {{&quot;column&quot;:&quot;{column}&quot;}}\"", StringComparison.Ordinal));

    [Fact]
    public void TheGrid_IsTheSameWhateverLanguageTheInterfaceSpeaks()
    {
        const string csv = "when,price,name\n2026-01-02,1.5,a\n2026-01-03 10:30:00,2.25,<b>\n2026-01-04,,c\n";
        void Schema(SchemaBuilder schema) => schema.Timestamp("when").Optional("price", ColumnKind.Number).Text("name");
        var before = CultureInfo.CurrentCulture;

        try
        {
            CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
            var invariant = Grid(csv, Schema);

            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("nl-NL");
            var dutch = Grid(csv, Schema);

            Assert.Equal(invariant, dutch);
            Assert.Contains(">1.5<", dutch, StringComparison.Ordinal);
            Assert.Contains(">2026-01-02<", dutch, StringComparison.Ordinal);
            Assert.Contains(">2026-01-03 10:30:00<", dutch, StringComparison.Ordinal);
            Assert.Contains("&lt;b&gt;", dutch, StringComparison.Ordinal);
        }
        finally
        {
            CultureInfo.CurrentCulture = before;
        }
    }

    [Fact]
    public void AnInfinity_DoesNotSquashTheRestOfItsColumnToOneEnd()
    {
        const string csv = "x\n1\n2\n3\nInfinity\n-Infinity\nNaN\n\n";
        var grid = Grid(csv, schema => schema.Optional("x", ColumnKind.Number));
        var map = EdgesColorMap.Coolwarm;

        Assert.Contains($"background:{Fill(0)}", Cell(grid, "1"), StringComparison.Ordinal);
        Assert.Contains($"background:{Fill(0.5)}", Cell(grid, "2"), StringComparison.Ordinal);
        Assert.Contains($"background:{Fill(1)}", Cell(grid, "3"), StringComparison.Ordinal);
        Assert.Contains($"background:{map.GetOverColor()!.Value.ToHex()}", Cell(grid, "Infinity"), StringComparison.Ordinal);
        Assert.Contains($"background:{map.GetUnderColor()!.Value.ToHex()}", Cell(grid, "-Infinity"), StringComparison.Ordinal);
        Assert.Contains($"background:{map.GetBadColor()!.Value.ToHex()}", Cell(grid, "NaN"), StringComparison.Ordinal);
        Assert.Equal(map.Name, ColorMaps.Coolwarm.Name);
        Assert.Equal(ColorMaps.Coolwarm.GetColor(0.3), map.GetColor(0.3));
    }

    [Fact]
    public void AValuePastTheTrainingRange_HasTheColourPastThatEnd_NotTheReddestTrainingValue()
    {
        // Twenty days, the value rising with them: training takes the first fourteen, so every later value lies
        // above everything a range was learned from.
        var csv = "t,x\n" + string.Concat(Enumerable.Range(1, 20).Select(day => $"{day},{day}\n"));
        var grid = Grid(csv, schema => schema.Integer("t").Number("x"), steps: 3, more: builder => builder.OrderBy("t").SplitByTime("t", 0.70, 0.15).Declaration);
        var over = EdgesColorMap.Coolwarm.GetOverColor()!.Value.ToHex();

        Assert.Contains("14 train", grid, StringComparison.Ordinal);
        Assert.Contains("training rows", grid, StringComparison.Ordinal);
        // The header is the first row, so the fourteenth row of data is the fifteenth piece after it.
        Assert.Contains($"background:{Fill(1)}", grid.Split("<tr>")[15], StringComparison.Ordinal);
        Assert.Contains($"background:{over}", grid.Split("<tr>")[16], StringComparison.Ordinal);
    }

    [Fact]
    public void AColumnWithOneValue_IsColouredInTheMiddle_AndOneWithNoNumberAtAll_IsNotColoured()
    {
        const string csv = "same,none,word\n5,,a\n5,,b\n";
        var grid = Grid(csv, schema => schema.Number("same").Optional("none", ColumnKind.Number).Text("word"));

        Assert.Contains($"background:{Fill(0.5)}", Cell(grid, "5"), StringComparison.Ordinal);
        Assert.Contains("<td>a</td>", grid, StringComparison.Ordinal);
        Assert.Contains("rows of its column", grid, StringComparison.Ordinal);
        Assert.Contains("2 undivided", grid, StringComparison.Ordinal);
    }

    [Fact]
    public void AColumnThatIsNotIn_IsBlack_ItsValuesAreNotOnThePage_AndItsBoxStillTakesItIn()
    {
        // The source's own rows, above the schema: every column as words, and one the schema does not take.
        const string csv = "kept,secret\n1,hidden-one\n2,hidden-two\n";
        var grid = Grid(csv, schema => schema.Number("kept"), steps: 1);
        var box = Box(grid, StepRenderer.Include, "secret");

        Assert.Equal(2, Count(grid, Black));
        Assert.DoesNotContain("hidden-one", grid, StringComparison.Ordinal);
        Assert.DoesNotContain("hidden-two", grid, StringComparison.Ordinal);
        Assert.Contains("<td>1</td><td class", grid, StringComparison.Ordinal);
        Assert.DoesNotContain(" checked", box, StringComparison.Ordinal);
        Assert.DoesNotContain(" disabled", box, StringComparison.Ordinal);
        Assert.Contains("; a column that is not in is black", grid, StringComparison.Ordinal);
        Assert.DoesNotContain("is black", Grid(csv, schema => schema.Number("kept").Text("secret"), steps: 1), StringComparison.Ordinal);
    }

    [Fact]
    public void AColumnDroppedBelowAStepThatReadsIt_IsBlackWhereItStillStands()
    {
        const string csv = "a,b\n11,17\n13,19\n";
        var grid = Grid(
            csv, schema => schema.Number("a", "b"), steps: 3,
            more: builder => new PipelineDeclaration(builder.AddFeature("c", "a", Arithmetic.Plus, "b").Declaration.Excluding("b")));

        Assert.Equal(2, Count(grid, Black));
        Assert.DoesNotContain(">17</td>", grid, StringComparison.Ordinal);
        Assert.DoesNotContain(">19</td>", grid, StringComparison.Ordinal);
        Assert.Contains($"background:{Fill(0)}", Cell(grid, "28"), StringComparison.Ordinal);
        Assert.Contains($"background:{Fill(1)}", Cell(grid, "32"), StringComparison.Ordinal);
    }

    [Fact]
    public void ACategory_IsOneColour_DarkerForAValueTheTrainingRowsNeverHeld_AndGreyForAGap()
    {
        // Twenty days: training takes the first fourteen, which hold x and y alone; the later rows bring z, and a gap.
        var csv = "t,kind\n" + string.Concat(Enumerable.Range(1, 20).Select(day => $"{day},{(day <= 14 ? (day % 2 == 0 ? "x" : "y") : day == 20 ? "" : "z")}\n"));
        var grid = Grid(csv, schema => schema.Integer("t").Category("kind"), steps: 4, more: builder => builder.OrderBy("t").SplitByTime("t", 0.70, 0.15).Declaration);
        var map = EdgesColorMap.Category;

        string Kind(string text) => grid.Split("<td").First(cell =>
            cell.Contains("deepsharp-category", StringComparison.Ordinal) && cell.Contains($">{text}</td>", StringComparison.Ordinal));

        Assert.Contains($"background:{map.GetColor(0).ToHex()}", Kind("x"), StringComparison.Ordinal);
        Assert.Contains($"background:{map.GetColor(0).ToHex()}", Kind("y"), StringComparison.Ordinal);
        Assert.Contains($"background:{map.GetOverColor()!.Value.ToHex()}", Kind("z"), StringComparison.Ordinal);
        Assert.Contains($"background:{map.GetBadColor()!.Value.ToHex()}", Kind(string.Empty), StringComparison.Ordinal);
        Assert.Contains("; a category is one colour, darker for a value those rows never held", grid, StringComparison.Ordinal);
        Assert.DoesNotContain("a category is", Grid(csv, schema => schema.Integer("t").Text("kind")), StringComparison.Ordinal);
    }

    [Fact]
    public void TheCategoryColour_IsOkabeItosReddishPurple_DarkerPastWhatTheTrainingRowsHeld_AndTheSameGreyForAGap()
    {
        var map = EdgesColorMap.Category;

        Assert.Equal("category", map.Name);
        Assert.Equal("#CC79A7", map.GetColor(0).ToHex());
        Assert.Equal(map.GetColor(0), map.GetColor(1));
        Assert.Equal("#7A4864", map.GetOverColor()!.Value.ToHex());
        Assert.Equal(map.GetOverColor(), map.GetUnderColor());
        Assert.Equal(EdgesColorMap.Coolwarm.GetBadColor(), map.GetBadColor());
    }

    [Fact]
    public void AnEmptyTable_IsOnePageWithNothingOnIt()
    {
        var grid = Grid("x\n", schema => schema.Number("x"), page: 3);

        Assert.Contains("0 rows", grid, StringComparison.Ordinal);
        Assert.Contains("no rows", grid, StringComparison.Ordinal);
        Assert.DoesNotContain($"data-action=\"{StepRenderer.Page}\"", grid, StringComparison.Ordinal);
    }
}
