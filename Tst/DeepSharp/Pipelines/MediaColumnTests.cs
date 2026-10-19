// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Images, sound and the ids of tokens are columns of a kind of their own: a fixed-length array of numbers to a row, read by a reader that
/// decodes them by arithmetic that learns nothing — a byte lands between minus one and one by one rule, wherever it stands. A cell is one
/// value in the source's text, so a row is known by what it holds and not by the file it came from, and a column holds the arrays whole
/// instead of a cell of text for each number. What is handed to a learner is the arrays flattened, with the shape of one beside them.
/// </summary>
public class MediaColumnTests
{
    private static string Pixels(int rows, int columns, Func<int, byte> by) =>
        MediaCell.Pixels([rows, columns, 1], [.. Enumerable.Range(0, rows * columns).Select(by)]);

    // ---- a cell ----

    [Fact]
    public void PixelsLandBetweenMinusOneAndOne_ByOneRule_AndComeBackAsTheyWereWritten()
    {
        var cell = MediaCell.Parse(MediaCell.Pixels([1, 3, 1], [0, 51, 255]));

        Assert.Equal([1, 3, 1], cell.Shape);
        Assert.Equal([-1f, (51 / 127.5f) - 1, 1f], cell.Values);
        Assert.False(cell.HoldsIds);
    }

    [Fact]
    public void SamplesOfSixteenBitsLandBetweenMinusOneAndOne_ByTheirOwnRule()
    {
        var cell = MediaCell.Parse(MediaCell.Samples([4], [-32768, 0, 16384, 32767]));

        Assert.Equal([4], cell.Shape);
        Assert.Equal([-1f, 0f, 0.5f, 32767 / 32768f], cell.Values);
    }

    [Fact]
    public void FloatsAndIds_AreKeptAsTheyAre_TheIdsAsIds()
    {
        var floats = MediaCell.Parse(MediaCell.Floats([2], [0.25f, -0.75f]));
        var ids = MediaCell.Parse(MediaCell.Ids([3], [5, 0, 17]));

        Assert.Equal([0.25f, -0.75f], floats.Values);
        Assert.False(floats.HoldsIds);
        Assert.Equal([5f, 0f, 17f], ids.Values);
        Assert.True(ids.HoldsIds);
    }

    [Theory]
    [InlineData("")]
    [InlineData("u8")]
    [InlineData("u8[2]AAAA")]
    [InlineData("u8[2]:!!!!")]
    [InlineData("u8[]:AAAA")]
    [InlineData("u8[0]:")]
    [InlineData("u8[2x]:AAAA")]
    [InlineData("u8[3]:AAA=")]
    [InlineData("zz[2]:AAAA")]
    [InlineData("f32[2]:AAAA")]
    [InlineData("u8[-1]:AAAA")]
    public void ACellThatIsNotAMediaCell_IsRefused(string text) =>
        Assert.Throws<FormatException>(() => MediaCell.Parse(text));

    [Fact]
    public void AShapeAndItsBytesThatDoNotAgree_AreRefusedWhereTheCellIsMade()
    {
        Assert.Throws<ArgumentException>(() => MediaCell.Pixels([2, 2, 1], [1, 2, 3]));
        Assert.Throws<ArgumentException>(() => MediaCell.Pixels([], []));
        Assert.Throws<ArgumentException>(() => MediaCell.Samples([2], [1]));
        Assert.Throws<ArgumentException>(() => MediaCell.Floats([0, 3], []));
        Assert.Throws<ArgumentException>(() => MediaCell.Ids([2], [1]));
        Assert.Throws<ArgumentException>(() => MediaCell.Ids([1], [-1]));
    }

    // ---- a column ----

    [Fact]
    public void ARowOfTheColumn_IsKnownByWhatItHolds_NotByWhereItCameFrom()
    {
        var column = new MediaColumn("pixels", [MediaCell.Parse(Pixels(2, 2, at => (byte)at)), MediaCell.Parse(Pixels(2, 2, at => (byte)at)), MediaCell.Parse(Pixels(2, 2, at => (byte)(at + 1)))]);

        Assert.Equal(ColumnKind.Media, column.Kind);
        Assert.Equal(3, column.Count);
        Assert.Equal([2, 2, 1], column.Shape);
        Assert.Equal(column.TextAt(0), column.TextAt(1));
        Assert.NotEqual(column.TextAt(0), column.TextAt(2));
        Assert.False(column.IsMissing(0));
        Assert.Equal(0, column.LeadingGaps());
        Assert.Equal([-1f, (1 / 127.5f) - 1, (2 / 127.5f) - 1, (3 / 127.5f) - 1], column.ValuesAt(0)!);
        var picked = column.Rows([2, 0]);

        Assert.Equal(column.TextAt(2), picked.TextAt(0));
        Assert.Equal(2, picked.Count);
    }

    [Fact]
    public void ACellOfAnotherSize_IsRefused_NamingTheRowAndBothSizes_AndNothingIsResized()
    {
        var refused = Assert.Throws<ArgumentException>(() => new MediaColumn("pixels", [MediaCell.Parse(Pixels(2, 2, _ => 1)), MediaCell.Parse(Pixels(3, 3, _ => 1))]));

        Assert.Contains("row 2", refused.Message, StringComparison.Ordinal);
        Assert.Contains("2x2x1", refused.Message, StringComparison.Ordinal);
        Assert.Contains("3x3x1", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void IdsAndPixelsInOneColumn_AreRefused_AndAColumnOfNoRowsHasNoShape()
    {
        Assert.Throws<ArgumentException>(() => new MediaColumn("x", [MediaCell.Parse(MediaCell.Ids([2], [1, 2])), MediaCell.Parse(MediaCell.Floats([2], [0, 1]))]));
        Assert.Throws<ArgumentException>(() => new MediaColumn("x", []));
    }

    // ---- read from a source, declared, and handed over ----

    private static InMemoryRowSource Digits(int count = 40) =>
        new(
            ["pixels", "digit"],
            [.. Enumerable.Range(0, count).Select(at => (IReadOnlyList<string?>)[Pixels(4, 4, pixel => (byte)((at * 7) + (pixel * 11) % 256)), (at % 3).ToString(CultureInfo.InvariantCulture)])]);

    private static PreparedData Prepared() =>
        Pdd.Create()
            .Read(Digits(), "forty digits")
            .Declare(schema => schema.Media("pixels").Integer("digit"))
            .SplitAtRandom(0.60, 0.20, 3)
            .Target("digit")
            .Build()
            .Run();

    [Fact]
    public void ADeclaredMediaColumn_IsReadFromTheCellsOfASource_ArraysWhole()
    {
        var prepared = Prepared();
        var column = Assert.IsType<MediaColumn>(prepared.Table["pixels"]);

        Assert.Equal(ColumnKind.Media, column.Kind);
        Assert.Equal([4, 4, 1], column.Shape);
        Assert.Equal(40, column.Count);
    }

    [Fact]
    public void AMediaCellThatIsNone_IsRefusedWhereTheRowsAreRead_NamingTheRow()
    {
        var rows = new InMemoryRowSource(["pixels", "digit"], [[Pixels(2, 2, _ => 1), "0"], ["not a cell", "1"]]);
        var refused = Assert.Throws<DeclarationException>(() => Pdd.Create().Read(rows, "two").Declare(schema => schema.Media("pixels").Integer("digit")).Build().Run());

        Assert.Contains("Row 2", refused.Message, StringComparison.Ordinal);
        Assert.Contains("pixels", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WhatIsHandedOver_IsTheArraysFlattened_WithTheShapeOfOneBesideThem()
    {
        var prepared = Prepared();
        var batch = prepared.Batch(Part.Train, Needs.OneScale);

        Assert.Equal(16, batch.Width);
        Assert.Equal("pixels.0", batch.FeatureNames[0]);
        Assert.Equal("pixels.15", batch.FeatureNames[15]);
        Assert.Equal([4, 4, 1], batch.ExampleShape);
        Assert.All(batch.Features, row => Assert.All(row, value => Assert.InRange(value, -1, 1)));

        var first = Assert.IsType<MediaColumn>(prepared.Table["pixels"]);
        var place = Enumerable.Range(0, prepared.Table.RowCount).First(row => prepared.Parts[row] == Part.Train);

        Assert.Equal(first.ValuesAt(place)!.Select(value => (double)value), batch.Features[0]);
    }

    [Fact]
    public void ARowServedLater_IsHandedOverTheSameWay_AndCarriesTheShapeToo()
    {
        var prepared = Prepared();
        var served = prepared.Served(Digits(3), Needs.OneScale);

        Assert.Equal(3, served.RowCount);
        Assert.Equal([4, 4, 1], served.ExampleShape);
        Assert.Equal(16, served.FeatureNames.Count);
    }

    [Fact]
    public void ARowOfSeveralColumns_ThatIsNotAloneAMediaOne_HasNoOneShapeToCarry()
    {
        var rows = new InMemoryRowSource(
            ["pixels", "age", "digit"],
            [.. Enumerable.Range(0, 20).Select(at => (IReadOnlyList<string?>)[Pixels(2, 2, pixel => (byte)(at + pixel)), at.ToString(CultureInfo.InvariantCulture), (at % 2).ToString(CultureInfo.InvariantCulture)])]);
        var prepared = Pdd.Create()
            .Read(rows, "twenty")
            .Declare(schema => schema.Media("pixels").Number("age").Integer("digit"))
            .SplitAtRandom(0.60, 0.20, 3)
            .Target("digit")
            .Normalise("age", Scale.MidRange)
            .Build()
            .Run();
        var batch = prepared.Batch(Part.Train, Needs.OneScale);

        Assert.Equal(5, batch.Width);
        Assert.Null(batch.ExampleShape);
    }

    // ---- what a steps that works on numbers does with it, and what looks at every column leaves out ----

    [Fact]
    public void AStepThatWorksOnNumbers_RefusesAMediaColumn()
    {
        var refused = Assert.ThrowsAny<Exception>(() =>
            Pdd.Create().Read(Digits(), "forty").Declare(schema => schema.Media("pixels").Integer("digit")).SplitAtRandom(0.60, 0.20, 3).Target("digit").Normalise("pixels", Scale.MidRange).Build().Run());

        Assert.Contains("pixels", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WhatWatchesEveryColumn_LeavesTheMediaOut_Unasked()
    {
        var prepared = Pdd.Create()
            .Read(Digits(), "forty digits")
            .Declare(schema => schema.Media("pixels").Integer("digit"))
            .SplitAtRandom(0.60, 0.20, 3)
            .Target("digit")
            .Watch(0.2)
            .Likeness(0.3)
            .Build()
            .Run();

        Assert.DoesNotContain("pixels", prepared.Table.Columns.Where(column => column.Kind != ColumnKind.Media).Select(column => column.Name));
        Assert.All(prepared.Evidence.Values.OfType<PartLikeness>(), likeness => Assert.DoesNotContain(likeness.Columns, column => column.Column == "pixels"));
    }

    [Fact]
    public void ADeclaredMediaColumn_ShowsInTheFile_AsItsKind_AndReadsBack()
    {
        var prepared = Prepared();
        var text = prepared.ToJson();

        Assert.Contains("\"media\"", text, StringComparison.Ordinal);

        var read = PreparedData.FromJson(text, StepCatalog.BuiltIn());

        Assert.Equal(prepared.Declaration.Steps.Count, read.Declaration.Steps.Count);
    }
}
