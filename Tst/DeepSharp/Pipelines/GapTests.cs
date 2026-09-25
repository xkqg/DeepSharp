// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Globalization;
using System.Text;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A split in time can keep the last moments of every part apart. A model asked what happens some rows ahead
/// learns each training row's answer from a later row, and without a gap the last training rows read theirs
/// across the line, from the rows it is measured on. The rows kept apart are fitted on by nothing and handed to
/// nothing, and the fit writes down how many there were.
/// </summary>
public class GapTests
{
    private static readonly SplitShares Halves = SplitShares.Of(0.50, 0.25);

    // One row a day, oldest first, with the day's number as its value; a day named twice is two rows of one moment.
    private static InMemoryRowSource Days(int count, params int[] twice) =>
        new(["Date", "value"], [.. Enumerable.Range(0, count)
            .SelectMany(day => twice.Contains(day) ? new[] { day, day } : [day])
            .Select((day, row) => (IReadOnlyList<string?>)
            [
                new DateTime(2026, 1, 1).AddDays(day).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
                row.ToString(CultureInfo.InvariantCulture),
            ])]);

    private static PreparedData Split(InMemoryRowSource rows, int gap) =>
        new Pipeline(
            new PipelineDeclaration(
            [
                new ReadRowsStep("days"),
                new DeclareStep([new ColumnDeclaration("Date", ColumnKind.Timestamp, false), new ColumnDeclaration("value", ColumnKind.Number, false)]),
                new SplitByTimeStep("Date", Halves, gap),
                new NormaliseStep("value"),
                new DropColumnsStep(["Date"]),
            ]),
            rows).Run();

    private static int[] ReadAt(PreparedData prepared, Part part) =>
        [.. Enumerable.Range(0, prepared.Table.RowCount).Where(row => prepared.Parts[row] == part).Select(row => prepared.Table.Identities[row].ReadAt)];

    [Fact]
    public void ATimeSplitWithAGap_SetsTheLastMomentsOfEveryPartApart()
    {
        // Twenty days in halves and quarters: ten, five and five, each losing its last two to the gap.
        var prepared = Split(Days(20), gap: 2);

        Assert.Equal([0, 1, 2, 3, 4, 5, 6, 7], ReadAt(prepared, Part.Train));
        Assert.Equal([10, 11, 12], ReadAt(prepared, Part.Validation));
        Assert.Equal([15, 16, 17], ReadAt(prepared, Part.Test));
        Assert.Equal([8, 9, 13, 14, 18, 19], ReadAt(prepared, Part.Gap));
    }

    [Fact]
    public void TheGapCountsMoments_SoAMomentIsNeverDivided()
    {
        // Day 9 is two rows of one moment, the last of training. A gap of one moment takes both, where a gap of
        // one row would have put the same moment on both sides.
        var prepared = Split(Days(19, twice: 9), gap: 1);

        Assert.Equal([9, 10], ReadAt(prepared, Part.Gap).Where(row => row < 11));
        Assert.DoesNotContain(9, ReadAt(prepared, Part.Train));
    }

    [Fact]
    public void TheFit_WritesDownHowManyRowsTheGapHeld_AndNothingWithoutOne()
    {
        Assert.Equal(6, Split(Days(20), gap: 2).Fitted[2].Number("rows.gap"));
        Assert.False(Split(Days(20), gap: 0).Fitted[2].Numbers.ContainsKey("rows.gap"));
    }

    [Fact]
    public void NothingIsFittedOnTheGap_AndNothingOfItIsHandedOver()
    {
        var prepared = Split(Days(20), gap: 2);

        // The mean the scale learned is that of the eight training rows, 0 to 7, and not of the ten before the line.
        Assert.Equal(3.5, prepared.Fitted[3].Number("centre"), 9);
        Assert.Equal(8, prepared.Batch(Part.Train).RowCount);
        Assert.Throws<ArgumentException>(() => prepared.Batch(Part.Gap));
    }

    [Fact]
    public void AGapOfNone_IsWrittenAsNothing_SoEveryFileWrittenBeforeKeepsItsKeys()
    {
        static string Written(IPipelineStep step)
        {
            var buffer = new ArrayBufferWriter<byte>();

            using (var writer = new Utf8JsonWriter(buffer))
            {
                step.WriteTo(writer);
            }

            return Encoding.UTF8.GetString(buffer.WrittenSpan);
        }

        Assert.Equal(Written(new SplitByTimeStep("Date", Halves)), Written(new SplitByTimeStep("Date", Halves, 0)));
        Assert.DoesNotContain("gap", Written(new SplitByTimeStep("Date", Halves)), StringComparison.Ordinal);
        Assert.Contains("\"gap\":2", Written(new SplitByTimeStep("Date", Halves, 2)), StringComparison.Ordinal);
    }

    [Fact]
    public void AGapIsReadFromAFile_AndANoughtIsWrittenBackAsNothing()
    {
        var catalog = StepCatalog.BuiltIn();
        const string Written = """{"step":"split.byTime","column":"Date","train":0.5,"validation":0.25,"test":0.25,"predict":0""";

        Assert.Equal(3, ((SplitByTimeStep)catalog.ReadStep(Written + ""","gap":3}""")).Gap);
        Assert.Equal(0, ((SplitByTimeStep)catalog.ReadStep(Written + ""","gap":0}""")).Gap);
        Assert.Equal(0, ((SplitByTimeStep)catalog.ReadStep(Written + "}")).Gap);
        Assert.Throws<ArgumentOutOfRangeException>(() => new SplitByTimeStep("Date", Halves, -1));
    }

    [Fact]
    public void TheChain_SplitsWithAGapToo()
    {
        var prepared = Pdd.Create()
            .Read(Days(20), "days")
            .Declare(schema => schema.Timestamp("Date").Number("value"))
            .SplitByTime("Date", 0.50, 0.25, gap: 2)
            .Build()
            .Run();

        Assert.Equal([8, 9, 13, 14, 18, 19], ReadAt(prepared, Part.Gap));
    }

    [Theory]
    [InlineData(10, "train", "learn from")]
    [InlineData(5, "test", "measure on")]
    public void AGapThatTakesAWholePart_ThatALearnerNeeds_IsRefused(int gap, string part, string purpose)
    {
        // Ten, five and five moments: a gap of five takes all of test, a gap of ten all of training. Validation may
        // come out empty; a model still learns from one part and is measured on another.
        var refused = Assert.Throws<InvalidOperationException>(() => Split(Days(20), gap));

        Assert.Contains($"every row of {part}", refused.Message, StringComparison.Ordinal);
        Assert.Contains(purpose, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheGapStandsApartInTheDataAtEveryBlock()
    {
        var view = new Pipeline(
            new PipelineDeclaration(
            [
                new ReadRowsStep("days"),
                new DeclareStep([new ColumnDeclaration("Date", ColumnKind.Timestamp, false), new ColumnDeclaration("value", ColumnKind.Number, false)]),
                new SplitByTimeStep("Date", Halves, 2),
            ]),
            Days(20)).ViewAt(2);

        Assert.Equal(6, view.Standings.Count(standing => standing == Standing.Gap));
    }
}
