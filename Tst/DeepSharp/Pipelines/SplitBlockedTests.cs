// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A split in blocks of time with an embargo: rows grouped by the week they fall in, whole weeks dealt at random to
/// training, validation and test, and every training row within the embargo of a row a model is measured on left out of
/// the training alone — kept apart as the gap a split in time already keeps, fitted on by nothing and handed to nothing.
/// </summary>
/// <remarks>
/// Seventy days, one row a day, from the first of January 2026, a Thursday: with blocks of seven days counted from the Unix
/// epoch, also a Thursday, the days fall in ten whole weeks.
/// </remarks>
public class SplitBlockedTests
{
    private static readonly DateTime First = new(2026, 1, 1);

    private static InMemoryRowSource Days(bool reversed = false)
    {
        var days = Enumerable.Range(0, 70).Select<int, IReadOnlyList<string?>>(day =>
        [
            First.AddDays(day).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture),
            (day % 3).ToString(CultureInfo.InvariantCulture),
            (10 + day).ToString(CultureInfo.InvariantCulture),
        ]);

        return new InMemoryRowSource(["date", "flock", "weight"], [.. reversed ? days.Reverse() : days]);
    }

    private static PreparedData Dealt(BlockedDeal deal, bool reversed = false, double validation = 0.2) =>
        Pdd.Create()
            .Read(Days(reversed), "seventy days")
            .Declare(schema => schema.Timestamp("date").Integer("flock").Number("weight"))
            .SplitBlocked("date", 0.6, validation, deal)
            .Normalise("weight")
            .Build()
            .Run();

    // The week a row falls in, counted from the Unix epoch, as the split counts it.
    private static long WeekOf(PreparedData prepared, int row) =>
        (long)Math.Floor((((Column<DateTime>)prepared.Table["date"])[row]!.Value - DateTime.UnixEpoch).TotalDays / 7);

    private static DateTime DayOf(PreparedData prepared, int row) => ((Column<DateTime>)prepared.Table["date"])[row]!.Value;

    private static Dictionary<DateTime, Part> PartsByDay(PreparedData prepared) =>
        Enumerable.Range(0, prepared.Table.RowCount).ToDictionary(row => DayOf(prepared, row), row => prepared.Parts[row]);

    [Fact]
    public void EveryRowOfAWeek_LandsInTheSamePart()
    {
        var prepared = Dealt(new BlockedDeal(Days: 7, EmbargoDays: 0));

        var weeks = Enumerable.Range(0, prepared.Table.RowCount).GroupBy(row => WeekOf(prepared, row)).ToArray();

        Assert.Equal(10, weeks.Length);
        Assert.All(weeks, week => Assert.Single(week.Select(row => prepared.Parts[row]).Distinct()));
        Assert.Contains(Part.Train, prepared.Parts);
        Assert.Contains(Part.Validation, prepared.Parts);
        Assert.Contains(Part.Test, prepared.Parts);
    }

    [Fact]
    public void TheSameRowsReadInAnotherOrder_AreDealtTheSameWay()
    {
        var forward = PartsByDay(Dealt(new BlockedDeal(7, 3)));
        var backward = PartsByDay(Dealt(new BlockedDeal(7, 3), reversed: true));

        Assert.Equal(forward.OrderBy(day => day.Key), backward.OrderBy(day => day.Key));
    }

    [Fact]
    public void TheEmbargo_LeavesOutExactlyTheTrainingRowsWithinItsDaysOfAMeasuredRow_AndTheFitSaysHowMany()
    {
        var without = Dealt(new BlockedDeal(7, 0));
        var with = Dealt(new BlockedDeal(7, 3));
        var measured = Enumerable.Range(0, without.Table.RowCount)
            .Where(row => without.Parts[row] is Part.Validation or Part.Test)
            .Select(row => DayOf(without, row))
            .ToArray();

        for (var row = 0; row < with.Table.RowCount; row++)
        {
            var near = measured.Any(day => Math.Abs((day - DayOf(with, row)).TotalDays) <= 3);
            var expected = without.Parts[row] == Part.Train && near ? Part.Gap : without.Parts[row];

            Assert.Equal(expected, with.Parts[row]);
        }

        Assert.Contains(Part.Gap, with.Parts);
        Assert.Equal(with.CountIn(Part.Gap), with.Fitted[2].Number("rows.gap"));
    }

    [Fact]
    public void TheEmbargo_NeverMovesARowAModelIsChosenOrMeasuredOn()
    {
        var without = Dealt(new BlockedDeal(7, 0));
        var with = Dealt(new BlockedDeal(7, 10));

        for (var row = 0; row < with.Table.RowCount; row++)
        {
            if (without.Parts[row] != Part.Train)
            {
                Assert.Equal(without.Parts[row], with.Parts[row]);
            }
        }
    }

    [Fact]
    public void AnEmbargoThatTakesEveryTrainingRow_IsRefused()
    {
        var refused = Assert.Throws<InvalidOperationException>(() => Dealt(new BlockedDeal(7, 400)));

        Assert.Contains("embargo of 400 days", refused.Message, StringComparison.Ordinal);
        Assert.Contains("learn from", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ATestSeed_KeepsTheRowsToBeMeasuredOnWhateverTheSeed()
    {
        var one = Dealt(new BlockedDeal(7, 0) { Seed = 1, TestSeed = 7 });
        var other = Dealt(new BlockedDeal(7, 0) { Seed = 2, TestSeed = 7 });

        Assert.Equal(
            Enumerable.Range(0, one.Table.RowCount).Where(row => one.Parts[row] == Part.Test),
            Enumerable.Range(0, other.Table.RowCount).Where(row => other.Parts[row] == Part.Test));
    }

    [Fact]
    public void AFold_DealsWholeWeeks_AndEveryWeekThatLearnsIsValidationOnce()
    {
        var folds = Enumerable.Range(0, 3).Select(index => Dealt(new BlockedDeal(7, 0) { Fold = new Fold(3, index) }, validation: 0)).ToArray();
        var learning = Enumerable.Range(0, folds[0].Table.RowCount).Where(row => folds[0].Parts[row] != Part.Test).Select(row => WeekOf(folds[0], row)).Distinct().ToArray();

        foreach (var fold in folds)
        {
            var weeks = Enumerable.Range(0, fold.Table.RowCount).GroupBy(row => WeekOf(fold, row));

            Assert.All(weeks, week => Assert.Single(week.Select(row => fold.Parts[row]).Distinct()));
        }

        Assert.All(learning, week => Assert.Single(folds, fold =>
            Enumerable.Range(0, fold.Table.RowCount).Any(row => WeekOf(fold, row) == week && fold.Parts[row] == Part.Validation)));
    }

    [Fact]
    public void TheFileReadsBackTheSplitItWrote_AndTheFitSaysHowManyWeeksEachPartHolds()
    {
        var prepared = Dealt(new BlockedDeal(7, 3) { Seed = 5, TestSeed = 9 });
        var declaration = prepared.Declaration;
        var split = Assert.IsType<SplitBlockedStep>(declaration.Steps[2]);

        Assert.Equal(declaration, PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn()));
        Assert.Equal(new BlockedDeal(7, 3) { Seed = 5, TestSeed = 9 }, split.Deal);
        Assert.Equal(10, Part.Dealt.Sum(part => prepared.Fitted[2].Numbers.TryGetValue($"blocks.{part.Word()}", out var weeks) ? weeks : 0));
    }

    [Theory]
    [InlineData(5, 5)]     // before every measured moment
    [InlineData(10, 0)]    // on one
    [InlineData(13, 3)]    // between two, nearer the earlier
    [InlineData(18, 2)]    // between two, nearer the later
    [InlineData(31, 11)]   // after every one
    public void HowFarAMomentIsFromTheNearestMeasuredOne_IsTheShortestDistance_OnEitherSide(long moment, long distance)
    {
        // What the embargo asks of every training row: how near the nearest row a model is measured on stands, before it or after.
        long[] measured = [10, 20];

        Assert.Equal(distance, measured.Nearest(moment));
    }

    [Fact]
    public void ABlockOfNoDays_OrAnEmbargoBelowNothing_IsRefusedWhereItIsWritten()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new SplitBlockedStep("date", new SplitShares(0.6, 0.2, 0.2), new BlockedDeal(0, 0)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new SplitBlockedStep("date", new SplitShares(0.6, 0.2, 0.2), new BlockedDeal(7, -1)));
    }

    [Fact]
    public void AColumnThatIsNotAMoment_IsRefusedAtTheSplit()
    {
        var refused = Assert.Throws<DeclarationException>(() => Pdd.Create()
            .Read(Days(), "seventy days")
            .Declare(schema => schema.Timestamp("date").Integer("flock").Number("weight"))
            .SplitBlocked("weight", 0.6, 0.2, new BlockedDeal(7, 0)));

        Assert.Equal("split.blocked", Assert.Single(refused.Faults).Verb);
    }
}
