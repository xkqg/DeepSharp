// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Settling a gap with a value no row decided, where the features are worked out.
/// </summary>
/// <remarks>
/// A fill learns its value from the training rows, so it stands below the split, and a column worked out above the split
/// from a column with a gap is itself a gap. Settling is the other half: a nought, a number you choose, or a refusal —
/// none of them a value any row could have told — so it may stand where the features are, and a feature worked out after
/// it is worked out from settled columns. It writes nothing down, because there is nothing it learned, and it marks where
/// the gaps were exactly as a fill does: filling destroys the difference between absent and measured whichever value went
/// in.
/// </remarks>
public class SettleGapsTests
{
    [Fact]
    public void SettlingWithNought_PutsNoughtInEveryGap_AndMarksWhereTheyWere()
    {
        var table = Numbers(1, null, 3);

        new SettleGapsStep("a", With.Zero).AddTo(table);

        Assert.Equal([1, 0, 3], table.NumbersOf("a").Select(value => value!.Value));
        Assert.Equal([0, 1, 0], table.NumbersOf("a_was_missing").Select(value => value!.Value));
    }

    [Fact]
    public void SettlingWithANumberYouChoose_PutsThatNumberIn()
    {
        var table = Numbers(1, null, 3);

        new SettleGapsStep("a", With.Constant(-1)).AddTo(table);

        Assert.Equal([1, -1, 3], table.NumbersOf("a").Select(value => value!.Value));
        Assert.Equal([0, 1, 0], table.NumbersOf("a_was_missing").Select(value => value!.Value));
    }

    [Fact]
    public void AColumnOfWholeNumbers_TakesTheNearestWholeNumber()
    {
        var table = new Table([new Column<long>("a", ColumnKind.Integer, [1, null, 3])]);

        new SettleGapsStep("a", With.Constant(2.5)).AddTo(table);

        Assert.Equal([1, 3, 3], table.NumbersOf("a").Select(value => value!.Value));
    }

    [Fact]
    public void SettlingWithARefusal_SaysWhichColumnAndWhichRow()
    {
        var refused = Assert.Throws<InvalidOperationException>(() => new SettleGapsStep("a", With.Refuse).AddTo(Numbers(1, null, 3)));

        Assert.Contains("'a'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("Row 2", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AColumnWithNoGaps_IsLeftAsItIs_AndStillSaysSo()
    {
        var table = Numbers(1, 2, 3);

        new SettleGapsStep("a", With.Refuse).AddTo(table);

        Assert.Equal([1, 2, 3], table.NumbersOf("a").Select(value => value!.Value));
        Assert.Equal([0, 0, 0], table.NumbersOf("a_was_missing").Select(value => value!.Value));
    }

    [Fact]
    public void WhatItSettles_IsDecidedByTheRowItself_AndByNoOtherRow()
    {
        // The whole reason it may stand above the split: no row anywhere in the table decides what goes into this one, so
        // dividing the rows afterwards cannot change what was settled.
        var one = Numbers(1, null, 3);
        var other = Numbers(1000, null, -4000);

        new SettleGapsStep("a", With.Zero).AddTo(one);
        new SettleGapsStep("a", With.Zero).AddTo(other);

        Assert.Equal(one.NumbersOf("a")[1], other.NumbersOf("a")[1]);
        Assert.Equal(one.NumbersOf("a_was_missing"), other.NumbersOf("a_was_missing"));
    }

    [Fact]
    public void TheStepSaysWhichColumnsItLeavesBehind()
    {
        var after = new SettleGapsStep("a", With.Zero).After(ColumnState.Of(Numbers(1, null, 3)));

        Assert.NotNull(after.Find("a"));
        Assert.NotNull(after.Find("a_was_missing"));
    }

    [Fact]
    public void ItLearnsNothing_SoItWritesNothingDown_AndStandsWhereTheFeaturesAre()
    {
        var step = new SettleGapsStep("a", With.Zero);

        Assert.IsNotAssignableFrom<IFittedStep>(step);
        Assert.IsAssignableFrom<IAddsColumns>(step);
        Assert.False(PreparedData.WritesAnEntry(step));
        Assert.Equal("settle.gaps", step.Verb);
        Assert.Equal(6, SettleGapsStep.Since);
    }

    [Fact]
    public void AWayOfFillingThatAnyRowDecides_IsRefusedWhereItIsWritten()
    {
        // Mean and median come from the rows, so they are a fill and not a settling; carrying the value before a gap
        // forward reads the rows in their order, which is the other form of the fill.
        Assert.Throws<ArgumentException>(() => new SettleGapsStep("a", With.Mean));
        Assert.Throws<ArgumentException>(() => new SettleGapsStep("a", With.Median));
        Assert.Throws<ArgumentException>(() => new SettleGapsStep("a", With.Previous));
    }

    [Fact]
    public void ItIsReadBackFromItsFile_AsItWroteItself()
    {
        var read = (SettleGapsStep)Shipped.Catalog().ReadStep("""{"step": "settle.gaps", "column": "a", "with": {"kind": "constant", "value": -1}}""");

        Assert.Equal("a", read.Column);
        Assert.Equal(With.Constant(-1), read.Strategy);
        Assert.Equal(new SettleGapsStep("a", With.Constant(-1)), read);
    }

    [Fact]
    public void ASplitThatDividesByASettledColumn_DividesByTheSettledValues_AndTheSameWayTwice()
    {
        // The one case a settling above the split can reach: the column the split divides BY. What goes in a gap is
        // decided by nobody but the person who wrote it, so the strata are the settled ones and no row of validation or
        // test taught them anything - and the division is the same on a second run, which is what a split promises.
        static PreparedData Run() =>
            Pdd.Create()
                .ReadCsv(Repository.Data("titanic.csv"))
                .Declare(schema => schema.Integer("survived").Optional("age", ColumnKind.Number).Number("fare"))
                .SettleGaps(gaps => gaps.Zero("age"))
                .SplitStratified("age", 0.70, 0.15)
                .Target("survived")
                .Build()
                .Run();

        var once = Run();
        var again = Run();

        Assert.Equal(once.CountIn(Part.Train), again.CountIn(Part.Train));
        Assert.Equal(once.Table.Digest(), again.Table.Digest());
        Assert.Equal(177, once.Table.NumbersOf("age_was_missing").Count(value => value!.Value == 1));
    }

    [Fact]
    public void AColumnThatHoldsNoNumber_IsRefusedWhereTheSettlingIsWritten()
    {
        // A moment in time or a word is not settled with a number, and the refusal comes at the step rather than from
        // the run: that is what keeps a split by time out of this altogether.
        var refused = Assert.Throws<DeclarationException>(() => Pdd.Create()
            .ReadCsv(Repository.Data("apple.csv"))
            .Declare(schema => schema.Timestamp("Date").Number("AAPL.Close"))
            .SettleGaps(gaps => gaps.Zero("Date"))
            .SplitByTime("Date", 0.70, 0.15)
            .Target("AAPL.Close")
            .Build());

        Assert.Equal("settle.gaps", Assert.Single(refused.Faults).Verb);
    }

    [Fact]
    public void AColumnSettledAboveTheSplit_CannotBeFilledBelowIt_BecauseThereIsNothingLeftToFill()
    {
        // Two answers to one question, and the second can do nothing: settling leaves no gap for a fill to find, so a
        // fill of the same column is a line that never acts. Said at the declaration, before a run, naming both.
        var refused = Assert.Throws<DeclarationException>(() => Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Optional("age", ColumnKind.Number).Number("fare"))
            .SettleGaps(gaps => gaps.Zero("age"))
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing(fill => fill.Median("age"))
            .Target("survived")
            .Build());

        var fault = Assert.Single(refused.Faults);

        Assert.Equal("fill.missing", fault.Verb);
        Assert.Contains("settle.gaps", fault.Message, StringComparison.Ordinal);
        Assert.Contains("age", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnotherColumnFilledBelowTheSplit_IsLeftAlone()
    {
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Optional("age", ColumnKind.Number).Number("fare"))
            .SettleGaps(gaps => gaps.Zero("age"))
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing(fill => fill.Median("fare"))
            .Target("survived")
            .Build()
            .Run();

        Assert.Equal(891, prepared.Table.RowCount);
    }

    [Fact]
    public void TheSameColumnSettledTwice_IsRefusedToo()
    {
        var refused = Assert.Throws<DeclarationException>(() => Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Optional("age", ColumnKind.Number))
            .SettleGaps(gaps => gaps.Zero("age"))
            .SettleGaps(gaps => gaps.Constant(-1, "age"))
            .SplitStratified("survived", 0.70, 0.15)
            .Target("survived")
            .Build());

        Assert.Equal("settle.gaps", Assert.Single(refused.Faults).Verb);
    }

    private static Table Numbers(params double?[] values) => new([new Column<double>("a", ColumnKind.Number, values)]);
}
