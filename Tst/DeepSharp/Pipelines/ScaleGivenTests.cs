// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Scaling a column into a range with bounds you give, above the split.
/// </summary>
/// <remarks>
/// Every other scaling reads its numbers from the training rows — a mean, a smallest and a largest — so it stands below
/// the split and is replayed from what it learned. This one is told the bounds: a fare runs from nothing to the most
/// anybody paid, an hour from nought to twenty-three, a share from nought to one. Nobody learned that from the rows, so
/// it may stand where the features are worked out, and the way back is exact without anything written down.
/// </remarks>
public class ScaleGivenTests
{
    [Fact]
    public void ItLandsTheColumnBetweenMinusOneAndOne_FromTheBoundsItWasGiven()
    {
        var table = Numbers(0, 50, 100);

        new ScaleGivenStep("a", new KnownRange(0, 100)).AddTo(table);

        Assert.Equal([-1, 0, 1], table.NumbersOf("a").Select(value => value!.Value));
    }

    [Fact]
    public void ItLandsItBetweenNothingAndOne_WhenThatIsWhatIsAskedFor()
    {
        var table = Numbers(0, 50, 100);

        new ScaleGivenStep("a", new KnownRange(0, 100)) { Lands = Form.Unit }.AddTo(table);

        Assert.Equal([0, 0.5, 1], table.NumbersOf("a").Select(value => value!.Value));
    }

    [Fact]
    public void AValueOutsideTheBounds_LandsOutsideTheRange_AndIsNotQuietlyMoved()
    {
        // The bounds are a statement about the column, not a clamp: a value beyond them says the statement was wrong,
        // and hiding that would make a model learn from a number nobody meant.
        var table = Numbers(-50, 150);

        new ScaleGivenStep("a", new KnownRange(0, 100)).AddTo(table);

        Assert.Equal([-2, 2], table.NumbersOf("a").Select(value => value!.Value));
    }

    [Fact]
    public void ItLearnsNothing_SoItWritesNothingDown_AndStandsWhereTheFeaturesAre()
    {
        var step = new ScaleGivenStep("a", new KnownRange(0, 100));

        Assert.IsNotAssignableFrom<IFittedStep>(step);
        Assert.IsAssignableFrom<IAddsColumns>(step);
        Assert.False(PreparedData.WritesAnEntry(step));
        Assert.Equal("scale.given", step.Verb);
        Assert.Equal(7, ScaleGivenStep.Since);
    }

    [Fact]
    public void ThePredictionComesBackInItsOwnUnits_WithoutAnythingFitted()
    {
        // What the owner's course needs at the end: the model answers between minus one and one, and the way back puts
        // it in euros again — exactly, because the bounds were given rather than learned.
        var step = new ScaleGivenStep("fare", new KnownRange(0, 512));

        Assert.IsAssignableFrom<IUndoesItself>(step);
        Assert.Equal(512, ((IUndoesItself)step).Undo(1, fitted: null), 9);
        Assert.Equal(256, ((IUndoesItself)step).Undo(0, fitted: null), 9);
        Assert.Equal(0, ((IUndoesItself)step).Undo(-1, fitted: null), 9);
    }

    [Fact]
    public void BoundsThatSayNothing_AreRefusedWhereTheyAreWritten()
    {
        Assert.Throws<ArgumentException>(() => new KnownRange(100, 100));
        Assert.Throws<ArgumentException>(() => new KnownRange(100, 0));
        Assert.Throws<ArgumentException>(() => new KnownRange(double.NaN, 1));
    }

    [Fact]
    public void OneLine_ScalesManyColumns_EachBetweenItsOwnBounds()
    {
        var declaration = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp").Number("fare"))
            .ScaleGiven(scale => scale.Between("fare", 0, 512).Between("sibsp", 0, 8))
            .SplitStratified("survived", 0.70, 0.15)
            .Target("survived")
            .Declaration;

        var scaled = declaration.Steps.OfType<ScaleGivenStep>().ToArray();

        Assert.Equal(["fare", "sibsp"], scaled.Select(step => step.Column));
        Assert.Equal([512, 8], scaled.Select(step => step.Range.Highest));
    }

    [Fact]
    public void ItIsReadBackFromItsFile_AsItWroteItself()
    {
        var read = (ScaleGivenStep)Shipped.Catalog().ReadStep(
            """{"step": "scale.given", "column": "fare", "lowest": 0, "highest": 512, "lands": "signed"}""");

        Assert.Equal("fare", read.Column);
        Assert.Equal(new KnownRange(0, 512), read.Range);
        Assert.Equal(Form.Signed, read.Lands);
    }

    [Fact]
    public void AColumnOfWholeNumbers_IsScaledToo_AndAGapStaysAGap()
    {
        var table = new Table([new Column<long>("a", ColumnKind.Integer, [0, null, 10])]);

        new ScaleGivenStep("a", new KnownRange(0, 10)) { Lands = Form.Unit }.AddTo(table);

        Assert.Equal([0, 1], table.NumbersOf("a").Where(value => value is not null).Select(value => value!.Value));
        Assert.True(table["a"].IsMissing(1));
    }

    [Fact]
    public void AColumnThatHoldsNoNumber_IsRefusedWhereItIsScaled()
    {
        var table = new Table([new TextColumn("a", ColumnKind.Category, ["one", "two"])]);

        var refused = Assert.Throws<InvalidOperationException>(() => new ScaleGivenStep("a", new KnownRange(0, 1)).AddTo(table));

        Assert.Contains("'a'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ThePredictionComesBackFromTheOtherRangeToo()
    {
        var unit = (IUndoesItself)new ScaleGivenStep("fare", new KnownRange(0, 512)) { Lands = Form.Unit };

        Assert.Equal(512, unit.Undo(1, fitted: null), 9);
        Assert.Equal(0, unit.Undo(0, fitted: null), 9);
        Assert.True(unit.Undoes("fare"));
        Assert.False(unit.Undoes("age"));
    }

    [Fact]
    public void ItSaysWhichColumnItLeavesBehind_AndWhereItLands()
    {
        var after = new ScaleGivenStep("a", new KnownRange(0, 100)).After(ColumnState.Of(Numbers(1, 2)));

        Assert.NotNull(after.Find("a"));
        Assert.Throws<ArgumentNullException>(() => new ScaleGivenStep("a", new KnownRange(0, 1)).After(null!));
        Assert.Throws<ArgumentNullException>(() => new ScaleGivenStep("a", new KnownRange(0, 1)).AddTo(null!));
    }

    private static Table Numbers(params double?[] values) => new([new Column<double>("a", ColumnKind.Number, values)]);
}
