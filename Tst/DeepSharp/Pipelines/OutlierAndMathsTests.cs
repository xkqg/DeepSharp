// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Holding the tail of a column, and pulling a column into another shape. The first learns its bounds from
/// the training rows like everything else below the split; the second learns nothing at all and therefore
/// stands above it. Both refuse what they cannot answer rather than handing back a not-a-number.
/// </summary>
public class OutlierAndMathsTests
{
    private static Table Numbers(params double[] values)
    {
        var source = CsvRowSource.FromText(
            "a\n" + string.Join("\n", values.Select(value => value.ToString("R", System.Globalization.CultureInfo.InvariantCulture))) + "\n");

        return SchemaBinding.Bind(new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, true)]), source);
    }

    private static Part[] AllTraining(Table table) => [.. Enumerable.Repeat(Part.Train, table.RowCount)];

    // ---- holding the tail ----------------------------------------------------------------------------

    [Fact]
    public void TheMiddleHalfDecidesTheBounds_AndOneSpikeCannotMoveThem()
    {
        // Nine ordinary values and one that is a hundred times the rest. The interquartile bounds barely
        // notice it, which is the whole reason to prefer them on data with a tail.
        var table = Numbers(1, 2, 3, 4, 5, 6, 7, 8, 9, 1000);
        var step = new ClipOutliersStep("a");

        var learned = step.Fit(table, AllTraining(table));
        step.ApplyTo(table, learned);

        var held = (Column<double>)table["a"];

        Assert.Equal(1, learned.Number("outside"));
        Assert.InRange(learned.Number("upper"), 10, 20);
        Assert.Equal(learned.Number("upper"), held[9]!.Value, 6);
        Assert.Equal(5, held[4]);
    }

    [Fact]
    public void ASpreadOfSoManySigmas_IsDraggedAroundByTheThingItShouldCatch()
    {
        // The same data, bounds by standard deviation: the spike inflates the spread that is supposed to
        // find it, and the bound lands far above anything else in the column.
        var table = Numbers(1, 2, 3, 4, 5, 6, 7, 8, 9, 1000);
        var step = new ClipOutliersStep("a", Bounds.Sigma, 3);

        var learned = step.Fit(table, AllTraining(table));

        Assert.True(learned.Number("upper") > 900, $"the bound was {learned.Number("upper")}");
        Assert.Equal(0, learned.Number("outside"));
    }

    [Fact]
    public void AQuantileBoundSetsAsideTheShareYouName()
    {
        var table = Numbers([.. Enumerable.Range(1, 100).Select(value => (double)value)]);
        var step = new ClipOutliersStep("a", Bounds.Quantile, 0.05);

        var learned = step.Fit(table, AllTraining(table));

        Assert.Equal(5.95, learned.Number("lower"), 2);
        Assert.Equal(95.05, learned.Number("upper"), 2);
    }

    [Fact]
    public void AValueOutsideCanBeBlankedInsteadOfHeld()
    {
        var table = Numbers(1, 2, 3, 4, 5, 6, 7, 8, 9, 1000);
        var step = new ClipOutliersStep("a", Bounds.Iqr, 1.5, Outlier.Blank);

        step.ApplyTo(table, step.Fit(table, AllTraining(table)));

        // A gap, for the step that fills gaps to answer — rather than a number this pipeline invented.
        Assert.True(table["a"].IsMissing(9));
        Assert.False(table["a"].IsMissing(0));
    }

    [Fact]
    public void OrRefused_WithTheRowAndHowFarOutItWas()
    {
        var table = Numbers(1, 2, 3, 4, 5, 6, 7, 8, 9, 1000);
        var step = new ClipOutliersStep("a", Bounds.Iqr, 1.5, Outlier.Refuse);

        var refused = Assert.Throws<InvalidOperationException>(
            () => step.ApplyTo(table, step.Fit(table, AllTraining(table))));

        Assert.Contains("Row 10", refused.Message);
        Assert.Contains("1000", refused.Message);
    }

    [Fact]
    public void TheBoundsComeFromTheTrainingRowsAlone()
    {
        var table = Numbers(1, 2, 3, 4, 5, 1000);
        var step = new ClipOutliersStep("a", Bounds.Quantile, 0.1);

        // The spike is in test, so it has no say in where the bounds sit — and it is held all the same.
        var learned = step.Fit(table, [Part.Train, Part.Train, Part.Train, Part.Train, Part.Train, Part.Test]);
        step.ApplyTo(table, learned);

        Assert.True(learned.Number("upper") < 10, $"the bound was {learned.Number("upper")}");
        Assert.Equal(learned.Number("upper"), ((Column<double>)table["a"])[5]!.Value, 6);
    }

    [Fact]
    public void ABoundThatIsNotADistance_IsRefusedWhereItIsWritten()
    {
        Assert.Throws<ArgumentException>(() => new ClipOutliersStep(" "));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ClipOutliersStep("a", Bounds.Iqr, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ClipOutliersStep("a", Bounds.Iqr, double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => new ClipOutliersStep("a", Bounds.Quantile, 0.5));
    }

    [Fact]
    public void AColumnOfNothingHasNoBoundsToLearn()
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, true)]),
            CsvRowSource.FromText("a\n\n\n"));

        Assert.Throws<InvalidOperationException>(
            () => new ClipOutliersStep("a").Fit(table, AllTraining(table)));
        Assert.Throws<ArgumentNullException>(() => new ClipOutliersStep("a").Fit(null!, []));
        Assert.Throws<ArgumentNullException>(() => new ClipOutliersStep("a").Fit(table, null!));
        Assert.Throws<ArgumentNullException>(() => new ClipOutliersStep("a").ApplyTo(null!, new FittedStepValues()));
        Assert.Throws<ArgumentNullException>(() => new ClipOutliersStep("a").ApplyTo(table, null!));
    }

    [Fact]
    public void ItBelongsBelowTheSplit_AndTheChainSaysSo()
    {
        var declaration = Pdd.Create()
            .ReadCsv("x.csv")
            .Declare(schema => schema.Number("a"))
            .SplitAtRandom(0.70, 0.15)
            .ClipOutliers("a", Bounds.Sigma, 2.5, Outlier.Blank)
            .Declaration;

        Assert.Equal(declaration, PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn()));
        Assert.IsType<IFittedStep>(declaration.Steps[3], exactMatch: false);
    }

    // ---- another shape -------------------------------------------------------------------------------

    [Theory]
    [InlineData(Maths.Log, 2.302585)]
    [InlineData(Maths.Log1P, 2.397895)]
    [InlineData(Maths.Sqrt, 3.162278)]
    [InlineData(Maths.Square, 100)]
    [InlineData(Maths.Reciprocal, 0.1)]
    [InlineData(Maths.Abs, 10)]
    [InlineData(Maths.Sign, 1)]
    public void EveryShapeIsTheArithmeticItSaysItIs(Maths maths, double expected)
    {
        var table = Numbers(10);

        new MathsStep("a", maths).AddTo(table);

        Assert.Equal(expected, ((Column<double>)table["a"])[0]!.Value, 6);
    }

    [Fact]
    public void AnArcSineIsForAColumnOfProportions()
    {
        var table = Numbers(0.25);

        new MathsStep("a", Maths.ArcSin).AddTo(table);

        Assert.Equal(Math.Asin(0.5), ((Column<double>)table["a"])[0]!.Value, 6);
    }

    [Theory]
    [InlineData(Maths.Log, 0)]
    [InlineData(Maths.Log, -1)]
    [InlineData(Maths.Log1P, -1)]
    [InlineData(Maths.Reciprocal, 0)]
    [InlineData(Maths.Sqrt, -1)]
    [InlineData(Maths.ArcSin, 2)]
    public void AValueTheShapeHasNoAnswerFor_IsRefusedWithItsRow(Maths maths, double value)
    {
        var table = Numbers(1, value);

        var refused = Assert.Throws<InvalidOperationException>(() => new MathsStep("a", maths).AddTo(table));

        Assert.Contains("Row 2", refused.Message);
    }

    [Fact]
    public void AGapStaysAGapWhateverTheShape()
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, true)]),
            CsvRowSource.FromText("a\n4\n\n"));

        new MathsStep("a", Maths.Sqrt).AddTo(table);

        Assert.Equal(2, ((Column<double>)table["a"])[0]);
        Assert.True(table["a"].IsMissing(1));
    }

    [Fact]
    public void TheResultCanGoIntoAColumnOfItsOwn()
    {
        var table = Numbers(100);

        new MathsStep("a", Maths.Log, into: "log_a").AddTo(table);

        Assert.Equal(100, ((Column<double>)table["a"])[0]);
        Assert.Equal(Math.Log(100), ((Column<double>)table["log_a"])[0]!.Value, 6);
    }

    [Fact]
    public void ItBelongsAboveTheSplitBecauseItLearnsNothing()
    {
        var declaration = Pdd.Create()
            .ReadCsv("x.csv")
            .Declare(schema => schema.Number("a"))
            .Reshape("a", Maths.Log1P, into: "log_a")
            .SplitAtRandom(0.70, 0.15)
            .Declaration;

        Assert.Equal(declaration, PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn()));
        Assert.IsNotType<IFittedStep>(declaration.Steps[2], exactMatch: false);
        Assert.Throws<ArgumentException>(() => new MathsStep(" ", Maths.Log));
        Assert.Throws<ArgumentNullException>(() => new MathsStep("a", Maths.Log).AddTo(null!));
    }
}
