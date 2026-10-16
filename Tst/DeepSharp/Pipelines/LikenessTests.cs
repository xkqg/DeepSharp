// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// How alike the parts a split made are, column by column: the two-sample Kolmogorov–Smirnov distance between the values of
/// a column of numbers in two parts — the most the share of one part at or below a value differs from the other's — and, for
/// a column of words, half the sum of how far each word's share differs. A part the model is measured on that is unlike the
/// rows it learned from explains a measurement that disappoints, so the evidence says each distance beside the bound the
/// person set, and flags the ones above it. It warns and never refuses, and it looks at features, never at the answer.
/// </summary>
/// <remarks>
/// The rows in time: the first six train, the next three validate, the last three test. x is 1 to 6, then 4 to 6, then 10 to
/// 12: training and validation are half apart (at three, half the training rows and none of the validation rows), and test
/// lies beyond both. c is three a and three b, then a b b, then b b b: a sixth, a half and a third apart.
/// </remarks>
public class LikenessTests
{
    private const string Rows =
        "t,x,c,y\n"
        + "1,1,a,0\n2,2,a,1\n3,3,a,0\n4,4,b,1\n5,5,b,0\n6,6,b,1\n"
        + "7,4,a,0\n8,5,b,1\n9,6,b,0\n"
        + "10,10,b,1\n11,11,b,0\n12,12,b,1\n";

    private static PartLikeness Measured(double warnAbove = 0.4, params string[] columns) =>
        (PartLikeness)Pdd.Create()
            .Read(CsvRowSource.FromText(Rows), "rows")
            .Declare(schema => schema.Integer("t").Number("x", "y").Category("c"))
            .SplitByTime("t", train: 0.5, validation: 0.25)
            .Target("y")
            .Likeness(warnAbove, columns)
            .Build()
            .Run()
            .Evidence[4];

    [Fact]
    public void AColumnOfNumbers_IsMeasuredByTheKolmogorovSmirnovDistance_BetweenEveryTwoParts()
    {
        var x = Measured().Columns.Single(column => column.Column == "x");

        Assert.Equal(Distance.KolmogorovSmirnov, x.Distance);
        Assert.Equal([(Standing.Train, Standing.Validation), (Standing.Train, Standing.Test), (Standing.Validation, Standing.Test)], x.Pairs.Select(pair => (pair.First, pair.Second)));
        Assert.Equal([0.5, 1, 1], x.Pairs.Select(pair => pair.Apart));
        Assert.Equal([(6, 3), (6, 3), (3, 3)], x.Pairs.Select(pair => (pair.FirstRows, pair.SecondRows)));
    }

    [Fact]
    public void ValuesThatTie_AreTakenWhole_AsScipysTwoSampleTestTakesThem()
    {
        // 1 2 2 3 against 2 2 3 3: at two, three quarters of the first and half of the second — a quarter, not the half that
        // stepping through the ties one at a time would find.
        Assert.Equal(0.25, PartLikeness.KolmogorovSmirnov([1, 2, 2, 3], [2, 2, 3, 3]), 12);
    }

    [Fact]
    public void AColumnOfWords_IsMeasuredByHalfTheSumOfHowFarEachWordsShareDiffers()
    {
        var c = Measured().Columns.Single(column => column.Column == "c");

        Assert.Equal(Distance.TotalVariation, c.Distance);
        Assert.Equal([1 / 6.0, 0.5, 1 / 3.0], c.Pairs.Select(pair => pair.Apart), new Close());
    }

    [Fact]
    public void EveryDistanceAboveTheBound_IsFlagged_AndTheAnswerIsNeverMeasured()
    {
        var likeness = Measured(0.4);

        Assert.Equal(["c", "t", "x"], likeness.Columns.Select(column => column.Column).Order(StringComparer.Ordinal));
        Assert.Equal(0.4, likeness.WarnAbove);
        Assert.Equal([false, true, false], likeness.Columns.Single(column => column.Column == "c").Pairs.Select(pair => pair.Above));
        Assert.Equal([true, true, true], likeness.Columns.Single(column => column.Column == "x").Pairs.Select(pair => pair.Above));
    }

    [Fact]
    public void TheColumnsItIsToldOf_AreTheOnesMeasured()
    {
        Assert.Equal(["x"], Measured(0.4, "x").Columns.Select(column => column.Column));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1.5)]
    public void ABoundThatIsNoShareAboveNothing_IsRefusedWhereTheStepIsMade(double wrong) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new LikenessStep(wrong));

    [Fact]
    public void TheBound_HasNoDefault()
    {
        Assert.Contains("warnAbove", StepCatalog.BuiltIn().Describe(LikenessStep.Name).Parameters.SelectMany(parameter => parameter.RequiredKeys));
    }

    [Fact]
    public void AMoment_IsMeasuredAsTheDaysItStandsFromTheEpoch_AndAPartWithoutAValueIsNoDistanceAndNoWarning()
    {
        // d is a day for every training and validation row, the same days in both, and no day at all on the test rows; the id
        // names each row and is never measured.
        var rows = "t,id,d,y\n"
            + "1,a,2025-01-01,0\n2,b,2025-01-02,1\n3,c,2025-01-03,0\n4,d,2025-01-04,1\n5,e,2025-01-05,0\n6,f,2025-01-06,1\n"
            + "7,g,2025-01-01,0\n8,h,2025-01-03,1\n9,i,2025-01-05,0\n"
            + "10,j,,1\n11,k,,0\n12,l,,1\n";
        var likeness = Pdd.Create()
            .Read(CsvRowSource.FromText(rows), "rows")
            .Declare(schema => schema.Integer("t").Id("id", ColumnKind.Text).Optional("d", ColumnKind.Timestamp).Number("y"))
            .SplitByTime("t", train: 0.5, validation: 0.25)
            .Target("y")
            .Likeness(0.4)
            .Build()
            .Run()
            .Evidence.Values.OfType<PartLikeness>().Single();
        var d = likeness.Columns.Single(column => column.Column == "d");

        Assert.DoesNotContain(likeness.Columns, column => column.Column == "id");
        Assert.Equal(Distance.KolmogorovSmirnov, d.Distance);
        Assert.Equal(1 / 6.0, d.Pairs[0].Apart, 12);
        Assert.True(double.IsNaN(d.Pairs[1].Apart));
        Assert.False(d.Pairs[1].Above);
        Assert.Equal((6, 0), (d.Pairs[1].FirstRows, d.Pairs[1].SecondRows));
    }

    [Fact]
    public void ADistance_IsBetweenTwoSetsThatEachHoldSomething()
    {
        Assert.Throws<ArgumentException>(() => PartLikeness.KolmogorovSmirnov([], [1]));
        Assert.Throws<ArgumentException>(() => PartLikeness.KolmogorovSmirnov([1], []));
        Assert.Throws<ArgumentException>(() => PartLikeness.TotalVariation([], ["a"]));
        Assert.Throws<ArgumentException>(() => PartLikeness.TotalVariation(["a"], []));
    }

    [Fact]
    public void TwoStepsAreTheSame_WhenTheyWarnAboveTheSameBound_OverTheSameColumns()
    {
        var step = new LikenessStep(0.4, ["x"]);

        Assert.Equal(step, new LikenessStep(0.4, ["x"]));
        Assert.Equal(step.GetHashCode(), new LikenessStep(0.4, ["x"]).GetHashCode());
        Assert.NotEqual(step, new LikenessStep(0.3, ["x"]));
        Assert.NotEqual(step, new LikenessStep(0.4));
        Assert.False(step.Equals(null));
    }

    [Fact]
    public void AVisitorIsHandedTheLikenessAsTheKindItIs()
    {
        Assert.Equal("likeness", Measured().Accept(new Named()));
    }

    private sealed class Named : IEvidenceVisitor<string>
    {
        public string Visit(DataProfile profile) => "profile";

        public string Visit(CorrelationInput correlation) => "correlation";

        public string Visit(Measures measures) => "measures";

        public string Visit(PartLikeness likeness) => "likeness";
    }

    private sealed class Close : IEqualityComparer<double>
    {
        public bool Equals(double left, double right) => Math.Abs(left - right) < 1e-12;

        public int GetHashCode(double value) => 0;
    }
}
