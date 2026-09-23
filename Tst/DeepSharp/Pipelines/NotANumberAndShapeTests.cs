// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A value that is not a number is not a gap, and the two get different verbs. And the two scales that need
/// more than a middle and a spread: one that keeps the whole shape of the training distribution, and one
/// that reshapes a lopsided column towards a bell curve.
/// </summary>
public class NotANumberAndShapeTests
{
    private static Table Read(string csv)
    {
        var source = CsvRowSource.FromText(csv);
        var schema = new SchemaBuilder();

        foreach (var name in source.ColumnNames)
        {
            schema.Column(name, ColumnKind.Number, optional: true);
        }

        return SchemaBinding.Bind(new DeclareStep(schema.Columns), source);
    }

    private static Split[] AllTraining(Table table) => [.. Enumerable.Repeat(Split.Train, table.RowCount)];

    // ---- not a number --------------------------------------------------------------------------------

    [Fact]
    public void ANotANumberStopsTheRunByDefault()
    {
        var table = Read("a\n1\nNaN\n");
        var step = new FillNaNStep("a");

        var learned = step.Fit(table, AllTraining(table));
        var refused = Assert.Throws<InvalidOperationException>(() => step.ApplyTo(table, learned));

        Assert.Equal("refuse", step.Strategy.Name);
        Assert.Equal(1, learned.Number("notNumbers"));
        Assert.Contains("Row 2", refused.Message);
        Assert.Contains("not a number", refused.Message);
    }

    [Theory]
    [InlineData("mean", 2)]
    [InlineData("median", 2)]
    [InlineData("zero", 0)]
    public void AStrategyPutsSomethingInItsPlace(string name, double expected)
    {
        var strategy = name switch
        {
            "mean" => With.Mean,
            "median" => With.Median,
            _ => With.Zero,
        };

        var table = Read("a\n1\n2\n3\nNaN\n");
        var step = new FillNaNStep("a", strategy);

        step.ApplyTo(table, step.Fit(table, AllTraining(table)));

        Assert.Equal(expected, ((Column<double>)table["a"])[3]);
    }

    [Fact]
    public void AConstantIsWrittenDownAsWhatWasUsed()
    {
        var table = Read("a\n1\nNaN\n");
        var step = new FillNaNStep("a", With.Constant(-9));

        var learned = step.Fit(table, AllTraining(table));
        step.ApplyTo(table, learned);

        Assert.Equal(-9, learned.Number("value"));
        Assert.Equal(-9, ((Column<double>)table["a"])[1]);
    }

    [Fact]
    public void AColumnWithNothingWrongWithIt_IsLeftAlone()
    {
        var table = Read("a\n1\n2\n");
        var step = new FillNaNStep("a");

        step.ApplyTo(table, step.Fit(table, AllTraining(table)));

        Assert.Equal(1, ((Column<double>)table["a"])[0]);
    }

    [Fact]
    public void CarryingTheLastValueForwardIsNotAnAnswerHere()
    {
        // A not-a-number is a fault upstream, and the row above it says nothing about what it should have
        // been. Refusing, or a number you choose, are the honest answers.
        Assert.Throws<ArgumentException>(() => new FillNaNStep("a", With.Previous));
        Assert.Throws<ArgumentException>(() => new FillNaNStep(" "));
        Assert.Throws<ArgumentException>(() => new FillNaNStep("a", new FillStrategy("sideways")));
    }

    [Fact]
    public void AWholeNumberCannotHoldOne()
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("a", ColumnKind.Integer, true)]),
            CsvRowSource.FromText("a\n1\n"));

        var step = new FillNaNStep("a", With.Zero);

        Assert.Throws<InvalidOperationException>(() => step.ApplyTo(table, step.Fit(table, AllTraining(table))));
    }

    [Fact]
    public void EveryTrainingRowBeingWrong_LeavesNothingToLearnFrom()
    {
        var table = Read("a\nNaN\nNaN\n");
        var step = new FillNaNStep("a", With.Mean);

        Assert.Throws<InvalidOperationException>(() => step.Fit(table, AllTraining(table)));
    }

    [Fact]
    public void TheWholeChainOffersIt_AndItSurvivesTheFile()
    {
        var declaration = Pdd.Create()
            .ReadCsv("x.csv")
            .Declare(schema => schema.Number("a"))
            .SplitAtRandom(0.70, 0.15, 0.15)
            .FillNaN("a")
            .FillNaN("a", With.Constant(0))
            .Target("a")
            .Declaration;

        Assert.Equal(declaration, PipelineDeclaration.FromJson(declaration.ToJson()));
    }

    [Theory]
    [InlineData("""{"declaration":[{"step":"fill.nan","column":"a"}]}""")]
    [InlineData("""{"declaration":[{"step":"fill.nan","column":"a","with":7}]}""")]
    public void AFileMissingTheStrategy_IsRefused(string json)
    {
        Assert.Throws<FormatException>(() => PipelineDeclaration.FromJson(json));
    }

    [Fact]
    public void AGapWhereThereShouldBeNone_IsRefusedTooWhenYouSaySo()
    {
        var table = SchemaBinding.Bind(
            new DeclareStep([new ColumnDeclaration("a", ColumnKind.Number, true)]),
            CsvRowSource.FromText("a\n1\n\n"));

        var refused = Assert.Throws<InvalidOperationException>(
            () => new FillMissingStep("a", With.Refuse).Fit(table, AllTraining(table)));

        Assert.Contains("should be none", refused.Message);
    }

    [Fact]
    public void AndSaysNothingWhenThereAreNone()
    {
        var table = Read("a\n1\n2\n");
        var step = new FillMissingStep("a", With.Refuse);

        step.ApplyTo(table, step.Fit(table, AllTraining(table)));

        Assert.Equal(1, ((Column<double>)table["a"])[0]);
    }

    // ---- the two scales that need the shape ----------------------------------------------------------

    [Fact]
    public void TheQuantileScalePutsEveryTrainingValueBetweenNothingAndOne()
    {
        var table = Read("a\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n");
        var step = new NormaliseStep("a", Scale.Quantile);

        var learned = step.Fit(table, AllTraining(table));
        step.ApplyTo(table, learned);

        var scaled = (Column<double>)table["a"];

        Assert.Equal(101, learned.Curve("knots").Count);
        Assert.Equal(0, scaled[0]);
        Assert.Equal(1, scaled[9]);
        Assert.All(Enumerable.Range(0, 10), row => Assert.InRange(scaled[row]!.Value, 0, 1));
        Assert.True(scaled[4] < scaled[5], "rank keeps the order it found");
    }

    [Fact]
    public void AValueOutsideWhatItSawHoldsAtTheEdge()
    {
        var table = Read("a\n1\n2\n3\n100\n-100\n");
        var step = new NormaliseStep("a", Scale.Quantile);

        var learned = step.Fit(table, [Split.Train, Split.Train, Split.Train, Split.Test, Split.Test]);
        step.ApplyTo(table, learned);

        var scaled = (Column<double>)table["a"];

        Assert.Equal(1, scaled[3]);
        Assert.Equal(0, scaled[4]);
    }

    [Fact]
    public void TheShapeIsKeptWholeThroughTheFile()
    {
        var table = Read("a\n1\n2\n3\n");
        var step = new NormaliseStep("a", Scale.Quantile);
        var prepared = new PreparedData(
            new PipelineDeclaration([new SplitAtRandomStep(new SplitShares(0.7, 0.15, 0.15), 1), step]),
            table,
            AllTraining(table),
            new Dictionary<int, FittedStepValues> { [1] = step.Fit(table, AllTraining(table)) });

        var loaded = PreparedData.FromJson(prepared.ToJson());

        Assert.Equal(prepared.Fitted[1].Curve("knots"), loaded.Fitted[1].Curve("knots"));
        Assert.Single(prepared.Fitted[1].Curves);
    }

    [Fact]
    public void WhatNoFitEverLearned_CannotBeReadBackAsACurve()
    {
        Assert.Throws<InvalidOperationException>(() => new FittedStepValues().Curve("knots"));
    }

    [Fact]
    public void ThePowerScaleReshapesALopsidedColumn()
    {
        // A column that leans hard to one side: most values small, a few very large.
        var table = Read("a\n1\n1\n1\n2\n2\n3\n5\n8\n13\n40\n");
        var step = new NormaliseStep("a", Scale.Power);

        var learned = step.Fit(table, AllTraining(table));
        step.ApplyTo(table, learned);

        var scaled = (Column<double>)table["a"];
        var values = Enumerable.Range(0, 10).Select(row => scaled[row]!.Value).ToArray();

        double Skew(double[] numbers)
        {
            var mean = numbers.Average();
            var spread = Math.Sqrt(numbers.Average(value => (value - mean) * (value - mean)));

            return numbers.Average(value => Math.Pow((value - mean) / spread, 3));
        }

        Assert.InRange(learned.Number("lambda"), -2, 2);
        Assert.Equal(0, values.Average(), 6);
        Assert.True(Math.Abs(Skew(values)) < Math.Abs(Skew([1, 1, 1, 2, 2, 3, 5, 8, 13, 40])),
            "the reshaped column should lean less than the one it came from");
    }

    [Fact]
    public void ThePowerScaleTakesNegativeValuesToo()
    {
        var table = Read("a\n-5\n-1\n0\n1\n20\n");
        var step = new NormaliseStep("a", Scale.Power);

        step.ApplyTo(table, step.Fit(table, AllTraining(table)));

        Assert.All(Enumerable.Range(0, 5), row => Assert.False(double.IsNaN(((Column<double>)table["a"])[row]!.Value)));
    }

    [Fact]
    public void BothNewScalesSurviveTheFile()
    {
        foreach (var scale in new[] { Scale.Quantile, Scale.Power })
        {
            var declaration = new PipelineDeclaration([
                new SplitAtRandomStep(new SplitShares(0.7, 0.15, 0.15), 1),
                new NormaliseStep("a", scale),
            ]);

            Assert.Equal(declaration, PipelineDeclaration.FromJson(declaration.ToJson()));
        }
    }
}
