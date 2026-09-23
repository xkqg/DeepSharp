// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What a fit learns from is the training rows of one column — and that was worked out four times, by four
/// steps, each with its own idea of a median and of a value that is not a number. The same column gave a
/// median of 5.5 to one step and 6 to another, and a min-max scale learned its middle from a not-a-number.
/// There is one set of training values now, every fit reads it, and a value that is not a finite number stops
/// a fit rather than sliding into what it learns.
/// </summary>
public class TrainingValuesTests
{
    private static Table Read(string csv)
    {
        var source = CsvRowSource.FromText(csv);

        return SchemaBinding.Bind(
            new DeclareStep([.. source.ColumnNames.Select(name => new ColumnDeclaration(name, ColumnKind.Number, true))]),
            source);
    }

    private static Part[] AllTraining(Table table) => [.. Enumerable.Repeat(Part.Train, table.RowCount)];

    private static PipelineBuilder Passengers() =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Optional("age", ColumnKind.Number).Number("fare"));

    [Fact]
    public void EveryFitLearnsFromTheSameTrainingValues()
    {
        // The value a fill learns as the median and the middle a robust scale learns are one number, because
        // they are one median of one set of training values.
        var filled = Passengers().SplitStratified("survived", 0.70, 0.15).FillMissing("age", With.Median).Build().Run();
        var scaled = Passengers().SplitStratified("survived", 0.70, 0.15).Normalise("age", Scale.Robust).Build().Run();

        var fill = filled.Fitted[3].Number("value");
        var centre = scaled.Fitted[3].Number("centre");
        var training = Passengers().SplitStratified("survived", 0.70, 0.15).Build().Run();

        Assert.Equal(fill, centre);
        Assert.Equal(fill, training.Table.TrainingValues("age", training.Parts).Median);
    }

    [Theory]
    [InlineData("normalise")]
    [InlineData("fill.missing")]
    [InlineData("outliers.clip")]
    public void AFitMeetingAValueThatIsNotAFiniteNumber_IsRefused_NamingTheStepThatDealsWithIt(string verb)
    {
        var table = Read("a\n1\nNaN\n3\n4\n");

        IFittedStep step = verb switch
        {
            "normalise" => new NormaliseStep("a", Scale.MinMax),
            "fill.missing" => FillMissingStep.Of("a", With.Mean),
            _ => new ClipOutliersStep("a"),
        };

        var refused = Assert.Throws<InvalidOperationException>(() => step.Fit(table, AllTraining(table)));

        Assert.Contains("'a'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("fill.nan", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AfterFillNaN_TheSameFitLearnsWhatItShould()
    {
        var prepared = Pdd.Create()
            .Read(CsvRowSource.FromText("a,b\n1,1\nNaN,2\n3,3\n4,4\n5,5\n6,6\n7,7\n8,8\n9,9\n10,10\n"), "ten rows")
            .Declare(schema => schema.Number("a", "b"))
            .SplitAtRandom(0.70, 0.15)
            .FillNaN("a", With.Median)
            .Normalise("a", Scale.MinMax)
            .Build()
            .Run();

        Assert.True(double.IsFinite(prepared.Fitted[4].Number("centre")));
    }

    [Theory]
    [InlineData("NaN", double.NaN)]
    [InlineData("Infinity", double.PositiveInfinity)]
    [InlineData("-Infinity", double.NegativeInfinity)]
    public void TheInvariantSpellingsOfNotANumberAndInfinity_AreReadAsWhatTheySay(string written, double read)
    {
        // Present in the file and parsed, so neither a gap nor an unreadable cell: a value that is not a
        // finite number, which fill.nan is the step for.
        var table = Read($"a\n{written}\n");

        Assert.Equal(read, ((Column<double>)table["a"])[0]);
    }

    [Theory]
    [InlineData("1e999")]
    [InlineData("-1e400")]
    public void ANumberTooLargeToHold_IsRefusedWhereItIsRead(string written)
    {
        // It is a finite number in the file and an infinity in memory: the value that reached the model would
        // not be the value somebody wrote.
        var refused = Assert.Throws<FormatException>(() => Read($"a\n1\n{written}\n"));

        Assert.Contains("Row 2, column 'a'", refused.Message, StringComparison.Ordinal);
        Assert.Contains(written, refused.Message, StringComparison.Ordinal);
        Assert.Contains("too large", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("inf")]
    [InlineData("∞")]
    [InlineData("nan")]
    [InlineData("infinity")]
    public void AnyOtherSpellingOfThem_IsNotANumber(string written)
    {
        var refused = Assert.Throws<FormatException>(() => Read($"a\n{written}\n"));

        Assert.Contains("is not a number", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EveryCountInTheFittedHalf_ComesFromTheTrainingRows()
    {
        // The fitted half is what the training rows taught, so the gaps a fill counts are the training rows'
        // gaps — 177 in the whole file, fewer among the rows a model learns from.
        var prepared = Passengers()
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .ClipOutliers("fare", Bounds.Iqr)
            .Build()
            .Run();

        var marker = (Column<double>)prepared.Table["age_was_missing"];
        var trainingGaps = Enumerable.Range(0, prepared.Table.RowCount).Count(row => prepared.Parts[row] == Part.Train && marker[row] == 1);
        var fare = prepared.Fitted[4];

        Assert.Equal(trainingGaps, prepared.Fitted[3].Number("gaps"));
        Assert.Equal(177, Enumerable.Range(0, prepared.Table.RowCount).Count(row => marker[row] == 1));
        Assert.Equal(133, trainingGaps);
        Assert.True(fare.Number("outside") > 0);
    }

    [Fact]
    public void WhatTheTrainingRowsHold_IsCountedAndMeasuredOnce()
    {
        var table = Read("a\n4\n\n1\nNaN\n3\n2\n9\n");
        var parts = new[] { Part.Train, Part.Train, Part.Train, Part.Train, Part.Train, Part.Train, Part.Test };

        var training = table.TrainingValues("a", parts);

        Assert.Equal([1.0, 2.0, 3.0, 4.0], training.Finite);
        Assert.Equal(1, training.Gaps);
        Assert.Equal(1, training.NotFinite);
        Assert.Equal(6, training.Rows);
        Assert.Equal(2.5, training.Mean);
        Assert.Equal(2.5, training.Median);
        Assert.Equal(1.75, training.Quantile(0.25));
        Assert.Equal(Math.Sqrt(1.25), training.StandardDeviation, 12);
        Assert.Equal("a", training.Column);
    }

    [Fact]
    public void TrainingValuesThatAreAllGaps_LearnNothing_AndSaySo()
    {
        var table = Read("a,b\n,1\n,2\n");
        var training = table.TrainingValues("a", AllTraining(table));

        var refused = Assert.Throws<InvalidOperationException>(() => training.Learnable("a scale"));

        Assert.Contains("Every training row of 'a' is a gap", refused.Message, StringComparison.Ordinal);
        Assert.Contains("a scale", refused.Message, StringComparison.Ordinal);
        Assert.Throws<InvalidOperationException>(() => training.Mean);
        Assert.Throws<InvalidOperationException>(() => training.Quantile(0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => table.TrainingValues("b", AllTraining(table)).Quantile(1.5));
        Assert.Throws<ArgumentException>(() => table.TrainingValues("a", [Part.Train]));
    }

    [Fact]
    public void AColumnIsReadAsNumbers_ByOneRule()
    {
        var table = new Table([
            new Column<bool>("flag", ColumnKind.Boolean, [true, false, null]),
            new Column<long>("count", ColumnKind.Integer, [3, null, 5]),
            new TextColumn("word", ["a", "b", "c"]),
        ]);

        Assert.Equal([1.0, 0.0, null], table.NumbersOf("flag"));
        Assert.Equal([3.0, null, 5.0], table.NumbersOf("count"));
        Assert.Throws<InvalidOperationException>(() => table.NumbersOf("word"));
    }
}
