// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The data as it stands after any number of steps — what the grid under a block shows — with every row
/// saying where it stands: in the part the split below will put it in, dropped before it gets there, or
/// undivided because nothing divides it. A range or a profile drawn over the rows above the split used to
/// read every row as a training row, 891 of 891; it reads the rows the split below will train on.
/// </summary>
public class ViewTests
{
    private static PipelineBuilder Passengers() =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp", "parch").Optional("age", ColumnKind.Number));

    [Fact]
    public void AViewAboveTheSplit_KnowsTheTrainingRowsOfTheSplitBelowIt()
    {
        var pipeline = Passengers()
            .AddFeature("family", "sibsp", Arithmetic.Plus, "parch")
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .Build();

        var view = pipeline.ViewAt(3);
        var run = pipeline.Run();

        Assert.Equal(891, view.Table.RowCount);
        Assert.True(view.Table.Has("family"));
        Assert.False(view.Table.Has("age_was_missing"));
        Assert.Equal(run.CountIn(Part.Train), view.Standings.Count(standing => standing == Standing.Train));
        Assert.Equal(Standing.Train, view.Measured);
        Assert.Equal(
            run.Table.TrainingValues("sibsp", run.Parts).Finite,
            view.MeasuredValues("sibsp").Finite);
    }

    [Fact]
    public void AViewIsKeyedByWhereItStands_AndEveryStepItIsWorkedOutFrom_DownToTheSplitWhenTheSplitStandsBelowIt()
    {
        // read, declare, feature, split, fill: the split at 3 places the rows at every step above it.
        FittingBuilder Built(int seed, FillStrategy fill) => Passengers()
            .AddFeature("family", "sibsp", Arithmetic.Plus, "parch")
            .SplitAtRandom(0.70, 0.15, seed)
            .FillMissing("age", fill);

        static string[] Keys(PipelineDeclaration declaration) =>
            [.. Enumerable.Range(0, declaration.Steps.Count).Select(declaration.ViewKeyAt)];

        var one = Built(1, With.Median).Declaration;
        var reseeded = Built(2, With.Median).Declaration;
        var refilled = Built(1, With.Mean).Declaration;

        // Every step shows rows of its own, and the same steps give the same keys wherever they are worked out.
        Assert.Equal(5, Keys(one).Distinct().Count());
        Assert.Equal(Keys(one), Keys(Built(1, With.Median).Declaration));
        Assert.All(Keys(one), key => Assert.Matches("^[0-9a-f]{64}$", key));

        // The split places every row above it, so every view down to it changes with it; none changes with a step
        // below it.
        Assert.All(Enumerable.Range(0, 4), at => Assert.NotEqual(one.ViewKeyAt(at), reseeded.ViewKeyAt(at)));
        Assert.Equal(Keys(one)[..4], Keys(refilled)[..4]);
        Assert.NotEqual(one.ViewKeyAt(4), refilled.ViewKeyAt(4));

        // Nothing divides these rows, so a step added below changes no view above it.
        var undivided = Passengers().AddFeature("family", "sibsp", Arithmetic.Plus, "parch").Declaration;
        var longer = Passengers().AddFeature("family", "sibsp", Arithmetic.Plus, "parch").AddFeature("twice", "family", Arithmetic.Plus, "family").Declaration;

        Assert.Equal(Keys(undivided), Keys(longer)[..3]);

        Assert.Throws<ArgumentOutOfRangeException>(() => one.ViewKeyAt(5));
        Assert.Throws<ArgumentOutOfRangeException>(() => one.ViewKeyAt(-1));

        // Two layers, one rule: the rows differ where the keys do.
        Assert.NotEqual(Built(1, With.Median).Build().ViewAt(3).Standings, Built(2, With.Median).Build().ViewAt(3).Standings);
    }

    [Fact]
    public void AViewChecksTheStepsItIsWorkedOutFrom_AndARunChecksEveryStep()
    {
        // A column the schema lets be absent, and the rows lack, is refused where a step reads it: at that step's
        // own view and at the run, never at a view above it that the step does not work out. So two declarations
        // whose view has the same key show the same rows there.
        const string csv = "t,a\n1,1\n2,2\n3,3\n4,4\n5,5\n6,6\n7,7\n8,8\n9,9\n10,10\n";

        FittingBuilder Divided() => Pdd.Create()
            .Read(CsvRowSource.FromText(csv), "rows")
            .Declare(schema => schema.Integer("t").Number("a").Optional("b", ColumnKind.Number))
            .SplitByTime("t", 0.70, 0.15);

        var alone = Divided().Declaration;
        var readingB = Divided().FillMissing("b", With.Median).Declaration;

        Assert.Equal(alone.ViewKeyAt(1), readingB.ViewKeyAt(1));
        Assert.Equal(10, new Pipeline(alone, CsvRowSource.FromText(csv)).ViewAt(2).Table.RowCount);
        Assert.Equal(10, new Pipeline(readingB, CsvRowSource.FromText(csv)).ViewAt(2).Table.RowCount);

        var atItsOwnView = Assert.Throws<DeclarationException>(() => new Pipeline(readingB, CsvRowSource.FromText(csv)).ViewAt(4));
        var atTheRun = Assert.Throws<DeclarationException>(() => new Pipeline(readingB, CsvRowSource.FromText(csv)).Run());

        Assert.Equal("fill.missing", Assert.Single(atItsOwnView.Faults).Verb);
        Assert.Equal("fill.missing", Assert.Single(atTheRun.Faults).Verb);
    }

    [Fact]
    public void AView_KnowsTheCategoriesItsMeasuredRowsHold_ByTheRuleAnEncoderLearnsThemBy()
    {
        var pipeline = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Category("embarked"))
            .SplitStratified("survived", 0.70, 0.15)
            .Build();
        var run = pipeline.Run();

        var view = pipeline.ViewAt(3);

        // What an encoder learns here, from the training rows, the gaps left out, in one order.
        Assert.Equal(new EncodeStep("embarked").Fit(run.Table, run.Parts).List("categories"), view.MeasuredCategories("embarked"));
        Assert.Equal(["C", "Q", "S"], view.MeasuredCategories("embarked"));

        // Nothing divides these rows, so every one of them is measured.
        var undivided = Pdd.Create().Read(CsvRowSource.FromText("kind\nb\na\n\nb\n"), "rows").Declare(schema => schema.Category("kind")).Build();

        Assert.Equal(["a", "b"], undivided.ViewAt(2).MeasuredCategories("kind"));
    }

    [Fact]
    public void AViewWithNoSplitAnywhere_IsUndivided_NeverTraining()
    {
        var view = Passengers().AddFeature("family", "sibsp", Arithmetic.Plus, "parch").Build().ViewAt(3);

        Assert.All(view.Standings, standing => Assert.Equal(Standing.Undivided, standing));
        Assert.Equal(Standing.Undivided, view.Measured);
        Assert.Equal(891, view.MeasuredValues("family").Rows);
    }

    [Fact]
    public void ARowDroppedBetweenTheViewAndTheSplit_StandsAsDropped_AndIsMeasuredNowhere()
    {
        // The gaps in age are dropped after the view and before the split, so they are in the grid and in no
        // part, and nothing measured at the view counts them.
        var pipeline = Passengers()
            .DropGaps("age")
            .SplitStratified("survived", 0.70, 0.15)
            .Build();

        var view = pipeline.ViewAt(2);

        Assert.Equal(891, view.Table.RowCount);
        Assert.Equal(177, view.Standings.Count(standing => standing == Standing.Dropped));
        Assert.Equal(0, view.MeasuredValues("age").Gaps);
    }

    [Fact]
    public void AViewBelowTheSplit_IsWhatTheRunHoldsThere()
    {
        var pipeline = Passengers().SplitStratified("survived", 0.70, 0.15).FillMissing("age", With.Median).Build();
        var run = pipeline.Run();

        var view = pipeline.ViewAt(pipeline.Declaration.Steps.Count);

        Assert.Equal(run.Parts.Select(part => part.ToString()), view.Standings.Select(standing => standing.ToString()));
        Assert.Equal(run.Table.NumbersOf("age"), view.Table.NumbersOf("age"));
    }

    [Fact]
    public void AViewOfTheSourceAlone_IsItsRowsAsText()
    {
        // Before the schema says which columns take part and what they hold, the rows are what the file says:
        // every column, as words.
        var pipeline = Passengers().SplitStratified("survived", 0.70, 0.15).Build();

        var view = pipeline.ViewAt(1);

        Assert.Equal(15, view.Table.Columns.Count);
        Assert.All(view.Table.Columns, column => Assert.Equal(ColumnKind.Text, column.Kind));
        Assert.Equal(pipeline.Run().CountIn(Part.Train), view.Standings.Count(standing => standing == Standing.Train));
        Assert.Equal(Standing.Undivided, new Pipeline(new PipelineDeclaration([new ReadCsvStep(Repository.Data("titanic.csv"))])).ViewAt(1).Measured);
    }

    [Fact]
    public void AViewAtNoStepsOrPastTheEnd_IsRefused()
    {
        var pipeline = Passengers().Build();

        Assert.Throws<ArgumentOutOfRangeException>(() => pipeline.ViewAt(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => pipeline.ViewAt(3));
    }

    [Fact]
    public void AViewAtAStepThatProducesEvidence_CarriesWhatItProduced_OverTheTrainingRowsOfTheSplitBelow()
    {
        // What a notebook block shows for a profile is what the walk to that block measured, and the walk goes on
        // to the split below so that it measures the training rows alone.
        var pipeline = Passengers().Profile("age").SplitStratified("survived", 0.70, 0.15).Build();

        var profile = Assert.IsType<DataProfile>(pipeline.ViewAt(3).Evidence[2]);

        Assert.Equal(Standing.Train, profile.Over);
        Assert.Equal(pipeline.Run().CountIn(Part.Train), profile.Rows);
        Assert.Equal(profile.Rows, ((DataProfile)pipeline.ViewAt(1).Evidence[2]).Rows);
        Assert.Empty(Passengers().Build().ViewAt(2).Evidence);
    }
}
