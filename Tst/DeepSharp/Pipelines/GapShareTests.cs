// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Two answers to a column full of gaps besides filling it. A row with a gap can be dropped — above the split,
/// where every part loses it alike, never below it, where the parts would shrink without knowing. And a fill
/// can be given a share of gaps above which it will not fill: past that point a filled column is invention,
/// so the column goes and the column that says where the gaps were speaks for it. The share is measured on
/// the training rows, decided once when the pipeline is fitted, and replayed unchanged.
/// </summary>
public class GapShareTests
{
    private static PipelineBuilder Passengers() =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Optional("age", ColumnKind.Number).Number("fare"));

    [Fact]
    public void DroppingTheRowsWithGaps_TakesThemFromEveryPart_BeforeTheSplit()
    {
        var prepared = Passengers()
            .DropGaps("age")
            .SplitStratified("survived", 0.70, 0.15)
            .Build()
            .Run();

        Assert.Equal(891 - 177, prepared.Table.RowCount);
        Assert.DoesNotContain(Enumerable.Range(0, prepared.Table.RowCount), prepared.Table["age"].IsMissing);
        Assert.Equal(prepared.Table.RowCount, prepared.CountIn(Part.Train) + prepared.CountIn(Part.Validation) + prepared.CountIn(Part.Test));
    }

    [Fact]
    public void DroppingRowsBelowTheSplit_IsRefused()
    {
        var refused = Assert.Throws<DeclarationException>(() => new PipelineDeclaration([
            new ReadCsvStep("titanic.csv"),
            new DeclareStep([new ColumnDeclaration("survived", ColumnKind.Integer, false), new ColumnDeclaration("age", ColumnKind.Number, true)]),
            new SplitStratifiedStep("survived", new SplitShares(0.70, 0.15, 0.15), 1),
            new DropGapsStep(["age"]),
        ]));

        Assert.Equal("drop.gaps", Assert.Single(refused.Faults).Verb);
    }

    [Fact]
    public void AFillAboveTheShareItWasGiven_LeavesOnlyTheColumnThatSaysWhereTheGapsWere()
    {
        // A fifth of the ages are gaps, above the tenth this fill allows: the ages go, and the column that
        // says which passengers had none speaks for them.
        var prepared = Passengers()
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median, refuseAbove: 0.10)
            .Build()
            .Run();

        var learned = prepared.Fitted[3];

        Assert.False(prepared.Table.Has("age"));
        Assert.True(prepared.Table.Has("age_was_missing"));
        Assert.Equal(0, learned.Number("filled"));
        Assert.Equal(133.0 / prepared.CountIn(Part.Train), learned.Number("share"), 12);
        Assert.False(learned.Numbers.ContainsKey("value"));
    }

    [Fact]
    public void AFillAtOrBelowTheShare_Fills()
    {
        var prepared = Passengers()
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median, refuseAbove: 0.50)
            .Build()
            .Run();

        Assert.True(prepared.Table.Has("age"));
        Assert.Equal(1, prepared.Fitted[3].Number("filled"));
        Assert.DoesNotContain(Enumerable.Range(0, prepared.Table.RowCount), prepared.Table["age"].IsMissing);
    }

    [Fact]
    public void WhatTheFitDecided_IsReplayedUnchanged()
    {
        // Rows served later without a single gap in them still meet the pipeline the model was trained with:
        // the ages were left out then, so they are left out now.
        var trained = Passengers()
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median, refuseAbove: 0.10)
            .Build()
            .Run();

        var served = trained.Replay(new InMemoryRowSource(["survived", "age", "fare"], [["1", "30", "7.25"]]));

        Assert.False(served.Has("age"));
        Assert.Equal(0, ((Column<double>)served["age_was_missing"])[0]);
    }

    [Fact]
    public void AStepReadingAColumnItsFillReplaced_IsRefusedWithTheReason()
    {
        // Whether the fill keeps the column is known only once it has seen the training rows, so the
        // declaration lets a later step read it; the run says why it cannot.
        var pipeline = Passengers()
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median, refuseAbove: 0.10)
            .Normalise("age")
            .Build();

        var refused = Assert.Throws<InvalidOperationException>(() => pipeline.Run());

        Assert.Contains("'age'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("normalise", refused.Message, StringComparison.Ordinal);
        Assert.Contains("fill.missing", refused.Message, StringComparison.Ordinal);
        Assert.Contains("age_was_missing", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AStepReadingAColumnNoStepLeftHere_IsRefusedWhenItRuns()
    {
        // The encoder's columns are known by the start of their names until it has seen the training rows; a
        // category the training rows never held is not among them.
        var pipeline = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Category("sex"))
            .SplitStratified("survived", 0.70, 0.15)
            .Encode("sex", unseen: Unseen.Refuse)
            .Normalise("sex_unknown")
            .Build();

        var refused = Assert.Throws<InvalidOperationException>(() => pipeline.Run());

        Assert.Contains("'sex_unknown'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheShareAndTheDrop_SurviveTheFile()
    {
        var declaration = Passengers()
            .DropGaps("fare")
            .SplitStratified("survived", 0.70, 0.15)
            .FillMissing("age", With.Median, refuseAbove: 0.25)
            .Declaration;

        var returned = PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn());

        Assert.Equal(declaration, returned);
        Assert.Equal(0.25, ((FillMissingStep)returned.Steps[4]).RefuseAbove);
        Assert.Equal(["fare"], ((DropGapsStep)returned.Steps[2]).Columns);

        using var written = JsonDocument.Parse(Pdd.Create().ReadCsv("x.csv").Declare(schema => schema.Number("a")).SplitAtRandom(0.7).FillMissing("a", With.Mean).Declaration.ToJson());

        Assert.False(written.RootElement.GetProperty("declaration")[3].TryGetProperty("refuseAbove", out _));
    }

    [Fact]
    public void TwoDropsOfTheSameColumns_AreTheSameStep()
    {
        var drop = new DropGapsStep(["age", "fare"]);

        Assert.Equal(drop, new DropGapsStep(["age", "fare"]));
        Assert.Equal(drop.GetHashCode(), new DropGapsStep(["age", "fare"]).GetHashCode());
        Assert.NotEqual(drop, new DropGapsStep(["fare", "age"]));
        Assert.False(drop.Equals(null));
        Assert.Throws<ArgumentNullException>(() => new DropGapsStep(null!));
        Assert.Throws<ArgumentException>(() => new DropGapsStep([]));
        Assert.Throws<ArgumentNullException>(() => drop.RowsToKeep(null!));
    }

    [Theory]
    [InlineData(-0.1)]
    [InlineData(1.5)]
    [InlineData(double.NaN)]
    public void AShareIsAShare(double share)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => FillMissingStep.Of("age", With.Median, share));
    }
}
