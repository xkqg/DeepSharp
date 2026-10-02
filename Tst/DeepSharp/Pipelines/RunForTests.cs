// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using DeepSharp.Tests.Learners;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A pipeline is run for the learner that learns from it: a step that only scales a feature is left out for a learner
/// indifferent to scale, and an encoder hands each category over as its place to a learner that takes categories, learning
/// the list it learns for every learner. What is left out is decided from the declaration and the need alone. A step stays
/// wherever leaving it out would change more than what the learner does without: an answer's way back runs through it, a
/// step below reads what it read or wrote, or it refuses what it was not fitted on. A run of every step is the run it always
/// was.
/// </summary>
public class RunForTests
{
    private static Pipeline Titanic => WikiTitanic.In(WikiTitanic.DataFolder);

    [Fact]
    public void ARunForALearnerThatTakesEveryStep_IsTheRunOfEveryStep_ByteForByte()
    {
        var every = Titanic.Run();

        Assert.Empty(every.Skipped);

        foreach (var run in new[] { Needs.Numbers, Needs.OneScale }.Select(needs => Titanic.RunFor(needs)))
        {
            Assert.Empty(run.Skipped);
            Assert.Equal(every.ToJson(), run.ToJson());
        }
    }

    [Fact]
    public void ARunForALearnerIndifferentToScale_LeavesOutTheFourScales_AndOneThatTakesCategories_TheEncoderToo()
    {
        var noScale = Titanic.RunFor(Needs.NoScale);
        var categories = Titanic.RunFor(Needs.Categories);
        var handedIn = new Pipeline(WikiTitanic.Declaration).RunFor(CsvRowSource.FromText(File.ReadAllText(Repository.Data("titanic.csv"))), Needs.NoScale);

        Assert.Equal([5, 6, 7, 8], noScale.Skipped);
        Assert.Equal([4, 5, 6, 7, 8], categories.Skipped);
        Assert.Equal([5, 6, 7, 8], handedIn.Skipped);

        // Left out, a scale is not applied: the fares are in pounds, the dearest ticket 512.3292 as it was read.
        Assert.Equal(512.3292, noScale.Table.NumbersOf("fare").Max());
        Assert.Equal(1, Titanic.Run().Table.NumbersOf("fare").Max());

        // The same rows in the same parts, whatever the learner: the split, the fill and the categories are learned alike.
        Assert.Equal(Titanic.Run().Parts, categories.Parts);
    }

    [Fact]
    public void AScaleThatRefuses_OneOnTheAnswersWayBack_AndOneAStepBelowReads_AreTakenForEveryLearner()
    {
        // Leaving one of these out would change more than the learner does without: a row it would have refused, a
        // prediction brought back through a scale never applied, or what the step below it computed.
        Assert.Empty(LeftOut(fitting => fitting.Normalise("fare", Scale.MidRange, OutOfRange.Refuse).Target("survived"), Needs.NoScale));
        Assert.Empty(LeftOut(fitting => fitting.Normalise("fare", Scale.MidRange).Target("fare"), Needs.NoScale));
        Assert.Empty(LeftOut(fitting => fitting.Normalise("age", Scale.MidRange).ClipOutliers("age").Target("survived"), Needs.NoScale));
        Assert.Empty(LeftOut(fitting => fitting.Normalise("sibsp", Scale.Standard).Add(new AddFeatureStep("family", "sibsp", Arithmetic.Plus, "parch")).Target("survived"), Needs.NoScale));
        Assert.Empty(LeftOut(fitting => fitting.Encode("sex").Drop("sex_other").Target("survived"), Needs.Categories));

        // A step from elsewhere that says nothing of the columns it reads may read any of them.
        Assert.Empty(LeftOut(fitting => fitting.Normalise("fare", Scale.MidRange).Add(new ScaleByStep("parch", 2)).Target("survived"), Needs.NoScale));

        // Nothing below reads them, so these are left out.
        Assert.Equal([4], LeftOut(fitting => fitting.Normalise("fare", Scale.MidRange, OutOfRange.Clip).Target("survived"), Needs.NoScale));
        Assert.Equal([4], LeftOut(fitting => fitting.Encode("sex").Drop("sibsp").Target("survived"), Needs.Categories));
    }

    [Fact]
    public void ADeclaredOrdinalEncoder_HandsItsCategoriesOverAsPlaces_ToALearnerThatTakesCategories_AGapAsNoPlace()
    {
        // The port a passenger embarked at, written down as its place in the list: a gap is the first place when the encoder
        // is declared that way, and no place at all when the categories are handed over as themselves.
        FittingBuilder Ports() => Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Category("embarked"))
            .SplitStratified("survived", 0.70, 0.15)
            .EncodeCategories(As.Ordinal);

        var declared = Ports().Target("survived").Build().Run();
        var places = Ports().Target("survived").Build().RunFor(Needs.Categories);

        Assert.Equal([3], places.Skipped);
        Assert.Equal<double?>([0, 0], Gaps(declared));
        Assert.Equal<double?>([-1, -1], Gaps(places));
        Assert.Equal(
            declared.Table.NumbersOf("embarked").Where(place => place > 0),
            places.Table.NumbersOf("embarked").Where(place => place > 0));

        static double?[] Gaps(PreparedData run) =>
        [
            .. Enumerable.Range(0, run.Table.RowCount)
                .Where(row => run.Table.NumbersOf("embarked_was_missing")[row] == 1)
                .Select(row => run.Table.NumbersOf("embarked")[row]),
        ];
    }

    [Fact]
    public void TheShippedStepsSomeLearnersDoWithout_AreTheScaleAndTheTwoEncoders()
    {
        var catalog = StepCatalog.BuiltIn();

        string[] meetANeed =
        [
            .. catalog.Descriptions
                .Where(description => catalog.ReadStep(description.Template) is IMeetsANeed)
                .Select(description => description.Verb),
        ];

        Assert.Equal(["encode", "encode.categories", "normalise"], meetANeed);
    }

    [Fact]
    public void ANeedNoValueNames_IsRefusedByTheRun()
    {
        Assert.Equal("needs", Assert.Throws<ArgumentOutOfRangeException>(() => Titanic.RunFor((Needs)42)).ParamName);
    }

    [Fact]
    public void AStepThatOffersInItsPlaceAStepWrittenOtherwise_IsRefused_ForTheRunsFileNamesEveryStepAsDeclared()
    {
        var centred = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Number("fare"))
            .SplitStratified("survived", 0.70, 0.15)
            .Add(new CentreStep("fare"))
            .Target("survived")
            .Build();

        var refused = Assert.Throws<InvalidOperationException>(() => centred.RunFor(Needs.NoScale));

        Assert.Contains("Step 4, 'centre'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'learn.nothing'", refused.Message, StringComparison.Ordinal);
        Assert.Empty(centred.RunFor(Needs.Numbers).Skipped);
    }

    [Fact]
    public void ACategoryTheTrainingRowsNeverHeld_IsRefusedNamingTheRowAsItWasHandedIn()
    {
        // The two rows before it have no age and are dropped, so the third row handed in is the first one encoded.
        var prepared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Category("embarked").Optional("age", ColumnKind.Number))
            .DropGaps("age")
            .SplitStratified("survived", 0.70, 0.15)
            .EncodeCategories(As.OneHot, Unseen.Refuse)
            .Target("survived")
            .Build()
            .Run();
        var served = new InMemoryRowSource(["embarked", "age"], [["S", ""], ["S", ""], ["X", "30"]]);

        var refused = Assert.Throws<InvalidOperationException>(() => prepared.Served(served));

        Assert.Contains("Row 3 of 'embarked' holds 'X'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AScaleDeclaredWithoutItsFit_IsRefused_WhileARunThatLeftItOutHasNone()
    {
        var every = Titanic.Run();
        var withoutScales = every.Fitted.Where(entry => entry.Key is < 5 or > 8).ToDictionary(entry => entry.Key, entry => entry.Value);

        var refused = Assert.Throws<ArgumentException>(() => new PreparedData(every.Declaration, every.Table, every.Parts, withoutScales));
        var noScale = Titanic.RunFor(Needs.NoScale);

        Assert.StartsWith("Step 6, 'normalise': learns from the data, and nothing it learned is here.", refused.Message, StringComparison.Ordinal);
        Assert.Equal([2, 3, 4], noScale.Fitted.Keys.Order());
    }

    [Fact]
    public void ARowServedBehindARunThatLeftOutTheScales_IsHandedOverAsItWasInTraining()
    {
        // The passenger on line 7 of the list: no age, which the fill makes 29, and a fare of 8.4583.
        var noScale = Titanic.RunFor(Needs.NoScale);
        var line = WikiTitanic.Line(7);

        var served = noScale.Served(line, Needs.NoScale);
        var train = noScale.Batch(Part.Train, Needs.NoScale);
        var trained = train.Features[train.Keys!.ToList().IndexOf(Assert.Single(served.Keys!))];

        Assert.Equal(29, served.Features[0][served.FeatureNames.ToList().IndexOf("age")]);
        Assert.Equal(8.4583, served.Features[0][served.FeatureNames.ToList().IndexOf("fare")]);
        Assert.Equal(train.FeatureNames, served.FeatureNames);
        Assert.Equal(trained, served.Features[0]);
    }

    [Fact]
    public void PredictionsForRowsServedBehindARunThatLeftOutAScale_ComeBackThroughTheWayBackItKept()
    {
        // The fare is the answer, scaled on its way to the learner and brought back in pounds; the age is a feature, whose
        // scale a learner indifferent to scale does without.
        var fares = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("pclass").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitStratified("pclass", 0.70, 0.15)
            .FillMissing("age", With.Median)
            .Normalise("age", Scale.MidRange)
            .Normalise("fare", Scale.MinMax)
            .Target("fare")
            .Build()
            .RunFor(Needs.NoScale);
        var rows = new InMemoryRowSource(["pclass", "age"], [["3", "30"], ["1", ""]]);

        var served = fares.Served(rows, Needs.NoScale);
        var back = fares.BackToOriginal([[0.0], [1.0]], served, rows);
        var fare = fares.Fitted[5];

        Assert.Equal([4], fares.Skipped);
        Assert.Equal([30.0, fares.Fitted[3].Number("value")], served.Features.Select(row => row[served.FeatureNames.ToList().IndexOf("age")]));
        Assert.Equal(fare.Number("centre"), back[0][0]);
        Assert.Equal(fare.Number("centre") + fare.Number("spread"), back[1][0], 1e-9);
    }

    // What a run for a learner leaves out of the Titanic passengers, their age filled, with these steps below.
    [Fact]
    public void ALearnerNamedInTheDeclaration_SaysWhatTheRunIsMadeFor_AndLeavesTheSameStepsOut()
    {
        // The learner the pipeline names says what it needs, so the run leaves out exactly what a run asked for that need
        // leaves out. A step that names a learner says which columns it leaves behind, as every step that changes none
        // does; one that said nothing of them would make every scale look read from below and nothing could be left out.
        var declared = LeftOut(fitting => fitting.Normalise("fare", Scale.MidRange).Target("survived").Add(new LearnStumpStep(1)), Needs.NoScale);

        Assert.Equal([4], declared);
        Assert.Equal([4], LeftOut(fitting => fitting.Normalise("fare", Scale.MidRange).Target("survived"), Needs.NoScale));
    }

    [Fact]
    public void AStepThatNamesALearner_AndSaysNothingOfItsColumns_LeavesNothingToLeaveOut()
    {
        // The reason the learner step describes its columns: a step below that says nothing of them may read any of them,
        // so every scale above it is taken for every learner. Pinned here because the cost is silent — the run is correct
        // and simply does more than it had to.
        Assert.Empty(LeftOut(fitting => fitting.Normalise("fare", Scale.MidRange).Target("survived").Add(new LearnQuietlyStep()), Needs.NoScale));
    }

    private static IReadOnlyList<int> LeftOut(Func<FittingBuilder, FittingBuilder> below, Needs needs) =>
        below(Pdd.Create()
                .ReadCsv(Repository.Data("titanic.csv"))
                .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("sex").Optional("age", ColumnKind.Number).Number("fare"))
                .SplitStratified("survived", 0.70, 0.15)
                .FillMissing("age", With.Median))
            .Build()
            .RunFor(needs)
            .Skipped;
}
