// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using DeepSharp.Tests.Learners;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A learner is handed only a run it can take: one that left out a step the learner needs is refused, each such step named,
/// and a run of every step goes to every learner. A feature is on one scale when the steps the run took land it there. A
/// learner that takes categories is handed each as its place in the list the training rows held — the place kept for one they
/// never held, no place for a gap — and the batch names each such feature with its list, served rows as much as training rows.
/// </summary>
public class HandoverForNeedsTests
{
    private static Pipeline Titanic => WikiTitanic.In(WikiTitanic.DataFolder);

    [Fact]
    public void ANetwork_IsRefusedARunThatLeftOutItsScalesOrItsEncoder_NamingEachStep()
    {
        var scales = Assert.Throws<InvalidOperationException>(() => Titanic.RunFor(Needs.NoScale).Batch(Part.Train, Needs.OneScale));
        var encoder = Assert.Throws<InvalidOperationException>(() => Titanic.RunFor(Needs.Categories).Batch(Part.Train, Needs.OneScale));

        Assert.Contains("Needs.OneScale", scales.Message, StringComparison.Ordinal);
        Assert.Contains("step 6, 'normalise'; step 7, 'normalise'; step 8, 'normalise'; step 9, 'normalise'", scales.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("encode.categories", scales.Message, StringComparison.Ordinal);
        Assert.Contains("step 5, 'encode.categories'; step 6, 'normalise'", encoder.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ALearnerOfNumbersOrOfAnySize_IsRefusedCategoriesHandedOverAsPlaces()
    {
        var categories = Titanic.RunFor(Needs.Categories);

        var numbers = Assert.Throws<InvalidOperationException>(() => categories.Batch(Part.Train, Needs.Numbers));
        var anySize = Assert.Throws<InvalidOperationException>(() => categories.Served(WikiTitanic.Line(7), Needs.NoScale));

        Assert.Contains("step 5, 'encode.categories'; step 6, 'normalise'", numbers.Message, StringComparison.Ordinal);
        Assert.Contains("step 5, 'encode.categories'.", anySize.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("normalise", anySize.Message, StringComparison.Ordinal);

        // Without saying what it needs, a learner is handed a batch for numbers, and so is refused the same.
        Assert.Contains("encode.categories", Assert.Throws<InvalidOperationException>(() => categories.Batch(Part.Train)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ARunOfEveryStep_GoesToEveryLearner_AndOneWithoutScales_ToALearnerThatTakesCategories()
    {
        var every = Titanic.Run();

        foreach (var needs in Enum.GetValues<Needs>())
        {
            Assert.Equal(623, every.Batch(Part.Train, needs).RowCount);
        }

        Assert.Empty(every.Batch(Part.Train, Needs.Categories).Categories!);
        Assert.Empty(Titanic.RunFor(Needs.NoScale).Batch(Part.Train, Needs.Categories).Categories!);
    }

    [Fact]
    public void AFeatureIsOnOneScale_WhenTheStepsTheRunTookLandItThere_NotTheStepsDeclared()
    {
        // A step every learner does without, that says it lands its column between minus one and one: a run for a learner on
        // one scale leaves it out, and the column lands nowhere said.
        var squashed = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Number("fare"))
            .SplitStratified("survived", 0.70, 0.15)
            .Add(new SquashStep("fare"))
            .Target("survived")
            .Build();

        var left = squashed.RunFor(Needs.OneScale);
        var refused = Assert.Throws<InvalidOperationException>(() => left.Batch(Part.Train, Needs.OneScale));

        Assert.Equal([3], left.Skipped);
        Assert.Contains("'fare'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("between minus one and one", refused.Message, StringComparison.Ordinal);
        Assert.Equal(623, squashed.Run().Batch(Part.Train, Needs.OneScale).RowCount);
    }

    [Fact]
    public void ALearnerThatTakesCategories_IsHandedEachAsItsPlace_AndTheBatchNamesEachWithTheCategoriesTheTrainingRowsHeld()
    {
        var categories = Titanic.RunFor(Needs.Categories);

        var train = categories.Batch(Part.Train, Needs.Categories);
        var served = categories.Served(WikiTitanic.Line(7), Needs.Categories);
        var passenger = train.Features[train.Keys!.ToList().IndexOf(Assert.Single(served.Keys!))];

        Assert.Equal(["pclass", "sex"], train.Categories!.Keys.Order());
        Assert.Equal(["1", "2", "3"], train.Categories["pclass"]);
        Assert.Equal(["female", "male"], train.Categories["sex"]);
        Assert.Equal(9, train.Width);

        // The passenger on line 7 of the list: in third class, the third place, and a man, the second.
        Assert.Equal(2, passenger[At(train.FeatureNames, "pclass")]);
        Assert.Equal(1, passenger[At(train.FeatureNames, "sex")]);
        Assert.Equal(passenger, served.Features[0]);
        Assert.Equal(train.Categories, served.Categories);
    }

    [Fact]
    public void AClassTheTrainingRowsNeverHeld_IsServedAtThePlaceKeptForIt_AndAGapAtNoPlace_WithItsMark()
    {
        var categories = Titanic.RunFor(Needs.Categories);
        var rows = new InMemoryRowSource(
            ["pclass", "sex", "age", "sibsp", "parch", "fare"],
            [["4", "male", "22", "1", "0", "7.25"], ["1", "", "30", "0", "0", "80"]]);

        var served = categories.Served(rows, Needs.Categories);

        Assert.Equal(3, served.Features[0][At(served.FeatureNames, "pclass")]);
        Assert.Equal(-1, served.Features[1][At(served.FeatureNames, "sex")]);
        Assert.Equal(1, served.Features[1][At(served.FeatureNames, "sex_was_missing")]);
        Assert.Equal(0, served.Features[0][At(served.FeatureNames, "sex_was_missing")]);
    }

    [Fact]
    public void ACategoryTheTrainingRowsNeverHeld_HandedOverAsAPlaceThatIsRefused_IsNamedAtTheRowAsItWasHandedIn()
    {
        // The two rows before it have no age and are dropped, so the third row handed in is the first one encoded.
        var places = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Category("embarked").Optional("age", ColumnKind.Number))
            .DropGaps("age")
            .SplitStratified("survived", 0.70, 0.15)
            .EncodeCategories(As.OneHot, Unseen.Refuse)
            .Target("survived")
            .Build()
            .RunFor(Needs.Categories);
        var served = new InMemoryRowSource(["embarked", "age"], [["S", ""], ["S", ""], ["X", "30"]]);

        var refused = Assert.Throws<InvalidOperationException>(() => places.Served(served, Needs.Categories));

        Assert.Contains("Row 3 of 'embarked' holds 'X'", refused.Message, StringComparison.Ordinal);
    }

    private static int At(IReadOnlyList<string> names, string name) => names.ToList().IndexOf(name);
}
