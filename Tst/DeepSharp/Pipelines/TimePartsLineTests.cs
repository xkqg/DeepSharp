// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// One line says which pieces are taken out of many moments, and the two verbs that take them share it.
/// </summary>
/// <remarks>
/// What a column is asked for here is a set of pieces rather than one kind, so a method per piece would write a step per
/// piece, which is another declaration than the verb for one column writes. The line names the pieces once and then the
/// columns they come from; what reaches the declaration is what it always was: one step a column. Pieces as groups and
/// pieces as numbers are two verbs that choose from one vocabulary, so it is written once.
/// </remarks>
public class TimePartsLineTests
{
    private static PipelineBuilder Moments() =>
        Pdd.Create()
            .ReadCsv("rides.csv")
            .Declare(schema => schema.Timestamp("start", "end", "created"));

    private static TimePartsStep[] Taken(PipelineBuilder chain) =>
        [.. chain.Declaration.Steps.OfType<TimePartsStep>()];

    [Fact]
    public void OneLine_TakesTheNamedPiecesOutOfEachColumnItNames()
    {
        var taken = Taken(Moments()
            .TimeParts(parts => parts
                .Taking(TimePart.Month, TimePart.Hour)
                .Of("start", "end")));

        Assert.Equal(["start", "end"], taken.Select(step => step.Column));
        Assert.All(taken, step => Assert.Equal([TimePart.Month, TimePart.Hour], step.Parts));
        Assert.All(taken, step => Assert.True(step.AsCategories));
    }

    [Fact]
    public void ALineThatNamesMoreThanOneGroup_KeepsTheOrderItWasWritten()
    {
        var taken = Taken(Moments()
            .TimeParts(parts => parts
                .Taking(TimePart.Month).Of("start")
                .Taking(TimePart.Season, TimePart.Quarter).Of("end", "created")));

        Assert.Equal(["start", "end", "created"], taken.Select(step => step.Column));
        Assert.Equal([TimePart.Month], taken[0].Parts);
        Assert.Equal([TimePart.Season, TimePart.Quarter], taken[1].Parts);
        Assert.Equal([TimePart.Season, TimePart.Quarter], taken[2].Parts);
    }

    [Fact]
    public void TheLineForNumbers_TakesThePiecesAsNumbersAndNotAsGroups()
    {
        var taken = Taken(Moments()
            .TimePartsAsNumbers(parts => parts
                .Taking(TimePart.Year)
                .Of("start", "created")));

        Assert.Equal(["start", "created"], taken.Select(step => step.Column));
        Assert.All(taken, step => Assert.Equal([TimePart.Year], step.Parts));
        Assert.All(taken, step => Assert.False(step.AsCategories));
    }

    [Fact]
    public void TheOneLineAndTheLinePerColumn_DeclareTheSameSteps_ForBothVerbs()
    {
        // The door is the only thing that changes: what the declaration holds is step for step what it held.
        var one = Moments()
            .TimeParts(parts => parts.Taking(TimePart.Month, TimePart.Hour).Of("start", "end"))
            .TimePartsAsNumbers(parts => parts.Taking(TimePart.Year).Of("created"))
            .Declaration;

        var each = Moments()
            .TimeParts("start", TimePart.Month, TimePart.Hour)
            .TimeParts("end", TimePart.Month, TimePart.Hour)
            .TimePartsAsNumbers("created", TimePart.Year)
            .Declaration;

        Assert.Equal(each.Steps, one.Steps);
        Assert.Equal(each.ToJson(), one.ToJson());
    }

    [Fact]
    public void TheTwoVerbs_ShareTheVocabularyAndNothingElse()
    {
        // One way of naming the pieces, written once: both lines offer it, and each makes its own step.
        var kinds = (Type line) => line.GetMethods()
            .Where(method => method.DeclaringType == line.BaseType)
            .Select(method => method.Name)
            .Order(StringComparer.Ordinal)
            .ToArray();

        Assert.Equal(kinds(typeof(CategoryPartLine)), kinds(typeof(NumberPartLine)));
        Assert.Equal(["Taking"], kinds(typeof(CategoryPartLine)));
    }

    [Fact]
    public void WhetherThePiecesAreGroupsOrNumbers_IsTheVerbAndNotAFlagOnTheLine()
    {
        // Said by choosing the verb, as it is for the verb of one column, so no method of either line takes it.
        Type[] offering = [typeof(CategoryPartLine), typeof(NumberPartLine), typeof(TimePartsTaken<CategoryPartLine>)];

        var methods = offering.SelectMany(line => line.GetMethods()).ToArray();

        Assert.NotEmpty(methods);
        Assert.DoesNotContain(
            methods,
            method => method.GetParameters().Any(parameter => parameter.ParameterType == typeof(bool)));
    }

    [Fact]
    public void AnEmptyLine_IsRefusedWhereItIsWritten_WhateverItNamedBefore()
    {
        // A line that takes nothing from anything is a line somebody meant to finish: with no group at all, with pieces
        // and no column to take them from, and with columns named by an empty list.
        Assert.Throws<ArgumentException>(() => Moments().TimeParts(parts => { }));
        Assert.Throws<ArgumentException>(() => Moments().TimePartsAsNumbers(parts => { }));
        Assert.Throws<ArgumentException>(() => Moments().TimeParts(parts => parts.Taking(TimePart.Month)));
        Assert.Throws<ArgumentException>(() => Moments().TimeParts(parts => parts.Taking(TimePart.Month).Of()));
    }

    [Fact]
    public void AGroupThatTakesNoPiece_IsRefusedAsTheVerbForOneColumnRefusesIt()
    {
        Assert.Throws<ArgumentException>(() => Moments().TimeParts(parts => parts.Taking().Of("start")));
        Assert.Throws<ArgumentException>(() => Moments().TimePartsAsNumbers(parts => parts.Taking().Of("start")));
    }

    [Fact]
    public void PiecesOrColumnsThatAreMissing_AreRefused()
    {
        Assert.Throws<ArgumentNullException>(() => Moments().TimeParts(parts => parts.Taking(null!)));
        Assert.Throws<ArgumentNullException>(() => Moments().TimeParts(parts => parts.Taking(TimePart.Month).Of(null!)));
    }
}
