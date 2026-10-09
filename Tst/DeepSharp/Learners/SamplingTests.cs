// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A sampler decides which candidate the next trial tries. The one that ships draws every number from the seed and the trial's
/// place alone, so a trial gets the same candidate whatever ran before it, a search that stops at twelve trials is the first
/// twelve of one that goes on to thirty, and a dimension added later leaves the values of the others where they were.
/// </summary>
public class SamplingTests
{
    private static SearchSpace Space() => new([new LogRange("rate", 1e-4, 1e-1), new WholeRange("units", 2, 32), new Choices("activation", ["relu", "tanh"])]);

    [Fact]
    public void TheSameSeedAndTrial_GiveTheSameCandidate_OnEverySampler()
    {
        var space = Space();

        Assert.Equal(new RandomSampler(11).Suggest(space, 4, []), new RandomSampler(11).Suggest(space, 4, []));
    }

    [Fact]
    public void AnotherTrial_OrAnotherSeed_GivesAnotherCandidate()
    {
        var space = Space();
        var sampler = new RandomSampler(11);

        Assert.NotEqual(sampler.Suggest(space, 4, []), sampler.Suggest(space, 5, []));
        Assert.NotEqual(sampler.Suggest(space, 4, []), new RandomSampler(12).Suggest(space, 4, []));
    }

    [Fact]
    public void WhatAsksFirstDoesNotShiftWhatAsksLater_ATrialIsTheSameWhateverTheSamplerWasAskedBefore()
    {
        var space = Space();
        var asked = new RandomSampler(11);

        for (var trial = 0; trial < 10; trial++)
        {
            _ = asked.Suggest(space, trial, []);
        }

        Assert.Equal(new RandomSampler(11).Suggest(space, 7, []), asked.Suggest(space, 7, []));
    }

    [Fact]
    public void ADimensionAdded_LeavesTheValuesOfTheOthersWhereTheyWere()
    {
        var narrow = new RandomSampler(11).Suggest(new SearchSpace([new LogRange("rate", 1e-4, 1e-1)]), 3, []);
        var wide = new RandomSampler(11).Suggest(Space(), 3, []);

        Assert.Equal(narrow.Number("rate"), wide.Number("rate"));
    }

    [Fact]
    public void ACandidate_HoldsAValueForEveryDimension_InTheOrderOfTheSpace()
    {
        var candidate = new RandomSampler(11).Suggest(Space(), 0, []);

        Assert.Equal(["rate", "units", "activation"], candidate.Settings.Select(setting => setting.Name));
        Assert.InRange(candidate.Number("rate"), 1e-4, 1e-1);
        Assert.InRange(candidate.Whole("units"), 2, 32);
        Assert.Contains(candidate.Choice("activation"), new[] { "relu", "tanh" });
    }

    [Fact]
    public void ATrialBelowNought_IsRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new RandomSampler(11).Suggest(Space(), -1, []));
    }

    [Fact]
    public void TheSeedIsKept_ForTheStudyToSayWhichSearchThisWas()
    {
        Assert.Equal(11, new RandomSampler(11).Seed);
    }
}
