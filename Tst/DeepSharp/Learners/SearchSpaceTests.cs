// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;
using DeepSharp.Networks;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// What a search may vary: a number between two bounds, a number between two bounds that spreads across decades, a whole
/// number, one of a few words. A dimension draws its value from numbers handed to it and nothing else, so the same numbers
/// give the same value on every machine, and the settings a draw gives are read by name.
/// </summary>
public class SearchSpaceTests
{
    private static Draws DrawsOf(int trial) => new RandomStream(20261009).Draw("test", trial, 0);

    [Fact]
    public void ASpace_IsAListOfItsDimensions_InTheOrderTheyWereGiven()
    {
        var space = new SearchSpace([new NumberRange("rate", 0, 1), new WholeRange("units", 1, 5), new Choices("activation", ["relu", "tanh"])]);

        Assert.Equal(3, space.Count);
        Assert.Equal("units", space[1].Name);
        Assert.Equal(["rate", "units", "activation"], space.Select(dimension => dimension.Name));
        Assert.Equal(3, ((System.Collections.IEnumerable)space).Cast<Dimension>().Count());
    }

    [Fact]
    public void ASpaceWithTwoDimensionsOfOneName_IsRefused_NamingIt()
    {
        var refused = Assert.Throws<ArgumentException>(() => new SearchSpace([new NumberRange("rate", 0, 1), new WholeRange("rate", 1, 5)]));

        Assert.Contains("'rate'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ASpaceOfNoDimensions_IsRefused_ForThereIsNothingToVary()
    {
        Assert.Contains("nothing to vary", Assert.Throws<ArgumentException>(() => new SearchSpace([])).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ADimensionNeedsAName()
    {
        Assert.Throws<ArgumentException>(() => new NumberRange(" ", 0, 1));
        Assert.Throws<ArgumentException>(() => new Choices("", ["a"]));
    }

    [Theory]
    [InlineData(5, 5)]
    [InlineData(6, 5)]
    [InlineData(double.NaN, 1)]
    [InlineData(0, double.PositiveInfinity)]
    public void ANumberRangeThatIsNotARange_IsRefused(double low, double high)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new NumberRange("rate", low, high));
    }

    [Theory]
    [InlineData(0, 1)]
    [InlineData(-1, 1)]
    [InlineData(2, 1)]
    [InlineData(1, 1)]
    public void ALogRangeNeedsBoundsAboveNoughtThatAreARange(double low, double high)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new LogRange("rate", low, high));
    }

    [Fact]
    public void AWholeRangeThatIsNotARange_IsRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new WholeRange("units", 5, 5));
        Assert.Throws<ArgumentOutOfRangeException>(() => new WholeRange("units", 9, 2));
    }

    [Fact]
    public void ChoicesNeedAtLeastTwoDifferentWords()
    {
        Assert.Throws<ArgumentException>(() => new Choices("optimizer", []));
        Assert.Throws<ArgumentException>(() => new Choices("optimizer", ["adam"]));
        Assert.Throws<ArgumentException>(() => new Choices("optimizer", ["adam", "adam"]));
        Assert.Throws<ArgumentException>(() => new Choices("optimizer", ["adam", " "]));
    }

    [Fact]
    public void ANumberRange_DrawsInsideItsBounds_AndFromBothHalves()
    {
        var range = new NumberRange("dropout", 0.2, 0.6);
        var values = Enumerable.Range(0, 400).Select(trial => range.Draw(DrawsOf(trial)).Number).ToArray();

        Assert.All(values, value => Assert.InRange(value, 0.2, 0.6));
        Assert.Contains(values, value => value < 0.4);
        Assert.Contains(values, value => value >= 0.4);
        Assert.Equal(0.4, values.Average(), 1);
    }

    [Fact]
    public void ALogRange_DrawsInsideItsBounds_AndSpreadsEvenlyAcrossTheDecades()
    {
        var range = new LogRange("rate", 1e-4, 1e-1);
        var values = Enumerable.Range(0, 900).Select(trial => range.Draw(DrawsOf(trial)).Number).ToArray();
        var perDecade = new[] { 1e-4, 1e-3, 1e-2 }.Select(floor => values.Count(value => value >= floor && value < floor * 10)).ToArray();

        Assert.All(values, value => Assert.InRange(value, 1e-4, 1e-1));
        Assert.All(perDecade, count => Assert.InRange(count, 240, 360));
    }

    [Fact]
    public void AWholeRange_DrawsEveryWholeNumberOfItsRange_BothEndsIncluded()
    {
        var range = new WholeRange("units", 4, 8);
        var values = Enumerable.Range(0, 300).Select(trial => range.Draw(DrawsOf(trial)).Number).ToArray();

        Assert.All(values, value => Assert.Equal(Math.Floor(value), value));
        Assert.Equal([4.0, 5.0, 6.0, 7.0, 8.0], values.Distinct().Order());
    }

    [Fact]
    public void Choices_DrawEveryWordAndNothingElse_TheSettingNamingThePlaceOfTheWord()
    {
        var choices = new Choices("optimizer", ["sgd", "adam", "rmsprop"]);
        var settings = Enumerable.Range(0, 300).Select(trial => choices.Draw(DrawsOf(trial))).ToArray();

        Assert.Equal(["adam", "rmsprop", "sgd"], settings.Select(setting => setting.Choice!).Distinct().Order());
        Assert.All(settings, setting => Assert.Equal((double)Array.IndexOf(["sgd", "adam", "rmsprop"], setting.Choice), setting.Number));
    }

    [Fact]
    public void TheSameNumbers_GiveTheSameValue_WhateverWasDrawnBefore()
    {
        var range = new LogRange("rate", 1e-4, 1e-1);

        _ = range.Draw(DrawsOf(0));

        Assert.Equal(range.Draw(DrawsOf(7)), range.Draw(DrawsOf(7)));
    }

    [Fact]
    public void ACandidate_ReadsItsSettingsByName()
    {
        var candidate = new Candidate([new Setting("rate", 0.003, null), new Setting("units", 16, null), new Setting("optimizer", 1, "adam")]);

        Assert.Equal(0.003, candidate.Number("rate"));
        Assert.Equal(16, candidate.Whole("units"));
        Assert.Equal("adam", candidate.Choice("optimizer"));
        Assert.Equal(["rate", "units", "optimizer"], candidate.Settings.Select(setting => setting.Name));
    }

    [Fact]
    public void AnUnknownName_IsRefused_NamingTheOnesTheCandidateHolds()
    {
        var candidate = new Candidate([new Setting("rate", 0.003, null), new Setting("units", 16, null)]);
        var refused = Assert.Throws<ArgumentException>(() => candidate.Number("rte"));

        Assert.Contains("'rte'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'rate'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'units'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ASettingIsReadAsWhatItIs_AndRefusedAsAnythingElse()
    {
        var candidate = new Candidate([new Setting("rate", 0.003, null), new Setting("optimizer", 1, "adam")]);

        Assert.Contains("not a whole number", Assert.Throws<InvalidOperationException>(() => candidate.Whole("rate")).Message, StringComparison.Ordinal);
        Assert.Contains("word", Assert.Throws<InvalidOperationException>(() => candidate.Number("optimizer")).Message, StringComparison.Ordinal);
        Assert.Contains("number", Assert.Throws<InvalidOperationException>(() => candidate.Choice("rate")).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACandidateWithTwoSettingsOfOneName_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => new Candidate([new Setting("rate", 1, null), new Setting("rate", 2, null)]));
    }

    [Fact]
    public void TwoCandidatesWithTheSameSettings_AreEqual()
    {
        var one = new Candidate([new Setting("rate", 0.003, null), new Setting("optimizer", 1, "adam")]);
        var other = new Candidate([new Setting("rate", 0.003, null), new Setting("optimizer", 1, "adam")]);

        Assert.Equal(one, other);
        Assert.Equal(one.GetHashCode(), other.GetHashCode());
        Assert.NotEqual(one, new Candidate([new Setting("rate", 0.004, null), new Setting("optimizer", 1, "adam")]));
        Assert.False(one.Equals(null));
    }
}
