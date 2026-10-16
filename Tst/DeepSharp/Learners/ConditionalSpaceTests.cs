// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A setting can exist only when another is chosen — a momentum only for the optimizer that takes one — as Optuna's spaces
/// written by running have it. A candidate then holds a value for every dimension that is there for it and for no other, a
/// dimension that is not there draws nothing, and a space without such a dimension draws what it always drew.
/// </summary>
public class ConditionalSpaceTests
{
    private static readonly Choices Optimizer = new("optimizer", ["adam", "sgd"]);
    private static readonly NumberRange Momentum = new("momentum", 0, 0.99) { OnlyWhen = new Chosen("optimizer", "sgd") };
    private static readonly LogRange Rate = new("rate", 1e-3, 1e-1);
    private static readonly SearchSpace Space = new([Optimizer, Momentum, Rate]);

    [Fact]
    public void ADimensionThatExistsOnlyWhenAWordIsChosen_IsDrawnThenAndOnlyThen()
    {
        var sampler = new RandomSampler(5);
        var candidates = Enumerable.Range(0, 30).Select(trial => sampler.Suggest(Space, trial, [])).ToArray();

        Assert.Contains(candidates, candidate => candidate.Choice("optimizer") == "sgd");
        Assert.Contains(candidates, candidate => candidate.Choice("optimizer") == "adam");
        Assert.All(candidates, candidate =>
        {
            var names = candidate.Settings.Select(setting => setting.Name).ToArray();

            Assert.Equal(candidate.Choice("optimizer") == "sgd" ? ["optimizer", "momentum", "rate"] : ["optimizer", "rate"], names);
        });
    }

    [Fact]
    public void ADimensionThatIsNotThere_MovesNoOtherDimensionsValue()
    {
        // Each dimension draws by its own name, so the rate a trial gets is the rate it gets in a space with no momentum at all.
        var sampler = new RandomSampler(5);
        var flat = new SearchSpace([Optimizer, Rate]);

        for (var trial = 0; trial < 10; trial++)
        {
            Assert.Equal(sampler.Suggest(flat, trial, []).Number("rate"), sampler.Suggest(Space, trial, []).Number("rate"));
        }
    }

    [Fact]
    public void ADimensionIsThereAmongWhatWasDrawn_WhenItsWordWasChosen()
    {
        Assert.True(Rate.IsThereAmong([new Setting("optimizer", 0, "adam")]));
        Assert.True(Momentum.IsThereAmong([new Setting("optimizer", 1, "sgd")]));
        Assert.False(Momentum.IsThereAmong([new Setting("optimizer", 0, "adam")]));
        Assert.False(Momentum.IsThereAmong([]));
    }

    [Theory]
    [InlineData("loss", "sgd")]
    [InlineData("rate", "sgd")]
    [InlineData("optimizer", "rmsprop")]
    public void ADimensionThatHangsOnNoEarlierChoiceOrOnAWordItDoesNotOffer_IsRefusedWhereTheSpaceIsMade(string dimension, string word)
    {
        var hanging = new WholeRange("units", 2, 8) { OnlyWhen = new Chosen(dimension, word) };

        var refused = Assert.Throws<ArgumentException>(() => new SearchSpace([Optimizer, Rate, hanging]));

        Assert.Contains("'units'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ADimensionThatHangsOnALaterChoice_IsRefused_ForItIsDrawnInOrder()
    {
        var early = new WholeRange("units", 2, 8) { OnlyWhen = new Chosen("optimizer", "sgd") };

        Assert.Throws<ArgumentException>(() => new SearchSpace([early, Optimizer]));
    }

    [Fact]
    public void AStudy_TakesACandidateOfTheDimensionsThatAreThereForIt_AndRefusesOneThatHoldsAnotherOrLacksOne()
    {
        var study = new Study(Space, (candidate, network) => network
            .Dense(2).Relu().Dense(1)
            .Adam(candidate.Number("rate"))
            .BinaryCrossEntropy()
            .Run(seed: 20260929, epochs: 1))
        {
            Sampler = new RandomSampler(5),
            Trials = 3,
        };

        Assert.Equal(3, study.Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken).Trials.Count);

        var holding = new Study(Space, (candidate, network) => network.Dense(1).Adam(0.01).BinaryCrossEntropy().Run(1, 1))
        {
            Sampler = new Fixed([new Setting("optimizer", 0, "adam"), new Setting("momentum", 0.5, null), new Setting("rate", 0.01, null)]),
            Trials = 1,
        };
        var lacking = new Study(Space, (candidate, network) => network.Dense(1).Adam(0.01).BinaryCrossEntropy().Run(1, 1))
        {
            Sampler = new Fixed([new Setting("optimizer", 1, "sgd"), new Setting("rate", 0.01, null)]),
            Trials = 1,
        };

        Assert.Contains("'momentum'", Assert.Throws<InvalidOperationException>(() => holding.Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken)).Message, StringComparison.Ordinal);
        Assert.Contains("'momentum'", Assert.Throws<InvalidOperationException>(() => lacking.Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken)).Message, StringComparison.Ordinal);
    }

    private sealed class Fixed(Setting[] settings) : ISampler
    {
        public Candidate Suggest(SearchSpace space, int trial, IReadOnlyList<Trial> finished) => new(settings);
    }
}
