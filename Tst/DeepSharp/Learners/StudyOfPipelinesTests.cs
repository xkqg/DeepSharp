// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A pipeline that depends on its candidate, and a trial of several networks judged together. A study handed one pipeline for each
/// deal prepares it once for each deal, as it always did; one handed a pipeline for each candidate and deal prepares it for every
/// trial, and holds what each prepares to the rows the first prepared were judged on, since scores are chosen between only where
/// the rows they were measured on are the same. A trial of several networks is judged by the one number a judge of the caller's
/// gives for the networks the deal trained, so a model of two networks is chosen as the one model it is.
/// </summary>
public class StudyOfPipelinesTests
{
    private static readonly SearchSpace Space = new([new LogRange("rate", 1e-3, 1e-1), new Choices("fill", ["median", "mean"])]);

    private static Pipeline Passengers(Candidate candidate, int deal, double validation = 0.15, int? testSeed = 7)
    {
        var declared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("sex").Optional("age", ColumnKind.Number).Number("fare"));

        return (testSeed is { } locked ? declared.SplitAtRandom(0.70, validation, 100 + deal, locked) : declared.SplitAtRandom(0.70, validation, 100 + deal))
            .FillMissing("age", candidate.Choice("fill") == "mean" ? With.Mean : With.Median)
            .EncodeCategories()
            .Normalise("age", Scale.MidRange)
            .Normalise("fare", Scale.MidRange)
            .Normalise("sibsp", Scale.MidRange)
            .Normalise("parch", Scale.MidRange)
            .Target("survived")
            .Build();
    }

    private static Study Studied(int trials = 4, int deals = 1) =>
        new(Space, (candidate, network) => network
            .Dense(4).Relu().Dense(1)
            .Adam(candidate.Number("rate"))
            .BinaryCrossEntropy()
            .Run(seed: 20260929, epochs: 2))
        {
            Sampler = new RandomSampler(11),
            Trials = trials,
            Deals = deals,
        };

    private static Study StudiedTogether(int trials = 3, int deals = 1, Func<IReadOnlyList<TrainedNetwork>, double>? judge = null) =>
        new(Space, candidate => (IReadOnlyList<LearnNetworkStep>)
        [
            new NetworkDeclaration(Vocabularies.Torch).Dense(4).Relu().Dense(1).Adam(candidate.Number("rate")).BinaryCrossEntropy().Run(seed: 1, epochs: 2).Step(),
            new NetworkDeclaration(Vocabularies.Torch).Dense(2).Relu().Dense(1).Adam(candidate.Number("rate")).BinaryCrossEntropy().Run(seed: 2, epochs: 2).Step(),
        ])
        {
            Sampler = new RandomSampler(11),
            Trials = trials,
            Deals = deals,
            Judge = judge ?? (networks => networks.Sum(network => network.ValidationLoss())),
        };

    [Fact]
    public void AStudyHandedAPipelineForEachCandidate_PreparesItForEveryTrial_AndTheNumbersAreTheStudyHandedThemAllInOneWhereTheyAreTheSame()
    {
        var prepared = new List<(string Fill, int Deal)>();
        var result = Studied(trials: 4).Run(
            (candidate, deal) =>
            {
                prepared.Add((candidate.Choice("fill"), deal));

                return Passengers(candidate, deal);
            },
            TestContext.Current.CancellationToken);
        var once = Studied(trials: 4).Run(deal => Passengers(new Candidate([new Setting("rate", 0.01, null), new Setting("fill", 0, "median")]), deal), TestContext.Current.CancellationToken);

        Assert.Equal(4, prepared.Count);
        Assert.Equal(4, result.Trials.Count);
        Assert.Equal(4, once.Trials.Count);
        Assert.Contains(result.Trials, trial => trial.Candidate.Choice("fill") == "mean");
        Assert.Contains(result.Trials, trial => trial.Candidate.Choice("fill") == "median");
    }

    [Fact]
    public void TheStudyHandedOnePipelineForEachDeal_StillPreparesItOnceForEachDeal_WhateverTheTrials()
    {
        var runs = new List<int>();

        Studied(trials: 3, deals: 2).Run(
            deal =>
            {
                runs.Add(deal);

                return Passengers(new Candidate([new Setting("rate", 0.01, null), new Setting("fill", 0, "median")]), deal);
            },
            TestContext.Current.CancellationToken);

        Assert.Equal([0, 1], runs);
    }

    [Fact]
    public void ACandidatesPipelineThatMovesTheRowsItIsJudgedOn_IsRefused_NamingTheTrialAndTheDeal()
    {
        var study = Studied(trials: 3);

        var refused = Assert.Throws<InvalidOperationException>(() => study.Run(
            (candidate, deal) => Passengers(candidate, deal, validation: candidate.Number("rate") > 0.02 ? 0.25 : 0.15),
            TestContext.Current.CancellationToken));

        Assert.Contains("trial", refused.Message, StringComparison.Ordinal);
        Assert.Contains("validation", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACandidatesPipelineThatMovesTheTestRows_IsRefused_NamingTheTrial()
    {
        var study = Studied(trials: 3);

        var refused = Assert.Throws<InvalidOperationException>(() => study.Run(
            (candidate, deal) => Passengers(candidate, deal, testSeed: candidate.Number("rate") > 0.02 ? 8 : 7),
            TestContext.Current.CancellationToken));

        Assert.Contains("test", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACandidatesPipelineThatDealsTheValidationRowsAnotherWay_ButKeepsTheTestRows_IsRefused_NamingTheTrialAndTheDeal()
    {
        var study = Studied(trials: 3);

        var refused = Assert.Throws<InvalidOperationException>(() => study.Run(
            (candidate, deal) => Passengers(candidate, deal + (candidate.Number("rate") > 0.02 ? 1 : 0)),
            TestContext.Current.CancellationToken));

        Assert.Contains("sets other rows aside as validation", refused.Message, StringComparison.Ordinal);
        Assert.Contains("deal 0", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AListOfPipelinesThatIsNothing_OrHoldsNone_IsRefused_NamingTheTrialAndTheDeal()
    {
        var nothing = Assert.Throws<InvalidOperationException>(() => StudiedTogether(trials: 1).Run((candidate, deal) => (IReadOnlyList<Pipeline>)null!, TestContext.Current.CancellationToken));
        var hole = Assert.Throws<InvalidOperationException>(() => StudiedTogether(trials: 1).Run((candidate, deal) => (IReadOnlyList<Pipeline>)[Passengers(candidate, deal), null!], TestContext.Current.CancellationToken));

        Assert.Contains("deal 0", nothing.Message, StringComparison.Ordinal);
        Assert.Contains("deal 0", hole.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void APipelineForACandidateThatIsNothing_IsRefused_NamingTheTrialAndTheDeal()
    {
        var refused = Assert.Throws<InvalidOperationException>(() => Studied(trials: 1).Run((candidate, deal) => (Pipeline)null!, TestContext.Current.CancellationToken));

        Assert.Contains("deal 0", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ATrialOfSeveralNetworks_IsJudgedByTheOneNumberTheJudgeGivesForTheNetworksADealTrained_AndKeepsEveryOneOfThem()
    {
        var judged = new List<int>();
        var together = StudiedTogether(trials: 3, deals: 2, networks =>
        {
            judged.Add(networks.Count);

            return networks.Sum(network => network.ValidationLoss());
        });

        var result = together.Run((candidate, deal) => (IReadOnlyList<Pipeline>)[Passengers(candidate, deal), Passengers(candidate, deal)], TestContext.Current.CancellationToken);

        Assert.All(judged, count => Assert.Equal(2, count));
        Assert.Equal(6, judged.Count);
        Assert.All(result.Trials, trial => Assert.Equal(2, trial.Members.Count));
        Assert.Equal(result.Best.Members[0], result.Best.Declared);
        Assert.Equal(2, result.Members.Count);
        Assert.All(result.Members, deal => Assert.Equal(2, deal.Count));
        Assert.All(result.Best.Scores, score => Assert.True(double.IsFinite(score)));
    }

    [Fact]
    public void ATrialOfSeveralNetworks_NeedsAJudge_ForNoScoreOfOneNetworkSaysHowTheyDoTogether()
    {
        var study = new Study(Space, candidate => (IReadOnlyList<LearnNetworkStep>)
        [
            new NetworkDeclaration(Vocabularies.Torch).Dense(1).Adam(0.01).BinaryCrossEntropy().Run(1, 1).Step(),
            new NetworkDeclaration(Vocabularies.Torch).Dense(1).Adam(0.01).BinaryCrossEntropy().Run(2, 1).Step(),
        ])
        {
            Trials = 1,
        };

        Assert.Contains("Judge", Assert.Throws<InvalidOperationException>(
            () => study.Run((candidate, deal) => (IReadOnlyList<Pipeline>)[Passengers(candidate, deal), Passengers(candidate, deal)], TestContext.Current.CancellationToken)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AJudgeBesideAPrunerOrAScore_IsRefused_ForTheyWouldJudgeTheTrialByTwoNumbers()
    {
        var pruned = new Study(Space, candidate => (IReadOnlyList<LearnNetworkStep>)[new NetworkDeclaration(Vocabularies.Torch).Dense(1).Adam(0.01).BinaryCrossEntropy().Run(1, 1).Step()])
        {
            Judge = networks => 0,
            Pruner = _ => false,
        };
        var scored = new Study(Space, candidate => (IReadOnlyList<LearnNetworkStep>)[new NetworkDeclaration(Vocabularies.Torch).Dense(1).Adam(0.01).BinaryCrossEntropy().Run(1, 1).Step()])
        {
            Judge = networks => 0,
            Score = network => 0,
        };

        Assert.Throws<InvalidOperationException>(() => pruned.Run((candidate, deal) => (IReadOnlyList<Pipeline>)[Passengers(candidate, deal)], TestContext.Current.CancellationToken));
        Assert.Throws<InvalidOperationException>(() => scored.Run((candidate, deal) => (IReadOnlyList<Pipeline>)[Passengers(candidate, deal)], TestContext.Current.CancellationToken));
    }

    [Fact]
    public void ASeveralNetworkTrial_NeedsAsManyPipelinesAsItHasNetworks()
    {
        var refused = Assert.Throws<InvalidOperationException>(() => StudiedTogether(trials: 1).Run(
            (candidate, deal) => (IReadOnlyList<Pipeline>)[Passengers(candidate, deal)],
            TestContext.Current.CancellationToken));

        Assert.Contains("2", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AStudyOfSeveralNetworks_GoneOnFromItsEarlierTrials_IsTheStudyThatNeverStopped()
    {
        Func<Candidate, int, IReadOnlyList<Pipeline>> pipelines = (candidate, deal) => [Passengers(candidate, deal), Passengers(candidate, deal)];
        var first = StudiedTogether(trials: 2).Run(pipelines, TestContext.Current.CancellationToken);
        var whole = StudiedTogether(trials: 3).Run(pipelines, TestContext.Current.CancellationToken);
        var resumed = StudiedTogether(trials: 3).Run(pipelines, first.Trials, TestContext.Current.CancellationToken);

        Assert.Equal(whole.Trials.Select(trial => trial.Scores), resumed.Trials.Select(trial => trial.Scores));
        Assert.Equal(whole.Best.Number, resumed.Best.Number);
    }

    [Fact]
    public void TheBestFewOfAStudyOfSeveralNetworks_CannotBeTrainedAgainAsOneEnsemble()
    {
        var result = StudiedTogether(trials: 2).Run((candidate, deal) => [Passengers(candidate, deal), Passengers(candidate, deal)], TestContext.Current.CancellationToken);

        Assert.Throws<InvalidOperationException>(() => result.Top(1));
    }
}
