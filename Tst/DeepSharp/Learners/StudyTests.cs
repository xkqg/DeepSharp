// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A study tries many candidates for one network, each trained on the rows a pipeline prepared and judged by its validation
/// rows, and keeps the one that was judged best. The rows to be measured on are never what it chooses by: a trial reports
/// nothing about them, and only the network the study keeps carries the report's measures of them. Trials do not share what
/// they change, so the same study run again gives the same numbers, and a study that stops early is the front of one that
/// goes on.
/// </summary>
public class StudyTests
{
    private static readonly SearchSpace Space = new([new LogRange("rate", 1e-3, 1e-1), new WholeRange("units", 2, 8)]);

    internal static Pipeline Passengers(int deal, int? testSeed = 7, double validation = 0.15)
    {
        var declared = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("sex").Optional("age", ColumnKind.Number).Number("fare"));

        return (testSeed is { } locked ? declared.SplitAtRandom(0.70, validation, 100 + deal, locked) : declared.SplitAtRandom(0.70, validation, 100 + deal))
            .FillMissing("age", With.Median)
            .EncodeCategories()
            .Normalise("age", Scale.MidRange)
            .Normalise("fare", Scale.MidRange)
            .Normalise("sibsp", Scale.MidRange)
            .Normalise("parch", Scale.MidRange)
            .Target("survived")
            .Report(report => report.Measure(Metric.Accuracy).On(Part.Train, Part.Validation, Part.Test).As(Shown.Numbers))
            .Build();
    }

    private static Study Studied(int trials = 4, int deals = 1) =>
        new(Space, (candidate, network) => network
            .Dense(candidate.Whole("units")).Relu().Dense(1)
            .Adam(candidate.Number("rate"))
            .BinaryCrossEntropy()
            .Run(seed: 20260929, epochs: 2))
        {
            Sampler = new RandomSampler(11),
            Trials = trials,
            Deals = deals,
        };

    [Fact]
    public void TheStudy_TriesEveryTrialInOrder_EachOnTheCandidateTheSamplerGave()
    {
        var result = Studied(trials: 4).Run(deal => Passengers(deal), TestContext.Current.CancellationToken);

        Assert.Equal([0, 1, 2, 3], result.Trials.Select(trial => trial.Number));

        foreach (var trial in result.Trials)
        {
            Assert.Equal(new RandomSampler(11).Suggest(Space, trial.Number, []), trial.Candidate);
            Assert.Equal(trial.Candidate.Whole("units"), trial.Declared.Layers[0].Whole("units"));
            Assert.Equal(trial.Candidate.Number("rate"), trial.Declared.Optimizer.Number("rate"));
            Assert.Single(trial.Scores);
        }
    }

    [Fact]
    public void TheBest_IsTheTrialJudgedLowest_AndTheWinnerIsTheNetworkItTrained()
    {
        var result = Studied(trials: 5).Run(deal => Passengers(deal), TestContext.Current.CancellationToken);
        var lowest = result.Trials.MinBy(trial => trial.Score)!;

        Assert.Equal(lowest.Number, result.Best.Number);
        Assert.Equal(lowest.Score, result.Best.Score);

        var winner = Assert.Single(result.Winner);
        var epoch = winner.History!.Epochs.Single(each => each.Number == winner.TrainedOn.Epoch);

        Assert.Equal(result.Best.Scores[0], epoch.ValidationLoss);
        Assert.Contains(winner.Measures!.Parts, part => part.Part == Part.Test);
    }

    [Fact]
    public void TheTrialsOfAShortStudy_AreTheFrontOfALongerOne_AndARunAgainIsTheSameNumberForNumber()
    {
        var short3 = Studied(trials: 3).Run(deal => Passengers(deal), TestContext.Current.CancellationToken);
        var long5 = Studied(trials: 5).Run(deal => Passengers(deal), TestContext.Current.CancellationToken);
        var again = Studied(trials: 5).Run(deal => Passengers(deal), TestContext.Current.CancellationToken);

        Assert.Equal(short3.Trials.Select(trial => trial.Scores[0]), long5.Trials.Take(3).Select(trial => trial.Scores[0]));
        Assert.Equal(long5.Trials.Select(trial => trial.Scores[0]), again.Trials.Select(trial => trial.Scores[0]));
        Assert.Equal(long5.Best.Number, again.Best.Number);
    }

    [Fact]
    public void ThePipelineIsRunOncePerDeal_NotOncePerTrial()
    {
        var runs = new List<int>();

        _ = Studied(trials: 3, deals: 2).Run(deal =>
        {
            runs.Add(deal);

            return Passengers(deal);
        }, TestContext.Current.CancellationToken);

        Assert.Equal([0, 1], runs);
    }

    [Fact]
    public void Deals_AreDealtOnTheirOwn_AroundOneTestPart_AndATrialIsScoredByTheirMean()
    {
        var result = Studied(trials: 2, deals: 3).Run(deal => Passengers(deal), TestContext.Current.CancellationToken);
        var trial = result.Trials[0];

        Assert.Equal(3, trial.Scores.Count);
        Assert.Equal(3, trial.Scores.Distinct().Count());
        Assert.Equal(trial.Scores.Average(), trial.Score, 12);

        var mean = trial.Scores.Average();

        Assert.Equal(Math.Sqrt(trial.Scores.Sum(score => (score - mean) * (score - mean)) / 2), trial.Spread, 12);
        Assert.Equal(3, result.Winner.Count);
    }

    [Fact]
    public void ATrialOfOneDeal_HasNoSpread()
    {
        Assert.Equal(0, Studied(trials: 1).Run(deal => Passengers(deal), TestContext.Current.CancellationToken).Best.Spread);
    }

    [Fact]
    public void DealsThatDoNotShareATestPart_AreRefused_SayingHowToLockIt()
    {
        var refused = Assert.Throws<InvalidOperationException>(() => Studied(trials: 2, deals: 2).Run(deal => Passengers(deal, testSeed: null), TestContext.Current.CancellationToken));

        Assert.Contains("test", refused.Message, StringComparison.Ordinal);
        Assert.Contains("testSeed", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void APipelineWithNoValidationRows_IsRefused_ForTheStudyChoosesByThem()
    {
        var refused = Assert.Throws<InvalidOperationException>(() => Studied(trials: 2).Run(deal => Passengers(deal, validation: 0), TestContext.Current.CancellationToken));

        Assert.Contains("validation", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ATrialThatDoesNotFinish_NeverWins_AndIsKeptAmongTheTrials()
    {
        var scores = new Queue<double>([3.0, double.NaN, 2.0, double.PositiveInfinity]);
        var result = new Study(Space, (candidate, network) => network.Dense(2).Relu().Dense(1).Adam(candidate.Number("rate")).BinaryCrossEntropy().Run(1, 1))
        {
            Sampler = new RandomSampler(11),
            Trials = 4,
            Score = _ => scores.Dequeue(),
        }.Run(deal => Passengers(deal), TestContext.Current.CancellationToken);

        Assert.Equal(2, result.Best.Number);
        Assert.Equal([true, false, true, false], result.Trials.Select(trial => trial.Finished));
    }

    [Fact]
    public void WhenNoTrialFinishes_TheStudyRefuses_SayingSo()
    {
        var study = new Study(Space, (candidate, network) => network.Dense(2).Relu().Dense(1).Adam(candidate.Number("rate")).BinaryCrossEntropy().Run(1, 1))
        {
            Trials = 2,
            Score = _ => double.NaN,
        };

        Assert.Contains("No trial finished", Assert.Throws<InvalidOperationException>(() => study.Run(deal => Passengers(deal), TestContext.Current.CancellationToken)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACandidateForAnotherSpace_IsRefused_NamingTheDimensionsTheSpaceHas()
    {
        var study = new Study(Space, (candidate, network) => network.Dense(2).Relu().Dense(1).Adam().BinaryCrossEntropy().Run(1, 1))
        {
            Sampler = new Foreign(),
            Trials = 1,
        };

        var refused = Assert.Throws<InvalidOperationException>(() => study.Run(deal => Passengers(deal), TestContext.Current.CancellationToken));

        Assert.Contains("'rate'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'units'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ASamplerIsHandedTheTrialsThatFinishedBeforeIt_SoOneThatLearnsFromThemCan()
    {
        var seen = new List<int>();
        var study = new Study(Space, (candidate, network) => network.Dense(2).Relu().Dense(1).Adam(candidate.Number("rate")).BinaryCrossEntropy().Run(1, 1))
        {
            Sampler = new Watching(seen),
            Trials = 3,
        };

        _ = study.Run(deal => Passengers(deal), TestContext.Current.CancellationToken);

        Assert.Equal([0, 1, 2], seen);
    }

    [Fact]
    public void AStepWrittenOutsideTheChain_CanBeTheCandidateToo()
    {
        var step = Declared();
        var study = new Study(Space, candidate => step with { Epochs = 1, Seed = candidate.Whole("units") })
        {
            Trials = 2,
        };

        var result = study.Run(deal => Passengers(deal), TestContext.Current.CancellationToken);

        Assert.Equal(2, result.Trials.Count);
        Assert.All(result.Trials, trial => Assert.Equal(1, trial.Declared.Epochs));
    }

    private static LearnNetworkStep Declared() => Assert.IsType<LearnNetworkStep>(
        Pdd.Create().ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("sex").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitAtRandom(0.70, 0.15)
            .FillMissing("age", With.Median)
            .EncodeCategories()
            .Normalise("age", Scale.MidRange)
            .Normalise("fare", Scale.MidRange)
            .Normalise("sibsp", Scale.MidRange)
            .Normalise("parch", Scale.MidRange)
            .Target("survived")
            .WithTorch(network => network.Dense(2).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy().Run(3, 2))
            .Build()
            .Declaration.Learner);

    [Fact]
    public void TheBestDeclaration_IsAStepAFileKeeps_SoTheWinnerLeavesTheStudyAsADeclaration()
    {
        var best = Studied(trials: 3).Run(deal => Passengers(deal), TestContext.Current.CancellationToken).Best.Declared;
        var written = System.Text.Json.JsonDocument.Parse(best.Canonical()).RootElement;

        Assert.Equal("learn.network", best.Verb);
        Assert.Equal(best, LearnNetworkStep.ReadFrom(written));
    }

    [Fact]
    public void AStudyIsToldOfEveryTrialAsItEnds_AndHearingThemChangesNoNumber()
    {
        var heard = new List<Trial>();
        var listened = new Study(Space, (candidate, network) => network
            .Dense(candidate.Whole("units")).Relu().Dense(1)
            .Adam(candidate.Number("rate"))
            .BinaryCrossEntropy()
            .Run(seed: 20260929, epochs: 2))
        {
            Sampler = new RandomSampler(11),
            Trials = 3,
            OnTrial = heard.Add,
        };
        var plain = Studied(trials: 3).Run(deal => Passengers(deal), TestContext.Current.CancellationToken);
        var result = listened.Run(deal => Passengers(deal), TestContext.Current.CancellationToken);

        Assert.Equal([0, 1, 2], heard.Select(trial => trial.Number));
        Assert.Equal(plain.Trials.Select(trial => trial.Scores[0]), result.Trials.Select(trial => trial.Scores[0]));
        Assert.Same(result.Trials[2], heard[2]);
    }

    [Fact]
    public void AStudyThatIsCancelled_StopsBeforeItsNextTrial_AndReturnsNothing()
    {
        using var source = new CancellationTokenSource();
        var heard = new List<int>();
        var study = new Study(Space, (candidate, network) => network.Dense(2).Relu().Dense(1).Adam(candidate.Number("rate")).BinaryCrossEntropy().Run(1, 1))
        {
            Trials = 5,
            OnTrial = trial =>
            {
                heard.Add(trial.Number);

                if (trial.Number == 1)
                {
                    source.Cancel();
                }
            },
        };

        var stopped = Assert.Throws<OperationCanceledException>(() => study.Run(deal => Passengers(deal), source.Token));

        Assert.Equal(source.Token, stopped.CancellationToken);
        Assert.Equal([0, 1], heard);
    }

    [Fact]
    public void AStudyCancelledBeforeItBegins_PreparesNothing()
    {
        var runs = 0;
        using var source = new CancellationTokenSource();

        source.Cancel();

        Assert.Throws<OperationCanceledException>(() => Studied(trials: 3).Run(
            deal =>
            {
                runs++;

                return Passengers(deal);
            },
            source.Token));
        Assert.Equal(0, runs);
    }

    [Fact]
    public void ABadStudy_IsRefusedWhereItIsMade()
    {
        Assert.Throws<ArgumentNullException>(() => new Study(Space, (Func<Candidate, DeepSharp.Learners.Networks.NetworkDeclaration, DeepSharp.Learners.Networks.NetworkDeclaration>)null!));
        Assert.Throws<ArgumentNullException>(() => new Study(null!, (candidate, network) => network));
    }

    [Fact]
    public void ATrialThatWasJudgedByNothing_DidNotFinish()
    {
        var trial = new Trial(0, new Candidate([new Setting("rate", 0.01, null)]), Declared(), []);

        Assert.False(trial.Finished);
        Assert.Equal(0, trial.Spread);
    }

    [Fact]
    public void ASamplerOrAScoreThatIsNothing_IsRefusedWhereItIsGiven()
    {
        Assert.Throws<ArgumentNullException>(() => new Study(Space, (candidate, network) => network) { Sampler = null! });
        Assert.Throws<ArgumentNullException>(() => new Study(Space, (candidate, network) => network) { Score = null! });
    }

    [Fact]
    public void APipelineThatIsNothing_IsRefused_NamingTheDeal()
    {
        var refused = Assert.Throws<InvalidOperationException>(() => Studied(trials: 1).Run(deal => null!, TestContext.Current.CancellationToken));

        Assert.Contains("deal 0", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TrialsAndDealsBelowOne_AreRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Study(Space, (candidate, network) => network) { Trials = 0 });
        Assert.Throws<ArgumentOutOfRangeException>(() => new Study(Space, (candidate, network) => network) { Deals = 0 });
    }

    private sealed class Foreign : ISampler
    {
        public Candidate Suggest(SearchSpace space, int trial, IReadOnlyList<Trial> finished) => new([new Setting("momentum", 0.9, null)]);
    }

    private sealed class Watching(List<int> seen) : ISampler
    {
        public Candidate Suggest(SearchSpace space, int trial, IReadOnlyList<Trial> finished)
        {
            seen.Add(finished.Count);
            Assert.Equal(Enumerable.Range(0, trial), finished.Select(each => each.Number));

            return new RandomSampler(3).Suggest(space, trial, finished);
        }
    }
}
