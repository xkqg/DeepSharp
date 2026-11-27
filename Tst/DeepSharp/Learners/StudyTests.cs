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

    // The letter's deals: five dealt at random around one test part, and one split in time, weighed alike.
    internal static readonly IReadOnlyList<DealKind> TwoKinds = [new DealKind("random", 5, 0.5), new DealKind("time", 1, 0.5)];

    // A deal of the bikes, dealt as its kind says: in time for the kind named "time", at random around one test part otherwise.
    internal static Pipeline Bikes(IReadOnlyList<DealKind> kinds, int deal, int? testSeed = 7)
    {
        var declared = Pdd.Create()
            .ReadCsv(Repository.Data("bikes.csv"))
            .Declare(schema => schema.Timestamp("dteday").Number("temp", "hum", "windspeed", "cnt"));
        var split = kinds.KindOf(deal).Kind.Name == "time"
            ? declared.SplitByTime("dteday", 0.70, 0.15)
            : testSeed is { } locked ? declared.SplitAtRandom(0.70, 0.15, 100 + deal, locked) : declared.SplitAtRandom(0.70, 0.15, 100 + deal);

        return split
            .Drop("dteday")
            .Target("cnt")
            .Normalise("temp", Scale.MidRange)
            .Normalise("hum", Scale.MidRange)
            .Normalise("windspeed", Scale.MidRange)
            .Normalise("cnt", Scale.MidRange)
            .Build();
    }

    // A study dealt in kinds, by either face: networks the study declares from the candidate, or a model of the caller's
    // judged by the mean of its validation answers, which moves with every deal.
    internal sealed record KindedStudy(bool ByTheCaller, IReadOnlyList<DealKind> Kinds)
    {
        public int Trials { get; init; } = 2;

        public Func<DealsJudged, bool>? DealStopper { get; init; }

        public IReadOnlyList<Trial> Run(Func<int, Pipeline> pipelineFor, IReadOnlyList<Trial> earlier) =>
            ByTheCaller
                ? new Study<double>(Space, deal => new Trained<double>(deal.Rows.Batch(Part.Validation).Labels!.Average() * deal.Candidate.Number("rate"), 0))
                {
                    Sampler = new RandomSampler(11),
                    Trials = Trials,
                    Kinds = Kinds,
                    DealStopper = DealStopper,
                }.Run(pipelineFor, earlier, TestContext.Current.CancellationToken).Trials
                : new Study(Space, (candidate, network) => network.Dense(candidate.Whole("units")).Relu().Dense(1).Adam(candidate.Number("rate")).MeanSquaredError().Run(seed: 1, epochs: 1))
                {
                    Sampler = new RandomSampler(11),
                    Trials = Trials,
                    Kinds = Kinds,
                    DealStopper = DealStopper,
                }.Run(pipelineFor, earlier, TestContext.Current.CancellationToken).Trials;
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DealsOfTwoKinds_EachHoldingItsOwnTestRows_AreAccepted_AndATrialIsScoredByTheWeightedMeanOfTheKindsMeans(bool byTheCaller)
    {
        var trials = new KindedStudy(byTheCaller, TwoKinds).Run(deal => Bikes(TwoKinds, deal), []);

        Assert.Equal(2, trials.Count);

        foreach (var trial in trials)
        {
            var random = trial.Scores.Take(5).ToArray();
            var mean = random.Average();
            var spread = Math.Sqrt(random.Sum(score => (score - mean) * (score - mean)) / 4);

            Assert.Equal(6, trial.Scores.Count);
            Assert.Equal(TwoKinds, trial.Kinds);
            Assert.Equal(2, trial.KindScores.Count);
            Assert.Equal(mean, trial.KindScores[0], 12);
            Assert.Equal(trial.Scores[5], trial.KindScores[1], 12);
            Assert.Equal([spread, 0], trial.KindSpreads, (one, other) => Math.Abs(one - other) < 1e-12);
            Assert.Equal(((0.5 * mean) + (0.5 * trial.Scores[5])) / 1.0, trial.Score, 12);
            Assert.Equal(((0.5 * spread) + (0.5 * 0)) / 1.0, trial.Spread, 12);
            Assert.True(trial.Finished);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DealsOfOneKind_ThatDoNotShareTheirTestRows_AreStillRefused_NamingTheKind(bool byTheCaller)
    {
        var refused = Assert.Throws<InvalidOperationException>(() => new KindedStudy(byTheCaller, TwoKinds).Run(deal => Bikes(TwoKinds, deal, testSeed: null), []));

        Assert.Contains("measures on other test rows", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'random'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("testSeed", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void EarlierTrialsOfAnotherLayoutOfDeals_AreRefused_NamingWhatDiffers(bool byTheCaller)
    {
        IReadOnlyList<DealKind> oneKind = [new DealKind("deals", 6, 1)];
        var earlier = new KindedStudy(byTheCaller, oneKind) { Trials = 1 }.Run(deal => Bikes(oneKind, deal), []);
        var kinded = new KindedStudy(byTheCaller, TwoKinds) { Trials = 2 };
        var plain = new KindedStudy(byTheCaller, oneKind) { Trials = 2 };

        var otherKinds = Assert.Throws<InvalidOperationException>(() => kinded.Run(deal => Bikes(TwoKinds, deal), earlier));
        var saidNone = Assert.Throws<InvalidOperationException>(() => kinded.Run(deal => Bikes(TwoKinds, deal), [earlier[0] with { Kinds = [] }]));
        var fewer = Assert.Throws<InvalidOperationException>(() => plain.Run(deal => Bikes(oneKind, deal), [earlier[0] with { Scores = [.. earlier[0].Scores.Take(3)] }]));
        var more = Assert.Throws<InvalidOperationException>(() => plain.Run(deal => Bikes(oneKind, deal), [earlier[0] with { Scores = [.. earlier[0].Scores, 0.5], Pruned = true }]));

        Assert.Contains("'deals'", otherKinds.Message, StringComparison.Ordinal);
        Assert.Contains("'random'", otherKinds.Message, StringComparison.Ordinal);
        Assert.Contains("'time'", saidNone.Message, StringComparison.Ordinal);
        Assert.Contains("3 deals", fewer.Message, StringComparison.Ordinal);
        Assert.Contains("6", fewer.Message, StringComparison.Ordinal);
        Assert.Contains("7 deals", more.Message, StringComparison.Ordinal);

        // A trial that says no kinds is of one kind, as every trial was before kinds were said: a study of one kind takes it.
        Assert.Equal(2, plain.Run(deal => Bikes(oneKind, deal), [earlier[0] with { Kinds = [] }]).Count);
    }

    [Fact]
    public void AKindOfNoDeals_AWeightAtOrBelowNothingOrNotANumber_OrNoName_IsRefusedWhereItIsWritten()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new DealKind("time", 0, 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DealKind("time", 1, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DealKind("time", 1, -0.5));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DealKind("time", 1, double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => new DealKind("time", 1, double.PositiveInfinity));
        Assert.Throws<ArgumentNullException>(() => new DealKind(null!, 1, 1));
        Assert.Throws<ArgumentException>(() => new DealKind(" ", 1, 1));
        Assert.Equal(("time", 2, 0.25), (new DealKind("time", 2, 0.25).Name, new DealKind("time", 2, 0.25).Deals, new DealKind("time", 2, 0.25).Weight));
    }

    [Fact]
    public void KindsThatAreNothing_None_OfNoDeals_OrTwiceNamed_AreRefused_AndKindsThatDisagreeWithTheDeals_AreRefusedNamingBoth()
    {
        Assert.Throws<ArgumentNullException>(() => new Study(Space, (candidate, network) => network) { Kinds = null! });
        Assert.Throws<ArgumentException>(() => new Study(Space, (candidate, network) => network) { Kinds = [] });
        Assert.Throws<ArgumentException>(() => new Study(Space, (candidate, network) => network) { Kinds = [default] });
        Assert.Contains("'time'", Assert.Throws<ArgumentException>(() => new Study(Space, (candidate, network) => network) { Kinds = [new DealKind("time", 1, 1), new DealKind("time", 2, 1)] }).Message, StringComparison.Ordinal);

        foreach (var refused in new[]
        {
            Assert.Throws<ArgumentException>(() => new Study(Space, (candidate, network) => network) { Deals = 3, Kinds = TwoKinds }),
            Assert.Throws<ArgumentException>(() => new Study(Space, (candidate, network) => network) { Kinds = TwoKinds, Deals = 3 }),
            Assert.Throws<ArgumentException>(() => new Study<double>(Space, deal => new Trained<double>(0, 0)) { Deals = 3, Kinds = TwoKinds }),
            Assert.Throws<ArgumentException>(() => new Study<double>(Space, deal => new Trained<double>(0, 0)) { Kinds = TwoKinds, Deals = 3 }),
        })
        {
            Assert.Contains("Deals", refused.Message, StringComparison.Ordinal);
            Assert.Contains("Kinds", refused.Message, StringComparison.Ordinal);
        }

        Assert.Equal(6, new Study(Space, (candidate, network) => network) { Deals = 6, Kinds = TwoKinds }.Deals);
        Assert.Equal(6, new Study<double>(Space, deal => new Trained<double>(0, 0)) { Kinds = TwoKinds, Deals = 6 }.Deals);
    }

    [Fact]
    public void WhichKindADealIs_AndItsPlaceWithinTheKind_IsReadFromTheKinds_InTheOrderTheyAreDealt()
    {
        Assert.Equal(new DealOfAKind(TwoKinds[0], 0), TwoKinds.KindOf(0));
        Assert.Equal(new DealOfAKind(TwoKinds[0], 4), TwoKinds.KindOf(4));
        Assert.Equal(new DealOfAKind(TwoKinds[1], 0), TwoKinds.KindOf(5));
        Assert.Throws<ArgumentOutOfRangeException>(() => TwoKinds.KindOf(6));
        Assert.Throws<ArgumentOutOfRangeException>(() => TwoKinds.KindOf(-1));
        Assert.Throws<ArgumentNullException>(() => ((IReadOnlyList<DealKind>)null!).KindOf(0));
    }

    [Fact]
    public void ATrialOfSeveralKinds_IsScoredByTheWeightedMeanOfItsKindsMeans_ItsSpreadAndObjectivesAlike_AndOfOneKindAsBefore()
    {
        var candidate = new Candidate([new Setting("rate", 0.01, null)]);
        IReadOnlyList<DealKind> kinds = [new DealKind("random", 3, 3), new DealKind("time", 1, 1)];
        var kinded = new Trial(0, candidate, Declared: null, [1, 2, 3, 10]) { Kinds = kinds };
        var weighed = new Trial(0, candidate, Declared: null, []) { Kinds = kinds, ObjectiveScores = [[1, 10], [2, 20], [3, 30], [10, 100]] };
        var stopped = new Trial(0, candidate, Declared: null, [1, 2]) { Kinds = kinds, Pruned = true };
        var plain = new Trial(0, candidate, Declared: null, [1, 2, 3, 10]);

        Assert.Equal([2, 10], kinded.KindScores);
        Assert.Equal([1, 0], kinded.KindSpreads);
        Assert.Equal(((3 * 2) + (1 * 10)) / 4.0, kinded.Score, 12);
        Assert.Equal(((3 * 1) + (1 * 0)) / 4.0, kinded.Spread, 12);
        Assert.Equal([((3 * 2) + 10) / 4.0, ((3 * 20) + 100) / 4.0], weighed.ObjectiveMean);

        // A trial stopped before a kind was judged is judged by the kinds it reached.
        Assert.Equal(1.5, stopped.Score, 12);
        Assert.True(double.IsNaN(stopped.KindScores[1]));

        // A trial that says no kinds is of one kind: the plain mean and spread of its deals, as before.
        Assert.Empty(plain.Kinds);
        Assert.Equal(new[] { 1, 2, 3, 10.0 }.Average(), plain.Score);
        Assert.Equal([plain.Score], plain.KindScores);
        Assert.Equal([plain.Spread], plain.KindSpreads);
        Assert.Throws<ArgumentNullException>(() => plain with { Kinds = null! });
    }

    [Fact]
    public void TheStudy_TriesEveryTrialInOrder_EachOnTheCandidateTheSamplerGave()
    {
        var result = Studied(trials: 4).Run(deal => Passengers(deal), TestContext.Current.CancellationToken);

        Assert.Equal([0, 1, 2, 3], result.Trials.Select(trial => trial.Number));

        foreach (var trial in result.Trials)
        {
            Assert.Equal(new RandomSampler(11).Suggest(Space, trial.Number, []), trial.Candidate);
            Assert.Equal(trial.Candidate.Whole("units"), trial.Declared!.Layers[0].Whole("units"));
            Assert.Equal(trial.Candidate.Number("rate"), trial.Declared!.Optimizer.Number("rate"));
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
        Assert.All(result.Trials, trial => Assert.Equal(1, trial.Declared!.Epochs));
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
        var best = Studied(trials: 3).Run(deal => Passengers(deal), TestContext.Current.CancellationToken).Best.Declared!;
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
        Assert.True(double.IsNaN(trial.Score));
        Assert.Equal(0, trial.Spread);
    }

    [Fact]
    public void ASamplerOrAScoreThatIsNothing_IsRefusedWhereItIsGiven()
    {
        Assert.Throws<ArgumentNullException>(() => new Study(Space, (candidate, network) => network) { Sampler = null! });
        Assert.Throws<ArgumentNullException>(() => new Study(Space, (candidate, network) => network) { Score = null! });
        Assert.Throws<ArgumentNullException>(() => new Study(Space, (candidate, network) => network) { Judge = null! });
        Assert.Throws<ArgumentNullException>(() => new Study(Space, (candidate, network) => network) { Objectives = null! });
        Assert.Throws<ArgumentNullException>(() => new Study<double>(Space, deal => new Trained<double>(0, 0)) { Sampler = null! });
    }

    [Fact]
    public void WhatAStudyIsSetBy_ReadsBackAsItWasSet_AndTheSameUnlessSaid_ForBothKindsOfStudy()
    {
        var sampler = new RandomSampler(5);
        Func<Progress, bool> pruner = _ => false;
        Func<DealsJudged, bool> stopper = _ => false;
        Action<Trial> onTrial = _ => { };
        var networks = new Study(Space, (candidate, network) => network) { Sampler = sampler, Trials = 3, Deals = 2, Pruner = pruner, DealStopper = stopper, OnTrial = onTrial };
        var models = new Study<double>(Space, deal => new Trained<double>(0, 0)) { Sampler = sampler, Trials = 3, Deals = 2, Pruner = pruner, DealStopper = stopper, OnTrial = onTrial };
        var plainNetworks = new Study(Space, (candidate, network) => network);
        var plainModels = new Study<double>(Space, deal => new Trained<double>(0, 0));

        Assert.Equal<object?>([sampler, 3, 2, pruner, stopper, onTrial], [networks.Sampler, networks.Trials, networks.Deals, networks.Pruner, networks.DealStopper, networks.OnTrial]);
        Assert.Equal<object?>([sampler, 3, 2, pruner, stopper, onTrial], [models.Sampler, models.Trials, models.Deals, models.Pruner, models.DealStopper, models.OnTrial]);
        Assert.Equal(plainNetworks.Sampler.GetType(), plainModels.Sampler.GetType());
        Assert.Equal<object?>([20, 1, null, null, null], [plainNetworks.Trials, plainNetworks.Deals, plainNetworks.Pruner, plainNetworks.DealStopper, plainNetworks.OnTrial]);
        Assert.Equal<object?>([20, 1, null, null, null], [plainModels.Trials, plainModels.Deals, plainModels.Pruner, plainModels.DealStopper, plainModels.OnTrial]);

        // Unless said, the deals are of one kind, weighed by one; said, the kinds read back as they were, and so do their deals.
        Assert.Equal([new DealKind("deals", 1, 1)], plainNetworks.Kinds);
        Assert.Equal([new DealKind("deals", 2, 1)], networks.Kinds);
        Assert.Equal([new DealKind("deals", 2, 1)], models.Kinds);
        Assert.Equal(TwoKinds, new Study(Space, (candidate, network) => network) { Kinds = TwoKinds }.Kinds);
        Assert.Equal(6, new Study<double>(Space, deal => new Trained<double>(0, 0)) { Kinds = TwoKinds }.Deals);
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

    [Theory]
    [InlineData(new[] { 0.4, 0.3, 0.9, 0.2, double.NaN, 0.1, 0.35, 0.3 })]
    [InlineData(new[] { 0.6, 0.45, 0.2, 0.25, 0.7, 0.1, 0.15, 0.05 })]
    public void TheTwoStudyFaces_RunOneLoop_TheSameScoresGivingTheSameTrialsStopsAndBest(double[] scores)
    {
        // A study of networks and a study of the caller's models, handed the same sampler, the same deals and the same score
        // for each deal in turn, try the same candidates, stop the same trials between the same deals, and keep the same best.
        var networkScores = new Queue<double>(scores);
        var modelScores = new Queue<double>(scores);
        var networkStops = new List<string>();
        var modelStops = new List<string>();

        static bool Stops(DealsJudged judged, List<string> heard)
        {
            heard.Add($"{judged.Earlier.Count}:{string.Join(",", judged.Scores)}");

            return judged.Scores[^1] > 0.5;
        }

        var networks = new Study(Space, (candidate, network) => network.Dense(2).Relu().Dense(1).Adam(candidate.Number("rate")).BinaryCrossEntropy().Run(1, 1))
        {
            Sampler = new RandomSampler(11),
            Trials = 4,
            Deals = 2,
            Score = _ => networkScores.Dequeue(),
            DealStopper = judged => Stops(judged, networkStops),
        }.Run(deal => Passengers(deal), TestContext.Current.CancellationToken);

        var models = new Study<double>(Space, deal => new Trained<double>(modelScores.Dequeue(), 0))
        {
            Sampler = new RandomSampler(11),
            Trials = 4,
            Deals = 2,
            DealStopper = judged => Stops(judged, modelStops),
        }.Run(deal => Passengers(deal), TestContext.Current.CancellationToken);

        Assert.Equal(networks.Trials.Select(trial => trial.Candidate), models.Trials.Select(trial => trial.Candidate));
        Assert.Equal(networks.Trials.Select(trial => trial.Scores), models.Trials.Select(trial => trial.Scores));
        Assert.Equal(networks.Trials.Select(trial => trial.Pruned), models.Trials.Select(trial => trial.Pruned));
        Assert.Equal(networks.Trials.Select(trial => trial.Finished), models.Trials.Select(trial => trial.Finished));
        Assert.Equal(networkStops, modelStops);
        Assert.Equal(networks.Best.Number, models.Best.Number);
        Assert.Equal(networkScores.Count, modelScores.Count);
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
