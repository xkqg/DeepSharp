// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A study whose candidates are trained by a function of the caller's rather than declared as a network step: the sampler,
/// the deals, a pruner heard through the deal, a deal stopper and resuming from earlier trials all stay exactly as a study
/// of networks has them, and only what trains, and the model inside, is the caller's.
/// </summary>
public class StudyOfModelsTests
{
    private static readonly SearchSpace Space = new([new LogRange("rate", 1e-3, 1e-1), new WholeRange("units", 2, 8)]);

    private static Study<double> Studied(int trials = 3, int deals = 1) =>
        new(Space, deal => new Trained<double>(deal.Candidate.Number("rate"), deal.Candidate.Number("rate")))
        {
            Sampler = new RandomSampler(11),
            Trials = trials,
            Deals = deals,
        };

    [Fact]
    public void TheTrainerIsCalledOncePerDeal_AndTheReturnedScoreBecomesTheTrialsScore()
    {
        var calls = new List<Candidate>();
        var study = new Study<double>(Space, deal =>
        {
            calls.Add(deal.Candidate);

            return new Trained<double>(deal.Candidate.Number("rate") * 100, deal.Candidate.Number("rate"));
        })
        {
            Sampler = new RandomSampler(11),
            Trials = 3,
            Deals = 2,
        };

        var result = study.Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);

        Assert.Equal(3, result.Trials.Count);
        Assert.Equal(6, calls.Count);
        Assert.All(result.Trials, trial => Assert.Equal(2, trial.Scores.Count));
        Assert.All(result.Trials, trial => Assert.Equal(trial.Candidate.Number("rate") * 100, trial.Scores[0], 12));
        Assert.All(result.Trials, trial => Assert.Null(trial.Declared));
    }

    [Fact]
    public void DealStopperAndOnTrial_SeeTheCallersTrials()
    {
        var onTrialSeen = new List<Candidate>();
        var dealsSeen = new List<int>();
        var study = new Study<double>(Space, deal => new Trained<double>(deal.Candidate.Number("rate"), deal.Candidate.Number("rate")))
        {
            Sampler = new RandomSampler(11),
            Trials = 2,
            Deals = 2,
            OnTrial = trial => onTrialSeen.Add(trial.Candidate),
            DealStopper = judged =>
            {
                dealsSeen.Add(judged.Deal);

                return judged.Earlier.Count == 0 && judged.Deal == 0;
            },
        };

        var result = study.Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);

        Assert.Equal(2, onTrialSeen.Count);
        Assert.Equal(result.Trials.Select(trial => trial.Candidate), onTrialSeen);
        Assert.Equal([0, 0], dealsSeen);
        Assert.True(result.Trials[0].Pruned);
        Assert.Single(result.Trials[0].Scores);
        Assert.False(result.Trials[1].Pruned);
        Assert.Equal(2, result.Trials[1].Scores.Count);
        Assert.Equal(1, result.Best.Number);
    }

    [Fact]
    public void APrunerFedThroughHeard_CancelsTheRun()
    {
        var reported = new List<double>();
        var study = new Study<double>(Space, deal =>
        {
            for (var step = 0; step < 5; step++)
            {
                deal.Heard?.Invoke(step);
                reported.Add(step);

                if (deal.Cancellation.IsCancellationRequested)
                {
                    throw new OperationCanceledException(deal.Cancellation);
                }
            }

            return new Trained<double>(0, 0);
        })
        {
            Sampler = new RandomSampler(11),
            Trials = 2,
            Pruner = progress => progress.Earlier.Count == 0 && progress.SoFar.Count >= 2,
        };

        var result = study.Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);

        Assert.True(result.Trials[0].Pruned);
        Assert.False(result.Trials[1].Pruned);
        Assert.Equal(1, result.Best.Number);
        Assert.True(reported.Count < 10);
    }

    [Fact]
    public void APrunerThatNeverHears_IsTheStudyWithoutOne()
    {
        var study = new Study<double>(Space, deal =>
        {
            Assert.Null(deal.Heard);

            return new Trained<double>(deal.Candidate.Number("rate"), deal.Candidate.Number("rate"));
        })
        {
            Sampler = new RandomSampler(11),
            Trials = 2,
        };

        var result = study.Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);

        Assert.All(result.Trials, trial => Assert.False(trial.Pruned));
    }

    [Fact]
    public void AStudyHandedItsEarlierTrials_IsTheStudyThatNeverStopped()
    {
        var whole = Studied(trials: 5).Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);
        var first = Studied(trials: 3).Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);
        var resumed = Studied(trials: 5).Run(deal => StudyTests.Passengers(deal), first.Trials, TestContext.Current.CancellationToken);

        Assert.Equal(whole.Trials.Select(trial => trial.Number), resumed.Trials.Select(trial => trial.Number));
        Assert.Equal(whole.Trials.Select(trial => trial.Candidate), resumed.Trials.Select(trial => trial.Candidate));
        Assert.Equal(whole.Trials.Select(trial => trial.Scores), resumed.Trials.Select(trial => trial.Scores));
        Assert.Equal(whole.Best.Number, resumed.Best.Number);
    }

    [Fact]
    public void WhenTheBestTrialIsAnEarlierOne_ItIsTrainedAgain_ThroughTheHandedInTrainer()
    {
        var first = Studied(trials: 3).Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);
        var resumed = Studied(trials: 3).Run(deal => StudyTests.Passengers(deal), first.Trials, TestContext.Current.CancellationToken);

        Assert.Equal(first.Best.Number, resumed.Best.Number);
        Assert.Equal(first.Winner, resumed.Winner);
    }

    [Fact]
    public void ANondeterministicTrainer_IsRefused_NamingTheCallersFunction()
    {
        var calls = 0;
        var study = new Study<double>(Space, _ => new Trained<double>(calls++ == 0 ? 1.0 : 2.0, 0.0))
        {
            Sampler = new RandomSampler(11),
            Trials = 1,
        };

        var first = study.Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);
        var refused = Assert.Throws<InvalidOperationException>(() => study.Run(deal => StudyTests.Passengers(deal), first.Trials, TestContext.Current.CancellationToken));

        Assert.Contains($"trial {first.Best.Number}", refused.Message, StringComparison.Ordinal);
        Assert.Contains("trainer", refused.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void AListOfTrialsThatDeclareANetwork_IsRefused_ByName()
    {
        var networked = new DeepSharp.Learners.Networks.Study(Space, (candidate, network) => network.Dense(candidate.Whole("units")).Relu().Dense(1).Adam(candidate.Number("rate")).BinaryCrossEntropy().Run(seed: 1, epochs: 1))
        {
            Sampler = new RandomSampler(11),
            Trials = 1,
        }.Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);

        var study = new Study<double>(Space, deal => new Trained<double>(deal.Candidate.Number("rate"), deal.Candidate.Number("rate")))
        {
            Sampler = new RandomSampler(11),
            Trials = 2,
        };

        var refused = Assert.Throws<InvalidOperationException>(() => study.Run(deal => StudyTests.Passengers(deal), networked.Trials, TestContext.Current.CancellationToken));

        Assert.Contains("declares a network", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACandidateForAnotherSpace_IsRefused_NamingTheDimensionsTheSpaceHas()
    {
        var study = new Study<double>(Space, deal => new Trained<double>(0, 0))
        {
            Sampler = new Foreign(),
            Trials = 1,
        };

        var refused = Assert.Throws<InvalidOperationException>(() => study.Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken));

        Assert.Contains("'rate'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'units'", refused.Message, StringComparison.Ordinal);
    }

    private sealed class Foreign : ISampler
    {
        public Candidate Suggest(SearchSpace space, int trial, IReadOnlyList<Trial> finished) => new([new Setting("momentum", 0.9, null)]);
    }

    private sealed class OutOfRange : ISampler
    {
        public Candidate Suggest(SearchSpace space, int trial, IReadOnlyList<Trial> finished) => new([new Setting("rate", 99, null), new Setting("units", 4, null)]);
    }

    [Fact]
    public void ACandidateOfASettingTheSpaceCannotTake_IsRefused_NamingIt()
    {
        var study = new Study<double>(Space, deal => new Trained<double>(0, 0)) { Sampler = new OutOfRange(), Trials = 1 };

        var refused = Assert.Throws<InvalidOperationException>(() => study.Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken));

        Assert.Contains("'rate'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WhenNoTrialFinishes_TheStudyRefuses_SayingSo()
    {
        var study = new Study<double>(Space, deal => new Trained<double>(double.NaN, 0)) { Sampler = new RandomSampler(11), Trials = 2 };

        Assert.Contains("No trial finished", Assert.Throws<InvalidOperationException>(() => study.Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void DealsThatDoNotShareTheirTestRows_AreRefused_NamingTestSeed()
    {
        var study = new Study<double>(Space, deal => new Trained<double>(deal.Candidate.Number("rate"), deal.Candidate.Number("rate"))) { Sampler = new RandomSampler(11), Trials = 2, Deals = 2 };

        var refused = Assert.Throws<InvalidOperationException>(() => study.Run(deal => StudyTests.Passengers(deal, testSeed: null), TestContext.Current.CancellationToken));

        Assert.Contains("testSeed", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EarlierTrialsNumberedWithAGap_AreRefused()
    {
        var study = new Study<double>(Space, deal => new Trained<double>(deal.Candidate.Number("rate"), deal.Candidate.Number("rate"))) { Sampler = new RandomSampler(11), Trials = 1 };
        var gap = new Trial(1, new Candidate([new Setting("rate", 0.01, null), new Setting("units", 4, null)]), Declared: null, [0.5]);

        var refused = Assert.Throws<InvalidOperationException>(() => study.Run(deal => StudyTests.Passengers(deal), [gap], TestContext.Current.CancellationToken));

        Assert.Contains("numbered from nought without a gap", refused.Message, StringComparison.Ordinal);
    }

    private static PreparedData CalibratedRows()
    {
        var noise = new Random(11);
        var rows = Enumerable.Range(1, 400).Select(t =>
        {
            var x = noise.NextDouble();
            var y = (2 * x) + ((noise.NextDouble() + noise.NextDouble() - 1) * 0.05);

            return new string?[] { $"{t}", x.ToString("R", CultureInfo.InvariantCulture), y.ToString("R", CultureInfo.InvariantCulture) };
        }).ToList();

        return Pdd.Create()
            .Read(new InMemoryRowSource(["t", "x", "y"], rows), "four hundred")
            .Declare(schema => schema.Integer("t").Number("x", "y"))
            .Calibrate(0.20)
            .SplitAtRandom(0.50, 0.10, 5, testSeed: 9)
            .Drop("t")
            .Target("y")
            .Normalise("x", Scale.MidRange)
            .WithTorch(network => network.Dense(2).Relu().Dense(1).Adam(0.02).MeanSquaredError().Run(seed: 1, epochs: 1))
            .Build()
            .Run();
    }

    private static PreparedData CalibratedRowsMeasuringCoverage()
    {
        var noise = new Random(11);
        var rows = Enumerable.Range(1, 400).Select(t =>
        {
            var x = noise.NextDouble();
            var y = (2 * x) + ((noise.NextDouble() + noise.NextDouble() - 1) * 0.05);

            return new string?[] { $"{t}", x.ToString("R", CultureInfo.InvariantCulture), y.ToString("R", CultureInfo.InvariantCulture) };
        }).ToList();

        return Pdd.Create()
            .Read(new InMemoryRowSource(["t", "x", "y"], rows), "four hundred")
            .Declare(schema => schema.Integer("t").Number("x", "y"))
            .Calibrate(0.20)
            .SplitAtRandom(0.50, 0.10, 5, testSeed: 9)
            .Drop("t")
            .Target("y")
            .Normalise("x", Scale.MidRange)
            .Report(report => report.Measure(Metric.Coverage, Metric.Width).On(Part.Validation, Part.Test).As(Shown.Numbers))
            .WithTorch(network => network.Dense(2).Relu().Dense(1).Adam(0.02).MeanSquaredError().Run(seed: 1, epochs: 1))
            .Build()
            .Run();
    }

    [Fact]
    public void WithInterval_MeasuresTheEnsembleAgain_WhenTheReportNamesCoverageOrWidth()
    {
        var study = new Study<double>(Space, deal => new Trained<double>(0, deal.Candidate.Number("rate")));
        var rows = CalibratedRowsMeasuringCoverage();
        Candidate[] candidates = [new([new Setting("rate", 0.3, null)]), new([new Setting("rate", 0.7, null)])];

        static IReadOnlyList<double[]> Answer(double weight, IReadOnlyList<double[]> features) => [.. features.Select(_ => new[] { weight })];

        var ensemble = study.TrainEnsemble(rows, candidates, Answer);
        var withInterval = ensemble.WithInterval(0.10);

        Assert.Contains(Metric.Coverage, withInterval.Measures!.Ensemble.Metrics);
        Assert.Contains(Metric.Width, withInterval.Measures!.Ensemble.Metrics);
    }

    private static InMemoryRowSource Asked(int rows = 5) =>
        new(["t", "x"], [.. Enumerable.Range(1, rows).Select(t => new string?[] { $"{t}", (t / 10.0).ToString(CultureInfo.InvariantCulture) })]);

    [Fact]
    public void ModelEnsemble_PredictsMeasuresAndGivesAnInterval_ThroughTheFourPipelinesMethods()
    {
        var study = new Study<double>(Space, deal => new Trained<double>(0, deal.Candidate.Number("rate")));
        var rows = CalibratedRows();
        Candidate[] candidates = [new([new Setting("rate", 0.3, null)]), new([new Setting("rate", 0.7, null)])];

        static IReadOnlyList<double[]> Answer(double weight, IReadOnlyList<double[]> features) => [.. features.Select(_ => new[] { weight })];

        var ensemble = study.TrainEnsemble(rows, candidates, Answer);

        Assert.Equal([0.3, 0.7], ensemble.Members);
        Assert.Same(rows, ensemble.Prepared);

        var predicted = ensemble.Predict(Asked());

        Assert.Equal(5, predicted.Answers.Count);
        Assert.All(predicted.Answers, answer => Assert.Equal(0.5, answer[0], 12));
        Assert.Equal(["y"], predicted.AnswerNames);

        var withInterval = ensemble.WithInterval(0.10);

        Assert.True(withInterval.Interval!.Margins[0] > 0);

        var intervaled = withInterval.Predict(Asked());

        Assert.All(intervaled.Answers, answer => Assert.Equal(0.5, answer[0], 12));
        Assert.Equal(intervaled.Answers[0][0] - withInterval.Interval.Margins[0], intervaled.Lower![0][0], 12);
        Assert.Equal(intervaled.Answers[0][0] + withInterval.Interval.Margins[0], intervaled.Upper![0][0], 12);
    }

    [Fact]
    public void TrainEnsemble_RefusesFewerThanTwoCandidates_AndEnginesInTheOptions_ByName()
    {
        var study = new Study<double>(Space, deal => new Trained<double>(0, deal.Candidate.Number("rate")));
        var rows = CalibratedRows();
        Candidate[] one = [new([new Setting("rate", 0.3, null)])];
        Candidate[] two = [new([new Setting("rate", 0.3, null)]), new([new Setting("rate", 0.7, null)])];

        static IReadOnlyList<double[]> Answer(double weight, IReadOnlyList<double[]> features) => [.. features.Select(_ => new[] { weight })];

        Assert.Throws<ArgumentException>(() => study.TrainEnsemble(rows, one, Answer));

        var refused = Assert.Throws<ArgumentException>(() => study.TrainEnsemble(rows, two, Answer, new EnsembleOptions { Engines = new Engines() }));

        Assert.Contains("Engines", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ATrialOfAModelStudy_DeclaresNoNetwork_AndHasNoMembers()
    {
        // A trial a Study<TModel> tried never declares a network, and never sets Members itself: the fallback that used to
        // read [Declared] unconditionally now has a second case to answer for, since Declared can be null.
        var trial = new Trial(0, new Candidate([new Setting("rate", 0.3, null)]), Declared: null, [0.1]);

        Assert.Empty(trial.Members);
    }

    [Fact]
    public void ATrialsObjectiveMean_IsEmptyWithNoObjectives_AndTheMeanOfEachObjectiveOtherwise()
    {
        var candidate = new Candidate([new Setting("rate", 0.3, null)]);
        var none = new Trial(0, candidate, Declared: null, []);
        var weighed = new Trial(0, candidate, Declared: null, []) { ObjectiveScores = [[1, 10], [3, 20]] };

        Assert.Empty(none.ObjectiveMean);
        Assert.Equal([2, 15], weighed.ObjectiveMean);
    }
}
