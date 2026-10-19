// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// Several networks trained behind one fit answer as one. The members are declarations of one pipeline — the best few of a study,
/// or one declaration under several seeds — trained on the rows it prepared once; each is measured alone and the ensemble beside
/// them on the same rows, the members' answers meeting in the answer's own units. What an ensemble predicts for served rows is
/// its members' answers combined, the rows replayed and named once for all of them.
/// </summary>
public class TrainedEnsembleTests
{
    private static readonly SearchSpace Space = new([new LogRange("rate", 1e-3, 1e-1), new WholeRange("units", 2, 8)]);

    private static Pipeline Declared(int seed = 1) =>
        Pdd.Create().ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("sex").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitAtRandom(0.70, 0.15, 100, 7)
            .FillMissing("age", With.Median)
            .EncodeCategories()
            .Normalise("age", Scale.MidRange)
            .Normalise("fare", Scale.MidRange)
            .Normalise("sibsp", Scale.MidRange)
            .Normalise("parch", Scale.MidRange)
            .Target("survived")
            .Report(report => report.Measure(Metric.Accuracy, Metric.Brier).On(Part.Train, Part.Validation, Part.Test).As(Shown.Numbers))
            .WithTorch(network => network.Dense(4).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy().Run(seed, 3))
            .Build();

    private static LearnNetworkStep Step(Pipeline pipeline) => Assert.IsType<LearnNetworkStep>(pipeline.Declaration.Learner);

    private readonly record struct Setup(PreparedData Prepared, LearnNetworkStep Step);

    private static Setup Prepared()
    {
        var pipeline = Declared();
        var step = Step(pipeline);

        return new Setup(pipeline.RunFor(step.Needs), step);
    }

    private static TrainedEnsemble Trained(PreparedData prepared, IReadOnlyList<LearnNetworkStep> steps, Blend blend = Blend.Average) =>
        prepared.TrainEnsemble(steps, new EnsembleOptions { Blend = blend, Cancellation = TestContext.Current.CancellationToken });

    private static TrainedNetwork Alone(PreparedData prepared, LearnNetworkStep step) =>
        prepared.Train(step, cancellation: TestContext.Current.CancellationToken);

    private static CsvRowSource Served(int rows = 12) =>
        CsvRowSource.FromText(string.Join('\n', File.ReadLines(Repository.Data("titanic.csv")).Take(rows + 1).Select(line => line)) + "\n");

    [Fact]
    public void TheMembers_AreTrainedBehindOneFit_EachMeasuredAloneAndTheEnsembleBesideThem()
    {
        var (prepared, step) = Prepared();
        var ensemble = Trained(prepared, [step, step with { Seed = 2 }, step with { Seed = 3 }]);

        Assert.Equal(3, ensemble.Members.Count);
        Assert.Same(prepared, ensemble.Prepared);
        Assert.All(ensemble.Members, member => Assert.Same(prepared, member.Prepared));
        Assert.Equal([1L, 2, 3], ensemble.Members.Select(member => member.TrainedOn.Seed));

        var measured = ensemble.Measures!;

        Assert.Equal(3, measured.Members.Count);

        for (var at = 0; at < 3; at++)
        {
            // A member measured in the ensemble is the member measured when it was trained, to the last bit.
            Assert.Equal(
                ensemble.Members[at].Measures!.Parts.SelectMany(part => part.Values.Select(value => value.Value)),
                measured.Members[at].Parts.SelectMany(part => part.Values.Select(value => value.Value)));
        }

        Assert.Equal(Blend.Average, ensemble.Blend);
        Assert.Equal(Blend.Average, measured.Blend);
    }

    [Fact]
    public void WhatTheEnsemblePredicts_IsItsMembersAnswersCombined_InTheAnswersOwnUnits()
    {
        var (prepared, step) = Prepared();
        var ensemble = Trained(prepared, [step, step with { Seed = 2 }]);
        var rows = Served();

        var predicted = ensemble.Predict(rows);
        var members = ensemble.Members.Select(member => member.Predict(rows)).ToArray();

        Assert.Equal(12, predicted.Answers.Count);
        Assert.Equal(members[0].AnswerNames, predicted.AnswerNames);
        Assert.Equal(members[0].HandedInAt, predicted.HandedInAt);
        Assert.Equal(members[0].Unfamiliar, predicted.Unfamiliar);

        for (var row = 0; row < 12; row++)
        {
            Assert.Equal((members[0].Answers[row][0] + members[1].Answers[row][0]) / 2, predicted.Answers[row][0], 1e-12);
        }
    }

    [Fact]
    public void AVotingEnsemble_AnswersWithTheClassMostMembersNamed()
    {
        var (prepared, step) = Prepared();
        var ensemble = Trained(prepared, [step, step with { Seed = 2 }, step with { Seed = 3 }], Blend.Vote);
        var rows = Served();

        var predicted = ensemble.Predict(rows);
        var members = ensemble.Members.Select(member => member.Predict(rows)).ToArray();

        for (var row = 0; row < 12; row++)
        {
            var yes = members.Count(member => member.Answers[row][0] >= 0.5);

            Assert.Equal(yes >= 2 ? 1.0 : 0.0, predicted.Answers[row][0]);
        }

        Assert.All(ensemble.Measures!.Ensemble.Parts, part => Assert.All(part.Predicted, row => Assert.True(row[0] is 0.0 or 1.0)));
    }

    [Fact]
    public void TheSameDeclarationUnderTheSameSeed_IsRefusedTwice_ForItTrainsTheSameNetworkTwice()
    {
        var (prepared, step) = Prepared();
        var refused = Assert.Throws<ArgumentException>(() => Trained(prepared, [step, step with { Seed = 2 }, step]));

        Assert.Contains("1", refused.Message, StringComparison.Ordinal);
        Assert.Contains("3", refused.Message, StringComparison.Ordinal);
        Assert.Contains("seed", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void FewerThanTwoMembers_AreNoEnsemble()
    {
        var (prepared, step) = Prepared();

        Assert.Throws<ArgumentException>(() => Trained(prepared, [step]));
        Assert.Throws<ArgumentException>(() => Trained(prepared, []));
        Assert.Throws<ArgumentNullException>(() => Trained(prepared, null!));
        Assert.Throws<ArgumentException>(() => TrainedEnsemble.Of([Alone(prepared, step)]));
        Assert.Contains("0 were given", Assert.Throws<ArgumentException>(() => TrainedEnsemble.Of([])).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void NetworksTrainedBehindOtherFits_AreRefusedAsMembersOfOneEnsemble()
    {
        var (prepared, step) = Prepared();
        var other = StudyTests.Passengers(3).RunFor(step.Needs);
        var refused = Assert.Throws<ArgumentException>(() => TrainedEnsemble.Of([Alone(prepared, step), Alone(other, step)]));

        Assert.Contains("another fit", refused.Message, StringComparison.Ordinal);
        Assert.Contains("2", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheBestFewOfAStudy_AreTrainedAgainAsAnEnsemble_BestFirst()
    {
        var study = new Study(Space, (candidate, network) => network
            .Dense(candidate.Whole("units")).Relu().Dense(1)
            .Adam(candidate.Number("rate"))
            .BinaryCrossEntropy()
            .Run(seed: 20260929, epochs: 2))
        {
            Sampler = new RandomSampler(11),
            Trials = 4,
        };

        var result = study.Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);
        var best = result.Top(3);

        Assert.Equal(3, best.Count);
        Assert.Equal(result.Best.Declared, best[0]);
        Assert.Equal(result.Trials.Where(trial => trial.Finished).OrderBy(trial => trial.Score).Take(3).Select(trial => trial.Declared), best);

        var prepared = StudyTests.Passengers(0).RunFor(best[0].Needs);

        Assert.Equal(3, Trained(prepared, best).Members.Count);
        Assert.Throws<ArgumentOutOfRangeException>(() => result.Top(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => result.Top(5));
    }
}
