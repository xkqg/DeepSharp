// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// Several networks answer as one. The members are declarations of one pipeline — the best few of a study, or one declaration under
/// several seeds — trained on the rows it prepared once; or whole models, each behind a pipeline of its own. Each is measured alone and
/// the ensemble beside them on the same rows, the members' answers meeting in the answer's own units. What an ensemble predicts for
/// served rows is its members' answers combined, each member served and brought back through its own pipeline; a row a member's
/// pipeline left out, members that answer another question, and votes on an amount are refused.
/// </summary>
public class TrainedEnsembleTests
{
    private static readonly SearchSpace Space = new([new LogRange("rate", 1e-3, 1e-1), new WholeRange("units", 2, 8)]);

    private static Pipeline Declared(int seed = 1) => Declared(seed, dropsAGap: false);

    // The passengers, a gap in age filled with the median; or every passenger whose age is a gap dropped, served rows too.
    private static Pipeline Declared(int seed, bool dropsAGap) =>
        (dropsAGap
            ? Pdd.Create().ReadCsv(Repository.Data("titanic.csv"))
                .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("sex").Optional("age", ColumnKind.Number).Number("fare"))
                .DropGaps("age")
                .SplitAtRandom(0.70, 0.15, 100, 7)
            : Pdd.Create().ReadCsv(Repository.Data("titanic.csv"))
                .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("sex").Optional("age", ColumnKind.Number).Number("fare"))
                .SplitAtRandom(0.70, 0.15, 100, 7)
                .FillMissing("age", With.Median))
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
        CsvRowSource.FromText(string.Join('\n', File.ReadLines(Repository.Data("titanic.csv")).Take(rows + 1)) + "\n");

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

        // Without options: averaged, on the light engine, unstoppable.
        Assert.Equal(Blend.Average, prepared.TrainEnsemble([step, step with { Seed = 2 }]).Blend);
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
    public void NetworksTrainedBehindOtherFits_AreMembersEachBehindItsOwn_AndTheEnsembleHasNoOnePipelineToHandBack()
    {
        // Once refused as members of one ensemble: now each answers behind the fit it was trained behind.
        var (prepared, step) = Prepared();
        var other = StudyTests.Passengers(3).RunFor(step.Needs);
        var ensemble = TrainedEnsemble.Of([Alone(prepared, step), Alone(other, step)]);

        Assert.Same(prepared, ensemble.Members[0].Prepared);
        Assert.Same(other, ensemble.Members[1].Prepared);
        Assert.Equal(12, ensemble.Predict(Served()).Answers.Count);
        Assert.Contains("2 pipelines", Assert.Throws<InvalidOperationException>(() => ensemble.Prepared).Message, StringComparison.Ordinal);
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

    // Six hundred rows of an amount above nought: as they are, or read on a log scale, so what a network says is a log; the same
    // rows divided alike either way — a fifth set aside to calibrate on, unless said — and the answer the one column or the other.
    internal static PreparedData Regression(bool logged = false, string answer = "y", bool calibrated = true, int testSeed = 9)
    {
        var noise = new Random(11);
        var rows = Enumerable.Range(1, 600).Select(t =>
        {
            var x = noise.NextDouble();
            var y = 1 + (3 * x) + ((noise.NextDouble() + noise.NextDouble() - 1) * 0.1);

            return new string?[] { $"{t}", x.ToString("R", System.Globalization.CultureInfo.InvariantCulture), y.ToString("R", System.Globalization.CultureInfo.InvariantCulture) };
        }).ToList();
        var read = Pdd.Create()
            .Read(new InMemoryRowSource(["t", "x", "y"], rows), "six hundred")
            .Declare(schema => schema.Integer("t").Number("x", "y"));

        var shaped = logged ? read.Reshape("y", Maths.Log) : read;

        return (calibrated ? shaped.Calibrate(0.20) : shaped)
            .SplitAtRandom(0.50, 0.10, 5, testSeed)
            .Drop("t")
            .Target(answer)
            .Normalise(answer == "y" ? "x" : "y", Scale.MidRange)
            .WithTorch(network => network.Dense(4).Relu().Dense(1).Adam(0.02).MeanSquaredError().Run(seed: 1, epochs: 3))
            .Build()
            .Run();
    }

    internal static TrainedNetwork Regressed(PreparedData prepared, int seed = 1) =>
        prepared.Train(((LearnNetworkStep)prepared.Declaration.Learner!) with { Seed = seed }, cancellation: TestContext.Current.CancellationToken);

    internal static InMemoryRowSource Asked(int rows = 8) =>
        new(["t", "x"], [.. Enumerable.Range(900, rows).Select(t => new string?[] { $"{t}", ((t % 10) / 10.0).ToString(System.Globalization.CultureInfo.InvariantCulture) })]);

    [Fact]
    public void AnEnsembleOfWholeModels_EachMemberReplaysTheRowsThroughItsOwnPipeline_AndTheAnswersMeetInOriginalUnits()
    {
        // One member was trained on the answer as it is, the other on its log: each comes back through its own pipeline's way
        // back — the log undone by an exponential — and only then are they averaged.
        var plain = Regressed(Regression());
        var logged = Regressed(Regression(logged: true));
        var ensemble = TrainedEnsemble.Of([plain, logged]);
        var rows = Asked();

        var predicted = ensemble.Predict(rows);
        var alone = new[] { plain.Predict(rows), logged.Predict(rows) };

        Assert.Equal(8, predicted.Answers.Count);
        Assert.Equal(alone[0].HandedInAt, predicted.HandedInAt);

        for (var row = 0; row < 8; row++)
        {
            Assert.Equal((alone[0].Answers[row][0] + alone[1].Answers[row][0]) / 2, predicted.Answers[row][0], 1e-12);
        }

        Assert.Same(logged.Prepared, ensemble.Members[1].Prepared);

        // There is no one pipeline to hand back: the refusal says where each member's own is.
        var refused = Assert.Throws<InvalidOperationException>(() => ensemble.Prepared);

        Assert.Contains("Members", refused.Message, StringComparison.Ordinal);
        Assert.Contains("2 pipelines", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AServedRowOneMemberDidNotAnswer_IsRefusedByName(bool droppingFirst)
    {
        // The sixth passenger handed in has no age: the member whose pipeline drops a gap in age answers nothing for that row,
        // and an ensemble answers a row with every member's answer or not at all.
        var filled = Alone(Declared(1).RunFor(Needs.OneScale), Step(Declared(1)));
        var dropping = Alone(Declared(1, dropsAGap: true).RunFor(Needs.OneScale), Step(Declared(1, dropsAGap: true)));
        var ensemble = TrainedEnsemble.Of(droppingFirst ? [dropping, filled] : [filled, dropping]);

        var refused = Assert.Throws<InvalidOperationException>(() => ensemble.Predict(Served()));

        Assert.Contains("Row 6 handed in", refused.Message, StringComparison.Ordinal);
        Assert.Contains($"not by member {(droppingFirst ? 1 : 2)}", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void MembersThatNameOtherAnswers_AreRefused_NamingBoth()
    {
        var refused = Assert.Throws<ArgumentException>(() => TrainedEnsemble.Of([Regressed(Regression()), Regressed(Regression(answer: "x"))]));

        Assert.Contains("Member 2", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'x'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'y'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void VotesOnAnAmount_AreRefusedWhereTheEnsembleIsMade_BeforeItServesOrCalibrates()
    {
        var prepared = Regression();
        var refused = Assert.Throws<InvalidOperationException>(() => TrainedEnsemble.Of([Regressed(prepared, 1), Regressed(prepared, 2)], Blend.Vote));

        Assert.Contains("no class", refused.Message, StringComparison.Ordinal);
        Assert.Contains("averaged", refused.Message, StringComparison.Ordinal);
    }
}
