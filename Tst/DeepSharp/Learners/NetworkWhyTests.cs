// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// Why a trained network answers as it does, asked of the rows it was read with: which columns its answers lean on, what they do as one
/// column moves through its range, and what one row's answer is made of. Each question perturbs the rows, serves them through the
/// network's own pipeline and reads the answers it gives, so a category, a scale and a log are all as the network knows them and none of
/// it is fitted again.
/// </summary>
public class NetworkWhyTests
{
    // Three of the columns carry the answer — x, and whether the group is "high" — and two carry nothing: a row number and noise.
    private static readonly string[] Names = ["t", "x", "group", "noise", "y"];

    private static string[][] Rows(int count)
    {
        var random = new Random(23);

        return [.. Enumerable.Range(1, count).Select(t =>
        {
            var x = random.NextDouble();
            var high = random.NextDouble() < 0.4;
            var y = (3 * x) + (high ? 2 : 0) + ((random.NextDouble() - 0.5) * 0.1);

            return new[]
            {
                t.ToString(CultureInfo.InvariantCulture),
                x.ToString("R", CultureInfo.InvariantCulture),
                high ? "high" : "low",
                random.NextDouble().ToString("R", CultureInfo.InvariantCulture),
                y.ToString("R", CultureInfo.InvariantCulture),
            };
        })];
    }

    private static InMemoryRowSource Source(IEnumerable<string[]> rows) => new(Names, rows.Select(row => (IReadOnlyList<string?>)row));

    private static PreparedData Prepared() =>
        Pdd.Create()
            .Read(Source(Rows(1200)), "twelve hundred")
            .Declare(schema => schema.Integer("t").Number("x", "noise", "y").Category("group"))
            .SplitAtRandom(0.70, 0.15, 5, testSeed: 9)
            .Target("y")
            .Drop("t")
            .EncodeCategories()
            .Normalise("x", Scale.MidRange)
            .Normalise("noise", Scale.MidRange)
            .WithTorch(network => network.Dense(12).Relu().Dense(1).Adam(0.02).MeanSquaredError().Run(seed: 1, epochs: 60))
            .Build()
            .Run();

    private static TrainedNetwork Trained(PreparedData prepared) =>
        prepared.Train(((LearnNetworkStep)prepared.Declaration.Learner!) with { Seed = 1 }, cancellation: TestContext.Current.CancellationToken);

    [Fact]
    public void TheAnswersLeanOnTheColumnsThatCarryThem_AndOnNothingElse()
    {
        var reliance = Trained(Prepared()).Leaning(Source(Rows(200)), new Redraws(seed: 3, repeats: 3));

        Assert.Equal(["group", "x"], reliance.Columns.Take(2).Select(column => column.Column).Order(StringComparer.Ordinal));
        Assert.Equal(0, reliance.Columns.Single(column => column.Column == "t").Mean);
        Assert.True(reliance.Columns.Single(column => column.Column == "noise").Mean < 0.2 * reliance.Columns[0].Mean, "the noise column moves the answers hardly at all");
        Assert.Equal(["y"], reliance.Answers);
    }

    [Fact]
    public void AsXMovesThroughItsRange_TheAnswersRiseWithIt()
    {
        var curve = Trained(Prepared()).Curve(Source(Rows(200)), new Moving("x", 5));

        Assert.Equal(5, curve.Values.Count);
        Assert.True(curve.Answers[^1][0] - curve.Answers[0][0] > 2, $"from {curve.Answers[0][0]} to {curve.Answers[^1][0]}");
    }

    [Fact]
    public void ARowsAnswerIsMadeOfWhatEachColumnAdds_AndTheyAddUpToItLessTheBackground()
    {
        var rows = Source(Rows(200));
        var trained = Trained(Prepared());
        var makeUp = trained.MakeUp(rows, 0, new Redraws(seed: 3, repeats: 40));
        var answer = trained.Predict(rows).Answers[0][0];

        Assert.Equal(answer, makeUp.Answer[0], 5);
        Assert.Equal(answer - makeUp.Background[0], makeUp.Columns.Sum(column => column.Contributions[0]), 5);
        Assert.Contains(makeUp.Columns.Take(2).Select(column => column.Column), name => name is "x" or "group");
    }

    [Fact]
    public void AskingWhyLearnsNothing_TheNetworkAndItsPipelineAreAsTheyWere()
    {
        var trained = Trained(Prepared());
        var file = trained.ToJson();

        _ = trained.Leaning(Source(Rows(100)), new Redraws(1, 2));
        _ = trained.Curve(Source(Rows(100)), new Moving("x", 4));
        _ = trained.MakeUp(Source(Rows(100)), 1, new Redraws(1, 5));

        Assert.Equal(file, trained.ToJson());
    }

    [Fact]
    public void AnEnsemblesReliance_IsThatOfItsCombinedAnswer()
    {
        var prepared = Prepared();
        var step = (LearnNetworkStep)prepared.Declaration.Learner!;
        var ensemble = prepared.TrainEnsemble([step with { Seed = 1 }, step with { Seed = 2 }], new EnsembleOptions { Cancellation = TestContext.Current.CancellationToken });
        var reliance = ensemble.Leaning(Source(Rows(150)), new Redraws(3, 2));
        var curve = ensemble.Curve(Source(Rows(150)), new Moving("group", 2));
        var makeUp = ensemble.MakeUp(Source(Rows(150)), 4, new Redraws(3, 10));

        Assert.Equal(["group", "x"], reliance.Columns.Take(2).Select(column => column.Column).Order(StringComparer.Ordinal));
        Assert.Equal(["high", "low"], curve.Values.Order(StringComparer.Ordinal));
        Assert.Equal(ensemble.Predict(Source(Rows(150))).Answers[4][0] - makeUp.Background[0], makeUp.Columns.Sum(column => column.Contributions[0]), 5);
    }
}
