// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Learners.ML;
using DeepSharp.Learners.MLNet;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners.MLNet;

/// <summary>
/// A tree is asked why as a network is: the questions are the pipeline's, so the second learner needs only a way of answering rows that
/// have arrived, which is the way it serves them.
/// </summary>
public class WhyTests
{
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

    private static InMemoryRowSource Source(int count = 1200) => new(Names, Rows(count).Select(row => (IReadOnlyList<string?>)row));

    private static TrainedMLModel Tree() =>
        Pdd.Create()
            .Read(Source(), "twelve hundred")
            .Declare(schema => schema.Integer("t").Number("x", "noise", "y").Category("group"))
            .SplitAtRandom(0.70, 0.15, 5, testSeed: 9)
            .Target("y")
            .Drop("t")
            .EncodeCategories()
            .WithML(trainer => trainer.FastTree(trees: 30))
            .Build()
            .TrainWithML();

    [Fact]
    public void ATreeServedRows_AnswersAsItAnswersThePartsRows()
    {
        var tree = Tree();
        var served = tree.Predict(Source());
        var test = tree.Predict(Part.Test);
        double[] expected = [.. Enumerable.Range(0, tree.Prepared.Table.RowCount).Where(row => tree.Prepared.Parts[row] == Part.Test).Select(row => served[tree.Prepared.Table.Identities[row].ReadAt][0])];

        Assert.Equal(1200, served.Count);
        Assert.Equal(test.Count, expected.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(test[at], expected[at], 5);
        }
    }

    [Fact]
    public void AFeatureTheRowsLack_IsRefusedWhenTheyAreServed()
    {
        var tree = Tree();

        Assert.Throws<InvalidOperationException>(() => tree.Predict(new InMemoryRowSource(["t", "x", "group"], [["1", "0.5", "low"]])));
        Assert.Throws<ArgumentNullException>(() => tree.Predict(null!));
    }

    [Fact]
    public void TheTreesAnswersLeanOnTheColumnsThatCarryThem_ItsCurveRises_AndARowsAnswerIsMadeOfParts()
    {
        var tree = Tree();
        var rows = Source(200);
        var reliance = tree.Leaning(rows, new Redraws(seed: 3, repeats: 3));
        var curve = tree.Curve(rows, new Moving("x", 5));
        var makeUp = tree.MakeUp(rows, 0, new Redraws(seed: 3, repeats: 40));

        Assert.Equal(["group", "x"], reliance.Columns.Take(2).Select(column => column.Column).Order(StringComparer.Ordinal));
        Assert.Equal(0, reliance.Columns.Single(column => column.Column == "t").Mean);
        Assert.True(curve.Answers[^1][0] - curve.Answers[0][0] > 2);
        Assert.Equal(tree.Predict(rows)[0][0] - makeUp.Background[0], makeUp.Columns.Sum(column => column.Contributions[0]), 5);
    }
}
