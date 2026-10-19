// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Why a model answers as it does, asked of the columns the rows were read with — never of the numbers a column was made into, since a
/// category is one-hot and a log is undone, and the columns a person knows are the ones they were read with. The model is perturbed and
/// replayed through the pipeline: nothing is fitted, nothing is learned, and every draw comes from a seed the caller names. Three
/// questions: which column the answers move with when its values are shuffled among the rows; what the answers do as one column moves
/// through its range with the rest left as they are; and what one row's answer is made of, column by column.
/// </summary>
public class WhyTests
{
    private static readonly string[] Columns = ["t", "x", "sex", "z", "y"];

    private static string[][] Rows(int count = 120) =>
        [.. Enumerable.Range(1, count).Select(t => new[]
        {
            t.ToString(CultureInfo.InvariantCulture),
            ((t * 37 % 100) / 10.0).ToString("R", CultureInfo.InvariantCulture),
            t % 3 == 0 ? "female" : t % 5 == 0 ? "unknown" : "male",
            ((t * 13 % 50) / 5.0).ToString("R", CultureInfo.InvariantCulture),
            "0",
        })];

    private static InMemoryRowSource Source(IEnumerable<string[]> rows) => new(Columns, rows.Select(row => (IReadOnlyList<string?>)row));

    // A model that leans on x and on whether the sex is female, and on nothing else: y = 2x + 5 if female.
    private static IReadOnlyList<double[]> Said(IRowSource rows) =>
        [.. rows.Rows.Select(row => new[] { (2 * double.Parse(row[1]!, CultureInfo.InvariantCulture)) + (row[2] == "female" ? 5 : 0) })];

    private static PreparedData Prepared() =>
        Pdd.Create()
            .Read(Source(Rows()), "a hundred and twenty")
            .Declare(schema => schema.Integer("t").Number("x", "z", "y").Category("sex"))
            .SplitAtRandom(0.70, 0.15, 3)
            .Drop("t")
            .Target("y")
            .EncodeCategories()
            .Build()
            .Run();

    private static RowsAnswered Handed(int count = 120) => new(Source(Rows(count)), Said);

    // ---- the columns the answers lean on ----

    [Fact]
    public void AColumnTheAnswersLeanOn_RelyMoreThanOneTheyIgnore_TheFirstMostAndTheIgnoredAtNothing()
    {
        var reliance = Prepared().RelianceOn(Handed(), new Redraws(seed: 7, repeats: 5));

        Assert.Equal(["sex", "x"], reliance.Columns.Take(2).Select(column => column.Column).Order(StringComparer.Ordinal));
        Assert.True(reliance.Columns[0].Mean >= reliance.Columns[1].Mean);
        Assert.Equal(["t", "z"], reliance.Columns.Skip(2).Select(column => column.Column).Order(StringComparer.Ordinal));
        Assert.All(reliance.Columns.Take(2), column => Assert.True(column.Mean > 0));
        Assert.All(reliance.Columns.Skip(2), column => Assert.Equal(0, column.Mean));
        Assert.Equal(["y"], reliance.Answers);
        Assert.Equal(7, reliance.Seed);
        Assert.Equal(5, reliance.Repeats);
        Assert.DoesNotContain(reliance.Columns, column => column.Column == "y");
    }

    [Fact]
    public void TheSameSeedGivesTheSameReliance_AndAnotherSeedAnother()
    {
        var prepared = Prepared();
        var first = prepared.RelianceOn(Handed(), new Redraws(7, 4));
        var again = prepared.RelianceOn(Handed(), new Redraws(7, 4));
        var other = prepared.RelianceOn(Handed(), new Redraws(8, 4));

        Assert.Equal(first.Columns, again.Columns, new ColumnComparer());
        Assert.NotEqual(first.Columns.First(column => column.Column == "x").Mean, other.Columns.First(column => column.Column == "x").Mean);
    }

    [Fact]
    public void ACategoryRespondsAsMuchAsItsValuesMoveTheAnswer_AndASingleRepeatSaysNoSpread()
    {
        var reliance = Prepared().RelianceOn(Handed(), new Redraws(7, 1));
        var sex = reliance.Columns.Single(column => column.Column == "sex");

        // Shuffling the sex moves the answer by five wherever a female takes a non-female's place or the other way round: between
        // 0 and 5 on average, and certainly something.
        Assert.InRange(sex.Mean, 0.5, 5);
        Assert.Equal(0, sex.Spread);
    }

    [Fact]
    public void WhatIsReplayedThroughThePipelineLearnsNothing_ItsFitIsTheOneItWas()
    {
        var prepared = Prepared();
        var before = prepared.ToJson();

        _ = prepared.RelianceOn(Handed(), new Redraws(1, 2));
        _ = prepared.DependenceOn(Handed(), new Moving("x", 4));
        _ = prepared.MakeUpOf(Handed(), 3, new Redraws(1, 10));

        Assert.Equal(before, prepared.ToJson());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    public void ARepeatCountBelowOne_IsRefused_AndRowsThereAreNone_AreRefusedToo(int repeats)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Redraws(1, repeats));
        Assert.Throws<ArgumentException>(() => Prepared().RelianceOn(Handed(0), new Redraws(1, 1)));
    }

    [Fact]
    public void AnAnswerThatCountsOtherRowsThanItWasGiven_IsRefused()
    {
        var few = new RowsAnswered(Source(Rows(10)), rows => [.. Said(rows).Take(3)]);

        Assert.Throws<InvalidOperationException>(() => Prepared().RelianceOn(few, new Redraws(1, 1)));
    }

    // ---- what the answers do as one column moves ----

    [Fact]
    public void ANumbersCurve_RunsFromTheSmallestToTheLargestOfTheRowsHandedIn_TheRestLeftAsTheyAre()
    {
        var rows = Rows();
        double[] values = [.. rows.Select(row => double.Parse(row[1], CultureInfo.InvariantCulture))];
        var dependence = Prepared().DependenceOn(Handed(), new Moving("x", 5));
        var females = rows.Count(row => row[2] == "female") / (double)rows.Length;

        Assert.Equal("x", dependence.Column);
        Assert.Equal(5, dependence.Values.Count);
        Assert.Equal(values.Min(), dependence.Points![0]);
        Assert.Equal(values.Max(), dependence.Points[^1]);

        for (var at = 0; at < 5; at++)
        {
            Assert.Equal((2 * dependence.Points[at]) + (5 * females), dependence.Answers[at][0], 9);
        }
    }

    [Fact]
    public void ACategorysCurve_HasAPointForEveryValueTheTrainingRowsHeld_EvenOneTheRowsHandedInLack()
    {
        var onlyMales = new RowsAnswered(Source(Rows().Where(row => row[2] != "female").Take(30)), Said);
        var dependence = Prepared().DependenceOn(onlyMales, new Moving("sex", 2));

        Assert.Equal(["female", "male", "unknown"], dependence.Values.Order(StringComparer.Ordinal));
        Assert.Null(dependence.Points);

        var female = dependence.Answers[dependence.Values.ToList().IndexOf("female")][0];
        var male = dependence.Answers[dependence.Values.ToList().IndexOf("male")][0];

        Assert.Equal(5, female - male, 9);
    }

    [Fact]
    public void ACurveOfAnIntegerColumn_IsOfWholeNumbers_NoneRepeated()
    {
        string[][] whole = [.. Rows().Select(row => new[] { row[0], ((int)double.Parse(row[1], CultureInfo.InvariantCulture)).ToString(CultureInfo.InvariantCulture), row[2], row[3], row[4] })];
        var prepared = Pdd.Create()
            .Read(Source(whole), "a hundred and twenty")
            .Declare(schema => schema.Integer("t", "x").Number("z", "y").Category("sex"))
            .SplitAtRandom(0.70, 0.15, 3)
            .Drop("t")
            .Target("y")
            .EncodeCategories()
            .Build()
            .Run();
        var dependence = prepared.DependenceOn(new RowsAnswered(Source(whole), Said), new Moving("x", 20));

        Assert.Equal(dependence.Values.Distinct().Count(), dependence.Values.Count);
        Assert.All(dependence.Points!, point => Assert.Equal(Math.Round(point), point));
        Assert.True(dependence.Values.Count < 20);
    }

    [Fact]
    public void ACurveIsRefusedOfWhatHasNoRangeToMoveThrough_OfTheAnswer_AndOfAColumnTheRowsLack()
    {
        var prepared = Prepared();

        Assert.Throws<ArgumentException>(() => prepared.DependenceOn(Handed(), new Moving("y", 4)));
        Assert.Throws<ArgumentException>(() => prepared.DependenceOn(Handed(), new Moving("nowhere", 4)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Moving("x", 1));
        Assert.Throws<ArgumentException>(() => new Moving(" ", 4));
    }

    // ---- what one row's answer is made of ----

    [Fact]
    public void AnAnswersContributions_AddUpToTheAnswerLessTheBackground_Whatever_TheSamplesDrawn()
    {
        var makeUp = Prepared().MakeUpOf(Handed(), 2, new Redraws(seed: 5, repeats: 25));

        // Row 3 of the list: t = 3, a female, so its answer carries the five.
        var answer = Said(Handed().Rows)[2][0];

        Assert.Equal(answer, makeUp.Answer[0], 9);
        Assert.Equal(answer - makeUp.Background[0], makeUp.Columns.Sum(column => column.Contributions[0]), 9);
        Assert.Equal(2, makeUp.Row);
        Assert.Equal(25, makeUp.Samples);
    }

    [Fact]
    public void WithManySamples_TheContributionsOfAnAdditiveModel_ApproachWhatEachColumnAdds()
    {
        var rows = Rows();
        var makeUp = Prepared().MakeUpOf(Handed(), 2, new Redraws(seed: 5, repeats: 4000));
        double[] xs = [.. rows.Select(row => double.Parse(row[1], CultureInfo.InvariantCulture))];
        var females = rows.Count(row => row[2] == "female") / (double)rows.Length;
        var x = makeUp.Columns.Single(column => column.Column == "x").Contributions[0];
        var sex = makeUp.Columns.Single(column => column.Column == "sex").Contributions[0];

        // y = 2x + 5 for a female: x adds 2 (x of the row less the average x), the sex adds 5 less the share of females, and the rest add nothing.
        Assert.Equal(2 * (xs[2] - xs.Average()), x, 0.35);
        Assert.Equal(5 * (1 - females), sex, 0.35);
        Assert.All(makeUp.Columns.Where(column => column.Column is "z" or "t"), column => Assert.Equal(0, column.Contributions[0]));
    }

    [Fact]
    public void ARowThatIsNotOneOfTheRowsHandedIn_IsRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Prepared().MakeUpOf(Handed(), 120, new Redraws(1, 1)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Prepared().MakeUpOf(Handed(), -1, new Redraws(1, 1)));
    }

    // Rows of a flag, a group, a number and an answer; the group is declared a category and dropped before anything encodes it.
    private static readonly string[] OtherColumns = ["flag", "group", "x", "y"];

    private static InMemoryRowSource OtherSource(Func<int, string?> x) => new(
        OtherColumns,
        [.. Enumerable.Range(1, 60).Select(t => (IReadOnlyList<string?>)[t % 2 == 0 ? "true" : "false", t % 3 == 0 ? "b" : "a", x(t), "0"])]);

    private static PreparedData OtherPrepared() =>
        Pdd.Create()
            .Read(OtherSource(t => (t / 10.0).ToString("R", CultureInfo.InvariantCulture)), "sixty")
            .Declare(schema => schema.Boolean("flag").Category("group").Number("x", "y"))
            .SplitAtRandom(0.70, 0.15, 3)
            .Drop("group")
            .Target("y")
            .Build()
            .Run();

    private static IReadOnlyList<double[]> SaidOfX(IRowSource rows) =>
        [.. rows.Rows.Select(row => new[] { string.IsNullOrWhiteSpace(row[2]) ? 0 : 2 * double.Parse(row[2]!, CultureInfo.InvariantCulture) })];

    [Fact]
    public void AColumnThatIsNeitherANumberNorACategory_HasNoRangeToMoveThrough()
    {
        var refused = Assert.Throws<ArgumentException>(
            () => OtherPrepared().DependenceOn(new RowsAnswered(OtherSource(t => (t / 10.0).ToString("R", CultureInfo.InvariantCulture)), SaidOfX), new Moving("flag", 3)));

        Assert.Contains("which has no range to move through", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AColumnThePipelineNeverDeclared_HasNoRangeToMoveThrough_SayingSo()
    {
        var rows = OtherSource(t => (t / 10.0).ToString("R", CultureInfo.InvariantCulture));
        var withExtra = new InMemoryRowSource([.. OtherColumns, "extra"], rows.Rows.Select(row => (IReadOnlyList<string?>)[.. row, "?"]));

        var refused = Assert.Throws<ArgumentException>(() => OtherPrepared().DependenceOn(new RowsAnswered(withExtra, SaidOfX), new Moving("extra", 3)));

        Assert.Contains("something the pipeline does not declare", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACategoryNoEncoderLearned_MovesThroughTheValuesTheRowsHandedInHold()
    {
        var dependence = OtherPrepared().DependenceOn(new RowsAnswered(OtherSource(t => (t / 10.0).ToString("R", CultureInfo.InvariantCulture)), SaidOfX), new Moving("group", 3));

        Assert.Equal(["a", "b"], dependence.Values);
    }

    [Fact]
    public void ANumberTheRowsHandedInNeverHold_HasNoRangeToMoveThrough()
    {
        var refused = Assert.Throws<ArgumentException>(
            () => OtherPrepared().DependenceOn(new RowsAnswered(OtherSource(_ => null), SaidOfX), new Moving("x", 3)));

        Assert.Contains("holds no number in the rows handed in", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void APipelineThatNamesNoAnswer_HasNoneToAskWhyOf()
    {
        var unanswered = Pdd.Create()
            .Read(Source(Rows()), "a hundred and twenty")
            .Declare(schema => schema.Integer("t").Number("x", "z", "y").Category("sex"))
            .SplitAtRandom(0.70, 0.15, 3)
            .Build()
            .Run();

        Assert.Contains("names no answer", Assert.Throws<ArgumentException>(() => unanswered.RelianceOn(Handed(), new Redraws(1, 1))).Message, StringComparison.Ordinal);
    }

    private sealed class ColumnComparer : IEqualityComparer<ColumnReliance>
    {
        public bool Equals(ColumnReliance? left, ColumnReliance? right) =>
            left is not null && right is not null && left.Column == right.Column && left.Mean.Equals(right.Mean) && left.Spread.Equals(right.Spread);

        public int GetHashCode(ColumnReliance value) => value.Column.GetHashCode(StringComparison.Ordinal);
    }
}
