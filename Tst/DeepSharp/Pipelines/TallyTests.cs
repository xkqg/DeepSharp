// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What a column holds over some rows, counted — each value once with how many rows hold it — is all either distance needs: the
/// two-sample Kolmogorov–Smirnov distance and the total variation worked out from the counts are, to the bit, the distances worked
/// out from the values one by one, ties included, so a reference kept as counts measures exactly what the rows would have.
/// </summary>
public class TallyTests
{
    // The distance as 0.11.0 worked it out from the values themselves, kept here as the oracle the counted form is held to.
    private static double KolmogorovSmirnovOneByOne(IReadOnlyList<double> first, IReadOnlyList<double> second)
    {
        double[] one = [.. first.Order()];
        double[] other = [.. second.Order()];
        int at = 0, elsewhere = 0;
        var most = 0d;

        while (at < one.Length || elsewhere < other.Length)
        {
            var value = elsewhere >= other.Length || (at < one.Length && one[at] <= other[elsewhere]) ? one[at] : other[elsewhere];

            while (at < one.Length && one[at] <= value)
            {
                at++;
            }

            while (elsewhere < other.Length && other[elsewhere] <= value)
            {
                elsewhere++;
            }

            most = Math.Max(most, Math.Abs(((double)at / one.Length) - ((double)elsewhere / other.Length)));
        }

        return most;
    }

    private static double TotalVariationOneByOne(IReadOnlyList<string> first, IReadOnlyList<string> second) =>
        first.Concat(second).Distinct(StringComparer.Ordinal)
            .Select(word => Math.Abs(((double)first.Count(each => each == word) / first.Count) - ((double)second.Count(each => each == word) / second.Count)))
            .Sum() / 2;

    [Fact]
    public void NumbersCounted_AreEachValueOnceInOrder_WithHowManyRowsHoldIt_AndTheGapsBeside()
    {
        var tally = NumbersTallied.Of([3, 1, 3, 2, 3], gaps: 2);

        Assert.Equal([1, 2, 3], tally.Values);
        Assert.Equal([1, 1, 3], tally.Counts);
        Assert.Equal(5, tally.Held);
        Assert.Equal(2, tally.Gaps);
        Assert.Equal(Distance.KolmogorovSmirnov, tally.Distance);
    }

    [Fact]
    public void WordsCounted_AreEachWordOnceInTheOrderItFirstStood_WithHowManyRowsHoldIt()
    {
        var tally = WordsTallied.Of(["b", "a", "b", "c", "b"], gaps: 0);

        Assert.Equal(["b", "a", "c"], tally.Words);
        Assert.Equal([3, 1, 1], tally.Counts);
        Assert.Equal(5, tally.Held);
        Assert.Equal(Distance.TotalVariation, tally.Distance);
    }

    [Theory]
    [InlineData(new double[] { 3 }, new double[] { 3, 3, 3 }, 0)]
    [InlineData(new double[] { 1, 2, 2, 3 }, new double[] { 2, 2, 3, 3 }, 0.25)]
    public void TheCountedKolmogorovSmirnov_TakesTiesWhole(double[] first, double[] second, double apart)
    {
        Assert.Equal(apart, NumbersTallied.Of(first, 0).ApartFrom(NumbersTallied.Of(second, 0)), 15);
        Assert.Equal(apart, PartLikeness.KolmogorovSmirnov(first, second), 15);
    }

    [Fact]
    public void NinetyNoughtsAndTenOnes_AgainstOneOne_AreATenthApart()
    {
        double[] reference = [.. Enumerable.Repeat(0d, 90), .. Enumerable.Repeat(1d, 10)];

        Assert.Equal(0.9, NumbersTallied.Of(reference, 0).ApartFrom(NumbersTallied.Of([1], 0)), 15);
        Assert.Equal(0.1, NumbersTallied.Of(reference, 0).ApartFrom(NumbersTallied.Of([0], 0)), 15);
    }

    [Fact]
    public void TheCountedDistances_AreTheDistancesOfTheValuesOneByOne_ToTheBit()
    {
        var random = new Random(20261017);

        for (var round = 0; round < 4000; round++)
        {
            double[] first = [.. Enumerable.Range(0, random.Next(1, 60)).Select(_ => (double)random.Next(0, 12) / (1 + random.Next(0, 3)))];
            double[] second = [.. Enumerable.Range(0, random.Next(1, 60)).Select(_ => (double)random.Next(0, 12) / (1 + random.Next(0, 3)))];
            string[] words = [.. Enumerable.Range(0, random.Next(1, 40)).Select(_ => ((char)('a' + random.Next(0, 6))).ToString())];
            string[] others = [.. Enumerable.Range(0, random.Next(1, 40)).Select(_ => ((char)('a' + random.Next(0, 7))).ToString())];

            Assert.Equal(KolmogorovSmirnovOneByOne(first, second), NumbersTallied.Of(first, 0).ApartFrom(NumbersTallied.Of(second, 0)));
            Assert.Equal(KolmogorovSmirnovOneByOne(first, second), PartLikeness.KolmogorovSmirnov(first, second));
            Assert.Equal(TotalVariationOneByOne(words, others), WordsTallied.Of(words, 0).ApartFrom(WordsTallied.Of(others, 0)));
            Assert.Equal(TotalVariationOneByOne(words, others), PartLikeness.TotalVariation(words, others));
        }
    }

    [Fact]
    public void ACountedSetOfNothing_HasNoDistance_AndTwoKindsAreNotCompared()
    {
        Assert.True(double.IsNaN(NumbersTallied.Of([], 3).ApartFrom(NumbersTallied.Of([1], 0))));
        Assert.True(double.IsNaN(WordsTallied.Of(["a"], 0).ApartFrom(WordsTallied.Of([], 1))));
        Assert.Throws<ArgumentException>(() => NumbersTallied.Of([1], 0).ApartFrom(WordsTallied.Of(["a"], 0)));
        Assert.Throws<ArgumentException>(() => WordsTallied.Of(["a"], 0).ApartFrom(NumbersTallied.Of([1], 0)));
    }

    [Fact]
    public void CountsWrittenDown_AreReadBackAsTheSameTally()
    {
        var numbers = new NumbersTallied([2, 5], [2, 1], 1);
        var words = new WordsTallied(["x", "y"], [2, 1], 4);

        Assert.Equal(NumbersTallied.Of([2, 2, 5], 1).ApartFrom(NumbersTallied.Of([5], 0)), numbers.ApartFrom(NumbersTallied.Of([5], 0)));
        Assert.Equal(WordsTallied.Of(["x", "y", "x"], 4).ApartFrom(WordsTallied.Of(["y"], 0)), words.ApartFrom(WordsTallied.Of(["y"], 0)));
        Assert.Equal((3, 1, 3, 4), (numbers.Held, numbers.Gaps, words.Held, words.Gaps));
        Assert.Throws<ArgumentException>(() => new NumbersTallied([1, 2], [1], 0));
        Assert.Throws<ArgumentException>(() => new WordsTallied(["a"], [1, 2], 0));
    }
}
