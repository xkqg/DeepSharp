// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// The ways beyond the four a layer's numbers can start: a normal and a truncated normal, an orthogonal matrix, ones and a
/// constant. Each is Keras's — its defaults, its truncation two deviations from the mean, its orthogonal matrix the one a QR
/// factorisation of random numbers gives with the signs of the diagonal put right — and each is drawn from the same counted
/// stream every other start is, so the same seed starts the same layer. A start from a random sampler is held to what makes it
/// that sampler — its mean, its spread, its bounds, the product of the matrix with its own transpose — and never to the
/// numbers drawn, which depend on the machine's logarithm in the last bit.
/// </summary>
public class NewInitialiserTests
{
    private static Draws DrawsFor(string purpose) => new RandomStream(42).Draw(purpose, 0, 0);

    private static (double Mean, double Spread) Moments(IEnumerable<float> values)
    {
        var all = values.Select(value => (double)value).ToArray();
        var mean = all.Average();

        return (mean, Math.Sqrt(all.Sum(value => (value - mean) * (value - mean)) / all.Length));
    }

    [Fact]
    public void ANextNormal_HasTheMeanAndTheSpreadOfTheStandardNormal_AndDrawsTheSameNumbersFromTheSameSeed()
    {
        var draws = DrawsFor("moments");
        var drawn = Enumerable.Range(0, 200_000).Select(_ => (float)draws.NextNormal()).ToArray();
        var (mean, spread) = Moments(drawn);

        Assert.InRange(mean, -0.01, 0.01);
        Assert.InRange(spread, 0.99, 1.01);
        Assert.Equal(NextEight(DrawsFor("moments")), NextEight(DrawsFor("moments")));
    }

    private static IEnumerable<float> NextEight(Draws draws) => Enumerable.Range(0, 8).Select(_ => (float)draws.NextNormal());

    [Fact]
    public void Normal_IsKerasRandomNormal_ZeroAndAFiftiethUnlessSaid()
    {
        var plain = new Normal().Draw(new Shape(400, 500), new Fans(400, 500), DrawsFor("0.weight"));
        var said = new Normal(deviation: 0.5, mean: 2).Draw(new Shape(400, 500), new Fans(400, 500), DrawsFor("0.weight"));
        var (plainMean, plainSpread) = Moments(plain.Values.ToArray());
        var (saidMean, saidSpread) = Moments(said.Values.ToArray());

        Assert.InRange(plainMean, -0.001, 0.001);
        Assert.InRange(plainSpread, 0.0495, 0.0505);
        Assert.InRange(saidMean, 1.99, 2.01);
        Assert.InRange(saidSpread, 0.495, 0.505);
        Assert.Equal(new Shape(400, 500), plain.Shape);
    }

    [Fact]
    public void TruncatedNormal_NeverReachesFurtherThanTwoDeviationsFromTheMean_AsKerasCutsIt()
    {
        var drawn = new TruncatedNormal(deviation: 0.5, mean: 1).Draw(new Shape(400, 500), new Fans(400, 500), DrawsFor("0.weight"));
        var (mean, spread) = Moments(drawn.Values.ToArray());

        Assert.All(drawn.Values.ToArray(), value => Assert.InRange(value, 0.0f, 2.0f));
        Assert.True(drawn.Values.ToArray().Max() > 1.9f && drawn.Values.ToArray().Min() < 0.1f);
        Assert.InRange(mean, 0.995, 1.005);

        // A normal cut at two deviations keeps 0.8796 of its spread.
        Assert.InRange(spread, 0.5 * 0.875, 0.5 * 0.885);
    }

    [Fact]
    public void ADeviationThatIsNoNumberAboveNothing_OrAMeanThatIsNoNumber_IsRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Normal(deviation: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Normal(deviation: double.NaN));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Normal(mean: double.PositiveInfinity));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TruncatedNormal(deviation: -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new TruncatedNormal(mean: double.NaN));
    }

    [Theory]
    [InlineData(40, 25)]
    [InlineData(25, 40)]
    [InlineData(30, 30)]
    public void Orthogonal_GivesAMatrixWhoseShorterSideIsOrthonormal_ScaledByTheGain(int rows, int columns)
    {
        var matrix = new Orthogonal(gain: 2).Draw(new Shape(rows, columns), new Fans(rows, columns), DrawsFor("0.weight"));
        var values = matrix.Values.ToArray();
        var shorter = Math.Min(rows, columns);

        // The columns are orthonormal where there are more rows than columns, the rows where there are more columns.
        for (var one = 0; one < shorter; one++)
        {
            for (var other = 0; other < shorter; other++)
            {
                var dot = 0d;

                for (var at = 0; at < Math.Max(rows, columns); at++)
                {
                    dot += rows >= columns
                        ? values[(at * columns) + one] * (double)values[(at * columns) + other]
                        : values[(one * columns) + at] * (double)values[(other * columns) + at];
                }

                Assert.Equal(one == other ? 4d : 0d, dot, 1e-4);
            }
        }
    }

    [Fact]
    public void Orthogonal_NeedsAMatrix_AndStartsTheSameWayFromTheSameSeed()
    {
        Assert.Throws<ArgumentException>(() => new Orthogonal().Draw(new Shape(5), new Fans(5, 1), DrawsFor("0.bias")));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Orthogonal(gain: 0));

        var first = new Orthogonal().Draw(new Shape(6, 4), new Fans(6, 4), DrawsFor("0.weight"));
        var again = new Orthogonal().Draw(new Shape(6, 4), new Fans(6, 4), DrawsFor("0.weight"));
        var other = new Orthogonal().Draw(new Shape(6, 4), new Fans(6, 4), DrawsFor("2.weight"));

        Assert.Equal(first.Values.ToArray(), again.Values.ToArray());
        Assert.NotEqual(first.Values.ToArray(), other.Values.ToArray());
    }

    [Fact]
    public void Ones_AreOne_AndAConstantIsWhatItIsGiven()
    {
        Assert.All(new Ones().Draw(new Shape(3, 2), new Fans(3, 2), DrawsFor("0.weight")).Values.ToArray(), value => Assert.Equal(1f, value));
        Assert.All(new Constant(0.25).Draw(new Shape(3, 2), new Fans(3, 2), DrawsFor("0.weight")).Values.ToArray(), value => Assert.Equal(0.25f, value));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Constant(double.NaN));
    }

    [Fact]
    public void WhateverANewInitialiserNeeds_IsHeldAsTheOthersAre()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Normal().Draw(new Shape(1), new Fans(0, 1), DrawsFor("x")));
        Assert.Throws<ArgumentNullException>(() => new Orthogonal().Draw(new Shape(2, 2), new Fans(2, 2), null!));
        Assert.Throws<ArgumentNullException>(() => new Ones().Draw(new Shape(1), new Fans(1, 1), null!));
    }
}
