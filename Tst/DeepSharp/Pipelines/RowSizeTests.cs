// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A row divided by its own size lands between minus one and one, whatever the numbers are.
/// </summary>
/// <remarks>
/// That is what the step declares of the columns it writes, and the handover holds every feature to what it is
/// declared to land in — so if the arithmetic can leave a value outside it, the declaration is a promise the run does
/// not keep. The length of a row squares its values, and a small enough value squares to nothing: a row of 1e-160 came
/// back at 1.0000055664551362, which is outside the range its own step promised. The sizes are therefore worked out
/// around the largest value in the row, which is what every library that computes a hypotenuse does.
/// </remarks>
public class RowSizeTests
{
    [Theory]
    [InlineData(Norm.L2, 1e-160)]
    [InlineData(Norm.L2, 2.6e-162)]
    [InlineData(Norm.L2, 1e-200)]
    [InlineData(Norm.L1, 1e-160)]
    [InlineData(Norm.Max, 1e-160)]
    [InlineData(Norm.L2, 1e160)]
    [InlineData(Norm.L2, 3.5)]
    public void ARowDividedByItsOwnSize_LandsWhereTheStepSaysItDoes(Norm norm, double value)
    {
        var prepared = Rows(norm, value, value * 2, -value);

        Assert.All(
            new[] { "a", "b", "c" }.SelectMany(column => prepared.Table.NumbersOf(column)),
            written => Assert.InRange(written!.Value, -1, 1));
    }

    [Fact]
    public void TheSizeOfARowOfLargeNumbers_DoesNotRunAway()
    {
        // The other end of the same arithmetic: squaring a large value overflows to infinity, and every value of the
        // row would then come back as nought.
        var prepared = Rows(Norm.L2, 1e200, 2e200, -1.5e200);

        // 1, 2 and -1.5 of one size: the row's length is that size times the root of 1 + 4 + 2.25.
        Assert.Equal(1 / Math.Sqrt(7.25), prepared.Table.NumbersOf("a")[0]!.Value, 10);
    }

    [Fact]
    public void ARowOfOrdinaryNumbers_ComesBackAsItAlwaysDid()
    {
        var prepared = Rows(Norm.L2, 3, 4, 0);

        Assert.Equal(0.6, prepared.Table.NumbersOf("a")[0]!.Value, 12);
        Assert.Equal(0.8, prepared.Table.NumbersOf("b")[0]!.Value, 12);
        Assert.Equal(0, prepared.Table.NumbersOf("c")[0]!.Value, 12);
    }

    private static PreparedData Rows(Norm norm, double first, double second, double third) =>
        Pdd.Create()
            .Read(
                new InMemoryRowSource(
                    ["key", "a", "b", "c"],
                    [["1", Written(first), Written(second), Written(third)], ["2", "1", "1", "1"]]),
                "two rows")
            .Declare(schema => schema.Integer("key").Number("a", "b", "c"))
            .SplitAtRandom(train: 0.50)
            .NormaliseRow(norm, "a", "b", "c")
            .Build()
            .Run();

    private static string Written(double value) => value.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
}
