// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// A shape is the first thing a network gets wrong, and the last thing it tells you about: a layer that
/// expects 2x3 and receives 3x2 produces numbers either way. These are the rules that make the mismatch a
/// refusal at the point it happens instead of a wrong answer several layers later.
/// </summary>
public class ShapeTests
{
    [Fact]
    public void AShape_KnowsHowManyAxesItHasAndHowManyValuesItHolds()
    {
        var shape = new Shape(2, 3);

        Assert.Equal(2, shape.Rank);
        Assert.Equal(6, shape.Count);
    }

    [Fact]
    public void AShapeWithNoAxes_IsASingleValue()
    {
        // A scalar is not "nothing": it is one value with no axes, and the product of no dimensions is one.
        // Getting this wrong makes every loss — which is a scalar — allocate an empty buffer.
        var scalar = new Shape();

        Assert.Equal(0, scalar.Rank);
        Assert.Equal(1, scalar.Count);
    }

    [Fact]
    public void AnAxisOfZeroLength_HoldsNothing()
    {
        var empty = new Shape(2, 0, 3);

        Assert.Equal(3, empty.Rank);
        Assert.Equal(0, empty.Count);
    }

    [Fact]
    public void ANegativeAxis_IsRefusedWhereItIsWritten()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Shape(2, -1));
    }

    [Fact]
    public void AShapeThatWouldOverflow_IsRefusedRatherThanWrappingAround()
    {
        // 2^31 values do not fit in an int, and a silently wrapped count allocates a buffer that is far too
        // small — which shows up as a memory-corruption bug thousands of operations later.
        Assert.Throws<ArgumentOutOfRangeException>(() => new Shape(65536, 65536));
    }

    [Fact]
    public void TwoShapes_AreTheSameWhenTheirAxesAre()
    {
        Assert.Equal(new Shape(2, 3), new Shape(2, 3));
        Assert.NotEqual(new Shape(2, 3), new Shape(3, 2));
        Assert.NotEqual(new Shape(2, 3), new Shape(2, 3, 1));
    }

    [Fact]
    public void TwoEqualShapes_HashTheSame()
    {
        Assert.Equal(new Shape(4, 5, 6).GetHashCode(), new Shape(4, 5, 6).GetHashCode());
    }

    [Fact]
    public void AShape_ReadsAsItIsSpoken()
    {
        Assert.Equal("2x3", new Shape(2, 3).ToString());
        Assert.Equal("scalar", new Shape().ToString());
    }

    [Fact]
    public void EachAxis_CanBeAskedForByNumber()
    {
        var shape = new Shape(2, 3, 4);

        Assert.Equal(2, shape[0]);
        Assert.Equal(4, shape[2]);
        Assert.Throws<ArgumentOutOfRangeException>(() => shape[3]);
    }

    [Fact]
    public void AShapeBuiltFromNoAxesAtAll_IsTheSameSingleValueAsOneThatWasNeverBuilt()
    {
        // Two ways to say "scalar" reach this type: the constructor with nothing in it, and a struct that
        // never ran one. They must not disagree, because a loss arrives by both routes.
        var built = new Shape([]);

        Assert.Equal(default, built);
        Assert.Equal(1, built.Count);
        Assert.Equal(0, built.Rank);
    }

    [Fact]
    public void AShape_IsNotEqualToSomethingThatIsNotAShape()
    {
        object notAShape = "2x3";

        Assert.False(new Shape(2, 3).Equals(notAShape));
        Assert.True(new Shape(2, 3).Equals((object)new Shape(2, 3)));
    }

    [Fact]
    public void ShapesCompareWithTheOperatorsToo()
    {
        Assert.True(new Shape(2, 3) == new Shape(2, 3));
        Assert.False(new Shape(2, 3) == new Shape(3, 2));
        Assert.True(new Shape(2, 3) != new Shape(3, 2));
        Assert.False(new Shape(2, 3) != new Shape(2, 3));
    }

    [Fact]
    public void AxesAreNeverNull()
    {
        Assert.Throws<ArgumentNullException>(() => new Shape(null!));
    }
}
