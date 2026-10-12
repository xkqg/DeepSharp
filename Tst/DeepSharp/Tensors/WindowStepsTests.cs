// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// What a window says for each axis rather than once for every axis: how far it moves, how far apart its cells are, and
/// what it pads with at each end of an axis. Each is a value of its own, so the two of a pair cannot be swapped unnoticed.
/// </summary>
public class WindowStepsTests
{
    [Fact]
    public void StepsAlikeOnEveryAxis_AreWrittenAsTheOneNumber_AndComeFromIt()
    {
        Steps steps = 2;

        Assert.Equal(new Steps(2, 2), steps);
        Assert.True(steps.Alike);
        Assert.Equal("2", steps.ToString());
        Assert.Equal(2, steps.Down);
        Assert.Equal(2, steps.Across);
    }

    [Fact]
    public void StepsThatDiffer_SayWhichAxisIsWhich()
    {
        var steps = new Steps(2, 1);

        Assert.False(steps.Alike);
        Assert.Equal("2 down and 1 across", steps.ToString());
        Assert.NotEqual(new Steps(1, 2), steps);
    }

    [Fact]
    public void StepsThroughAVolume_AreThreeAndSayWhichAxisIsWhich()
    {
        Steps3D alike = 3;

        Assert.Equal(new Steps3D(3, 3, 3), alike);
        Assert.True(alike.Alike);
        Assert.Equal("3", alike.ToString());

        var steps = new Steps3D(1, 2, 3);

        Assert.False(steps.Alike);
        Assert.Equal("1 deep, 2 down and 3 across", steps.ToString());
        Assert.Equal(1, steps.Deep);
        Assert.Equal(2, steps.Down);
        Assert.Equal(3, steps.Across);
        Assert.NotEqual(new Steps3D(3, 2, 1), steps);
    }

    [Fact]
    public void StepsThroughAVolume_AreAlikeOnlyWhenEveryAxisAgrees()
    {
        Assert.False(new Steps3D(1, 2, 2).Alike);
        Assert.False(new Steps3D(2, 2, 1).Alike);
        Assert.False(new Steps3D(2, 1, 2).Alike);
        Assert.True(new Steps3D(2, 2, 2).Alike);
    }

    [Fact]
    public void TheSidesOfAnAxis_AreABeforeAndAnAfter_AndAddUp()
    {
        var sides = new Sides(1, 2);

        Assert.Equal(1, sides.Before);
        Assert.Equal(2, sides.After);
        Assert.Equal(3, sides.Total);
        Assert.NotEqual(new Sides(2, 1), sides);
        Assert.Equal(0, default(Sides).Total);
    }

    [Fact]
    public void ABorderStatedPerAxis_IsAPairForEachAxis_AndNeverFourLooseNumbers()
    {
        var margins = new Margins(new Sides(1, 2), new Sides(3, 0));

        Assert.Equal(new Sides(1, 2), margins.Down);
        Assert.Equal(new Sides(3, 0), margins.Across);
        Assert.NotEqual(new Margins(new Sides(3, 0), new Sides(1, 2)), margins);

        var through = new Margins3D(new Sides(1, 0), new Sides(0, 2), new Sides(1, 1));

        Assert.Equal(new Sides(1, 0), through.Deep);
        Assert.Equal(new Sides(0, 2), through.Down);
        Assert.Equal(new Sides(1, 1), through.Across);
    }

    [Fact]
    public void WhereAWindowStands_IsOneRule_AndItSaysTheBorderTheStrideAndTheDilationTogether()
    {
        // The one thing every engine reads: it cannot take the border and miss the stride, or take the stride and miss the
        // dilation, because they arrive together.
        var window = new Window(3, 3) { Stride = new Steps(2, 1), Dilation = new Steps(2, 1), Padding = 1 };
        var placing = window.PlacingOver(9, 9);

        Assert.Equal(new Borders(1, 1, 1, 1), placing.Border);
        Assert.Equal(new Steps(2, 1), placing.Stride);
        Assert.Equal(new Steps(2, 1), placing.Dilation);
        Assert.Equal(window.BordersOver(9, 9), placing.Border);
    }

    [Fact]
    public void TheOneRule_SaysTheSameBorderItAlwaysSaid_ForEveryWindowTheLibraryCouldAlreadyMake()
    {
        for (var size = 1; size <= 4; size++)
        {
            for (var stride = 1; stride <= 3; stride++)
            {
                for (var length = 1; length <= 10; length++)
                {
                    foreach (var mode in new[] { PaddingMode.Stated, PaddingMode.Same, PaddingMode.Causal })
                    {
                        var padding = mode == PaddingMode.Stated ? 1 : 0;
                        var window = new Window(size, size + 1) { Stride = stride, PaddingMode = mode, Padding = padding };

                        Assert.Equal(window.BordersOver(length, length), window.PlacingOver(length, length).Border);
                        Assert.Equal((Steps)stride, window.PlacingOver(length, length).Stride);
                        Assert.Equal((Steps)1, window.PlacingOver(length, length).Dilation);
                    }
                }
            }
        }
    }
}
