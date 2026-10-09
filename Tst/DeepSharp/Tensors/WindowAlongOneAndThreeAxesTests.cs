// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// A window along one axis, over a series, and one along three, over a volume. Both stand on the engine's two-dimensional
/// unfolding, so each says how it pads in the two-dimensional window's words, and the one rule for where a window stands
/// stays the two-dimensional window's.
/// </summary>
public class WindowAlongOneAndThreeAxesTests
{
    [Fact]
    public void AWindowAlongOneAxis_MovesOnePlaceAtATime_OverASeriesWithNoBorder_UnlessSaid()
    {
        var window = new Window1D(5);

        Assert.Equal(5, window.Length);
        Assert.Equal(1, window.Stride);
        Assert.Equal(0, window.Padding);
        Assert.Equal(PaddingMode.Stated, window.PaddingMode);
        Assert.Equal(6, window.PlacesOver(10));
        Assert.Equal(0, window.PlacesOver(4));
    }

    [Fact]
    public void AWindowAlongOneAxis_StandsAtPyTorchsPlaces_ForAStatedBorder()
    {
        // floor((length + 2 * padding - size) / stride) + 1, which is what torch.nn.Conv1d and MaxPool1d count.
        for (var size = 1; size <= 5; size++)
        {
            for (var stride = 1; stride <= 4; stride++)
            {
                for (var padding = 0; padding <= 2; padding++)
                {
                    for (var length = 1; length <= 12; length++)
                    {
                        var expected = length + (2 * padding) < size ? 0 : ((length + (2 * padding) - size) / stride) + 1;

                        Assert.Equal(expected, new Window1D(size) { Stride = stride, Padding = padding }.PlacesOver(length));
                    }
                }
            }
        }
    }

    [Fact]
    public void AWindowAlongOneAxis_PaddedAsSame_StandsAtAsManyPlacesAsTheStrideFits_AndAsCausal_AtTheSameNumber()
    {
        for (var size = 1; size <= 5; size++)
        {
            for (var stride = 1; stride <= 3; stride++)
            {
                for (var length = 1; length <= 10; length++)
                {
                    Assert.Equal((length + stride - 1) / stride, new Window1D(size) { Stride = stride, PaddingMode = PaddingMode.Same }.PlacesOver(length));
                    Assert.Equal((length + stride - 1) / stride, new Window1D(size) { Stride = stride, PaddingMode = PaddingMode.Causal }.PlacesOver(length));
                }
            }
        }
    }

    [Fact]
    public void AWindowAlongOneAxis_WalksTheEnginesUnfoldingAsOneRowTall_PaddedAlongTheSeriesAndNeverAcrossTheRow()
    {
        var stated = new Window1D(3) { Stride = 2, Padding = 2 }.Over(9);
        var same = new Window1D(4) { Stride = 2, PaddingMode = PaddingMode.Same }.Over(7);
        var causal = new Window1D(3) { PaddingMode = PaddingMode.Causal }.Over(6);

        Assert.Equal(new Window(1, 3) { Stride = 2, Border = new Borders(0, 0, 2, 2) }, stated);
        Assert.Equal(new Borders(0, 0, 1, 2), same.BordersOver(1, 7));
        Assert.Equal(new Borders(0, 0, 2, 0), causal.BordersOver(1, 6));
        Assert.Equal(1, stated.RowsOver(1));
        Assert.Equal(1, same.RowsOver(1));
        Assert.Equal(1, causal.RowsOver(1));
    }

    [Fact]
    public void AWindowAlongOneAxis_SaysHowItPads_AndIsNotEqualToOneThatPadsOtherwise()
    {
        Assert.Equal("window 3 (stride 2, padding 1)", new Window1D(3) { Stride = 2, Padding = 1 }.ToString());
        Assert.Equal("window 3 (stride 1, padding 'same')", new Window1D(3) { PaddingMode = PaddingMode.Same }.ToString());
        Assert.Equal("window 3 (stride 1, padding 'causal')", new Window1D(3) { PaddingMode = PaddingMode.Causal }.ToString());
        Assert.NotEqual(new Window1D(3), new Window1D(3) { PaddingMode = PaddingMode.Same });
    }

    [Theory]
    [InlineData(0, 1, 0, PaddingMode.Stated)]
    [InlineData(3, 0, 0, PaddingMode.Stated)]
    [InlineData(3, 1, -1, PaddingMode.Stated)]
    [InlineData(3, 1, 1, PaddingMode.Same)]
    [InlineData(3, 1, 1, PaddingMode.Causal)]
    public void AWindowAlongOneAxis_ThatCannotStand_IsRefused(int length, int stride, int padding, PaddingMode mode)
    {
        var window = new Window1D(length) { Stride = stride, Padding = padding, PaddingMode = mode };

        var wrong = Assert.Throws<ArgumentException>(window.RequireStanding);

        Assert.Equal("window", wrong.ParamName);
        Assert.StartsWith($"A {window} ", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AWindowAlongThreeAxes_MovesOnePlaceAtATime_OverAVolumeWithNoBorder_UnlessSaid()
    {
        var window = new Window3D(2, 3, 4);

        Assert.Equal(2, window.Depth);
        Assert.Equal(3, window.Height);
        Assert.Equal(4, window.Width);
        Assert.Equal(1, window.Stride);
        Assert.Equal(0, window.Padding);
        Assert.Equal(PaddingMode.Stated, window.PaddingMode);
        Assert.Equal(9, window.DepthsOver(10));
        Assert.Equal(6, window.RowsOver(8));
        Assert.Equal(3, window.ColumnsOver(6));
    }

    [Fact]
    public void AWindowAlongThreeAxes_StandsAtPyTorchsPlaces_OnEachAxisAlone()
    {
        for (var stride = 1; stride <= 3; stride++)
        {
            for (var padding = 0; padding <= 2; padding++)
            {
                for (var length = 1; length <= 9; length++)
                {
                    var window = new Window3D(2, 3, 4) { Stride = stride, Padding = padding };

                    Assert.Equal(Places(length, 2, stride, padding), window.DepthsOver(length));
                    Assert.Equal(Places(length, 3, stride, padding), window.RowsOver(length));
                    Assert.Equal(Places(length, 4, stride, padding), window.ColumnsOver(length));
                }
            }
        }
    }

    [Fact]
    public void AWindowAlongThreeAxes_PaddedAsSame_WorksEachAxisOutFromItsOwnLength()
    {
        var window = new Window3D(2, 3, 4) { Stride = 2, PaddingMode = PaddingMode.Same };

        Assert.Equal(2, window.DepthsOver(4));
        Assert.Equal(3, window.RowsOver(5));
        Assert.Equal(3, window.ColumnsOver(6));
    }

    [Fact]
    public void AWindowAlongThreeAxes_WalksTheEnginesUnfoldingTwice_PlanesFirstAndTheWidthAfterThem_PaddedAlongTheWidthAlone()
    {
        var window = new Window3D(2, 3, 4) { Stride = 2, Padding = 1 };

        var walk = window.Over(9);

        Assert.Equal(new Window(2, 3) { Stride = 2, Padding = 1 }, walk.Planes);
        Assert.Equal(new Window(6, 4) { Stride = 2, Border = new Borders(0, 0, 1, 1) }, walk.Lines);
        Assert.Equal(1, walk.Lines.RowsOver(6));
        Assert.Equal(window.ColumnsOver(9), walk.Lines.ColumnsOver(9));
    }

    [Fact]
    public void AWindowAlongThreeAxes_PaddedAsSame_PadsTheWidthAsTheOtherTwoAxesAre()
    {
        var walk = new Window3D(2, 3, 4) { Stride = 2, PaddingMode = PaddingMode.Same }.Over(7);

        var across = new Window(1, 4) { Stride = 2, PaddingMode = PaddingMode.Same }.BordersOver(1, 7);

        Assert.Equal(new Borders(0, 0, 1, 2), across);
        Assert.Equal(across, walk.Lines.BordersOver(6, 7));
        Assert.Equal(new Window(2, 3) { Stride = 2, PaddingMode = PaddingMode.Same }, walk.Planes);
    }

    [Fact]
    public void AWindowAlongThreeAxes_SaysHowItPads_AndIsNotEqualToOneThatPadsOtherwise()
    {
        Assert.Equal("window 2x3x4 (stride 2, padding 1)", new Window3D(2, 3, 4) { Stride = 2, Padding = 1 }.ToString());
        Assert.Equal("window 2x3x4 (stride 1, padding 'same')", new Window3D(2, 3, 4) { PaddingMode = PaddingMode.Same }.ToString());
        Assert.Equal("window 2x3x4 (stride 1, padding 'causal')", new Window3D(2, 3, 4) { PaddingMode = PaddingMode.Causal }.ToString());
        Assert.NotEqual(new Window3D(2, 3, 4), new Window3D(2, 3, 4) { Padding = 1 });
    }

    [Theory]
    [InlineData(0, 1, 1, 1, 0, PaddingMode.Stated)]
    [InlineData(1, 0, 1, 1, 0, PaddingMode.Stated)]
    [InlineData(1, 1, 0, 1, 0, PaddingMode.Stated)]
    [InlineData(1, 1, 1, 0, 0, PaddingMode.Stated)]
    [InlineData(1, 1, 1, 1, -1, PaddingMode.Stated)]
    [InlineData(1, 1, 1, 1, 1, PaddingMode.Same)]
    [InlineData(1, 1, 1, 1, 1, PaddingMode.Causal)]
    public void AWindowAlongThreeAxes_ThatCannotStand_IsRefused(int depth, int height, int width, int stride, int padding, PaddingMode mode)
    {
        var window = new Window3D(depth, height, width) { Stride = stride, Padding = padding, PaddingMode = mode };

        var wrong = Assert.Throws<ArgumentException>(window.RequireStanding);

        Assert.Equal("window", wrong.ParamName);
        Assert.StartsWith($"A {window} ", wrong.Message, StringComparison.Ordinal);
    }

    private static int Places(int length, int size, int stride, int padding) =>
        length + (2 * padding) < size ? 0 : ((length + (2 * padding) - size) / stride) + 1;
}
