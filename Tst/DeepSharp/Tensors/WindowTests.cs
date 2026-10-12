// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// The patch a convolution looks at and how it walks: the border it pads an image with — stated for every side, or worked
/// out as TensorFlow's 'same' does — and the places it stands at. One rule gives both, and every engine reads it.
/// </summary>
public class WindowTests
{
    [Fact]
    public void AWindowPaddedAsSame_BordersEveryImageAsTensorFlowDoes_AndStandsAtAsManyPlacesAsTheStrideFits()
    {
        // TensorFlow's kernel_shape_util.cc, as written there: out = ceil(in / stride); the border needed for it, never below
        // nothing, split with the odd row or column after the image.
        for (var size = 1; size <= 5; size++)
        {
            for (var stride = 1; stride <= 4; stride++)
            {
                for (var length = 1; length <= 12; length++)
                {
                    var places = (length + stride - 1) / stride;
                    var needed = Math.Max(0, ((places - 1) * stride) + size - length);
                    var window = new Window(size, size + 1) { Stride = stride, PaddingMode = PaddingMode.Same };
                    var across = Math.Max(0, ((places - 1) * stride) + size + 1 - length);

                    Assert.Equal(new Borders(needed / 2, needed - (needed / 2), across / 2, across - (across / 2)), window.BordersOver(length, length));
                    Assert.Equal(places, window.RowsOver(length));
                    Assert.Equal(places, window.ColumnsOver(length));
                }
            }
        }
    }

    [Fact]
    public void AWindowPaddedAsSame_WorksEachSideOutFromItsOwnLength()
    {
        // Four rows by three columns at a stride of two over seven rows and eight columns: one row before and two after, no
        // column before and one after — Keras's own example of a border no stated padding gives.
        var window = new Window(4, 3) { Stride = 2, PaddingMode = PaddingMode.Same };

        Assert.Equal(new Borders(1, 2, 0, 1), window.BordersOver(7, 8));
        Assert.Equal(4, window.RowsOver(7));
        Assert.Equal(4, window.ColumnsOver(8));
    }

    [Fact]
    public void AStatedBorder_IsTheSameOnEverySide_WhateverTheImage()
    {
        var window = new Window(3, 3) { Stride = 2, Padding = 1 };

        Assert.Equal(new Borders(1, 1, 1, 1), window.BordersOver(28, 5));
        Assert.Equal(14, window.RowsOver(28));
        Assert.Equal(3, window.ColumnsOver(5));
        Assert.Equal(new Borders(0, 0, 0, 0), new Window(2, 2).BordersOver(3, 3));
        Assert.Equal(PaddingMode.Stated, new Window(2, 2).PaddingMode);
    }

    [Fact]
    public void AWindowThatCannotStand_HasNoBorder_AndStandsNowhere()
    {
        var window = new Window(3, 3) { Stride = 0, PaddingMode = PaddingMode.Same };

        Assert.Equal(new Borders(0, 0, 0, 0), window.BordersOver(7, 7));
        Assert.Equal(0, window.RowsOver(7));
        Assert.Equal(0, window.ColumnsOver(7));
    }

    [Fact]
    public void AWindowSaysHowItPads()
    {
        Assert.Equal("window 3x3 (stride 2, padding 1)", new Window(3, 3) { Stride = 2, Padding = 1 }.ToString());
        Assert.Equal("window 4x3 (stride 2, padding 'same')", new Window(4, 3) { Stride = 2, PaddingMode = PaddingMode.Same }.ToString());
        Assert.NotEqual(new Window(3, 3), new Window(3, 3) { PaddingMode = PaddingMode.Same });
    }

    [Fact]
    public void AWindowPaddedAsSame_ThatIsGivenABorderToo_CannotStand()
    {
        var images = Tensor.Zeros(new Shape(1, 3, 3, 1));

        var wrong = Assert.Throws<ArgumentException>(() => images.RequireImagesFor(new Window(2, 2) { Padding = 1, PaddingMode = PaddingMode.Same }));

        Assert.Equal("window", wrong.ParamName);
        Assert.StartsWith("A window 2x2 (stride 1, padding 'same') works its border out from each image it stands on", wrong.Message, StringComparison.Ordinal);
        images.RequireImagesFor(new Window(5, 5) { PaddingMode = PaddingMode.Same });
    }

    [Fact]
    public void AWindowPaddedAsCausal_PadsEachSideBySizeLessOneBeforeAndNothingAfter_SoNoPlaceSeesPastItsOwn()
    {
        // Keras's padding='causal': output[t] does not depend on input[t+1:]. Along a side of one place nothing is padded.
        for (var size = 1; size <= 5; size++)
        {
            for (var stride = 1; stride <= 3; stride++)
            {
                for (var length = 1; length <= 10; length++)
                {
                    var window = new Window(1, size) { Stride = stride, PaddingMode = PaddingMode.Causal };

                    Assert.Equal(new Borders(0, 0, size - 1, 0), window.BordersOver(1, length));
                    Assert.Equal(1, window.RowsOver(1));
                    Assert.Equal(((length - 1) / stride) + 1, window.ColumnsOver(length));
                }
            }
        }
    }

    [Fact]
    public void AWindowPaddedAsCausal_SaysSo_AndIsNotTheOneStatedOrTheOneWorkedOut()
    {
        Assert.Equal("window 1x3 (stride 1, padding 'causal')", new Window(1, 3) { PaddingMode = PaddingMode.Causal }.ToString());
        Assert.NotEqual(new Window(1, 3) { PaddingMode = PaddingMode.Causal }, new Window(1, 3) { PaddingMode = PaddingMode.Same });
        Assert.Equal(new Borders(2, 0, 2, 0), new Window(3, 3) { PaddingMode = PaddingMode.Causal }.BordersOver(9, 9));
    }

    [Fact]
    public void AWindowPaddedAsCausal_ThatIsGivenABorderToo_CannotStand()
    {
        var wrong = Assert.Throws<ArgumentException>(() => Tensor.Zeros(new Shape(1, 3, 3, 1)).RequireImagesFor(new Window(2, 2) { Padding = 1, PaddingMode = PaddingMode.Causal }));

        Assert.Equal("window", wrong.ParamName);
        Assert.StartsWith("A window 2x2 (stride 1, padding 'causal') works its border out from each image it stands on", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AWindowThatStridesOtherwiseAlongOneAxisThanAlongAnother_StandsAtEachAxisOwnPlaces()
    {
        var window = new Window(3, 2) { Stride = new Steps(2, 1) };

        Assert.Equal(3, window.RowsOver(7));
        Assert.Equal(7, window.ColumnsOver(8));
        Assert.Equal("window 3x2 (stride 2 down and 1 across, padding 0)", window.ToString());
        Assert.NotEqual(new Window(3, 2) { Stride = new Steps(1, 2) }, window);
    }

    [Fact]
    public void ADilatedWindow_ReachesFurtherThanItCovers_AndStandsAtFewerPlaces()
    {
        // A window of three at a dilation of three covers three places and reaches over seven, so over twelve it stands at six.
        var window = new Window(1, 3) { Dilation = new Steps(1, 3) };

        Assert.Equal(6, window.ColumnsOver(12));
        Assert.Equal(1, window.RowsOver(1));
        Assert.Equal("window 1x3 (stride 1, dilated by 1 down and 3 across, padding 0)", window.ToString());

        // Dilated alike on both axes, it says the one number, and a window that is not dilated says nothing of it.
        Assert.Equal("window 3x3 (stride 1, dilated by 2, padding 0)", new Window(3, 3) { Dilation = 2 }.ToString());
        Assert.Equal("window 3x3 (stride 1, padding 0)", new Window(3, 3).ToString());
        Assert.Equal(5, new Window(3, 3) { Dilation = 2 }.RowsOver(9));
    }

    [Fact]
    public void ADilatedWindowPaddedAsSameOrCausal_WorksItsBorderOutOverWhatItReaches()
    {
        // TensorFlow's rule reads the window's reach, not its size: three cells at a dilation of two reach over five.
        Assert.Equal(new Borders(2, 2, 2, 2), new Window(3, 3) { Dilation = 2, PaddingMode = PaddingMode.Same }.BordersOver(9, 9));
        Assert.Equal(9, new Window(3, 3) { Dilation = 2, PaddingMode = PaddingMode.Same }.RowsOver(9));
        Assert.Equal(new Borders(0, 0, 4, 0), new Window(1, 3) { Dilation = new Steps(1, 2), PaddingMode = PaddingMode.Causal }.BordersOver(1, 9));
    }

    [Theory]
    // Measured against PyTorch's own place count over 1644 cases with no mismatch — scratchpad/probe_places.py.
    [InlineData(8, 3, 2, 0, 1, 3, 4)]
    [InlineData(8, 3, 2, 1, 1, 4, 5)]
    [InlineData(7, 3, 2, 0, 1, 3, 3)]
    [InlineData(7, 2, 3, 0, 1, 2, 3)]
    [InlineData(11, 3, 2, 1, 2, 5, 5)]
    [InlineData(9, 3, 1, 0, 2, 5, 5)]
    [InlineData(12, 3, 1, 0, 3, 6, 6)]
    public void AWindowThatRoundsUp_StandsAtAPlaceWhoseWindowOverhangsTheEnd(int length, int size, int stride, int padding, int dilation, int down, int up)
    {
        var walking = new Window(1, size) { Stride = new Steps(1, stride), Dilation = new Steps(1, dilation), Padding = padding };

        Assert.Equal(down, walking.ColumnsOver(length));
        Assert.Equal(up, (walking with { RoundsUp = true }).ColumnsOver(length));
    }

    [Fact]
    public void AWindowThatRoundsUp_LeavesOutAPlaceThatWouldStartInTheBorderAfterTheImage()
    {
        // PyTorch's own rule: a window that would begin in or past the border after the image is not stood at, however the
        // rounding counts. One place of two at a stride of two over a single value padded on both sides.
        var window = new Window(1, 2) { Stride = new Steps(1, 2), Padding = 1, RoundsUp = true };

        Assert.Equal(1, window.ColumnsOver(1));
        Assert.Equal(1, (window with { RoundsUp = false }).ColumnsOver(1));
        Assert.Equal("window 1x2 (stride 1 down and 2 across, rounding up, padding 1)", window.ToString());
    }

    [Fact]
    public void ADilationBelowOne_CannotStand()
    {
        var images = Tensor.Zeros(new Shape(1, 5, 5, 1));

        var wrong = Assert.Throws<ArgumentException>(() => images.RequireImagesFor(new Window(2, 2) { Dilation = new Steps(1, 0) }));

        Assert.Equal("window", wrong.ParamName);
        Assert.StartsWith("A window 2x2 (stride 1, dilated by 1 down and 0 across, padding 0) cannot stand anywhere: its dilation is at least one.",
                          wrong.Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentException>(() => images.RequireImagesFor(new Window(2, 2) { Dilation = new Steps(0, 1) }));
        images.RequireImagesFor(new Window(2, 2) { Dilation = 2 });
    }

    [Fact]
    public void ABorderStatedForEachSide_IsTheOnlyOneTheWindowHas_WhateverTheImage()
    {
        // How a one-dimensional or a three-dimensional window walks a two-dimensional engine: one axis padded, another not.
        var window = new Window(1, 3) { Stride = 2, Border = new Margins(new Sides(0, 0), new Sides(2, 2)) };

        Assert.Equal(new Borders(0, 0, 2, 2), window.BordersOver(1, 9));
        Assert.Equal(1, window.RowsOver(1));
        Assert.Equal(6, window.ColumnsOver(9));
        Assert.Equal("window 1x3 (stride 2, padding 0 above, 0 below, 2 before, 2 after)", window.ToString());
        Assert.NotEqual(new Window(1, 3) { Stride = 2, Padding = 2 }, window);
    }

    [Theory]
    [InlineData(-1, 0, 0, 0)]
    [InlineData(0, -1, 0, 0)]
    [InlineData(0, 0, -1, 0)]
    [InlineData(0, 0, 0, -1)]
    public void ABorderBelowNothing_CannotStand(int top, int bottom, int left, int right)
    {
        var window = new Window(2, 2) { Border = new Margins(new Sides(top, bottom), new Sides(left, right)) };

        Assert.Throws<ArgumentException>(() => Tensor.Zeros(new Shape(1, 5, 5, 1)).RequireImagesFor(window));
    }

    [Fact]
    public void ABorderStatedForEachSide_ThatIsGivenAPaddingOrAModeToo_CannotStand()
    {
        var images = Tensor.Zeros(new Shape(1, 5, 5, 1));
        var border = new Margins(new Sides(0, 0), new Sides(1, 1));

        Assert.Throws<ArgumentException>(() => images.RequireImagesFor(new Window(2, 2) { Border = border, Padding = 1 }));
        Assert.Throws<ArgumentException>(() => images.RequireImagesFor(new Window(2, 2) { Border = border, PaddingMode = PaddingMode.Same }));
        Assert.Throws<ArgumentException>(() => images.RequireImagesFor(new Window(2, 2) { Border = border, PaddingMode = PaddingMode.Causal }));
        images.RequireImagesFor(new Window(2, 2) { Border = border });
    }
}
