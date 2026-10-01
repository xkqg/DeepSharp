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
}
