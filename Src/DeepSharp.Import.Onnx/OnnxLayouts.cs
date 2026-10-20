// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Import.Onnx;

/// <summary>
/// How ONNX lays a number out and how a slot here keeps it, in both directions: the one home of every such rule, compiled into the
/// package that reads a graph and the package that writes one, so the two cannot come to disagree.
/// </summary>
/// <remarks>
/// Each rule is a pure move of numbers from one order to another, row-major both sides. A move to the slot's order and the move back
/// are written side by side, and a round trip through both is the numbers it began with.
/// </remarks>
internal static class OnnxLayouts
{
    /// <summary>A matrix turned round: rows by columns to columns by rows. It is its own inverse, with the lengths exchanged.</summary>
    /// <param name="values">The numbers, row-major.</param>
    /// <param name="rows">How many rows they are written in.</param>
    /// <param name="columns">How many columns.</param>
    /// <returns>The numbers of the matrix turned round.</returns>
    public static float[] Transposed(float[] values, int rows, int columns)
    {
        var turned = new float[values.Length];

        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                turned[(column * rows) + row] = values[(row * columns) + column];
            }
        }

        return turned;
    }

    /// <summary>
    /// A kernel written channels out by channels in by the window's places, as the window's places times the channels in by the
    /// channels out: the order a convolution's slot keeps.
    /// </summary>
    /// <param name="values">The numbers as ONNX writes them.</param>
    /// <param name="outputs">The channels out.</param>
    /// <param name="inputs">The channels in.</param>
    /// <param name="places">How many places the window has.</param>
    /// <returns>The numbers in the slot's order.</returns>
    public static float[] KernelToSlot(float[] values, int outputs, int inputs, int places)
    {
        var laid = new float[values.Length];

        for (var output = 0; output < outputs; output++)
        {
            for (var input = 0; input < inputs; input++)
            {
                for (var place = 0; place < places; place++)
                {
                    laid[(((place * inputs) + input) * outputs) + output] = values[(((output * inputs) + input) * places) + place];
                }
            }
        }

        return laid;
    }

    /// <summary>The move back of <see cref="KernelToSlot"/>: a convolution's slot as ONNX writes a kernel.</summary>
    /// <param name="values">The numbers in the slot's order.</param>
    /// <param name="outputs">The channels out.</param>
    /// <param name="inputs">The channels in.</param>
    /// <param name="places">How many places the window has.</param>
    /// <returns>The numbers as ONNX writes them: channels out by channels in by the window's places.</returns>
    public static float[] SlotToKernel(float[] values, int outputs, int inputs, int places)
    {
        var written = new float[values.Length];

        for (var output = 0; output < outputs; output++)
        {
            for (var input = 0; input < inputs; input++)
            {
                for (var place = 0; place < places; place++)
                {
                    written[(((output * inputs) + input) * places) + place] = values[(((place * inputs) + input) * outputs) + output];
                }
            }
        }

        return written;
    }

    /// <summary>
    /// A transposed convolution's kernel written channels in by the channels out of one group by the window's places, as a row for
    /// every channel in holding the window's places by those channels out.
    /// </summary>
    /// <param name="values">The numbers as ONNX writes them.</param>
    /// <param name="inputs">The channels in.</param>
    /// <param name="outputs">The channels out of one group.</param>
    /// <param name="places">How many places the window has.</param>
    /// <returns>The numbers in the slot's order.</returns>
    public static float[] GivenBackToSlot(float[] values, int inputs, int outputs, int places)
    {
        var laid = new float[values.Length];

        for (var input = 0; input < inputs; input++)
        {
            for (var output = 0; output < outputs; output++)
            {
                for (var place = 0; place < places; place++)
                {
                    laid[(((input * places) + place) * outputs) + output] = values[(((input * outputs) + output) * places) + place];
                }
            }
        }

        return laid;
    }

    /// <summary>The move back of <see cref="GivenBackToSlot"/>.</summary>
    /// <param name="values">The numbers in the slot's order.</param>
    /// <param name="inputs">The channels in.</param>
    /// <param name="outputs">The channels out of one group.</param>
    /// <param name="places">How many places the window has.</param>
    /// <returns>The numbers as ONNX writes them.</returns>
    public static float[] SlotToGivenBack(float[] values, int inputs, int outputs, int places)
    {
        var written = new float[values.Length];

        for (var input = 0; input < inputs; input++)
        {
            for (var output = 0; output < outputs; output++)
            {
                for (var place = 0; place < places; place++)
                {
                    written[(((input * outputs) + output) * places) + place] = values[(((input * places) + place) * outputs) + output];
                }
            }
        }

        return written;
    }

    /// <summary>
    /// The numbers along the first axis of a value laid out over a flattened image, from ONNX's order — channel, row, column — to the
    /// order here: row, column, channel. Whatever follows that axis moves with it.
    /// </summary>
    /// <param name="values">The numbers, with the flattened image's axis first.</param>
    /// <param name="rows">The image's rows.</param>
    /// <param name="columns">The image's columns.</param>
    /// <param name="channels">The image's channels.</param>
    /// <returns>The numbers with the first axis in the order here.</returns>
    public static float[] PlaceByPlace(float[] values, int rows, int columns, int channels)
    {
        var each = values.Length / (rows * columns * channels);
        var turned = new float[values.Length];

        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                for (var channel = 0; channel < channels; channel++)
                {
                    var here = (((row * columns) + column) * channels) + channel;
                    var there = (((channel * rows) + row) * columns) + column;
                    values.AsSpan(there * each, each).CopyTo(turned.AsSpan(here * each, each));
                }
            }
        }

        return turned;
    }

    /// <summary>The move back of <see cref="PlaceByPlace"/>: the first axis from the order here to ONNX's, channel by channel.</summary>
    /// <param name="values">The numbers, with the flattened image's axis first.</param>
    /// <param name="rows">The image's rows.</param>
    /// <param name="columns">The image's columns.</param>
    /// <param name="channels">The image's channels.</param>
    /// <returns>The numbers with the first axis in ONNX's order.</returns>
    public static float[] ChannelByChannel(float[] values, int rows, int columns, int channels)
    {
        var each = values.Length / (rows * columns * channels);
        var turned = new float[values.Length];

        for (var row = 0; row < rows; row++)
        {
            for (var column = 0; column < columns; column++)
            {
                for (var channel = 0; channel < channels; channel++)
                {
                    var here = (((row * columns) + column) * channels) + channel;
                    var there = (((channel * rows) + row) * columns) + column;
                    values.AsSpan(here * each, each).CopyTo(turned.AsSpan(there * each, each));
                }
            }
        }

        return turned;
    }
}
