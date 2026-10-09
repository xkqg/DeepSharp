// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>How a window pads, as a network file says it: a number of steps, or the word for a border worked out.</summary>
/// <param name="Mode">How the border is had.</param>
/// <param name="Padding">The number of steps stated, read when the mode is <see cref="PaddingMode.Stated"/>.</param>
internal readonly record struct PaddingSetting(PaddingMode Mode, int Padding);

/// <summary>How a window's padding is written to a network file and read back, alike for every layer that walks a window.</summary>
internal static class WindowSettingsExtensions
{
    // The words a window's padding is written with besides a number: Keras's own.
    private const string SamePadding = "same";
    private const string CausalPadding = "causal";

    extension(Utf8JsonWriter writer)
    {
        /// <summary>Writes a window along a series: its length, its stride and its padding.</summary>
        internal void WriteWindow(Window1D window)
        {
            writer.WriteNumber("length", window.Length);
            writer.WriteNumber("stride", window.Stride);
            writer.WritePadding(window.PaddingMode, window.Padding);
        }

        /// <summary>Writes a window over an image: its sides, its stride and its padding.</summary>
        internal void WriteWindow(Window window)
        {
            writer.WriteNumber("height", window.Height);
            writer.WriteNumber("width", window.Width);
            writer.WriteNumber("stride", window.Stride);
            writer.WritePadding(window.PaddingMode, window.Padding);
        }

        /// <summary>Writes a window through a volume: its sides, its stride and its padding.</summary>
        internal void WriteWindow(Window3D window)
        {
            writer.WriteNumber("depth", window.Depth);
            writer.WriteNumber("height", window.Height);
            writer.WriteNumber("width", window.Width);
            writer.WriteNumber("stride", window.Stride);
            writer.WritePadding(window.PaddingMode, window.Padding);
        }

        /// <summary>Writes a window's padding: the number of steps on every side, or — for one worked out — the word for it.</summary>
        internal void WritePadding(PaddingMode mode, int padding)
        {
            switch (mode)
            {
                case PaddingMode.Same:
                    writer.WriteString("padding", SamePadding);
                    break;
                case PaddingMode.Causal:
                    writer.WriteString("padding", CausalPadding);
                    break;
                default:
                    writer.WriteNumber("padding", padding);
                    break;
            }
        }
    }

    extension(Rebuilding rebuilding)
    {
        /// <summary>Reads a window along a series back.</summary>
        /// <exception cref="FormatException">A setting is missing or of the wrong kind.</exception>
        internal Window1D Window1DIn(JsonElement settings)
        {
            var padding = rebuilding.PaddingIn(settings);

            return new Window1D(rebuilding.Whole(settings, "length")) { Stride = rebuilding.Whole(settings, "stride"), PaddingMode = padding.Mode, Padding = padding.Padding };
        }

        /// <summary>Reads a window over an image back.</summary>
        /// <exception cref="FormatException">A setting is missing or of the wrong kind.</exception>
        internal Window Window2DIn(JsonElement settings)
        {
            var padding = rebuilding.PaddingIn(settings);

            return new Window(rebuilding.Whole(settings, "height"), rebuilding.Whole(settings, "width"))
            {
                Stride = rebuilding.Whole(settings, "stride"),
                PaddingMode = padding.Mode,
                Padding = padding.Padding,
            };
        }

        /// <summary>Reads a window through a volume back.</summary>
        /// <exception cref="FormatException">A setting is missing or of the wrong kind.</exception>
        internal Window3D Window3DIn(JsonElement settings)
        {
            var padding = rebuilding.PaddingIn(settings);

            return new Window3D(rebuilding.Whole(settings, "depth"), rebuilding.Whole(settings, "height"), rebuilding.Whole(settings, "width"))
            {
                Stride = rebuilding.Whole(settings, "stride"),
                PaddingMode = padding.Mode,
                Padding = padding.Padding,
            };
        }

        /// <summary>Reads a window's padding back: the number of steps on every side, or the word for a border worked out.</summary>
        /// <exception cref="FormatException">The padding is missing, or is neither a whole number nor one of the words.</exception>
        internal PaddingSetting PaddingIn(JsonElement settings) =>
            rebuilding.SaysOneOf(settings, "padding", "a whole number, 'same' or 'causal',", SamePadding, CausalPadding) switch
            {
                SamePadding => new PaddingSetting(PaddingMode.Same, 0),
                CausalPadding => new PaddingSetting(PaddingMode.Causal, 0),
                _ => new PaddingSetting(PaddingMode.Stated, rebuilding.Whole(settings, "padding")),
            };
    }
}
