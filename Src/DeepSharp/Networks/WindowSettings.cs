// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>How a window pads, as a network file says it: a number of steps, the word for a border worked out, or a pair of numbers for each axis.</summary>
/// <param name="Mode">How the border is had.</param>
/// <param name="Padding">The number of steps stated, read when the mode is <see cref="PaddingMode.Stated"/> and no side is stated apart.</param>
/// <param name="Sides">The steps at each end of each axis, when the file states them apart; nothing otherwise.</param>
internal readonly record struct PaddingSetting(PaddingMode Mode, int Padding, IReadOnlyList<int>? Sides);

/// <summary>
/// How a window's stride, dilation, rounding and padding are written to a network file and read back, alike for every layer
/// that walks a window.
/// </summary>
/// <remarks>
/// Every setting is written only when it says something the file did not say before it: a window that strides alike on every
/// axis writes its stride as the one number it always wrote, one that is not dilated writes no dilation at all, and one that
/// does not round up says nothing of rounding. So a window the release before this one could make writes what it wrote,
/// byte for byte, and a library that does not know a setting names the version of the part rather than the key.
/// </remarks>
internal static class WindowSettingsExtensions
{
    // The words a window's padding is written with besides a number: Keras's own.
    private const string SamePadding = "same";
    private const string CausalPadding = "causal";

    extension(Utf8JsonWriter writer)
    {
        /// <summary>Writes a window along a series: its length, its stride, its dilation, its rounding and its padding.</summary>
        internal void WriteWindow(Window1D window)
        {
            writer.WriteNumber("length", window.Length);
            writer.WriteNumber("stride", window.Stride);
            writer.WriteSpread("dilation", window.Dilation);
            writer.WriteRounding(window.RoundsUp);
            writer.WritePadding(window.PaddingMode, window.Padding, window.Border is { } sides ? [sides.Before, sides.After] : null);
        }

        /// <summary>Writes a window over an image: its sides, its stride, its dilation, its rounding and its padding.</summary>
        internal void WriteWindow(Window window)
        {
            writer.WriteNumber("height", window.Height);
            writer.WriteNumber("width", window.Width);
            writer.WriteSteps("stride", window.Stride, always: true);
            writer.WriteSteps("dilation", window.Dilation, always: false);
            writer.WriteRounding(window.RoundsUp);
            writer.WritePadding(
                window.PaddingMode,
                window.Padding,
                window.Border is { } sides ? [sides.Down.Before, sides.Down.After, sides.Across.Before, sides.Across.After] : null);
        }

        /// <summary>Writes a window through a volume: its sides, its stride, its dilation, its rounding and its padding.</summary>
        internal void WriteWindow(Window3D window)
        {
            writer.WriteNumber("depth", window.Depth);
            writer.WriteNumber("height", window.Height);
            writer.WriteNumber("width", window.Width);
            writer.WriteSteps("stride", window.Stride, always: true);
            writer.WriteSteps("dilation", window.Dilation, always: false);
            writer.WriteRounding(window.RoundsUp);
            writer.WritePadding(
                window.PaddingMode,
                window.Padding,
                window.Border is { } sides
                    ? [sides.Deep.Before, sides.Deep.After, sides.Down.Before, sides.Down.After, sides.Across.Before, sides.Across.After]
                    : null);
        }

        /// <summary>
        /// Writes a window's padding: the number of steps on every side, the word for a border worked out, or the steps at each
        /// end of each axis, a pair an axis in the axes' own order.
        /// </summary>
        internal void WritePadding(PaddingMode mode, int padding, IReadOnlyList<int>? sides)
        {
            if (sides is not null)
            {
                writer.WriteStartArray("padding");

                foreach (var side in sides)
                {
                    writer.WriteNumberValue(side);
                }

                writer.WriteEndArray();

                return;
            }

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

        // A stride or a dilation over an image: the one number where both axes agree, a list where they differ. A dilation of
        // one is what a window means by saying nothing of it, so it is not written then.
        private void WriteSteps(string key, Steps steps, bool always)
        {
            if (steps.Alike)
            {
                if (always || steps.Down != 1)
                {
                    writer.WriteNumber(key, steps.Down);
                }

                return;
            }

            writer.WriteStartArray(key);
            writer.WriteNumberValue(steps.Down);
            writer.WriteNumberValue(steps.Across);
            writer.WriteEndArray();
        }

        // The same through a volume.
        private void WriteSteps(string key, Steps3D steps, bool always)
        {
            if (steps.Alike)
            {
                if (always || steps.Deep != 1)
                {
                    writer.WriteNumber(key, steps.Deep);
                }

                return;
            }

            writer.WriteStartArray(key);
            writer.WriteNumberValue(steps.Deep);
            writer.WriteNumberValue(steps.Down);
            writer.WriteNumberValue(steps.Across);
            writer.WriteEndArray();
        }

        // A window along a series has one axis, so its dilation is one number, written only when it is not one.
        private void WriteSpread(string key, int spread)
        {
            if (spread != 1)
            {
                writer.WriteNumber(key, spread);
            }
        }

        /// <summary>Writes the places of nothing at each end of each axis, a pair an axis in the axes' own order.</summary>
        internal void WriteBorder(IReadOnlyList<int> sides) => writer.WritePadding(PaddingMode.Stated, 0, sides);

        // A window that stands only where it fits whole says nothing of rounding.
        private void WriteRounding(bool roundsUp)
        {
            if (roundsUp)
            {
                writer.WriteBoolean("roundsUp", true);
            }
        }
    }

    extension(Window1D window)
    {
        /// <summary>
        /// The oldest version of a network's part that says this window: the second, unless it is dilated, rounds its places
        /// up or states its border at each end apart, none of which a library before this release could read.
        /// </summary>
        internal int FileVersion =>
            window.Dilation != 1 || window.RoundsUp || window.Border is not null
                ? NetworkDocument.FourthVersion
                : NetworkDocument.SecondVersion;
    }

    extension(Window window)
    {
        /// <summary>
        /// The oldest version of a network's part that says this window: the second, unless it strides otherwise along one
        /// axis than along another, is dilated, rounds its places up or states its border at each end of each axis apart.
        /// </summary>
        internal int FileVersion =>
            !window.Stride.Alike || window.Dilation != (Steps)1 || window.RoundsUp || window.Border is not null
                ? NetworkDocument.FourthVersion
                : NetworkDocument.SecondVersion;
    }

    extension(Window3D window)
    {
        /// <summary>
        /// The oldest version of a network's part that says this window: the second, unless it strides otherwise along one
        /// axis than along another, is dilated, rounds its places up or states its border at each end of each axis apart.
        /// </summary>
        internal int FileVersion =>
            !window.Stride.Alike || window.Dilation != (Steps3D)1 || window.RoundsUp || window.Border is not null
                ? NetworkDocument.FourthVersion
                : NetworkDocument.SecondVersion;
    }

    extension(Rebuilding rebuilding)
    {
        /// <summary>Reads a window along a series back.</summary>
        /// <exception cref="FormatException">A setting is missing or of the wrong kind.</exception>
        internal Window1D Window1DIn(JsonElement settings)
        {
            var padding = rebuilding.PaddingIn(settings, 1);

            return new Window1D(rebuilding.Whole(settings, "length"))
            {
                Stride = rebuilding.Whole(settings, "stride"),
                Dilation = rebuilding.Whole(settings, "dilation", 1),
                RoundsUp = rebuilding.Holds(settings, "roundsUp"),
                PaddingMode = padding.Mode,
                Padding = padding.Padding,
                Border = padding.Sides is { } sides ? new Sides(sides[0], sides[1]) : null,
            };
        }

        /// <summary>Reads a window over an image back.</summary>
        /// <exception cref="FormatException">A setting is missing or of the wrong kind.</exception>
        internal Window Window2DIn(JsonElement settings)
        {
            var padding = rebuilding.PaddingIn(settings, 2);

            return new Window(rebuilding.Whole(settings, "height"), rebuilding.Whole(settings, "width"))
            {
                Stride = rebuilding.StepsIn(settings, "stride", 1),
                Dilation = rebuilding.StepsIn(settings, "dilation", 1),
                RoundsUp = rebuilding.Holds(settings, "roundsUp"),
                PaddingMode = padding.Mode,
                Padding = padding.Padding,
                Border = padding.Sides is { } sides ? new Margins(new Sides(sides[0], sides[1]), new Sides(sides[2], sides[3])) : null,
            };
        }

        /// <summary>Reads a window through a volume back.</summary>
        /// <exception cref="FormatException">A setting is missing or of the wrong kind.</exception>
        internal Window3D Window3DIn(JsonElement settings)
        {
            var padding = rebuilding.PaddingIn(settings, 3);

            return new Window3D(rebuilding.Whole(settings, "depth"), rebuilding.Whole(settings, "height"), rebuilding.Whole(settings, "width"))
            {
                Stride = rebuilding.Steps3DIn(settings, "stride", 1),
                Dilation = rebuilding.Steps3DIn(settings, "dilation", 1),
                RoundsUp = rebuilding.Holds(settings, "roundsUp"),
                PaddingMode = padding.Mode,
                Padding = padding.Padding,
                Border = padding.Sides is { } sides
                    ? new Margins3D(new Sides(sides[0], sides[1]), new Sides(sides[2], sides[3]), new Sides(sides[4], sides[5]))
                    : null,
            };
        }

        /// <summary>
        /// Reads a window's padding back: the number of steps on every side, the word for a border worked out, or the steps at
        /// each end of each axis.
        /// </summary>
        /// <param name="settings">The object the kind was written as.</param>
        /// <param name="rank">How many axes the window walks, which is how many pairs a border stated apart holds.</param>
        /// <exception cref="FormatException">The padding is missing, or is none of a whole number, one of the words, and a pair for each axis.</exception>
        internal PaddingSetting PaddingIn(JsonElement settings, int rank)
        {
            if (rebuilding.Lists(settings, "padding"))
            {
                var sides = rebuilding.Wholes(settings, "padding");

                return sides.Count == 2 * rank
                    ? new PaddingSetting(PaddingMode.Stated, 0, sides)
                    : throw rebuilding.Refuse("padding", $"A window of {rank} {(rank == 1 ? "axis" : "axes")} that states its padding at each end states {2 * rank} numbers, and {sides.Count} are written.");
            }

            return rebuilding.SaysOneOf(settings, "padding", "a whole number, 'same', 'causal' or a pair for each axis,", SamePadding, CausalPadding) switch
            {
                SamePadding => new PaddingSetting(PaddingMode.Same, 0, null),
                CausalPadding => new PaddingSetting(PaddingMode.Causal, 0, null),
                _ => new PaddingSetting(PaddingMode.Stated, rebuilding.Whole(settings, "padding"), null),
            };
        }

        /// <summary>Reads the places of nothing a layer that only pads adds at each end of each axis.</summary>
        /// <param name="settings">The object the kind was written as.</param>
        /// <param name="rank">How many axes it pads, which is how many pairs it states.</param>
        /// <exception cref="FormatException">The padding is missing, or is not a pair for each axis.</exception>
        internal IReadOnlyList<int> BorderIn(JsonElement settings, int rank)
        {
            var sides = rebuilding.Wholes(settings, "padding");

            return sides.Count == 2 * rank
                ? sides
                : throw rebuilding.Refuse(
                    "padding",
                    $"A layer that pads {rank} {(rank == 1 ? "axis" : "axes")} states the places at each end of each, which is {2 * rank} numbers, and {sides.Count} are written.");
        }

        // A stride or a dilation over an image: the one number for both axes, or one for each.
        private Steps StepsIn(JsonElement settings, string key, int whenAbsent)
        {
            if (!rebuilding.Lists(settings, key))
            {
                return rebuilding.Whole(settings, key, whenAbsent);
            }

            var steps = rebuilding.Wholes(settings, key);

            return steps.Count == 2
                ? new Steps(steps[0], steps[1])
                : throw rebuilding.Refuse(key, $"A window over an image writes its '{key}' as one number for both axes or as one for each, and {steps.Count} are written.");
        }

        // The same through a volume.
        private Steps3D Steps3DIn(JsonElement settings, string key, int whenAbsent)
        {
            if (!rebuilding.Lists(settings, key))
            {
                return rebuilding.Whole(settings, key, whenAbsent);
            }

            var steps = rebuilding.Wholes(settings, key);

            return steps.Count == 3
                ? new Steps3D(steps[0], steps[1], steps[2])
                : throw rebuilding.Refuse(key, $"A window through a volume writes its '{key}' as one number for every axis or as one for each, and {steps.Count} are written.");
        }
    }
}
