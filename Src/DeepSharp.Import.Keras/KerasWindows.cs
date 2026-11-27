// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Import.Keras;

/// <summary>Where a Keras layer's window stands and how it walks, as the layer says it, before its border is had.</summary>
/// <param name="Sides">The length of each side of the window, outermost first.</param>
/// <param name="Strides">How many places it moves between one patch and the next, along each axis.</param>
/// <param name="Dilation">How many places apart its cells are, along each axis.</param>
internal readonly record struct KerasWindow(int[] Sides, int[] Strides, int[] Dilation)
{
    /// <summary>The window with its border had as given.</summary>
    /// <param name="padding">How the border is had.</param>
    /// <returns>The footprint the layer walks.</returns>
    public Footprint Padded(PaddingMode padding) => new(Sides, Strides, Dilation, padding);
}

/// <summary>A layer's rules for its numbers, and whether it adds a bias: what a layer with a kernel and a bias says of them.</summary>
/// <param name="Bias">Whether the layer adds a bias.</param>
/// <param name="Rules">The rules for its kernel and its bias, and whether it is held still.</param>
internal readonly record struct KerasBiasedRules(bool Bias, KerasKind.Rules Rules);

/// <summary>What a Keras layer whose window walks one, two or three axes says, read in one place for every layer that has one.</summary>
internal static class KerasWindowExtensions
{
    extension(KerasSettings layer)
    {
        /// <summary>The window a convolution walks: Keras's <c>kernel_size</c>, <c>strides</c> and <c>dilation_rate</c>, each noted when it is not one length for every axis.</summary>
        /// <param name="axes">How many axes the window walks: one, two or three.</param>
        /// <returns>The window.</returns>
        internal KerasWindow Window(int axes) => new(layer.Axes("kernel_size", axes), layer.Axes("strides", axes), layer.Axes("dilation_rate", axes));

        /// <summary>
        /// How the border of a window is had: Keras's <c>padding</c>, <c>valid</c> unless it says otherwise. A window here pads as
        /// valid or same, and along a series, where the layer allows it, as causal; any other is refused here.
        /// </summary>
        /// <param name="axes">How many axes the window walks.</param>
        /// <param name="allowCausal">Whether the layer may pad as causal, which only a window along one axis can.</param>
        /// <returns>The border as a window here has it; the stated one when the padding is refused, which nothing is built from while a fault is noted.</returns>
        internal PaddingMode Padding(int axes, bool allowCausal)
        {
            var padding = layer.Setting("padding", "valid");
            var causal = allowCausal && axes == 1;

            if (padding is not ("valid" or "same") && !(causal && padding == "causal"))
            {
                layer.Refuse($"it pads as '{padding.Quoted()}', and a window here pads as {(causal ? "'valid', 'same' or 'causal'" : "'valid' or 'same'")}.");
            }

            return padding switch
            {
                "same" => PaddingMode.Same,
                "causal" when causal => PaddingMode.Causal,
                _ => PaddingMode.Stated,
            };
        }

        /// <summary>Whether a layer with a kernel adds a bias, and the rules it puts on its kernel, its bias and its being held still.</summary>
        /// <param name="kernel">What Keras calls the kernel in the settings: <c>kernel</c>, <c>depthwise</c>.</param>
        /// <param name="rank">How many axes the kernel has in Keras's layout.</param>
        /// <param name="bounded">The axes a bound over each unit's numbers is over; nothing for a kernel a bound cannot be held on.</param>
        /// <returns>Whether it adds a bias, and the rules.</returns>
        internal KerasBiasedRules Biased(string kernel, int rank, int[]? bounded)
        {
            var bias = KerasKind.KeepsBias(layer);

            return new(bias, new KerasKind.Rules(KerasKind.Governing(layer, kernel, rank, bounded), bias ? KerasKind.Governing(layer, "bias", 1, [0]) : default, KerasKind.Held(layer)));
        }
    }
}
