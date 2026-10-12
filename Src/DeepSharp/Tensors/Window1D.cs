// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Tensors;

/// <summary>
/// The patch a one-dimensional convolution or pooling looks at, and how it walks along a series: so many steps long, moved
/// so many places at a time, over a series with so many steps of nothing added at its ends.
/// </summary>
/// <param name="Length">How many steps of the series the window covers.</param>
/// <remarks>
/// A window that moves one place at a time over a series with no border, as PyTorch's <c>Conv1d</c> and Keras's
/// <c>Conv1D</c> leave it, unless said: <c>new Window1D(5) { Stride = 2, Padding = 2 }</c>. Its border is stated, the same
/// number of steps at each end; stated at each end apart; worked out as TensorFlow's <c>padding='same'</c> works it out; or
/// <c>causal</c>, as Keras's <c>padding='causal'</c> is, with the window's reach less one steps before the series and none
/// after it. Every one of them is the two-dimensional <see cref="Window"/>'s own rule, applied to a window one row tall.
/// <para>
/// What a window covers is what <see cref="Window"/> covers: a patch of neighbouring steps unless it is dilated, every
/// channel of a step at once. There is one axis, so a stride and a dilation are one number each.
/// </para>
/// </remarks>
public readonly record struct Window1D(int Length)
{
    /// <summary>How many places the window moves between one patch and the next; one unless said.</summary>
    public int Stride { get; init; } = 1;

    /// <summary>How many steps apart the cells of a patch are; one — neighbouring steps — unless said.</summary>
    public int Dilation { get; init; } = 1;

    /// <summary>How many steps of nothing surround the series at each end; none unless said.</summary>
    /// <remarks>Read only when <see cref="PaddingMode"/> is <see cref="PaddingMode.Stated"/>.</remarks>
    public int Padding { get; init; }

    /// <summary>How the border is had: the one <see cref="Padding"/> states at each end, unless said.</summary>
    public PaddingMode PaddingMode { get; init; }

    /// <summary>Whether the window also stands at a last place whose patch overhangs the end of the series, as PyTorch's <c>ceil_mode</c> has it.</summary>
    public bool RoundsUp { get; init; }

    /// <summary>The steps of nothing at each end of the series, stated apart; nothing, unless said.</summary>
    /// <remarks>Instead of <see cref="Padding"/> and of every mode but <see cref="PaddingMode.Stated"/>: a window given both is refused.</remarks>
    public Sides? Border { get; init; }

    /// <summary>How many places the window stands at along a series so many steps long.</summary>
    /// <param name="length">How many steps the series has.</param>
    /// <returns>The places, nought when the window does not fit.</returns>
    public int PlacesOver(int length) => Over(length).ColumnsOver(length);

    /// <inheritdoc />
    public override string ToString()
    {
        var dilated = Dilation == 1 ? string.Empty : $", dilated by {Dilation}";
        var rounding = RoundsUp ? ", rounding up" : string.Empty;
        var border = Border is { } sides ? $"{sides.Before} before, {sides.After} after" : PaddingMode.Said(Padding);

        return $"window {Length} (stride {Stride}{dilated}{rounding}, padding {border})";
    }

    /// <summary>Refuses a window that cannot stand anywhere, in the words the two-dimensional window refuses with.</summary>
    /// <exception cref="ArgumentException">
    /// The length or the stride is below one, the dilation below one, the border below nothing, or a border is given besides
    /// one worked out from the series.
    /// </exception>
    internal void RequireStanding()
    {
        if (Length < 1 || Stride < 1 || Padding < 0 || (Border is { } sides && (sides.Before < 0 || sides.After < 0)))
        {
            throw new ArgumentException($"A {this} cannot stand anywhere: its length and its stride are at least one, and its border at least nothing.", "window");
        }

        if (Dilation < 1)
        {
            throw new ArgumentException($"A {this} cannot stand anywhere: its dilation is at least one.", "window");
        }

        if (Border is not null && (Padding != 0 || PaddingMode != PaddingMode.Stated))
        {
            throw new ArgumentException($"A {this} states its border at each end, and cannot be given a padding or a way to work one out besides.", "window");
        }

        if (PaddingMode != PaddingMode.Stated && Padding != 0)
        {
            throw new ArgumentException($"A {this} works its border out from each series it stands on, and cannot be given one of {Padding} besides.", "window");
        }
    }

    /// <summary>
    /// The window as the engine's two-dimensional unfolding walks it over a series laid out one row tall: one row of patches,
    /// padded along the series and never across the row.
    /// </summary>
    /// <param name="length">How many steps the series has.</param>
    /// <returns>A window one row tall, its border stated at each end of the series.</returns>
    internal Window Over(int length)
    {
        var like = new Window(1, Length)
        {
            Stride = new Steps(1, Stride),
            Dilation = new Steps(1, Dilation),
            Padding = Padding,
            PaddingMode = PaddingMode,
            RoundsUp = RoundsUp,
        };

        var worked = like.BordersOver(1, length);
        var along = Border ?? new Sides(worked.Left, worked.Right);

        return like with { Padding = 0, PaddingMode = PaddingMode.Stated, Border = new Margins(new Sides(0, 0), along) };
    }
}

/// <summary>What a padding is called, in the words a window says how it pads with.</summary>
internal static class PaddingModeExtensions
{
    extension(PaddingMode mode)
    {
        /// <summary>The padding as a window says it: the word for a border worked out, the number for one stated.</summary>
        /// <param name="padding">The stated padding, read when the mode is <see cref="PaddingMode.Stated"/>.</param>
        /// <returns><c>'same'</c>, <c>'causal'</c>, or the number.</returns>
        internal string Said(int padding) => mode switch
        {
            PaddingMode.Same => "'same'",
            PaddingMode.Causal => "'causal'",
            _ => $"{padding}",
        };
    }
}
