// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Networks;

/// <summary>
/// How a run scales its loss so that small gradients survive an engine that keeps its work in a narrow width: PyTorch's
/// <c>GradScaler</c>. Every gradient is worked back from a seed that is a power of two, so it is scaled and scaled down again without
/// losing a digit; a step whose gradients are not numbers is skipped and a dynamic scale lowered; a dynamic scale is raised after a
/// run of clean steps.
/// </summary>
/// <remarks>
/// A scale is part of what a run is: a checkpoint records it, with where the scale had got to, and a run going on from the
/// checkpoint under another, or under none, is refused. An engine at full width needs none, and a run that scales is the run that
/// does not, to the bit, wherever nothing overflows. The scale, its growth and its backoff are powers of two, so that is true.
/// </remarks>
public sealed record LossScale
{
    private LossScale()
    {
    }

    /// <summary>The scale the run begins at.</summary>
    public double Initial { get; private init; }

    /// <summary>What a scale is multiplied by after <see cref="Interval"/> clean steps in a row; one for a scale that never changes.</summary>
    public double Growth { get; private init; } = 1;

    /// <summary>What a scale is multiplied by after a step whose gradients were not numbers; one for a scale that never changes.</summary>
    public double Backoff { get; private init; } = 1;

    /// <summary>How many clean steps in a row raise the scale; one for a scale that never changes.</summary>
    public int Interval { get; private init; } = 1;

    /// <summary>Whether the scale changes as the run goes: lowered after a step that was skipped, raised after clean ones.</summary>
    public bool Changes => Growth is not 1 || Backoff is not 1;

    /// <summary>A scale that stays where it is for the whole run.</summary>
    /// <param name="scale">The scale: a power of two.</param>
    /// <returns>The scale.</returns>
    /// <exception cref="ArgumentOutOfRangeException">It is not a power of two above nothing.</exception>
    public static LossScale Fixed(double scale) => new() { Initial = RequirePowerOfTwo(scale, nameof(scale)) };

    /// <summary>A scale that follows the run, as PyTorch's does: <c>init_scale</c>, <c>growth_factor</c>, <c>growth_interval</c> and <c>backoff_factor</c>.</summary>
    /// <param name="initial">The scale the run begins at: a power of two; 65536, unless said.</param>
    /// <param name="growth">What the scale is multiplied by after <paramref name="interval"/> clean steps: a power of two above one; two, unless said.</param>
    /// <param name="interval">How many clean steps in a row raise it: at least one; two thousand, unless said.</param>
    /// <param name="backoff">What it is multiplied by after a skipped step: a power of two below one; a half, unless said.</param>
    /// <returns>The scale.</returns>
    /// <exception cref="ArgumentOutOfRangeException">A number is not what it is described as.</exception>
    public static LossScale Dynamic(double initial = 65536, double growth = 2, int interval = 2000, double backoff = 0.5)
    {
        RequirePowerOfTwo(growth, nameof(growth));
        RequirePowerOfTwo(backoff, nameof(backoff));
        ArgumentOutOfRangeException.ThrowIfLessThan(interval, 1);

        if (growth <= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(growth), growth, "A scale that grows is multiplied by a power of two above one.");
        }

        if (backoff >= 1)
        {
            throw new ArgumentOutOfRangeException(nameof(backoff), backoff, "A scale that backs off is multiplied by a power of two below one.");
        }

        return new LossScale { Initial = RequirePowerOfTwo(initial, nameof(initial)), Growth = growth, Interval = interval, Backoff = backoff };
    }

    /// <summary>
    /// A scale as a file read back says it: the same numbers a scale is made of, held to the same rules.
    /// </summary>
    /// <param name="initial">The scale the run began at.</param>
    /// <param name="growth">What it is multiplied by after the interval's clean steps; one for a scale that never changes.</param>
    /// <param name="interval">How many clean steps in a row raise it.</param>
    /// <param name="backoff">What it is multiplied by after a skipped step; one for a scale that never changes.</param>
    /// <returns>The scale.</returns>
    /// <exception cref="ArgumentOutOfRangeException">A number is not what a scale's is.</exception>
    internal static LossScale Read(double initial, double growth, int interval, double backoff) =>
        growth is 1 && backoff is 1 ? Fixed(initial) : Dynamic(initial, growth, interval, backoff);

    /// <inheritdoc />
    public override string ToString() =>
        Changes
            ? string.Create(CultureInfo.InvariantCulture, $"a dynamic loss scale from {Initial}, times {Growth} after {Interval} clean steps and times {Backoff} after a skipped one")
            : string.Create(CultureInfo.InvariantCulture, $"a fixed loss scale of {Initial}");

    private static double RequirePowerOfTwo(double scale, string name) =>
        double.IsFinite(scale) && scale > 0 && double.IsInteger(double.Log2(scale))
            ? scale
            : throw new ArgumentOutOfRangeException(name, scale, "A scale is a power of two, so that scaling and scaling down again costs no digit.");
}

/// <summary>Where a loss scale had got to: the scale the next step is worked back from, and how many clean steps in a row it has had.</summary>
/// <param name="Scale">The scale.</param>
/// <param name="Clean">How many steps in a row were clean, since the scale last changed.</param>
internal readonly record struct ScalerState(double Scale, int Clean);

/// <summary>A loss scale at work: the scale of the step in hand, lowered after a skipped step and raised after enough clean ones.</summary>
/// <param name="options">What the run was told.</param>
/// <param name="state">Where it begins: the scale the options begin at, or where a checkpoint left it.</param>
internal sealed class LossScaler(LossScale options, ScalerState state)
{
    private double _scale = state.Scale;
    private int _clean = state.Clean;

    /// <summary>The scale the batches of the window in hand are worked back from.</summary>
    public double Scale => _scale;

    /// <summary>Where the scale stands, for a checkpoint.</summary>
    public ScalerState State => new(_scale, _clean);

    /// <summary>A step was taken: one more clean step, and the scale raised when enough of them have followed one another.</summary>
    public void Stepped()
    {
        if (!options.Changes)
        {
            return;
        }

        _clean++;

        if (_clean >= options.Interval)
        {
            var raised = _scale * options.Growth;

            if (double.IsFinite(raised))
            {
                _scale = raised;
            }

            _clean = 0;
        }
    }

    /// <summary>A step's gradients were not numbers: the scale is lowered, or the run is stopped where there is nothing left to lower.</summary>
    /// <exception cref="InvalidOperationException">The scale is fixed, or has been lowered to one: nothing a scaler does would help.</exception>
    public void Overflowed()
    {
        if (!options.Changes)
        {
            throw new InvalidOperationException(string.Create(
                CultureInfo.InvariantCulture,
                $"The gradients of a step are not finite numbers at a fixed loss scale of {_scale}, and a fixed scale is not lowered: the rows, the rate or the network's start have driven them past what the width holds, or the scale is too high for it."));
        }

        if (_scale <= 1)
        {
            throw new InvalidOperationException(string.Create(
                CultureInfo.InvariantCulture,
                $"The gradients of a step are not finite numbers at a loss scale of {_scale}, and there is nothing left to lower: the rows, the rate or the network's start have driven them past what the width holds."));
        }

        _scale *= options.Backoff;
        _clean = 0;
    }
}
