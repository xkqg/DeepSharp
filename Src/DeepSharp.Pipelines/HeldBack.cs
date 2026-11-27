// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>The shares a pipeline holds back before it is split: to predict on, and to calibrate an interval on.</summary>
/// <param name="Predict">The share held back to predict on; nought for none.</param>
/// <param name="Calibration">The share set aside to calibrate on; nought for none.</param>
/// <remarks>
/// Said above the split and carried by it, so every split verb the builder writes hands them on the same way, and a split
/// of the caller's — which deals its own parts — is refused when either was said, rather than left holding a promise
/// nothing keeps.
/// </remarks>
internal readonly record struct HeldBack(double Predict, double Calibration)
{
    /// <summary>Whether anything is held back.</summary>
    internal bool Any => Predict > 0 || Calibration > 0;

    /// <summary>The shares a split the builder writes divides the rows into, with what is held back.</summary>
    /// <param name="train">The share the model learns from.</param>
    /// <param name="validation">The share used while choosing between models; none, unless said.</param>
    /// <returns>The shares, the one to be measured on worked out.</returns>
    internal SplitShares Of(double train, double validation = 0) => SplitShares.Of(train, validation, Predict, Calibration);

    /// <summary>These shares, with a share held back to predict on.</summary>
    /// <param name="share">How much to hold back.</param>
    /// <returns>The shares held back from here on.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The share is not a share.</exception>
    /// <exception cref="InvalidOperationException">A share is already held back to predict on.</exception>
    internal HeldBack Predicting(double share) =>
        this with { Predict = Checked(share, Predict, "predict on", $"holds {Predict} back to predict on") };

    /// <summary>These shares, with a share set aside to calibrate on.</summary>
    /// <param name="share">How much to set aside.</param>
    /// <returns>The shares held back from here on.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The share is not a share.</exception>
    /// <exception cref="InvalidOperationException">A share is already set aside to calibrate on.</exception>
    internal HeldBack Calibrating(double share) =>
        this with { Calibration = Checked(share, Calibration, "calibrate on", $"sets {Calibration} aside to calibrate on") };

    /// <summary>Refuses a pipeline that holds a share back and never splits the rows, which is what sets it aside.</summary>
    /// <exception cref="InvalidOperationException">A share is held back.</exception>
    internal void ThrowIfNothingSplits()
    {
        ThrowIfNothingSplits(Predict, $"holds {Predict} back to predict on");
        ThrowIfNothingSplits(Calibration, $"sets {Calibration} aside to calibrate on");
    }

    /// <summary>Refuses a split of the caller's while a share is held back, which only a split the builder writes is given.</summary>
    /// <param name="verb">The verb of the caller's split.</param>
    /// <exception cref="InvalidOperationException">A share is held back.</exception>
    internal void ThrowIfHeldFrom(string verb)
    {
        ThrowIfHeldFrom(verb, Predict, $"holds {Predict} back to predict on");
        ThrowIfHeldFrom(verb, Calibration, $"sets {Calibration} aside to calibrate on");
    }

    // A share said twice would quietly replace the first, and one that is not a share is refused where it is written.
    private static double Checked(double share, double already, string purpose, string holding)
    {
        if (!double.IsFinite(share) || share <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(share), share, $"A share to {purpose} is more than none of the data.");
        }

        if (already > 0)
        {
            throw new InvalidOperationException(
                $"This pipeline already {holding}, and a second share would quietly replace the first.");
        }

        return share;
    }

    private static void ThrowIfNothingSplits(double share, string holding)
    {
        if (share > 0)
        {
            throw new InvalidOperationException(
                $"This pipeline {holding} but never splits the rows, and the split is what sets that share aside.");
        }
    }

    private static void ThrowIfHeldFrom(string verb, double share, string holding)
    {
        if (share > 0)
        {
            throw new InvalidOperationException(
                $"This pipeline {holding}, and '{verb}' deals its own parts: a share held back is given to a split this library "
                + "writes. Leave the share out, or let the split you write set it aside.");
        }
    }
}
