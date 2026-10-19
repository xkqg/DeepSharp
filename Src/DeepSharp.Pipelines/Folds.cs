// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// The i-th of k slices of the rows a model learns from and is chosen with: the slice is the validation rows and the others
/// are the training rows.
/// </summary>
/// <remarks>
/// Every split here cut once, so a small table was measured by the luck of one deal. A fold is an ordinary split that says which
/// slice is validation: the rows that are neither test nor predict are dealt in the order the seed ranks them into <see cref="Folds"/>
/// slices of about as many rows, a repeated record staying in one, and the test rows are the same on every fold — the seed deals
/// them alone, or the test seed does. Every fold is a pipeline of its own with a fit of its own, so a network trained behind one is
/// refused beside another.
/// </remarks>
public readonly record struct Fold
{
    /// <summary>The i-th of k slices.</summary>
    /// <param name="folds">How many slices there are: at least two.</param>
    /// <param name="index">Which of them is validation, counted from nought.</param>
    /// <exception cref="ArgumentOutOfRangeException">There are fewer than two slices, or the index is not one of them.</exception>
    public Fold(int folds, int index)
    {
        if (folds < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(folds), folds, "Folds cut the rows into at least two slices.");
        }

        if (index < 0 || index >= folds)
        {
            throw new ArgumentOutOfRangeException(nameof(index), index, $"The fold is one of the {folds} slices, counted from nought, and {index} is none of them.");
        }

        Folds = folds;
        Index = index;
    }

    /// <summary>How many slices there are.</summary>
    public int Folds { get; }

    /// <summary>Which of them is validation, counted from nought.</summary>
    public int Index { get; }

    /// <summary>The key a file names how many slices there are under.</summary>
    internal static WholeNumberParameter FoldsKey { get; } = new(
        "folds",
        "How many slices the rows that are neither test nor predict are cut into, at least two; one of them is validation and the others are training. Left out, the rows are cut once, as shares say.",
        5)
    {
        AtLeast = 2,
        LeftOut = 0,
    };

    /// <summary>The key a file names which slice is validation under.</summary>
    internal static WholeNumberParameter FoldKey { get; } = new(
        "fold",
        "Which of the slices is validation, counted from nought; said with folds, and below them.",
        0)
    {
        AtLeast = 0,
        LeftOut = -1,
    };

    /// <summary>Which slice a row belongs to, when so many rows are dealt in order into this many slices.</summary>
    /// <param name="place">The row's place in the order, from nought.</param>
    /// <param name="count">How many rows are dealt.</param>
    /// <returns>The slice, from nought.</returns>
    internal int SliceOf(int place, int count) => (int)((long)place * Folds / count);

    /// <summary>The fold a step's file names; nothing when it names none.</summary>
    /// <param name="step">The step as the file writes it.</param>
    /// <returns>The fold.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The file names a slice it does not have.</exception>
    /// <exception cref="FormatException">The file names one of the two keys without the other.</exception>
    internal static Fold? ReadFrom(JsonElement step)
    {
        var folds = step.TryGetProperty(FoldsKey.Key, out _);
        var fold = step.TryGetProperty(FoldKey.Key, out _);

        return !folds && !fold ? null
            : folds && fold ? new Fold(FoldsKey.Read(step), FoldKey.Read(step))
            : throw new FormatException($"A fold says how many slices there are, '{FoldsKey.Key}', and which of them is validation, '{FoldKey.Key}': this file says only '{(folds ? FoldsKey.Key : FoldKey.Key)}'.");
    }

    /// <summary>Refuses shares that say a validation share where folds say what validation is.</summary>
    /// <param name="shares">The shares the step is given.</param>
    /// <exception cref="ArgumentException">They say a validation share.</exception>
    internal static void RequireNoValidationShare(SplitShares shares)
    {
        if (shares.Validation != 0)
        {
            throw new ArgumentException(
                "Folds say which rows are validation — one slice of the rows that are neither test nor predict — so the shares beside folds say no validation share: leave it out.",
                nameof(shares));
        }
    }
}

/// <summary>
/// The i-th of k places the cut between the rows a model learns from and the rows it is chosen with stands at, in a series: the
/// training rows grow with the index, validation is the block after them, and the test rows are one block at the end.
/// </summary>
/// <remarks>
/// A series is not measured as forecasting is by dealing it again: it is measured by moving the cut forward and fitting again. The
/// rows that are neither test nor predict, in the order of their time, are cut into one block more than there are origins; the
/// training rows of origin <c>i</c> are the first <c>i + 1</c> blocks and validation is the next, so what a model learned from always
/// came before what it is chosen on. The rows after validation, up to the test rows, are fitted on by nothing and handed to nothing,
/// and a gap keeps the last moments of training and of validation apart, as it does for a single cut. A moment is never divided.
/// </remarks>
public readonly record struct Origin
{
    /// <summary>The i-th of k places.</summary>
    /// <param name="origins">How many places there are: at least two.</param>
    /// <param name="index">Which of them this is, counted from nought.</param>
    /// <exception cref="ArgumentOutOfRangeException">There are fewer than two places, or the index is not one of them.</exception>
    public Origin(int origins, int index)
    {
        if (origins < 2)
        {
            throw new ArgumentOutOfRangeException(nameof(origins), origins, "A cut is moved to at least two places.");
        }

        if (index < 0 || index >= origins)
        {
            throw new ArgumentOutOfRangeException(nameof(index), index, $"The origin is one of the {origins} places, counted from nought, and {index} is none of them.");
        }

        Origins = origins;
        Index = index;
    }

    /// <summary>How many places there are.</summary>
    public int Origins { get; }

    /// <summary>Which of them this is, counted from nought.</summary>
    public int Index { get; }

    /// <summary>The key a file names how many places there are under.</summary>
    internal static WholeNumberParameter OriginsKey { get; } = new(
        "origins",
        "How many places the cut between training and validation is moved to, at least two; the rows that are neither test nor predict are cut into one block more. Left out, the rows are cut once, as shares say.",
        3)
    {
        AtLeast = 2,
        LeftOut = 0,
    };

    /// <summary>The key a file names which place this is under.</summary>
    internal static WholeNumberParameter OriginKey { get; } = new(
        "origin",
        "Which of the places this is, counted from nought; said with origins, and below them.",
        0)
    {
        AtLeast = 0,
        LeftOut = -1,
    };

    /// <summary>Which block a row belongs to, when so many rows are dealt in time order into one block more than there are origins.</summary>
    /// <param name="place">The row's place in the order, from nought.</param>
    /// <param name="count">How many rows are dealt.</param>
    /// <returns>The block, from nought.</returns>
    internal int BlockOf(int place, int count) => (int)((long)place * (Origins + 1) / count);

    /// <summary>The place a step's file names; nothing when it names none.</summary>
    /// <param name="step">The step as the file writes it.</param>
    /// <returns>The place.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The file names a place it does not have.</exception>
    /// <exception cref="FormatException">The file names one of the two keys without the other.</exception>
    internal static Origin? ReadFrom(JsonElement step)
    {
        var origins = step.TryGetProperty(OriginsKey.Key, out _);
        var origin = step.TryGetProperty(OriginKey.Key, out _);

        return !origins && !origin ? null
            : origins && origin ? new Origin(OriginsKey.Read(step), OriginKey.Read(step))
            : throw new FormatException($"An origin says how many places there are, '{OriginsKey.Key}', and which of them this is, '{OriginKey.Key}': this file says only '{(origins ? OriginsKey.Key : OriginKey.Key)}'.");
    }
}
