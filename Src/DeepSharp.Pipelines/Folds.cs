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
/// refused beside another. The shares beside a fold say how many rows are cut — training and validation together — and a validation
/// share is not asked for: the slice is what validation is, so one that is said only adds to the rows that are cut.
/// </remarks>
public readonly record struct Fold
{
    /// <summary>The i-th of k slices.</summary>
    /// <param name="folds">How many slices there are: at least two.</param>
    /// <param name="index">Which of them is validation, counted from nought.</param>
    /// <exception cref="ArgumentOutOfRangeException">There are fewer than two slices, or the index is not one of them.</exception>
    public Fold(int folds, int index)
    {
        Cut.Check(folds, index, nameof(folds));

        Folds = folds;
        Index = index;
    }

    /// <summary>How many slices there are.</summary>
    public int Folds { get; }

    /// <summary>Which of them is validation, counted from nought.</summary>
    public int Index { get; }

    /// <summary>The words and the keys of this cut: how many slices there are, and which of them is validation.</summary>
    private static CountedCut Cut { get; } = new()
    {
        Noun = "fold",
        Article = "A",
        Plural = "slices",
        Which = "is validation",
        TooFew = "Folds cut the rows into at least two slices.",
        CountKey = new WholeNumberParameter(
            "folds",
            "How many slices the rows that are neither test nor predict are cut into, at least two; one of them is validation and the others are training. Left out, the rows are cut once, as shares say.",
            5)
        {
            AtLeast = 2,
            LeftOut = 0,
            OnlyWhenMeant = true,
        },
        IndexKey = new WholeNumberParameter(
            "fold",
            "Which of the slices is validation, counted from nought; said with folds, and below them.",
            0)
        {
            AtLeast = 0,
            LeftOut = -1,
            OnlyWhenMeant = true,
        },
    };

    /// <summary>The key a file names how many slices there are under.</summary>
    internal static WholeNumberParameter FoldsKey => Cut.CountKey;

    /// <summary>The key a file names which slice is validation under.</summary>
    internal static WholeNumberParameter FoldKey => Cut.IndexKey;

    /// <summary>How many slices the rows are dealt into.</summary>
    internal int Pieces => Folds;

    /// <summary>The fold a step's file names; nothing when it names none.</summary>
    /// <param name="step">The step as the file writes it.</param>
    /// <returns>The fold.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The file names a slice it does not have.</exception>
    /// <exception cref="FormatException">The file names one of the two keys without the other.</exception>
    internal static Fold? ReadFrom(JsonElement step) =>
        Cut.ReadFrom(step) is { } said ? new Fold(said.Count, said.Index) : null;
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
        Cut.Check(origins, index, nameof(origins));

        Origins = origins;
        Index = index;
    }

    /// <summary>How many places there are.</summary>
    public int Origins { get; }

    /// <summary>Which of them this is, counted from nought.</summary>
    public int Index { get; }

    /// <summary>The words and the keys of this cut: how many places there are, and which of them this is.</summary>
    private static CountedCut Cut { get; } = new()
    {
        Noun = "origin",
        Article = "An",
        Plural = "places",
        Which = "this is",
        TooFew = "A cut is moved to at least two places.",
        CountKey = new WholeNumberParameter(
            "origins",
            "How many places the cut between training and validation is moved to, at least two; the rows that are neither test nor predict are cut into one block more. Left out, the rows are cut once, as shares say.",
            3)
        {
            AtLeast = 2,
            LeftOut = 0,
            OnlyWhenMeant = true,
        },
        IndexKey = new WholeNumberParameter(
            "origin",
            "Which of the places this is, counted from nought; said with origins, and below them.",
            0)
        {
            AtLeast = 0,
            LeftOut = -1,
            OnlyWhenMeant = true,
        },
    };

    /// <summary>The key a file names how many places there are under.</summary>
    internal static WholeNumberParameter OriginsKey => Cut.CountKey;

    /// <summary>The key a file names which place this is under.</summary>
    internal static WholeNumberParameter OriginKey => Cut.IndexKey;

    /// <summary>How many blocks the rows are dealt into: one more than there are origins.</summary>
    internal int Pieces => Origins + 1;

    /// <summary>The place a step's file names; nothing when it names none.</summary>
    /// <param name="step">The step as the file writes it.</param>
    /// <returns>The place.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The file names a place it does not have.</exception>
    /// <exception cref="FormatException">The file names one of the two keys without the other.</exception>
    internal static Origin? ReadFrom(JsonElement step) =>
        Cut.ReadFrom(step) is { } said ? new Origin(said.Count, said.Index) : null;
}

/// <summary>
/// What every "the i-th of k" cut has in common: how many there are at least, which of them is meant, the two keys a file says
/// it with, and the rule that a file says both or neither. <see cref="Fold"/> and <see cref="Origin"/> differ only in the words.
/// </summary>
/// <remarks>The words are what an error says; the shape of the refusal is the same for every cut, and written once.</remarks>
internal sealed class CountedCut
{
    /// <summary>What one of them is called, in the singular: "fold".</summary>
    public required string Noun { get; init; }

    /// <summary>"A" or "An", before the noun at the head of a sentence.</summary>
    public required string Article { get; init; }

    /// <summary>What all of them are called: "slices".</summary>
    public required string Plural { get; init; }

    /// <summary>What it is to be the one that is meant: "is validation".</summary>
    public required string Which { get; init; }

    /// <summary>What is said when fewer than two are asked for.</summary>
    public required string TooFew { get; init; }

    /// <summary>The key a file names how many there are under.</summary>
    public required WholeNumberParameter CountKey { get; init; }

    /// <summary>The key a file names which of them is meant under.</summary>
    public required WholeNumberParameter IndexKey { get; init; }

    /// <summary>Refuses a count below two and an index that is not one of them.</summary>
    /// <param name="count">How many there are.</param>
    /// <param name="index">Which of them is meant, counted from nought.</param>
    /// <param name="countName">The name of the argument the count came in under.</param>
    /// <exception cref="ArgumentOutOfRangeException">There are fewer than two, or the index is not one of them.</exception>
    internal void Check(int count, int index, string countName)
    {
        if (count < 2)
        {
            throw new ArgumentOutOfRangeException(countName, count, TooFew);
        }

        if (index < 0 || index >= count)
        {
            throw new ArgumentOutOfRangeException(nameof(index), index, $"The {Noun} is one of the {count} {Plural}, counted from nought, and {index} is none of them.");
        }
    }

    /// <summary>What a step's file names; nothing when it names neither key.</summary>
    /// <param name="step">The step as the file writes it.</param>
    /// <returns>How many there are and which is meant, checked by the caller's constructor.</returns>
    /// <exception cref="FormatException">The file names one of the two keys without the other.</exception>
    internal CutAsWritten? ReadFrom(JsonElement step)
    {
        var counted = step.TryGetProperty(CountKey.Key, out _);
        var indexed = step.TryGetProperty(IndexKey.Key, out _);

        return !counted && !indexed ? null
            : counted && indexed ? new CutAsWritten(CountKey.Read(step), IndexKey.Read(step))
            : throw new FormatException($"{Article} {Noun} says how many {Plural} there are, '{CountKey.Key}', and which of them {Which}, '{IndexKey.Key}': this file says only '{(counted ? CountKey.Key : IndexKey.Key)}'.");
    }
}

/// <summary>What a file says of a cut: how many there are, and which of them is meant.</summary>
/// <param name="Count">How many there are.</param>
/// <param name="Index">Which of them is meant, counted from nought.</param>
internal readonly record struct CutAsWritten(int Count, int Index);
