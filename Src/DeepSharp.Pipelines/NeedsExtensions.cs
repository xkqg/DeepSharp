// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Pipelines;

/// <summary>What each need a learner states says of the features it is handed: the one place a need is read.</summary>
/// <remarks>
/// A step that some learners do without, the handover and a learner of its own all ask these rather than comparing a need
/// with another, so a need added later is read the same way everywhere: <see cref="ScaleExtensions"/> does the same for where
/// a scale lands its rows.
/// </remarks>
public static class NeedsExtensions
{
    /// <summary>Whether a learner with this need takes numbers of any size, so a step that only scales a feature can be left out for it.</summary>
    /// <param name="needs">The need.</param>
    /// <returns><see langword="true"/> for <see cref="Needs.NoScale"/> and <see cref="Needs.Categories"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException">No need is named by this value.</exception>
    public static bool DoesWithoutScaling(this Needs needs) => needs.Named() is Needs.NoScale or Needs.Categories;

    /// <summary>Whether a learner with this need takes each category as its place in the list the training rows held.</summary>
    /// <param name="needs">The need.</param>
    /// <returns><see langword="true"/> for <see cref="Needs.Categories"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException">No need is named by this value.</exception>
    public static bool TakesCategories(this Needs needs) => needs.Named() == Needs.Categories;

    /// <summary>Whether a learner with this need takes every feature on one scale, between minus one and one.</summary>
    /// <param name="needs">The need.</param>
    /// <returns><see langword="true"/> for <see cref="Needs.OneScale"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException">No need is named by this value.</exception>
    public static bool NeedsOneScale(this Needs needs) => needs.Named() == Needs.OneScale;

    /// <summary>The need, when a need is named by it.</summary>
    /// <param name="needs">The value stated.</param>
    /// <returns>The same value.</returns>
    /// <exception cref="ArgumentOutOfRangeException">No need is named by this value: a learner that stated it would be handed something nobody decided.</exception>
    internal static Needs Named(this Needs needs) =>
        Enum.IsDefined(needs)
            ? needs
            : throw new ArgumentOutOfRangeException(
                nameof(needs),
                needs,
                string.Create(CultureInfo.InvariantCulture, $"No need is numbered {(int)needs}: a learner needs {string.Join(", ", Enum.GetNames<Needs>())}."));
}
