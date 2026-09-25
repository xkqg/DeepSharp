// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// The boxes a column is drawn with, wherever it is drawn: the grid's header and the list of the source's columns ask the
/// same rule, so a box says the same thing in both.
/// </summary>
internal static class ColumnBoxExtensions
{
    /// <summary>The box for whether a column is in.</summary>
    /// <param name="column">How the column stands, and what the rules offer it.</param>
    /// <returns>Ticked when it takes part, is kept or is made; clickable when the rules allow what the click asks for.</returns>
    public static HeaderBox IncludedBox(this ColumnChoice column) =>
        Box(column.Standing is ColumnStanding.Taking or ColumnStanding.Kept or ColumnStanding.Made, column.Offers, ColumnOffers.Exclude, ColumnOffers.Include);

    /// <summary>The box for whether a column is a category.</summary>
    /// <param name="column">How the column stands, and what the rules offer it.</param>
    /// <returns>Ticked when it is one; clickable when the rules allow what the click asks for.</returns>
    /// <remarks>A category can be unticked only when it says which kind it was.</remarks>
    public static HeaderBox CategoryBox(this ColumnChoice column) =>
        Box(column.Kind == ColumnKind.Category, column.Offers, ColumnOffers.BackToWas, ColumnOffers.MakeCategory);

    // A box, ticked or not, that can be clicked when the rules offer what the click asks for: unticking it when it is
    // ticked, ticking it when it is not.
    private static HeaderBox Box(bool ticked, ColumnOffers offers, ColumnOffers untick, ColumnOffers tick) =>
        new(ticked, offers.HasFlag(ticked ? untick : tick));
}
