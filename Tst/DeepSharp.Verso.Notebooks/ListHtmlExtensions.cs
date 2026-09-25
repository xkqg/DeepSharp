// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using System.Text.RegularExpressions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>One row of the list of the source's columns, as a person reads it.</summary>
/// <param name="Column">The column.</param>
/// <param name="Values">Its first values, as the list writes them.</param>
/// <param name="Kind">The kind it shows.</param>
/// <param name="Mark">What the row says of itself: new in the source, not in the source, or nothing.</param>
/// <param name="Included">Its box for whether it is in.</param>
internal readonly record struct ListedRow(string Column, string Values, string Kind, string Mark, DrawnBox Included);

/// <summary>What a test reads off the list of a source's columns.</summary>
internal static partial class ListHtmlExtensions
{
    /// <summary>Every row the list draws, in the order it draws them.</summary>
    /// <param name="list">The list's page.</param>
    /// <returns>The rows.</returns>
    public static IReadOnlyList<ListedRow> Rows(this string list) =>
        [.. Row().Matches(list).Select(row => new ListedRow(
            WebUtility.HtmlDecode(row.Groups[1].Value),
            Cell(row.Groups[2].Value, "values"),
            Cell(row.Groups[2].Value, "kind"),
            Cell(row.Groups[2].Value, "mark"),
            row.Groups[2].Value.Boxes().Single()))];

    /// <summary>The row of one column.</summary>
    /// <param name="list">The list's page.</param>
    /// <param name="column">The column.</param>
    /// <returns>The row.</returns>
    public static ListedRow Row(this string list, string column) => list.Rows().Single(row => row.Column == column);

    private static string Cell(string row, string name) =>
        WebUtility.HtmlDecode(Regex.Match(row, $"<td class=\"deepsharp-{name}\">(.*?)</td>", RegexOptions.Singleline).Groups[1].Value);

    [GeneratedRegex("<tr data-column=\"([^\"]*)\">(.*?)</tr>", RegexOptions.Singleline)]
    private static partial Regex Row();
}
