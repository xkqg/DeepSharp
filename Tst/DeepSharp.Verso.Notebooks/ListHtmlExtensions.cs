// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>A select a row of the list draws, as Verso's router reads it.</summary>
/// <param name="Action">Its <c>data-action</c>, as the router hands it on: the attribute's text, decoded.</param>
/// <param name="Value">The value it is drawn at: the option drawn selected.</param>
/// <param name="Options">The value of every option it offers, in the order it draws them.</param>
/// <param name="Enabled">Whether it can be used.</param>
internal readonly record struct DrawnSelect(string Action, string Value, IReadOnlyList<string> Options, bool Enabled);

/// <summary>One row of the list of the source's columns, as a person reads it.</summary>
/// <param name="Column">The column.</param>
/// <param name="Values">Its first values, as the list writes them.</param>
/// <param name="Kind">Its kind select.</param>
/// <param name="ShownKind">The kind its box takes the column in with.</param>
/// <param name="Mark">What the row says of itself: new in the source, not in the source, or nothing.</param>
/// <param name="Included">Its box for whether it is in.</param>
internal readonly record struct ListedRow(string Column, string Values, DrawnSelect Kind, string ShownKind, string Mark, DrawnBox Included);

/// <summary>What a test reads off the list of a source's columns.</summary>
internal static partial class ListHtmlExtensions
{
    /// <summary>Every row the list draws, in the order it draws them.</summary>
    /// <param name="list">The list's page.</param>
    /// <returns>The rows.</returns>
    public static IReadOnlyList<ListedRow> Rows(this string list) =>
        [.. Row().Matches(list).Select(row =>
        {
            var included = row.Groups[2].Value.Boxes().Single();

            return new ListedRow(
                WebUtility.HtmlDecode(row.Groups[1].Value),
                Cell(row.Groups[2].Value, "values"),
                Select(row.Groups[2].Value),
                KindOf(included.Action),
                Cell(row.Groups[2].Value, "mark"),
                included);
        })];

    /// <summary>The row of one column.</summary>
    /// <param name="list">The list's page.</param>
    /// <param name="column">The column.</param>
    /// <returns>The row.</returns>
    public static ListedRow Row(this string list, string column) => list.Rows().Single(row => row.Column == column);

    private static string Cell(string row, string name) =>
        WebUtility.HtmlDecode(Regex.Match(row, $"<td class=\"deepsharp-{name}\">(.*?)</td>", RegexOptions.Singleline).Groups[1].Value);

    private static DrawnSelect Select(string row)
    {
        var select = SelectTag().Match(row);
        var options = Option().Matches(select.Groups[2].Value).ToArray();
        var selected = options.FirstOrDefault(option => option.Groups[2].Success) ?? options.First();

        return new DrawnSelect(
            WebUtility.HtmlDecode(Regex.Match(select.Groups[1].Value, "data-action=\"([^\"]*)\"").Groups[1].Value),
            WebUtility.HtmlDecode(selected.Groups[1].Value),
            [.. options.Select(option => WebUtility.HtmlDecode(option.Groups[1].Value))],
            !select.Groups[1].Value.Contains(" disabled", StringComparison.Ordinal));
    }

    // The kind a box's action carries, which a tick takes its column in with.
    private static string KindOf(string action) =>
        JsonNode.Parse(action[(action.IndexOf(' ', StringComparison.Ordinal) + 1)..])!["kind"]!.GetValue<string>();

    [GeneratedRegex("<tr data-column=\"([^\"]*)\">(.*?)</tr>", RegexOptions.Singleline)]
    private static partial Regex Row();

    [GeneratedRegex("<select\\b([^>]*)>(.*?)</select>", RegexOptions.Singleline)]
    private static partial Regex SelectTag();

    [GeneratedRegex("<option value=\"([^\"]*)\"( selected)?>", RegexOptions.Singleline)]
    private static partial Regex Option();
}
