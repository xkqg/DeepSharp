// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Verso.Abstractions;

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
/// <param name="Output">Its box for whether it is an answer of the output.</param>
/// <param name="Role">What it is to the output, as the row says it: an answer, read by an answer's way back, or nothing.</param>
internal readonly record struct ListedRow(
    string Column, string Values, DrawnSelect Kind, string ShownKind, string Mark, DrawnBox Included, DrawnBox Output, string Role);

/// <summary>A list as drawn: the block it was drawn on, and its page.</summary>
/// <param name="On">The block.</param>
/// <param name="Html">The page.</param>
internal readonly record struct DrawnList(CellModel On, string Html);

/// <summary>What a test reads off the list of a source's columns.</summary>
internal static partial class ListHtmlExtensions
{
    /// <summary>Every row the list draws, in the order it draws them.</summary>
    /// <param name="list">The list's page.</param>
    /// <returns>The rows.</returns>
    public static IReadOnlyList<ListedRow> Rows(this string list) =>
        [.. Row().Matches(list).Select(row =>
        {
            var boxes = row.Groups[2].Value.Boxes();
            var included = boxes.Single(box => box.Action.StartsWith("deepsharp.list.include ", StringComparison.Ordinal));

            return new ListedRow(
                WebUtility.HtmlDecode(row.Groups[1].Value),
                Cell(row.Groups[2].Value, "values"),
                Select(row.Groups[2].Value),
                KindOf(included.Action),
                Cell(row.Groups[2].Value, "mark"),
                included,
                boxes.Single(box => box.Action.StartsWith("deepsharp.list.output ", StringComparison.Ordinal)),
                Regex.Match(row.Groups[2].Value, "<span class=\"deepsharp-role\">(.*?)</span>").Groups[1].Value);
        })];

    /// <summary>The row of one column.</summary>
    /// <param name="list">The list's page.</param>
    /// <param name="column">The column.</param>
    /// <returns>The row.</returns>
    public static ListedRow Row(this string list, string column) => list.Rows().Single(row => row.Column == column);

    /// <summary>The select that picks the kind of output the list's boxes make.</summary>
    /// <param name="list">The list's page.</param>
    /// <returns>The select.</returns>
    public static DrawnSelect TypeSelect(this string list) => Select(Section().Match(list).Groups[1].Value);

    /// <summary>The labels of the type select's options, as a person reads them.</summary>
    /// <param name="list">The list's page.</param>
    /// <returns>Each option's value and its label.</returns>
    public static IReadOnlyDictionary<string, string> TypeLabels(this string list) =>
        OptionLabel().Matches(Section().Match(list).Groups[1].Value)
            .ToDictionary(option => WebUtility.HtmlDecode(option.Groups[1].Value), option => WebUtility.HtmlDecode(option.Groups[2].Value));

    /// <summary>The box that takes the output away.</summary>
    /// <param name="list">The list's page.</param>
    /// <returns>The box.</returns>
    public static DrawnBox RemovalBox(this string list) => Section().Match(list).Groups[1].Value.Boxes().Single();

    /// <summary>The selects that set the output's parameters, by the parameter each sets.</summary>
    /// <param name="list">The list's page.</param>
    /// <returns>Each select, under its parameter's key; none when the list draws no parameters.</returns>
    public static IReadOnlyDictionary<string, DrawnSelect> ParameterSelects(this string list) =>
        SelectTag().Matches(Parameters().Match(list).Groups[1].Value)
            .Select(match => Select(match.Value))
            .ToDictionary(select => JsonNode.Parse(select.Action[(select.Action.IndexOf(' ', StringComparison.Ordinal) + 1)..])!["key"]!.GetValue<string>());

    /// <summary>What the list says of the output's parameters it cannot set, as a person reads it.</summary>
    /// <param name="list">The list's page.</param>
    /// <returns>One line per such parameter.</returns>
    public static IReadOnlyList<string> ParametersSetElsewhere(this string list) =>
        [.. ParameterNote().Matches(Parameters().Match(list).Groups[1].Value).Select(note => WebUtility.HtmlDecode(note.Groups[1].Value))];

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

    [GeneratedRegex("<option value=\"([^\"]*)\"( selected)?[^>]*>", RegexOptions.Singleline)]
    private static partial Regex Option();

    [GeneratedRegex("<option value=\"([^\"]*)\"[^>]*>(.*?)</option>", RegexOptions.Singleline)]
    private static partial Regex OptionLabel();

    [GeneratedRegex("<div class=\"deepsharp-output\">(.*?)</div>", RegexOptions.Singleline)]
    private static partial Regex Section();

    [GeneratedRegex("<div class=\"deepsharp-parameters\">(.*?)</div>", RegexOptions.Singleline)]
    private static partial Regex Parameters();

    [GeneratedRegex("<span class=\"deepsharp-parameter\">(.*?)</span>", RegexOptions.Singleline)]
    private static partial Regex ParameterNote();
}
