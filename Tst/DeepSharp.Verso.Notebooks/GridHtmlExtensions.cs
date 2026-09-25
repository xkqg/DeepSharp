// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>A box a grid's header draws, as Verso's router reads it.</summary>
/// <param name="Action">Its <c>data-action</c>, as the router hands it on: the attribute's text, decoded.</param>
/// <param name="Ticked">Whether it is drawn ticked.</param>
/// <param name="Enabled">Whether it can be clicked.</param>
/// <param name="CarriesAPayload">Whether it carries a <c>data-payload</c>, which the router would send in place of its state.</param>
internal readonly record struct DrawnBox(string Action, bool Ticked, bool Enabled, bool CarriesAPayload);

/// <summary>What a test reads off a grid's page.</summary>
internal static partial class GridHtmlExtensions
{
    /// <summary>Whether the grid has a column of that name: its header, followed by the column's boxes.</summary>
    /// <param name="grid">The grid's page.</param>
    /// <param name="column">The column's name, as the page writes it.</param>
    /// <returns><see langword="true"/> when the column is a header of the grid.</returns>
    public static bool Heads(this string grid, string column) =>
        grid.Contains($"<th>{column} <label>", StringComparison.Ordinal);

    /// <summary>The box a grid's header draws for a column and a gesture, found the way the router reads it.</summary>
    /// <param name="grid">The grid's page.</param>
    /// <param name="gesture">The gesture the box makes.</param>
    /// <param name="column">The column the box is about.</param>
    /// <returns>The box, or nothing when the grid draws none.</returns>
    public static DrawnBox? Box(this string grid, string gesture, string column) =>
        grid.Boxes().Cast<DrawnBox?>().FirstOrDefault(box => box!.Value.Action.StartsWith($"{gesture} ", StringComparison.Ordinal)
                                                              && ColumnOf(box.Value.Action) == column);

    /// <summary>Every box a grid's page draws, in the order it draws them.</summary>
    /// <param name="grid">The grid's page.</param>
    /// <returns>The boxes.</returns>
    public static IReadOnlyList<DrawnBox> Boxes(this string grid) =>
        [.. Inputs().Matches(grid).Select(input => input.Value).Select(input => new DrawnBox(
            WebUtility.HtmlDecode(Action().Match(input).Groups[1].Value),
            input.Contains(" checked", StringComparison.Ordinal),
            !input.Contains(" disabled", StringComparison.Ordinal),
            input.Contains("data-payload=", StringComparison.Ordinal)))];

    // The column a box's action names: the JSON after the gesture's name.
    private static string? ColumnOf(string action) =>
        action.IndexOf(' ', StringComparison.Ordinal) is var space and > 0
        && JsonNode.Parse(action[(space + 1)..]) is JsonObject carried
        && carried["column"] is JsonValue column
            ? column.GetValue<string>()
            : null;

    [GeneratedRegex("<input\\b[^>]*>")]
    private static partial Regex Inputs();

    [GeneratedRegex("data-action=\"([^\"]*)\"")]
    private static partial Regex Action();
}
