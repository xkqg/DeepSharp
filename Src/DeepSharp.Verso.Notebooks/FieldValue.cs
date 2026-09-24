// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Verso.Notebooks;

/// <summary>What a person set in one field of the form, whatever shape the front end handed it over in.</summary>
/// <param name="Text">The value as text: what was typed or picked, or a number in its invariant spelling.</param>
/// <param name="Flag">The value as true or false, when it reads as one.</param>
/// <param name="Items">The values picked, when several were.</param>
/// <remarks>
/// Verso hands a field's value back as the control holds it — text, true or false, a list, or the JSON a remote
/// front end sent — and a choice of several joined with commas. Every shape is read here, once, so each kind of
/// field reads a value the same way whichever front end the notebook is open in.
/// </remarks>
internal readonly record struct FieldValue(string? Text, bool? Flag, IReadOnlyList<string>? Items)
{
    /// <summary>Reads a value as the front end handed it over.</summary>
    /// <param name="value">The value.</param>
    /// <returns>The value in every shape it has.</returns>
    public static FieldValue Of(object? value) => value switch
    {
        null => new(null, null, null),
        string text => new(text, FlagOf(text), null),
        bool flag => new(flag ? "true" : "false", flag, null),
        JsonElement element => Of(element),
        IFormattable number => new(number.ToString(null, CultureInfo.InvariantCulture), null, null),
        IEnumerable items => new(null, null, [.. items.Cast<object?>().Select(item => Of(item).Text ?? string.Empty)]),
        _ => new(value.ToString(), null, null),
    };

    /// <summary>The values picked: the list handed over, or the text split where Verso joined them with commas.</summary>
    public IReadOnlyList<string> Picked =>
        Items ?? (Text ?? string.Empty).Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);

    private static FieldValue Of(JsonElement element) => element.ValueKind switch
    {
        JsonValueKind.String => Of(element.GetString()),
        JsonValueKind.True => Of(true),
        JsonValueKind.False => Of(false),
        JsonValueKind.Array => new(null, null, [.. element.EnumerateArray().Select(item => Of(item).Text ?? string.Empty)]),
        JsonValueKind.Null or JsonValueKind.Undefined => new(null, null, null),
        _ => new(element.GetRawText(), null, null),
    };

    private static bool? FlagOf(string text) => bool.TryParse(text, out var flag) ? flag : null;
}
