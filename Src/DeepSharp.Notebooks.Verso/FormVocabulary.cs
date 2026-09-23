// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Notebooks.Verso;

/// <summary>
/// How the form names its fields and spells its values: each name built and read back in one place.
/// </summary>
/// <remarks>
/// A parameter with one value is one field under its own key. The others are named from their key and what the field
/// is about — one column of a set, one place of a list of roles, the kind a schema gives one column, whether that
/// column may be absent, the number a way of filling carries — so a field the form draws is always one it reads back.
/// </remarks>
internal static class FormVocabulary
{
    /// <summary>The pick that leaves a source's column out of the schema.</summary>
    public const string NotTaken = "not taken";

    private const string Kinds = "kind";

    private const string Absence = "optional";

    private const string Number = "value";

    /// <summary>The switch for one column of a set of columns.</summary>
    /// <param name="key">The set's key.</param>
    /// <param name="column">The column.</param>
    /// <returns>The field's name.</returns>
    public static string Member(string key, string column) => $"{key}/{column}";

    /// <summary>Whether a field is the switch for one column of a set, and which.</summary>
    /// <param name="field">The field's name.</param>
    /// <param name="key">The set's key.</param>
    /// <param name="column">The column, when it is.</param>
    /// <returns><see langword="true"/> when it is.</returns>
    public static bool IsMember(string field, string key, out string column) => After(field, $"{key}/", out column);

    /// <summary>The pick of one place of a list whose places are roles.</summary>
    /// <param name="key">The list's key.</param>
    /// <param name="place">The place, counting from nought.</param>
    /// <returns>The field's name.</returns>
    public static string Place(string key, int place) => string.Create(CultureInfo.InvariantCulture, $"{key}/{place}");

    /// <summary>Whether a field is the pick of one place of a list, and which.</summary>
    /// <param name="field">The field's name.</param>
    /// <param name="key">The list's key.</param>
    /// <param name="place">The place, when it is one.</param>
    /// <returns><see langword="true"/> when it is.</returns>
    public static bool IsPlace(string field, string key, out int place)
    {
        place = -1;

        return After(field, $"{key}/", out var rest) && int.TryParse(rest, NumberStyles.None, CultureInfo.InvariantCulture, out place);
    }

    /// <summary>The pick of the kind a schema gives one column, or not taken.</summary>
    /// <param name="key">The declarations' key.</param>
    /// <param name="column">The column.</param>
    /// <returns>The field's name.</returns>
    public static string Kind(string key, string column) => $"{key}/{Kinds}/{column}";

    /// <summary>Whether a field is the pick of a column's kind, and which column.</summary>
    /// <param name="field">The field's name.</param>
    /// <param name="key">The declarations' key.</param>
    /// <param name="column">The column, when it is.</param>
    /// <returns><see langword="true"/> when it is.</returns>
    public static bool IsKind(string field, string key, out string column) => After(field, $"{key}/{Kinds}/", out column);

    /// <summary>The switch for whether one declared column may be absent.</summary>
    /// <param name="key">The declarations' key.</param>
    /// <param name="column">The column.</param>
    /// <returns>The field's name.</returns>
    public static string Absent(string key, string column) => $"{key}/{Absence}/{column}";

    /// <summary>Whether a field is the switch for a column's absence, and which column.</summary>
    /// <param name="field">The field's name.</param>
    /// <param name="key">The declarations' key.</param>
    /// <param name="column">The column, when it is.</param>
    /// <returns><see langword="true"/> when it is.</returns>
    public static bool IsAbsent(string field, string key, out string column) => After(field, $"{key}/{Absence}/", out column);

    /// <summary>The field of the number a way of filling carries.</summary>
    /// <param name="key">The strategy's key.</param>
    /// <returns>The field's name.</returns>
    public static string StrategyValue(string key) => $"{key}/{Number}";

    /// <summary>A list of names as the form writes it in a text field: as JSON, so a name may hold anything.</summary>
    /// <param name="names">The names.</param>
    /// <returns>The list, written as JSON.</returns>
    public static string ListText(IEnumerable<string> names) =>
        $"[{string.Join(", ", names.Select(name => JsonSerializer.Serialize(name)))}]";

    private static bool After(string field, string prefix, out string rest)
    {
        rest = field.StartsWith(prefix, StringComparison.Ordinal) ? field[prefix.Length..] : string.Empty;

        return rest.Length > 0;
    }
}
