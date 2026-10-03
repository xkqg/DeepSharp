// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// How a parameter's form names its fields and spells its values: each name built and read back in one place.
/// </summary>
/// <remarks>
/// A parameter with one value is one field under its own key. The others are named from the parameter and what the field
/// is about — one column of a set, one place of a list of roles, the kind a schema gives one column, whether that column
/// may be absent, how its moments are written, which of its values stands for a gap, the number a way of filling carries —
/// so a field the form draws is always one it reads back. The parameter is what each name is asked of, rather than its key
/// handed over beside what the field is about: two names of the same kind, side by side, are a pair a caller can hand over
/// the wrong way round, and a receiver cannot be.
/// </remarks>
internal static class FormVocabulary
{
    /// <summary>The pick that leaves a source's column out of the schema.</summary>
    public const string NotTaken = "not taken";

    private const string Kinds = "kind";

    private const string Absence = "optional";

    private const string Written = "format";

    private const string Gap = "missing";

    private const string TheNumber = "value";

    extension(StepParameter parameter)
    {
        /// <summary>The switch for one column of a set of columns.</summary>
        /// <param name="column">The column.</param>
        /// <returns>The field's name.</returns>
        public string Member(string column) => $"{parameter.Key}/{column}";

        /// <summary>Whether a field is the switch for one column of a set, and which.</summary>
        /// <param name="field">The field's name.</param>
        /// <param name="column">The column, when it is.</param>
        /// <returns><see langword="true"/> when it is.</returns>
        public bool IsMember(string field, out string column) => After(field, $"{parameter.Key}/", out column);

        /// <summary>The pick of one place of a list whose places are roles.</summary>
        /// <param name="place">The place, counting from nought.</param>
        /// <returns>The field's name.</returns>
        public string Place(int place) => string.Create(CultureInfo.InvariantCulture, $"{parameter.Key}/{place}");

        /// <summary>Whether a field is the pick of one place of a list, and which.</summary>
        /// <param name="field">The field's name.</param>
        /// <param name="place">The place, when it is one.</param>
        /// <returns><see langword="true"/> when it is.</returns>
        public bool IsPlace(string field, out int place)
        {
            place = -1;

            return After(field, $"{parameter.Key}/", out var rest)
                && int.TryParse(rest, NumberStyles.None, CultureInfo.InvariantCulture, out place);
        }

        /// <summary>One setting of the thing at one place of a list.</summary>
        /// <param name="place">The place, counting from nought.</param>
        /// <param name="setting">The setting's own key.</param>
        /// <returns>The field's name.</returns>
        public string Setting(int place, string setting) =>
            string.Create(CultureInfo.InvariantCulture, $"{parameter.Key}/{place}/{setting}");

        /// <summary>Whether a field is one setting of the thing at one place of a list, and which setting of which place.</summary>
        /// <param name="field">The field's name.</param>
        /// <param name="place">The place, when it is one.</param>
        /// <param name="setting">The setting's own key, when it is one.</param>
        /// <returns><see langword="true"/> when it is.</returns>
        public bool IsSetting(string field, out int place, out string setting)
        {
            place = -1;
            setting = string.Empty;

            if (!After(field, $"{parameter.Key}/", out var rest) || rest.IndexOf('/', StringComparison.Ordinal) is var at && at <= 0)
            {
                return false;
            }

            setting = rest[(at + 1)..];

            return setting.Length > 0 && int.TryParse(rest[..at], NumberStyles.None, CultureInfo.InvariantCulture, out place);
        }

        /// <summary>The kind a schema gives one column.</summary>
        /// <param name="column">The column.</param>
        /// <returns>The field's name.</returns>
        public string Kind(string column) => $"{parameter.Key}/{Kinds}/{column}";

        /// <summary>Whether a field is the kind a schema gives one column, and which column.</summary>
        /// <param name="field">The field's name.</param>
        /// <param name="column">The column, when it is.</param>
        /// <returns><see langword="true"/> when it is.</returns>
        public bool IsKind(string field, out string column) => After(field, $"{parameter.Key}/{Kinds}/", out column);

        /// <summary>Whether one column may be absent from a source.</summary>
        /// <param name="column">The column.</param>
        /// <returns>The field's name.</returns>
        public string Absent(string column) => $"{parameter.Key}/{Absence}/{column}";

        /// <summary>Whether a field says one column may be absent, and which column.</summary>
        /// <param name="field">The field's name.</param>
        /// <param name="column">The column, when it is.</param>
        /// <returns><see langword="true"/> when it is.</returns>
        public bool IsAbsent(string field, out string column) => After(field, $"{parameter.Key}/{Absence}/", out column);

        /// <summary>How one column's moments are written.</summary>
        /// <param name="column">The column.</param>
        /// <returns>The field's name.</returns>
        public string Format(string column) => $"{parameter.Key}/{Written}/{column}";

        /// <summary>Whether a field says how one column's moments are written, and which column.</summary>
        /// <param name="field">The field's name.</param>
        /// <param name="column">The column, when it is.</param>
        /// <returns><see langword="true"/> when it is.</returns>
        public bool IsFormat(string field, out string column) => After(field, $"{parameter.Key}/{Written}/", out column);

        /// <summary>Which of one column's values stands for a gap.</summary>
        /// <param name="column">The column.</param>
        /// <returns>The field's name.</returns>
        public string Missing(string column) => $"{parameter.Key}/{Gap}/{column}";

        /// <summary>Whether a field says which of one column's values stands for a gap, and which column.</summary>
        /// <param name="field">The field's name.</param>
        /// <param name="column">The column, when it is.</param>
        /// <returns><see langword="true"/> when it is.</returns>
        public bool IsMissing(string field, out string column) => After(field, $"{parameter.Key}/{Gap}/", out column);

        /// <summary>The number a way of filling carries.</summary>
        /// <returns>The field's name.</returns>
        public string Number() => $"{parameter.Key}/{TheNumber}";
    }

    extension(IEnumerable<string> names)
    {
        /// <summary>A list of names as the form writes it in a text field: as JSON, so a name may hold anything.</summary>
        /// <returns>The list, written as JSON.</returns>
        public string AsListText() => $"[{string.Join(", ", names.Select(name => JsonSerializer.Serialize(name)))}]";
    }

    private static bool After(string field, string prefix, out string rest)
    {
        rest = field.StartsWith(prefix, StringComparison.Ordinal) ? field[prefix.Length..] : string.Empty;

        return rest.Length > 0;
    }
}
