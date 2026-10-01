// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// Rows read from a JSON file holding an array of records, one object a row.
/// </summary>
/// <remarks>
/// JSON is text, as a comma-separated file is, so every value is handed over as the file writes it: words as the words
/// they are, a number in its own spelling — <c>22.0</c> stays <c>22.0</c>, since a row is known by what its file says —
/// and true and false as the file spells them. The columns are the keys the records use, in the order they first
/// appear; a key a record leaves out is a gap on its row, and so is a null. A value that is itself an object or a list is
/// not one value to a cell, and it is refused by name with its record rather than flattened by a guess about what it was
/// meant to be. The whole file is read when it is opened, so its rows are the ones it held at that moment.
/// </remarks>
internal sealed class JsonRowSource : IRowSource
{
    // What each kind of value is, in words, for a refusal that says what stood where a record or a cell should.
    private static readonly Dictionary<JsonValueKind, string> Kinds = new()
    {
        [JsonValueKind.Object] = "an object",
        [JsonValueKind.Array] = "a list",
        [JsonValueKind.String] = "words",
        [JsonValueKind.Number] = "a number",
        [JsonValueKind.True] = "true or false",
        [JsonValueKind.False] = "true or false",
        [JsonValueKind.Null] = "nothing",
    };

    private readonly List<IReadOnlyList<string?>> _rows = [];

    /// <summary>Reads a JSON file's bytes.</summary>
    /// <param name="bytes">Every byte of the file.</param>
    /// <param name="file">What a refusal names the file as.</param>
    /// <exception cref="FormatException">
    /// The bytes are not JSON, not an array of records, or a record names a key twice or holds a value that is not one
    /// value to a cell.
    /// </exception>
    public JsonRowSource(byte[] bytes, string file)
    {
        // Read as text, as the file is, so a byte-order mark is taken as what it is rather than as the start of the first value.
        var text = bytes.AsText();
        JsonDocument document;

        try
        {
            document = JsonDocument.Parse(text);
        }
        catch (JsonException notJson)
        {
            throw new FormatException($"{file} is not JSON: {notJson.Message}", notJson);
        }

        using (document)
        {
            var records = Records(file, document.RootElement);
            var names = Names(file, records);

            ColumnNames = names;

            foreach (var record in records)
            {
                var cells = new string?[names.Count];

                foreach (var property in record.EnumerateObject())
                {
                    cells[names.IndexOf(property.Name)] = Cell(property.Value);
                }

                _rows.Add(cells);
            }
        }
    }

    /// <inheritdoc />
    public IReadOnlyList<string> ColumnNames { get; }

    /// <inheritdoc />
    public IEnumerable<IReadOnlyList<string?>> Rows => _rows;

    // The records: the array the file holds, every one of them an object.
    private static JsonElement[] Records(string file, JsonElement root)
    {
        if (root.ValueKind != JsonValueKind.Array)
        {
            throw new FormatException($"{file} holds {Kinds[root.ValueKind]} where a table's records stand: an array of objects, one a row.");
        }

        JsonElement[] records = [.. root.EnumerateArray()];

        if (records.Length == 0)
        {
            throw new FormatException($"{file} holds no records, so its columns have no names.");
        }

        for (var at = 0; at < records.Length; at++)
        {
            if (records[at].ValueKind != JsonValueKind.Object)
            {
                throw new FormatException(
                    $"Record {at + 1} of {file} is {Kinds[records[at].ValueKind]}, where each record is an object of its cells.");
            }
        }

        return records;
    }

    // The columns: every key the records use, in the order each first appears, and every value one value to a cell.
    private static List<string> Names(string file, JsonElement[] records)
    {
        var names = new List<string>();

        for (var at = 0; at < records.Length; at++)
        {
            var keys = new HashSet<string>(StringComparer.Ordinal);

            foreach (var property in records[at].EnumerateObject())
            {
                // Readers of JSON disagree about which of two values under one key wins, so the file says which or neither.
                if (!keys.Add(property.Name))
                {
                    throw new FormatException($"Record {at + 1} of {file} names '{property.Name}' twice.");
                }

                if (property.Value.ValueKind is JsonValueKind.Object or JsonValueKind.Array)
                {
                    throw new FormatException(
                        $"Record {at + 1} of {file} holds {Kinds[property.Value.ValueKind]} under '{property.Name}', and a cell holds one "
                        + "value: words, a number, true, false or null. Write it as keys of one value each, or leave it out of the file.");
                }

                if (!names.Contains(property.Name, StringComparer.Ordinal))
                {
                    names.Add(property.Name);
                }
            }
        }

        return names;
    }

    // A value as the file writes it: words as the words they are, anything else as its own text; null is a gap.
    private static string? Cell(JsonElement value) => value.ValueKind switch
    {
        JsonValueKind.String => value.GetString(),
        JsonValueKind.Null => null,
        _ => value.GetRawText(),
    };
}
