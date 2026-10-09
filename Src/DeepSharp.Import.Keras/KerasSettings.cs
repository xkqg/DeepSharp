// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Networks;

namespace DeepSharp.Import.Keras;

/// <summary>
/// The settings a Keras file writes for one thing — a layer, the loss, the model — read in the file's own words, each one
/// that cannot be read, or says what nothing here does, noted as a fault at the place the thing stands.
/// </summary>
/// <remarks>
/// A fault is noted and the reading goes on, so every fault of a file is named at once; what could not be read stands at
/// the value it would have had, which nothing is built from while a fault is noted.
/// </remarks>
internal sealed class KerasSettings
{
    // The settings of something that writes none.
    private static readonly JsonElement None = KerasSaved.Json("{}", "{}");

    // How a setting of each kind is read from what the file writes: the value, or nothing when it is not one. Read by hand
    // rather than by the serializer, which an application that trims or compiles ahead of time has no reflection for.
    private static readonly Dictionary<Type, Func<JsonElement, object?>> Readers = new()
    {
        [typeof(int)] = value => IsWhole(value) ? value.GetInt32() : null,
        [typeof(double)] = value => value.ValueKind == JsonValueKind.Number ? value.GetDouble() : null,
        [typeof(bool)] = value => value.ValueKind is JsonValueKind.True or JsonValueKind.False ? value.GetBoolean() : null,
        [typeof(string)] = value => value.ValueKind == JsonValueKind.String ? value.GetString() : null,
        [typeof(int[])] = Wholes,
        [typeof(int?[])] = Lengths,
        [typeof(JsonElement)] = value => value,
    };

    private readonly JsonElement _settings;
    private readonly string _place;
    private readonly List<string> _faults;

    /// <summary>Settings to read, where they stand, and the faults of the file they are noted among.</summary>
    /// <param name="settings">The settings as the file writes them: an object; anything else writes none.</param>
    /// <param name="place">Where they stand, as a fault names it: <c>config.json, layer 'dense' (Dense)</c>.</param>
    /// <param name="faults">The faults of the file.</param>
    public KerasSettings(JsonElement settings, string place, List<string> faults)
    {
        _settings = settings.ValueKind == JsonValueKind.Object ? settings : None;
        _place = place;
        _faults = faults;
    }

    /// <summary>A setting Keras always writes; its fault noted when it writes none, or one that cannot be read.</summary>
    public T Setting<T>(string key) => Said(key) is { } value ? Read(key, value, default(T)!) : Missing<T>(key);

    /// <summary>A setting Keras may leave out, which then means what it means when left out; its fault noted when it cannot be read.</summary>
    public T Setting<T>(string key, T otherwise) => Said(key) is { } value ? Read(key, value, otherwise) : otherwise;

    /// <summary>Whether a setting says something: it is written, and written as more than nothing.</summary>
    public bool Says(string key) => Said(key) is not null;

    /// <summary>A setting as the file writes it, a list's items apart; nothing when it says nothing.</summary>
    public string? Written(string key) => Said(key)?.Written();

    /// <summary>
    /// A setting Keras writes as one length for each axis a window walks, outermost first — a pair for an image — noted when it
    /// is not that many lengths.
    /// </summary>
    /// <param name="key">The setting.</param>
    /// <param name="count">How many axes the window walks: one, two or three.</param>
    /// <returns>The lengths; one of each when the setting cannot be read, which nothing is built from while a fault is noted.</returns>
    public int[] Axes(string key, int count)
    {
        var lengths = Setting<int[]?>(key);

        if (lengths is not null && lengths.Length != count)
        {
            var wanted = count switch { 1 => "one length", 2 => "a pair", _ => "three lengths" };

            Refuse($"its '{key}' is written as {Written(key)!.Quoted()}, and Keras writes {wanted} there.");
        }

        return lengths is not null && lengths.Length == count ? lengths : [.. Enumerable.Repeat(1, count)];
    }

    /// <summary>
    /// A setting Keras writes as one length for each axis a window walks, or as one whole number that stands for every axis
    /// alike — a pooling's size or stride — noted when it is neither.
    /// </summary>
    /// <param name="key">The setting.</param>
    /// <param name="count">How many axes the window walks: one, two or three.</param>
    /// <returns>The lengths, one for each axis.</returns>
    public int[] Alike(string key, int count) =>
        Setting(key, default(JsonElement)) is { ValueKind: JsonValueKind.Number } number && IsWhole(number)
            ? [.. Enumerable.Repeat(number.GetInt32(), count)]
            : Axes(key, count);

    /// <summary>Notes a fault at the place these settings stand.</summary>
    public void Refuse(string fault) => _faults.Add($"{_place.Quoted()}: {fault}");

    private JsonElement? Said(string key) =>
        _settings.TryGetProperty(key, out var value) && value.ValueKind != JsonValueKind.Null ? value : null;

    // A whole number, as Keras writes a count or a length.
    private static bool IsWhole(JsonElement value) => value.ValueKind == JsonValueKind.Number && value.TryGetInt32(out _);

    // A list of whole numbers; nothing when it is no list, or an item is no whole number.
    private static object? Wholes(JsonElement value) =>
        value.ValueKind == JsonValueKind.Array && value.EnumerateArray().All(IsWhole) ? value.EnumerateArray().Select(item => item.GetInt32()).ToArray() : null;

    // A list of lengths, each a whole number or null, which leaves it open; nothing when it is no list, or an item is neither.
    private static object? Lengths(JsonElement value) =>
        value.ValueKind == JsonValueKind.Array && value.EnumerateArray().All(item => item.ValueKind == JsonValueKind.Null || IsWhole(item))
            ? value.EnumerateArray().Select(item => item.ValueKind == JsonValueKind.Null ? (int?)null : item.GetInt32()).ToArray()
            : null;

    private T Read<T>(string key, JsonElement value, T otherwise)
    {
        if (Readers[typeof(T)](value) is T read)
        {
            return read;
        }

        Refuse($"its '{key}' is written as {value.Written().Quoted()}, which is not what Keras writes there.");

        return otherwise;
    }

    private T Missing<T>(string key)
    {
        Refuse($"it says no '{key}'.");

        return default!;
    }
}

/// <summary>How a Keras file writes a value, as a fault names it.</summary>
internal static class WrittenJsonExtensions
{
    extension(JsonElement value)
    {
        /// <summary>The value as the file writes it, a list's items apart whatever space the file wrote between them: <c>[null, 14]</c>.</summary>
        internal string Written() =>
            value.ValueKind == JsonValueKind.Array ? $"[{string.Join(", ", value.EnumerateArray().Select(item => item.Written()))}]" : value.GetRawText();
    }
}
