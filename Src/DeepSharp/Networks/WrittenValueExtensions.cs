// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Networks;

/// <summary>
/// What a network file holds at a place, read as the value it should be — or nothing, when what is written there is not
/// one, so a reader names the fault rather than throwing on it.
/// </summary>
/// <remarks>
/// A <see cref="JsonElement"/> asked for a number it does not hold throws, and one that is not an object throws when asked
/// for a key; a file is text anybody may have edited, so every read of it goes through these, and a number read is always
/// a finite one.
/// </remarks>
internal static class WrittenValueExtensions
{
    extension(JsonElement element)
    {
        /// <summary>The value under a key of the object written here; nothing when this is no object, or has no such key.</summary>
        public JsonElement? Member(string key) =>
            element.ValueKind == JsonValueKind.Object && element.TryGetProperty(key, out var value) ? value : null;

        /// <summary>The whole number written here; nothing when it is not one, or too large for one.</summary>
        public int? AsWhole() => element.ValueKind == JsonValueKind.Number && element.TryGetInt32(out var whole) ? whole : null;

        /// <summary>The whole number written here, as large as a seed may be; nothing when it is not one.</summary>
        public long? AsLong() => element.ValueKind == JsonValueKind.Number && element.TryGetInt64(out var whole) ? whole : null;

        /// <summary>The finite number written here; nothing when it is not one.</summary>
        public double? AsNumber() =>
            element.ValueKind == JsonValueKind.Number && element.TryGetDouble(out var number) && double.IsFinite(number) ? number : null;

        /// <summary>The finite number written here, as a float holds it; nothing when it is not one, or past what a float holds.</summary>
        public float? AsFloat() =>
            element.ValueKind == JsonValueKind.Number && element.TryGetSingle(out var number) && float.IsFinite(number) ? number : null;

        /// <summary>The true or false written here; nothing when it is neither.</summary>
        public bool? AsTruth() => element.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => null,
        };
    }
}
