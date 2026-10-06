// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// What a member of a JSON object says, read as far as it goes: a member that is not there, or is not the kind it should be,
/// or an element that is not an object at all, says nothing — and never throws, since the venue's own words are read by a
/// program that has to tell a refusal apart from an answer it cannot make sense of.
/// </summary>
internal static class VenueJsonExtensions
{
    extension(JsonElement element)
    {
        /// <summary>The text a member holds.</summary>
        /// <param name="key">The member's name.</param>
        /// <returns>The text; nothing when the element is not an object, the member is missing or it holds something else.</returns>
        public string? TextOf(string key) =>
            element.ValueKind == JsonValueKind.Object && element.TryGetProperty(key, out var found) && found.ValueKind == JsonValueKind.String ? found.GetString() : null;

        /// <summary>The whole number a member holds.</summary>
        /// <param name="key">The member's name.</param>
        /// <returns>The number; nothing when the element is not an object, the member is missing, holds something else or holds a number that is not a whole one of this size.</returns>
        public int? WholeOf(string key) =>
            element.ValueKind == JsonValueKind.Object && element.TryGetProperty(key, out var found) && found.ValueKind == JsonValueKind.Number && found.TryGetInt32(out var number) ? number : null;
    }
}
