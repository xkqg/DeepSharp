// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Net;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>What the venue said in a refusal, in its own words: a code and a message, either of which may be missing.</summary>
/// <param name="Code">The venue's own code, such as -1121.</param>
/// <param name="Message">The venue's own message, such as <c>Invalid symbol.</c></param>
internal readonly record struct VenueWords(int? Code, string? Message);

/// <summary>One answer of the venue, as it came, before anyone has decided what it means.</summary>
/// <param name="Status">The status.</param>
/// <param name="Body">The bytes of the body.</param>
/// <param name="Used">What the venue says this address has spent this minute, when it says.</param>
/// <param name="RetryAfter">How long the venue asks to be left alone, when it asks.</param>
/// <param name="Date">The venue's <c>Date</c> header as it was written, when it came with one.</param>
/// <param name="Uuid">The venue's <c>x-mbx-uuid</c>, when it came with one.</param>
internal sealed record VenueReply(HttpStatusCode Status, byte[] Body, int? Used, TimeSpan? RetryAfter, string? Date, string? Uuid)
{
    /// <summary>The venue's own clock as its <c>Date</c> header says it, to the second.</summary>
    public DateTimeOffset? DateValue =>
        DateTimeOffset.TryParse(Date, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var parsed) ? parsed : null;

    /// <summary>What the venue said in the body, when it said it as it says a refusal.</summary>
    /// <returns>Its code and its message; nothing in either when the body is not that.</returns>
    public VenueWords Words()
    {
        try
        {
            using var document = JsonDocument.Parse(Body);
            var root = document.RootElement;

            return new VenueWords(root.WholeOf("code"), root.TextOf("msg"));
        }
        catch (JsonException)
        {
            return default;
        }
    }
}
