// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;

namespace DeepSharp.Pipelines;

/// <summary>
/// The venue answered a request with a status that is not an answer: a refusal, a wait, a fault of its own.
/// </summary>
/// <param name="status">The status.</param>
/// <param name="wait">How long the venue asked to be left alone, when it asked.</param>
/// <param name="words">What the venue said of it.</param>
internal sealed class VenueStatusException(HttpStatusCode status, TimeSpan? wait, VenueWords words) : Exception($"The venue answered {(int)status}.")
{
    /// <summary>The status.</summary>
    public HttpStatusCode Status { get; } = status;

    /// <summary>How long the venue asked to be left alone, when it asked.</summary>
    public TimeSpan? Wait { get; } = wait;

    /// <summary>What the venue said of it.</summary>
    public VenueWords Words { get; } = words;

    /// <summary>
    /// Whether the status says stop: a firewall that can mean a rate limit was broken, a ban, a region the venue does not
    /// answer from. A walk that marched on after any of them is how one rejected call becomes a storm.
    /// </summary>
    public bool Stops => Status is HttpStatusCode.Forbidden or (HttpStatusCode)418 or (HttpStatusCode)451;

    /// <summary>Whether the same request may be asked again after a wait: the venue says it is busy or it has a fault of its own.</summary>
    public bool Retryable => Status is HttpStatusCode.TooManyRequests or HttpStatusCode.RequestTimeout || (int)Status is >= 500 and <= 599;
}
