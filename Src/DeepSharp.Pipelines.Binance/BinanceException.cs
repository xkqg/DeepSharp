// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;

namespace DeepSharp.Pipelines;

/// <summary>
/// Binance stopped a landing, or answered it in its own words that the request could not be met.
/// </summary>
/// <remarks>
/// <para>
/// Thrown for what the venue said and no more: a ban, a wall, a region it does not answer from, a refusal of the request as
/// it was made, a wait it asked for that is longer than a landing waits, and a page it could not be made to answer with after
/// every try a unit of work gets. A unit that failed is one failure however many tries it took.
/// </para>
/// <para>
/// A landing that stops on any of these leaves nothing behind, and a venue that has said stop is not asked again: a walk that
/// marched on after a refusal is how one rejected call becomes a storm, and the address is shared.
/// </para>
/// </remarks>
public sealed class BinanceException : Exception
{
    /// <summary>A refusal in Binance's words, or the refusal of a request that never reached it.</summary>
    /// <param name="message">What happened, with what Binance said.</param>
    /// <param name="status">The status of the answer; nothing when no answer came.</param>
    /// <param name="code">Binance's own code for what was wrong, when its answer carried one.</param>
    /// <param name="inner">What went wrong underneath, when something did.</param>
    public BinanceException(string message, HttpStatusCode? status = null, int? code = null, Exception? inner = null)
        : base(message, inner)
    {
        Status = status;
        Code = code;
    }

    /// <summary>The status of the answer that ended the landing; nothing when no answer came at all.</summary>
    public HttpStatusCode? Status { get; }

    /// <summary>Binance's own code, such as -1121 for a symbol it does not know; nothing when its answer carried none.</summary>
    public int? Code { get; }
}
