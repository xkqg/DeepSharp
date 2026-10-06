// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// The venue answered, and what it answered with is not a page of candles that can be taken.
/// </summary>
/// <remarks>
/// A unit of work that fails like this is tried again, as one that never came is, because a page garbled in transit is the
/// same page asked for again; what is refused at the end is the page that was never a page, after every try.
/// </remarks>
/// <param name="why">What is wrong with it.</param>
internal sealed class PageFormatException(string why) : Exception(why);
