// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>What a word in a cell's text means, as its kernel says when the cursor rests on it.</summary>
/// <param name="Content">What it means.</param>
/// <param name="ContentType">What the content is: plain text or Markdown.</param>
/// <param name="Range">The stretch of text it is about, when the kernel says; the word alone otherwise.</param>
public readonly record struct HostedHover(string Content, string ContentType, HostedRange? Range);
