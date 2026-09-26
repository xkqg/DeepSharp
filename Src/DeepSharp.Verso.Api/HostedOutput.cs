// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>One thing a cell shows.</summary>
/// <param name="ContentType">What the content is: HTML, plain text, an image.</param>
/// <param name="Content">The content itself.</param>
/// <param name="IsError">Whether it reports a fault.</param>
public readonly record struct HostedOutput(string ContentType, string Content, bool IsError);
