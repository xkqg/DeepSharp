// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>One thing a cell shows, as Verso's engine holds it.</summary>
/// <param name="ContentType">What the content is: HTML, plain text, an image, JSON, a table, a widget, progress.</param>
/// <param name="Content">The content itself.</param>
/// <param name="IsError">Whether it reports a failure — one that stopped the cell; text on standard error is none by itself.</param>
/// <param name="ErrorName">What failed, for a failure: the exception's name, or the kind of error the kernel named.</param>
/// <param name="ErrorStack">Where it failed, for a failure the kernel knows that of.</param>
/// <param name="Stream">
/// Which of the kernel's text streams it came on, when it is stream text the kernel said so of; nothing otherwise.
/// </param>
public readonly record struct HostedOutput(string ContentType, string Content, bool IsError, string? ErrorName, string? ErrorStack, OutputStream? Stream);
