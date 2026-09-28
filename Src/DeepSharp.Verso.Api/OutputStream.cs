// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>
/// Which of a kernel's text streams an output came on. Where text came from is not whether anything went wrong: programs
/// that work write progress, logs and warnings to standard error.
/// </summary>
public enum OutputStream
{
    /// <summary>The kernel's standard output.</summary>
    StandardOutput,

    /// <summary>The kernel's standard error, shown apart from ordinary output but not as a failure.</summary>
    StandardError,
}
