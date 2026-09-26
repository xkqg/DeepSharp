// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>A stretch of a cell's text, from where it starts to where it ends, as its kernel counts lines and columns.</summary>
/// <param name="StartLine">The line it starts on.</param>
/// <param name="StartColumn">The column it starts at.</param>
/// <param name="EndLine">The line it ends on.</param>
/// <param name="EndColumn">The column it ends at.</param>
public readonly record struct HostedRange(int StartLine, int StartColumn, int EndLine, int EndColumn);
