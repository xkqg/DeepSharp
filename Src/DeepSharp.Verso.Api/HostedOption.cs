// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>One choice a field offers.</summary>
/// <param name="Value">What choosing it sends.</param>
/// <param name="Label">What it says.</param>
public readonly record struct HostedOption(string Value, string Label);
