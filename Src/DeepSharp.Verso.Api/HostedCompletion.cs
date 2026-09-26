// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>What a cell's kernel offers to write next where the cursor stands, as Verso's editors list it.</summary>
/// <param name="Text">What the list shows.</param>
/// <param name="Insert">What is written when it is taken.</param>
/// <param name="Kind">What it is — a method, a keyword, a verb, a column — as the kernel names it.</param>
/// <param name="Description">What it means, when the kernel says.</param>
/// <param name="SortText">Where it stands in the list, when the kernel says; otherwise by its text.</param>
public readonly record struct HostedCompletion(string Text, string Insert, string Kind, string? Description, string? SortText);
