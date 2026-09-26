// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>A toolbar button, as a page draws it.</summary>
/// <param name="Id">What pressing it names.</param>
/// <param name="Label">What it says.</param>
/// <param name="Icon">Its icon, as SVG; nothing for a button without one.</param>
/// <param name="IconOnly">Whether it shows the icon alone.</param>
/// <param name="IsPrimary">Whether it is the notebook's main button.</param>
/// <param name="Confirmation">What to ask before pressing it; nothing when it asks nothing.</param>
/// <param name="Place">Where it belongs.</param>
/// <param name="Order">Its place among the buttons of that place.</param>
/// <param name="IsEnabled">Whether it can be pressed now.</param>
public readonly record struct HostedToolbarAction(
    string Id, string Label, string? Icon, bool IconOnly, bool IsPrimary, string? Confirmation, ToolbarPlace Place, int Order, bool IsEnabled);
