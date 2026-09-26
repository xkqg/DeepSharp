// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>A click on a control a cell drew, as the page it was drawn on hands it over.</summary>
/// <param name="Cell">The cell the control stands in.</param>
/// <param name="ExtensionId">The part that answers it: the control's <c>data-extension-id</c>.</param>
/// <param name="Action">What it asks for: the control's <c>data-action</c>.</param>
/// <param name="Payload">
/// What it carries: the control's <c>data-payload</c>, or, for a control that carries none, the state it is in —
/// <c>true</c> or <c>false</c> for a box, the value it is at for a select — as Verso's own editor sends it.
/// </param>
public readonly record struct HostedGesture(Guid Cell, string ExtensionId, string Action, string Payload);
