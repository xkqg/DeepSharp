// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>A layout the engine can show a notebook in, as Verso's View panel lists it.</summary>
/// <param name="Id">The layout's id, which a switch names.</param>
/// <param name="Name">The name a person knows it by.</param>
/// <param name="Allows">What it lets a person do to the cells; one that lets nobody write them is read only.</param>
public readonly record struct HostedLayoutChoice(string Id, string Name, LayoutAllows Allows);
