// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Notebooks.Verso;

/// <summary>
/// The verbs a notebook understands: the library's own, and those of the packages this one brings along.
/// </summary>
/// <remarks>
/// A fresh catalog every time, never a shared one: the parts of a notebook are separate instances that Verso makes
/// on its own, and a catalog one of them taught a verb must not change what another reads. The vocabulary is what
/// this package ships, frozen when it is packed — a block naming a verb it does not carry is refused, and says so.
/// </remarks>
internal static class NotebookVerbs
{
    /// <summary>A new catalog of every verb a notebook can hold.</summary>
    /// <returns>The catalog.</returns>
    public static StepCatalog Catalog() => StepCatalog.BuiltIn().WithIndicators();
}
