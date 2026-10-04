// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.ML;
using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// The verbs a notebook understands: the library's own, and those of the packages this one brings along — the indicators,
/// the readers of a Parquet file, an Excel workbook and a JSON file, and the network a pipeline is declared for.
/// </summary>
/// <remarks>
/// A fresh catalog every time, never a shared one: the parts of a notebook are separate instances that Verso makes
/// on its own, and a catalog one of them taught a verb must not change what another reads. The vocabulary is what
/// this package ships, frozen when it is packed and the same in every host — a block naming a verb it does not carry is
/// refused, and says so. No host adds to it: an extension Verso installs is loaded apart from every other, so another
/// package's steps would not be the notebook's steps even under the same names.
/// </remarks>
internal static class NotebookVerbs
{
    /// <summary>A new catalog of every verb a notebook can hold.</summary>
    /// <returns>The catalog.</returns>
    public static StepCatalog Catalog() => StepCatalog.BuiltIn().WithIndicators().WithParquet().WithExcel().WithJson().WithNetworks().WithML();
}
