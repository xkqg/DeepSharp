// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Api;

/// <summary>
/// A cell something was asked of no longer stands in the notebook: a change that came before rewrote it or took it
/// away, so nothing the request meant holds any more, and it is not carried out.
/// </summary>
public sealed class CellGoneException : KeyNotFoundException
{
    /// <summary>Says which cell is gone, and from which version of the notebook on.</summary>
    /// <param name="cell">The cell's id.</param>
    /// <param name="version">The version of the notebook the cell is not in.</param>
    public CellGoneException(Guid cell, long version)
        : base($"The cell {cell} no longer stands in the notebook as of its version {version}: a change before this one rewrote it or took it away.")
    {
        Cell = cell;
        Version = version;
    }

    /// <summary>The cell that is gone.</summary>
    public Guid Cell { get; }

    /// <summary>The version of the notebook the cell is not in; a view that has it knows the cell is gone.</summary>
    public long Version { get; }
}
