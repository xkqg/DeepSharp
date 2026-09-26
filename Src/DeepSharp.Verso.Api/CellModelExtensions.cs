// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Verso.Api;

/// <summary>A cell of Verso's engine as a value an application holds, with nothing of the engine inside it.</summary>
internal static class CellModelExtensions
{
    /// <summary>The cell as it stands now.</summary>
    /// <param name="cell">The engine's cell.</param>
    /// <returns>Its value.</returns>
    public static HostedCell Hosted(this CellModel cell) =>
        new(cell.Id, cell.Type, cell.Language, cell.Source, [.. cell.Outputs.Select(output => new HostedOutput(output.MimeType, output.Content, output.IsError))]);
}
