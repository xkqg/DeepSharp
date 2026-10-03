// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using Verso.Abstractions;

namespace DeepSharp.Verso.Api;

/// <summary>A cell of Verso's engine as a value an application holds, with nothing of the engine inside it.</summary>
internal static class CellModelExtensions
{
    extension(CellModel cell)
    {
        /// <summary>The cell as it stands now.</summary>
        /// <param name="shownBefore">What it showed at the notebook's last version; kept when what it shows now is caught half written.</param>
        /// <returns>Its value.</returns>
        public HostedCell Hosted(IReadOnlyList<HostedOutput> shownBefore) =>
            new(cell.Id, cell.Type, cell.Language, cell.Source, cell.Outputs.Shown() ?? shownBefore, cell.Metadata.Written(), cell.ExecutionCount, cell.LastStatus, cell.LastElapsed);
    }

    extension(List<CellOutput> outputs)
    {
        /// <summary>What a cell shows, copied whole, or not at all.</summary>
        /// <returns>The copy; nothing when the list was caught half written.</returns>
        /// <remarks>
        /// Copied as Verso's engine copies a list a run may be writing (Scaffold.ReadOutputs): the copy fails when the list
        /// grows its storage while it is read, and a copy can hold a place the list has counted and not yet filled. Neither is
        /// a state the cell was ever in, so neither is shown (measured: both happen within a millisecond against a writer).
        /// </remarks>
        public IReadOnlyList<HostedOutput>? Shown()
        {
            CellOutput[] copy;

            try
            {
                copy = outputs.ToArray();
            }
            catch (ArgumentException)
            {
                return null;
            }

            return Array.IndexOf(copy, null) >= 0 ? null : [.. copy.Select(Hosted)];
        }
    }

    // An output as a view is told it: the stream it came on in the Api's own words, since the engine's are not handed out.
    private static HostedOutput Hosted(CellOutput output) => new(
        output.MimeType,
        output.Content,
        output.IsError,
        output.ErrorName,
        output.ErrorStackTrace,
        output.Channel switch
        {
            OutputChannel.Stdout => OutputStream.StandardOutput,
            OutputChannel.Stderr => OutputStream.StandardError,
            _ => null,
        });

    extension(Dictionary<string, object> metadata)
    {
        /// <summary>What is kept with a cell, each value written as JSON.</summary>
        /// <returns>Each key with its value's JSON.</returns>
        /// <remarks>
        /// Written the way Verso's engine writes a value to compare it (CellAligner), so a value read from the file and the same
        /// value set since look the same; a value JSON cannot write is written as its text.
        /// </remarks>
        public IReadOnlyDictionary<string, string> Written() =>
            metadata.ToDictionary(pair => pair.Key, pair => Json(pair.Value), StringComparer.Ordinal);
    }

    private static string Json(object value)
    {
        try
        {
            return JsonSerializer.Serialize(value);
        }
        catch (NotSupportedException)
        {
            return $"{value}";
        }
    }
}
