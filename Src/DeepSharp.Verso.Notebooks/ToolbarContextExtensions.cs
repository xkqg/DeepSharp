// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// What a toolbar button that writes blocks does to the notebook it is pressed in.
/// </summary>
/// <remarks>
/// Verso hands a button the notebook's cells and the operations that change them. Making a cell holding text is the
/// operation, then the text, and every button that writes blocks does the two the same way, so they are said once.
/// </remarks>
internal static class ToolbarContextExtensions
{
    extension(IToolbarActionContext context)
    {
        /// <summary>Makes a cell where it is asked for, holding what it is asked to hold.</summary>
        /// <param name="at">Where the cell goes, counting from nought.</param>
        /// <param name="wanted">The cell's type, its language and its text.</param>
        /// <returns>The id of the cell made.</returns>
        public async Task<Guid> InsertedAsync(int at, CellModel wanted)
        {
            var id = Guid.Parse(await context.Notebook.InsertCellAsync(at, wanted.Type, wanted.Language));

            context.NotebookCells[context.IndexOf(id)].Source = wanted.Source;

            return id;
        }

        /// <summary>Where a cell stands among the notebook's cells.</summary>
        /// <param name="id">The cell's id.</param>
        /// <returns>Its place, counting from nought.</returns>
        /// <exception cref="InvalidOperationException">The cell is no longer in the notebook.</exception>
        public int IndexOf(Guid id)
        {
            for (var at = 0; at < context.NotebookCells.Count; at++)
            {
                if (context.NotebookCells[at].Id == id)
                {
                    return at;
                }
            }

            throw new InvalidOperationException("A cell the notebook was asked to change is no longer in it.");
        }
    }
}
