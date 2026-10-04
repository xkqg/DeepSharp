// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// The column that says where the gaps in another column were: named in one place, written in one place.
/// </summary>
/// <remarks>
/// Three verbs write it — filling a gap with a value learned from the training rows, settling one with a value no row
/// decided, and encoding a column whose cell was empty. Each of the three destroys the difference between "absent" and
/// "the value happened to be that", permanently, so the difference is written down before it goes; and each wrote the
/// name for itself until the third made it a sentence said three times.
/// </remarks>
internal static class Marking
{
    /// <summary>What the column marking where the value was missing is named after, beside the column's own name.</summary>
    internal const string WasMissing = "was_missing";

    extension(string column)
    {
        /// <summary>What the column saying where this column's gaps were is called.</summary>
        internal string Marked => $"{column}_{WasMissing}";
    }

    extension(Table table)
    {
        /// <summary>Puts the column that says where a column's gaps were onto the table, one value a row.</summary>
        /// <param name="column">The column whose gaps are marked; it is read, never changed.</param>
        /// <remarks>Written before the gaps are settled or filled, since afterwards there is nothing left to see.</remarks>
        internal void Marks(string column)
        {
            var values = table[column];

            table.Put(new Column<double>(
                values.Name.Marked, ColumnKind.Number,
                Enumerable.Range(0, table.RowCount).Select(row => (double?)(values.IsMissing(row) ? 1 : 0))));
        }
    }
}
