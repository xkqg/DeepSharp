// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;

namespace DeepSharp.Pipelines;

/// <summary>The one key a step that ranks rows takes to say it ranks by the declared id instead of the whole record.</summary>
/// <remarks>
/// Shared by every step <see cref="SplitPlacement.Ranked"/> serves, so a file says "by the declared id" once, the same
/// way, wherever it is read. Left out, as every file before this key existed is, a step ranks by the whole record exactly
/// as it always has — declaring an id changes nothing on its own, until a step is asked to rank by it.
/// </remarks>
internal static class Ranking
{
    /// <summary>Whether a step ranks by the declared id rather than the whole record; left out, it does not.</summary>
    internal static readonly TrueOrFalseParameter ByIdKey = new(
        "byId", "Whether to rank by the declared id rather than the whole record. Left out, by the whole record.", example: false)
    {
        LeftOut = false,
    };

    extension<TStep>(TStep step)
        where TStep : IPipelineStep
    {
        /// <summary>This step, ranking by the declared id where it is a step that ranks rows and the steps above declare an id.</summary>
        /// <param name="above">The steps written before it.</param>
        /// <returns>The step ranking by the id, or the step as it is.</returns>
        internal TStep RankedWhereAnIdStands(IReadOnlyList<IPipelineStep> above) =>
            above.IdColumn() is not null && step is IRanksRows ranks ? (TStep)ranks.RankedById() : step;
    }
}

/// <summary>A step that ranks the rows by a digest of what each says, and may rank them by the declared id instead.</summary>
/// <remarks>
/// The one thing the builder and the rule about an id need to know of such a step, said by the step itself: there used to
/// be one switch over the step types for setting it and another for reading it, and the first had forgotten the shuffle.
/// </remarks>
internal interface IRanksRows : IPipelineStep
{
    /// <summary>Whether this step ranks by the declared id rather than the whole record.</summary>
    bool ById { get; }

    /// <summary>This step, ranking by the declared id.</summary>
    /// <returns>The same step with <see cref="ById"/> set.</returns>
    IRanksRows RankedById();
}

/// <summary>A row's rank when a declared id is in force: by the id first, by the whole record only to break a tie.</summary>
/// <param name="Primary">The seeded rank of the id.</param>
/// <param name="Tie">The seeded rank of the whole record, which a repeated id leaves to break.</param>
/// <remarks>
/// An id is a group, named by the person who declared it — "a flock number, a customer code" — and never
/// promised to be one row: two rows of one id rank together by it, and only between two different ids does
/// the whole record decide which comes first.
/// </remarks>
internal readonly record struct CompositeRank(RowKey Primary, RowKey Tie) : IComparable<CompositeRank>
{
    /// <inheritdoc />
    public int CompareTo(CompositeRank other)
    {
        var primary = Primary.CompareTo(other.Primary);

        return primary != 0 ? primary : Tie.CompareTo(other.Tie);
    }
}

/// <summary>
/// Putting a run of parts back onto the rows they belong to.
/// </summary>
internal static class SplitPlacement
{
    extension(Table table)
    {
        /// <summary>The column the schema says names each row, when it still stands; nothing when none does, or it was dropped.</summary>
        internal IColumn? IdColumnInForce => table.Id is { } id && table.Has(id) ? table[id] : null;

        /// <summary>The id column, when a step asks to rank by it; nothing when it does not ask.</summary>
        /// <param name="byId">Whether the step ranks by the declared id.</param>
        /// <returns>The id column, when <paramref name="byId"/> asks for it; nothing otherwise.</returns>
        /// <exception cref="InvalidOperationException"><paramref name="byId"/> asks for it, and no id stands here.</exception>
        internal IColumn? IdColumnFor(bool byId) =>
            !byId ? null
            : table.IdColumnInForce is { } column ? column
            : throw new InvalidOperationException(
                "This step ranks by the declared id, and no id stands here: a step above it took the id column away. "
                + "Keep the id above this step, or do not ask to rank by it.");

        /// <summary>The text a row holds in the id column, refusing a gap.</summary>
        /// <param name="column">The id column, as read off this table.</param>
        /// <param name="row">The row.</param>
        /// <returns>The text.</returns>
        /// <exception cref="InvalidOperationException">The row has no value there.</exception>
        /// <remarks>The one guard an id is read through, so a split, a shuffle and the handover refuse a gap the same way.</remarks>
        internal string IdAt(IColumn column, int row) =>
            column.TextAt(row) ?? throw new InvalidOperationException(
                string.Create(
                    CultureInfo.InvariantCulture,
                    $"Row {table.Identities[row].ReadAt + 1} has no '{column.Name}': the id names every row it is carried with, so a gap there is refused. Fill it in the rows, or drop the rows without one."));

        /// <summary>Rows in the order of a digest of the seed and what each row says — or, when a step ranks by the declared id, of the seed and the id, with the whole record only to break a tie.</summary>
        /// <param name="rows">The rows to rank.</param>
        /// <param name="seed">The number that makes the ranking one of many possible ones, and the same one each time.</param>
        /// <param name="byId">Whether to rank by the declared id rather than the whole record.</param>
        /// <returns>The rows, ranked; rows that say the same thing, or share the id a step ranks by, stand side by side.</returns>
        /// <remarks>
        /// SHA-256 rather than a generator of random numbers: a generator shuffles places, so the same rows in
        /// another order were dealt differently, while a digest of what a row says ranks it the same wherever it
        /// stands, on every machine. Ranking by the id uses the same digest, over the id alone, so adding or
        /// dropping a feature column never moves a row to another part.
        /// </remarks>
        internal int[] Ranked(IEnumerable<int> rows, int seed, bool byId = false)
        {
            var ranked = rows.ToArray();

            if (table.IdColumnFor(byId) is { } column)
            {
                var ranks = ranked
                    .Select(row => new CompositeRank(RowKey.Of([column.Name], [table.IdAt(column, row)]).Ranked(seed), table.Identities[row].Key.Ranked(seed)))
                    .ToArray();

                Array.Sort(ranks, ranked);
            }
            else
            {
                var ranks = ranked.Select(row => table.Identities[row].Key.Ranked(seed)).ToArray();

                Array.Sort(ranks, ranked);
            }

            return ranked;
        }

        /// <summary>Whether two rows belong together: the same declared id, when a step ranks by it; the same record otherwise.</summary>
        /// <param name="one">A row.</param>
        /// <param name="other">Another row.</param>
        /// <param name="byId">Whether the step ranks by the declared id.</param>
        internal bool Together(int one, int other, bool byId = false) =>
            table.IdColumnFor(byId) is { } column
                ? table.IdAt(column, one) == table.IdAt(column, other)
                : table.Identities[one].Key == table.Identities[other].Key;
    }

    extension(Part[] parts)
    {
        /// <summary>Gives every row the part of the row before it, when the two belong together.</summary>
        /// <param name="ordered">The rows in the order the parts were handed out.</param>
        /// <param name="together">Whether a row belongs with the one before it: the same row twice, the same moment.</param>
        internal void KeptTogether(int[] ordered, Func<int, int, bool> together)
        {
            for (var at = 1; at < ordered.Length; at++)
            {
                if (together(ordered[at - 1], ordered[at]))
                {
                    parts[ordered[at]] = parts[ordered[at - 1]];
                }
            }
        }

        /// <summary>Deals the rows that learn into slices in the order given: the fold's slice is validation and the others are training.</summary>
        /// <param name="ordered">The rows that learn, in the order the seed ranks them.</param>
        /// <param name="slice">Which slice is validation.</param>
        /// <param name="together">Whether a row is a copy of the one before it, which stays in its slice.</param>
        internal void Sliced(int[] ordered, Fold slice, Func<int, int, bool> together)
        {
            var slices = new int[ordered.Length];

            for (var at = 0; at < ordered.Length; at++)
            {
                slices[at] = at > 0 && together(ordered[at - 1], ordered[at]) ? slices[at - 1] : slice.SliceOf(at, ordered.Length);
                parts[ordered[at]] = slices[at] == slice.Index ? Part.Validation : Part.Train;
            }
        }

        /// <summary>Keeps the last moments of every part apart: before each line, and at the end.</summary>
        /// <param name="ordered">The rows in the order the parts were handed out.</param>
        /// <param name="gap">How many moments of each part are kept apart.</param>
        /// <param name="together">Whether a row belongs with the one before it: the same moment.</param>
        /// <remarks>Moments rather than rows, so a moment is kept apart whole, as a split keeps it whole.</remarks>
        internal void Gapped(int[] ordered, int gap, Func<int, int, bool> together)
        {
            var end = ordered.Length;

            while (end > 0)
            {
                var part = parts[ordered[end - 1]];
                var start = end - 1;

                while (start > 0 && parts[ordered[start - 1]] == part)
                {
                    start--;
                }

                var moments = 0;

                for (var at = end - 1; at >= start && moments < gap; at--)
                {
                    parts[ordered[at]] = Part.Gap;

                    // The row that begins its moment completes one moment kept apart.
                    if (at == start || !together(ordered[at - 1], ordered[at]))
                    {
                        moments++;
                    }
                }

                end = start;
            }
        }
    }

    extension(Part[] inOrder)
    {
        /// <summary>Lays a run of parts out over the rows in a given order.</summary>
        /// <param name="rows">The rows, in the order the parts should be handed out.</param>
        /// <returns>One part per row, in row order.</returns>
        internal Part[] Placed(int[] rows)
        {
            var parts = new Part[inOrder.Length];

            for (var at = 0; at < rows.Length; at++)
            {
                parts[rows[at]] = inOrder[at];
            }

            return parts;
        }
    }
}
