// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// The rows a study measures on and the rows it chooses on, held to the first that were prepared: every deal of a kind
/// measures on the same test rows, and on a deal every candidate is chosen on the same validation rows.
/// </summary>
/// <remarks>
/// Scores are chosen between only where the rows they were measured on are the same: a candidate chosen on one deal's
/// validation rows and measured on rows another deal of its kind learned from would be chosen by luck. The promise is made
/// within a kind and not across kinds: a split in time and a split at random measure on other rows by design, and a row one
/// kind measures on may be learned from by another. Which rows a part holds is read by their keys alone, never their features,
/// so the guard is the same whatever trains behind the pipeline.
/// </remarks>
/// <param name="kinds">The kinds the study's deals are of, kind after kind.</param>
internal sealed class MeasuredRows(IReadOnlyList<DealKind> kinds)
{
    private readonly Dictionary<int, HashSet<RowKey>> _validation = [];
    private readonly Dictionary<string, HashSet<RowKey>> _test = new(StringComparer.Ordinal);

    /// <summary>Holds what a pipeline prepared to the rows the first of its kind measured on, and to those its deal was first chosen on.</summary>
    /// <param name="rows">What the pipeline prepared.</param>
    /// <param name="place">The deal it was prepared for, and the trial where it was prepared for one.</param>
    /// <exception cref="InvalidOperationException">It measures on other test rows than its kind, or sets other rows aside as validation on its deal.</exception>
    public void Hold(PreparedData rows, DealPlace place)
    {
        var kind = kinds.KindOf(place.Deal).Kind.Name;
        var test = rows.KeysOf(Part.Test).ToHashSet();
        var validation = rows.KeysOf(Part.Validation).ToHashSet();

        if (!_test.TryAdd(kind, test) && !_test[kind].SetEquals(test))
        {
            throw new InvalidOperationException(
                $"{place.Pipeline} measures on other test rows than the first of its kind, '{kind}', prepared did: a candidate would be chosen on rows another deal of that kind learned from. Give the split a test seed — SplitAtRandom(train, validation, seed, testSeed) — so the rows to be measured on are the same on every deal of a kind, and move it by nothing a candidate chooses; deals that measure on other rows by design, such as a split at random and a split in time, are told apart as kinds of their own.");
        }

        if (!_validation.TryAdd(place.Deal, validation) && !_validation[place.Deal].SetEquals(validation))
        {
            throw new InvalidOperationException(
                $"{place.Pipeline} sets other rows aside as validation than the first trial's does: its score would be compared with scores measured on other rows. Deal the validation rows by nothing a candidate chooses.");
        }
    }
}
