// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// The rows a study measures on and the rows it chooses on, held to the first that were prepared: every deal measures on the
/// same test rows, and on a deal every candidate is chosen on the same validation rows.
/// </summary>
/// <remarks>
/// Scores are chosen between only where the rows they were measured on are the same: a candidate chosen on one deal's
/// validation rows and measured on rows another deal learned from would be chosen by luck. Which rows a part holds is read by
/// their keys alone, never their features, so the guard is the same whatever trains behind the pipeline.
/// </remarks>
internal sealed class MeasuredRows
{
    private readonly Dictionary<int, HashSet<RowKey>> _validation = [];
    private HashSet<RowKey>? _test;

    /// <summary>Holds what a pipeline prepared to the rows the first one measured on, and to those its deal was first chosen on.</summary>
    /// <param name="rows">What the pipeline prepared.</param>
    /// <param name="place">The deal it was prepared for, and the trial where it was prepared for one.</param>
    /// <exception cref="InvalidOperationException">It measures on other test rows, or sets other rows aside as validation on its deal.</exception>
    public void Hold(PreparedData rows, DealPlace place)
    {
        var test = rows.KeysOf(Part.Test).ToHashSet();
        var validation = rows.KeysOf(Part.Validation).ToHashSet();

        if (_test is null)
        {
            _test = test;
        }
        else if (!_test.SetEquals(test))
        {
            throw new InvalidOperationException(
                $"{place.Pipeline} measures on other test rows than the first one prepared did: a candidate would be chosen on rows another deal learned from. Give the split a test seed — SplitAtRandom(train, validation, seed, testSeed) — so the rows to be measured on are the same on every deal, and move it by nothing a candidate chooses.");
        }

        if (!_validation.TryAdd(place.Deal, validation) && !_validation[place.Deal].SetEquals(validation))
        {
            throw new InvalidOperationException(
                $"{place.Pipeline} sets other rows aside as validation than the first trial's does: its score would be compared with scores measured on other rows. Deal the validation rows by nothing a candidate chooses.");
        }
    }
}
