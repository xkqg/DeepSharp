// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// What a search needs of a pipeline, of its space and of a trained network: rows to judge by, candidates of the space, and the judgement.
/// </summary>
internal static class SearchExtensions
{
    extension(Pipeline? pipeline)
    {
        // The pipeline for a place of a search, run for what trains behind it, with validation rows to judge it by: a search
        // chooses between finished networks by those rows and by no others. A pipeline that is nothing is refused by its place.
        internal PreparedData RunToBeJudged(DealPlace place, Needs needs)
        {
            var prepared = (pipeline ?? throw new InvalidOperationException($"{place.Pipeline} is nothing.")).RunFor(needs);

            if (prepared.KeysOf(Part.Validation).Count == 0)
            {
                throw new InvalidOperationException(
                    "This pipeline sets no rows aside as validation, and a search chooses between networks by the validation rows: divide the rows with a validation share.");
            }

            return prepared;
        }
    }

    extension(SearchSpace space)
    {
        // A candidate holds a setting for every dimension that is there for it, in the space's order, and for no other, and each
        // value is one its dimension takes: a sampler is the caller's, and nothing is to learn from what it gave by mistake.
        internal void Require(Candidate candidate)
        {
            var there = space.Where(dimension => dimension.IsThereAmong(candidate.Settings)).Select(dimension => dimension.Name).ToArray();

            if (!candidate.Settings.Select(each => each.Name).SequenceEqual(there, StringComparer.Ordinal))
            {
                throw new InvalidOperationException(
                    $"The sampler gave a candidate of other dimensions than the space has for it: the space has {there.Quoted()}, "
                    + $"and the candidate {candidate.Settings.Select(each => each.Name).Quoted()}.");
            }

            foreach (var setting in candidate.Settings)
            {
                if (space.Single(dimension => dimension.Name == setting.Name).Refusal(setting) is { } refusal)
                {
                    throw new InvalidOperationException($"The sampler gave a setting the space cannot take: '{setting.Name}' {refusal}.");
                }
            }
        }
    }

    extension(TrainedNetwork trained)
    {
        // The validation loss of the numbers the network holds: the epoch it ended at, or the best one when it went back to it.
        internal double ValidationLoss() =>
            trained.History!.Epochs.FirstOrDefault(each => each.Number == trained.TrainedOn.Epoch).ValidationLoss ?? double.NaN;
    }
}
