// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// What a search needs of a pipeline and of a trained network: rows to judge by, and the judgement.
/// </summary>
internal static class SearchExtensions
{
    extension(Pipeline pipeline)
    {
        // The pipeline run for the networks, with validation rows to judge them by: a search chooses between finished
        // networks by those rows and by no others.
        internal PreparedData RunToBeJudged()
        {
            ArgumentNullException.ThrowIfNull(pipeline);

            var prepared = pipeline.RunFor(TrainedNetwork.FeatureNeeds);

            if (prepared.Batch(Part.Validation, TrainedNetwork.FeatureNeeds).RowCount == 0)
            {
                throw new InvalidOperationException(
                    "This pipeline sets no rows aside as validation, and a search chooses between networks by the validation rows: divide the rows with a validation share.");
            }

            return prepared;
        }
    }

    extension(TrainedNetwork trained)
    {
        // The validation loss of the numbers the network holds: the epoch it ended at, or the best one when it went back to it.
        internal double ValidationLoss() =>
            trained.History!.Epochs.FirstOrDefault(each => each.Number == trained.TrainedOn.Epoch).ValidationLoss ?? double.NaN;
    }
}
