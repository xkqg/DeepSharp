// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// What every file of a trained model is held to once its parts are read: that each network says what it was trained on, that
/// it was trained for the answers of the pipeline beside it and behind that very fit, and that the interval it carries has a
/// margin for each answer. The kind of the file says how a fault is worded and where it stands.
/// </summary>
internal static class VerifiedFileExtensions
{
    extension(FileText file)
    {
        /// <summary>What the network at a place in the file records it was trained on.</summary>
        /// <param name="saved">The network, as its part of the file was read.</param>
        /// <param name="at">Which of the file's networks it is, from nought: the ensemble's member, or a boosted pair's basis (0) or correction (1).</param>
        /// <returns>What it was trained on.</returns>
        /// <exception cref="NetworkFileException">The network records nothing of it: the fault, placed at the network.</exception>
        public TrainedOn TrainedOnOf(SavedNetwork saved, int at = 0) =>
            saved.TrainedOn
            ?? throw file.Refused(
                file.Kind.KeyOf(at),
                $"{file.Kind.NameOf(at)} names nothing it was trained on: the file of a trained network says its features, its answers and the fit of the pipeline it was trained behind.");

        /// <summary>Holds a network to the pipeline its file carries, and to the pipeline its file was read into.</summary>
        /// <param name="trainedOn">What the network records it was trained on.</param>
        /// <param name="prepared">The pipeline the file was read into.</param>
        /// <param name="at">Which of the file's networks it is, from nought.</param>
        /// <exception cref="NetworkFileException">The network was trained for other answers than the pipeline gives, or behind another fit than the pipeline beside it: the fault, placed where the kind of file says.</exception>
        public void HeldTo(TrainedOn trainedOn, PreparedData prepared, int at = 0)
        {
            if ((TrainedNetwork.Mismatch(trainedOn, prepared) ?? TrainedNetwork.Behind(trainedOn, file.Pipeline())) is not { } fault)
            {
                return;
            }

            throw file.Kind switch
            {
                FileKind.Ensemble => file.Refused(file.Kind.KeyOf(at), $"Network {at + 1}: {fault.Message}"),
                FileKind.Boosted => file.Refused(file.Kind.KeyOf(at), fault.Message),
                _ => file.Refused(fault.Key, fault.Message),
            };
        }

        /// <summary>The interval the file carries, held to the number of answers the networks give.</summary>
        /// <param name="answers">How many answers each network gives.</param>
        /// <returns>The interval; nothing when the file carries none.</returns>
        /// <exception cref="NetworkFileException">What is under the key is no interval, or it has another number of margins than the networks have answers.</exception>
        public ConformalInterval? IntervalHeldTo(int answers)
        {
            var interval = file.Interval();

            if (interval is not null && interval.Margins.Count != answers)
            {
                var who = file.Kind == FileKind.Ensemble ? "the networks answer" : "the network answers";

                throw file.Refused(TrainedNetwork.IntervalKey, $"The interval names {interval.Margins.Count} margins, and {who} {answers}: it has one margin for each answer.");
            }

            return interval;
        }
    }

    extension(FileKind kind)
    {
        // The key a network of the file stands under: the ensemble's members stand in one list.
        private string KeyOf(int at) => kind switch
        {
            FileKind.Ensemble => NetworkDocument.NetworksKey,
            FileKind.Boosted => at == 0 ? BoostedNetwork.BasisKey : BoostedNetwork.CorrectionKey,
            _ => TrainedNetwork.NetworkKey,
        };

        // What a network of the file is called where a fault is about it.
        private string NameOf(int at) => kind switch
        {
            FileKind.Ensemble => $"Network {at + 1}",
            FileKind.Boosted => at == 0 ? "This file's basis" : "This file's correction",
            _ => "This file's network",
        };
    }
}
