// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// A checkpoint as a file: the one file of the network as it stood at the end of an epoch, with what the run needs to go
/// on from there — and read back beside the fit of the pipeline it was taken behind, and no other.
/// </summary>
/// <remarks>
/// Written from inside a run, where the checkpoint is handed over, so the network written beside it is the network it was
/// taken of: <c>Checkpoints = new Checkpoints(checkpoint =&gt; File.WriteAllText(path, CheckpointFile.Write(compiled, prepared, checkpoint)))</c>.
/// A checkpoint is a whole trained network too, and <see cref="TrainedNetwork.FromJson"/> serves it as it stood.
/// </remarks>
public static class CheckpointFile
{
    /// <summary>Writes a checkpoint as the one file, with what the run needs to go on.</summary>
    /// <param name="compiled">The network being trained, as the checkpoint left it.</param>
    /// <param name="prepared">The pipeline it is trained behind.</param>
    /// <param name="checkpoint">The checkpoint the run handed over.</param>
    /// <returns>The file, as JSON, indented, with a line feed between lines on every system.</returns>
    /// <exception cref="InvalidOperationException">
    /// The network has moved on since the checkpoint was taken; or the pipeline names no answer, was run for another learner
    /// and left out a step a network needs, or does not hand over every feature on one scale.
    /// </exception>
    /// <exception cref="OutOfMemoryException">
    /// The file would be longer than .NET makes a text: about 14 million parameters or more trained under Adam, or about 10
    /// million when the run keeps its best epoch's slots.
    /// </exception>
    /// <remarks>
    /// Beside each parameter's number the file holds what the optimizer remembers of it — two numbers under Adam — and, when
    /// the run keeps its best epoch's slots, that epoch's number too: 76 characters a parameter under Adam, and 104 with the
    /// best epoch's, where the one file of a trained network takes about 24. .NET makes no text longer than 1,073,741,791
    /// characters, and that sets the largest checkpoint there is.
    /// </remarks>
    public static string Write(CompiledNetwork compiled, PreparedData prepared, Checkpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(compiled);
        ArgumentNullException.ThrowIfNull(prepared);
        ArgumentNullException.ThrowIfNull(checkpoint);

        var trainedOn = TrainedNetwork.TrainedOnOf(prepared, prepared.Batch(Part.Train, TrainedNetwork.FeatureNeeds), checkpoint.Seed, checkpoint.Epochs - 1);

        return new TrainedNetwork(compiled.Network, compiled.Loss, prepared, trainedOn).Json(compiled, checkpoint);
    }

    /// <summary>Reads a checkpoint back, to go on behind the pipeline it was taken behind.</summary>
    /// <param name="json">The file.</param>
    /// <param name="networks">The kinds its network, its optimizer and its schedule may be.</param>
    /// <param name="prepared">The pipeline to go on behind: the very fit the checkpoint was taken behind.</param>
    /// <returns>The network compiled as it was, and the checkpoint to hand to <see cref="FitOptions.ResumeFrom"/>.</returns>
    /// <exception cref="NetworkFileException">
    /// The file is not one this library reads, or anything in it is wrong — at its line and column: its network was trained
    /// behind another fit than the pipeline the file carries, it carries none, or that pipeline names a version of the
    /// pipeline file no network's file of this library carries.
    /// </exception>
    /// <exception cref="ArgumentException">
    /// The pipeline is another fit than the one the checkpoint was taken behind — a run made for a learner that left steps out
    /// is one — or names other answers.
    /// </exception>
    /// <remarks>
    /// The network is held to the pipeline the file carries, as <see cref="TrainedNetwork.FromJson"/> holds it, and that
    /// pipeline to the one handed over by passing both through one writer and leaving out the version of the pipeline file
    /// each names: a checkpoint taken under an earlier version goes on behind the same fit run again under this one. The
    /// pipeline the file carries is compared, never read, so the version it names has to be one in which every step means
    /// what it means now: from the first version a network's file carried, 0.4.0's, to the one this library writes.
    /// The network and the run are read from one reading of the text, by <see cref="NetworkDocument.ReadCheckpoint"/>, before
    /// either is held to a pipeline, so a file that is wrong itself is refused for that, whichever pipeline it is handed.
    /// </remarks>
    public static ResumedRun Read(string json, NetworkCatalog networks, PreparedData prepared)
    {
        ArgumentNullException.ThrowIfNull(json);
        ArgumentNullException.ThrowIfNull(networks);
        ArgumentNullException.ThrowIfNull(prepared);

        var file = new FileText(json);
        var read = NetworkDocument.ReadCheckpoint(json, TrainedNetwork.NetworkKey, TrainedNetwork.TrainingKey, networks);
        var trainedOn = read.Network.TrainedOn
            ?? throw file.Refused(TrainedNetwork.NetworkKey, "This file's network names nothing it was trained on: a checkpoint says its features, its answers and the fit of the pipeline it was taken behind.");
        var carried = file.Pipeline();

        if (TrainedNetwork.Behind(trainedOn, carried) is { } unlike)
        {
            throw file.Refused(unlike.Key, unlike.Message);
        }

        if (Uncomparable(carried) is { } unread)
        {
            throw file.Refused(TrainedNetwork.PipelineKey, unread);
        }

        var handed = PipelineText.Of(prepared);
        var mismatch = TrainedNetwork.Mismatch(trainedOn, prepared)?.Message
            ?? (carried.IsTheFitOf(handed) ? null : TrainedNetwork.AnotherFit(trainedOn, handed));

        if (mismatch is not null)
        {
            throw new ArgumentException(
                $"This checkpoint cannot go on behind this pipeline, and going on would train on rows prepared another way. {mismatch}", nameof(prepared));
        }

        return read.Run;
    }

    // Why the pipeline a checkpoint carries cannot be held to one handed over here; nothing when it can. It is compared and
    // not read, so its words are taken to mean what they mean now only in a version in which no step has meant anything else.
    private static string? Uncomparable(PipelineText carried) =>
        PipelineText.IsComparable(carried.Version) ? null
        : carried.Version > PipelineDeclaration.Version
            ? string.Create(
                CultureInfo.InvariantCulture,
                $"The pipeline this checkpoint was taken behind was written against version {carried.Version} of the pipeline file, by a newer DeepSharp than this one, which reads up to version {PipelineDeclaration.Version}: go on with that DeepSharp.")
            : string.Create(
                CultureInfo.InvariantCulture,
                $"The pipeline this checkpoint was taken behind names no version of the pipeline file a network's file is written with, from {PipelineText.FirstComparable} to {PipelineDeclaration.Version}: no DeepSharp wrote this checkpoint.");
}
