// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

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
    /// <returns>The file, as JSON.</returns>
    /// <exception cref="InvalidOperationException">
    /// The network has moved on since the checkpoint was taken; or the pipeline names no answer, or does not hand over
    /// every feature on one scale.
    /// </exception>
    public static string Write(CompiledNetwork compiled, PreparedData prepared, Checkpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(compiled);
        ArgumentNullException.ThrowIfNull(prepared);
        ArgumentNullException.ThrowIfNull(checkpoint);

        var trainedOn = TrainedNetwork.TrainedOnOf(prepared, prepared.Batch(Part.Train, Needs.OneScale), checkpoint.Seed, checkpoint.Epochs - 1);

        return new TrainedNetwork(compiled.Network, compiled.Loss, prepared, trainedOn).Json(compiled, checkpoint);
    }

    /// <summary>Reads a checkpoint back, to go on behind the pipeline it was taken behind.</summary>
    /// <param name="json">The file.</param>
    /// <param name="networks">The kinds its network, its optimizer and its schedule may be.</param>
    /// <param name="prepared">The pipeline to go on behind: the very fit the checkpoint was taken behind.</param>
    /// <returns>The network compiled as it was, and the checkpoint to hand to <see cref="FitOptions.ResumeFrom"/>.</returns>
    /// <exception cref="NetworkFileException">The file is not one this library reads, or anything in it is wrong — at its line and column.</exception>
    /// <exception cref="ArgumentException">The pipeline is another fit than the one the checkpoint was taken behind, or names other answers.</exception>
    public static ResumedRun Read(string json, NetworkCatalog networks, PreparedData prepared)
    {
        ArgumentNullException.ThrowIfNull(json);
        ArgumentNullException.ThrowIfNull(networks);
        ArgumentNullException.ThrowIfNull(prepared);

        var file = new FileText(json);
        var saved = NetworkDocument.ReadNetwork(json, TrainedNetwork.NetworkKey, networks);
        var trainedOn = saved.TrainedOn
            ?? throw file.Refused(TrainedNetwork.NetworkKey, "This file's network names nothing it was trained on: a checkpoint says its features, its answers and the fit of the pipeline it was taken behind.");

        if (TrainedNetwork.Mismatch(trainedOn, prepared) is { } mismatch)
        {
            throw new ArgumentException(
                $"This checkpoint cannot go on behind this pipeline, and going on would train on rows prepared another way. {mismatch.Message}", nameof(prepared));
        }

        return NetworkDocument.ReadTraining(json, TrainedNetwork.TrainingKey, networks, saved);
    }
}
