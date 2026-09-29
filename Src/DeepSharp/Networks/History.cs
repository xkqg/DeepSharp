// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Networks;

/// <summary>What a run did, epoch by epoch: the metrics the loop keeps, and what the charts are drawn from.</summary>
/// <remarks>
/// Keras's <c>History</c>: the training loss, the validation loss and the learning rate of every epoch, the seed the run was
/// worked out from, the epoch the validation rows judged best, and why the run stopped. Nobody assembles an array to draw a
/// loss curve; it is here.
/// </remarks>
public sealed class History
{
    internal History(long seed, IReadOnlyList<Epoch> epochs, int? bestEpoch, Stopping stopped)
    {
        Seed = seed;
        Epochs = epochs;
        BestEpoch = bestEpoch;
        Stopped = stopped;
    }

    /// <summary>The seed the run was worked out from.</summary>
    public long Seed { get; }

    /// <summary>Every epoch the run went through, in order.</summary>
    public IReadOnlyList<Epoch> Epochs { get; }

    /// <summary>The epoch whose validation loss was the best, by the rule the run was stopped by; nothing without validation rows.</summary>
    public int? BestEpoch { get; }

    /// <summary>Why the run stopped.</summary>
    public Stopping Stopped { get; }
}

/// <summary>One epoch of a run.</summary>
/// <param name="Number">Which epoch it was, counted from nought.</param>
/// <param name="Loss">The training loss: every batch's loss, weighted by the rows it held.</param>
/// <param name="ValidationLoss">The loss on the validation rows at the end of the epoch; nothing without them.</param>
/// <param name="LearningRate">The rate the epoch's steps took.</param>
public readonly record struct Epoch(int Number, double Loss, double? ValidationLoss, double LearningRate);

/// <summary>Why a run stopped.</summary>
public enum Stopping
{
    /// <summary>It went through every epoch it was given.</summary>
    AllEpochsRan,

    /// <summary>Early stopping ended it: the validation loss stopped falling for longer than its patience.</summary>
    NoLongerImproving,
}
