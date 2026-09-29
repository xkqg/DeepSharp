// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>How a loop trains: the seed every random draw is worked out from, how many epochs, how many rows a batch.</summary>
/// <remarks>
/// The seed is asked for and never assumed: it decides how the training rows are shuffled, which values a dropout leaves
/// out and — for a network described in Keras's words — what the layers start at, so it is what makes a run the same run
/// again, and the history and the saved network both say which it was. The rest follows Keras's <c>fit</c>: one epoch and
/// batches of thirty-two, unless said.
/// </remarks>
public sealed class FitOptions
{
    private readonly int _epochs = 1;
    private readonly int _batchSize = 32;

    /// <summary>Options for a run worked out from the given seed.</summary>
    /// <param name="seed">The number every random draw of the run is worked out from.</param>
    public FitOptions(long seed) => Seed = seed;

    /// <summary>The number every random draw of the run is worked out from.</summary>
    public long Seed { get; }

    /// <summary>How many times the loop goes over the training rows, at most; one, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is below one.</exception>
    public int Epochs
    {
        get => _epochs;
        init => _epochs = value >= 1 ? value : throw new ArgumentOutOfRangeException(nameof(value), value, "A run is at least one epoch.");
    }

    /// <summary>How many rows a batch holds, the last of an epoch holding what is left; thirty-two, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is below one.</exception>
    public int BatchSize
    {
        get => _batchSize;
        init => _batchSize = value >= 1 ? value : throw new ArgumentOutOfRangeException(nameof(value), value, "A batch holds at least one row.");
    }

    /// <summary>When the run stops before its last epoch, judged by the validation rows; never, unless said.</summary>
    public EarlyStopping? EarlyStopping { get; init; }

    /// <summary>Where the arithmetic runs; a light engine of the run's own, unless said.</summary>
    public ITensorBackend? Backend { get; init; }

    /// <summary>Where a checkpoint goes at the end of an epoch, and which epochs keep one; none, unless said.</summary>
    public Checkpoints? Checkpoints { get; init; }

    /// <summary>A checkpoint to go on from, as if the run had never stopped there; a run from its first epoch, unless said.</summary>
    public Checkpoint? ResumeFrom { get; init; }
}

/// <summary>
/// Stops a run once the validation loss stops falling, and brings back the weights of its best epoch when asked: Keras's
/// <c>EarlyStopping</c>, watching the validation loss.
/// </summary>
/// <remarks>
/// As Keras has it: every epoch the wait grows by one; an epoch whose validation loss is below the best by more than
/// <see cref="MinDelta"/> is the new best and the wait starts again; an epoch after the first that is not, once the wait
/// has reached <see cref="Patience"/>, is the last. With <see cref="RestoreBest"/>, the network ends the run holding what
/// it held at the end of its best epoch — every slot, running statistics too — whether the run stopped early or not.
/// </remarks>
public sealed class EarlyStopping
{
    private readonly double _minDelta;
    private readonly int _patience;

    /// <summary>How far the validation loss has to fall below the best for an epoch to count as better; nothing, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is below nothing, or not a number.</exception>
    public double MinDelta
    {
        get => _minDelta;
        init => _minDelta = double.IsFinite(value) && value >= 0
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value), value, "The least fall that counts is a number of at least nothing.");
    }

    /// <summary>How many epochs without a better one the run waits before it stops; none, unless said.</summary>
    /// <exception cref="ArgumentOutOfRangeException">It is below nothing.</exception>
    public int Patience
    {
        get => _patience;
        init => _patience = value >= 0 ? value : throw new ArgumentOutOfRangeException(nameof(value), value, "Patience is a number of epochs, at least none.");
    }

    /// <summary>Whether the run ends holding its best epoch's weights rather than its last; its last, unless said.</summary>
    public bool RestoreBest { get; init; }
}

/// <summary>Where a run's checkpoints go, and which epochs keep one: every epoch, or those that improved on the best.</summary>
/// <param name="keep">What is done with each checkpoint: kept in a list, written to a file.</param>
/// <remarks>
/// A checkpoint is taken at the end of an epoch and is everything the run needs to go on from there, as Keras's
/// <c>ModelCheckpoint</c> keeps the model and its optimizer; every epoch keeps one, unless only the best are asked for, which
/// the validation rows judge.
/// </remarks>
public sealed class Checkpoints(Action<Checkpoint> keep)
{
    private readonly Action<Checkpoint> _keep = keep ?? throw new ArgumentNullException(nameof(keep));

    /// <summary>Whether only an epoch whose validation loss is the best so far keeps a checkpoint; every epoch, unless said.</summary>
    public bool BestOnly { get; init; }

    internal void Keep(Checkpoint checkpoint) => _keep(checkpoint);
}
