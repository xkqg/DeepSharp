// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
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

    /// <summary>The most norm a step's gradients may have together before the optimizer takes them; no clipping, unless said.</summary>
    /// <remarks>
    /// It is part of what a run is: a checkpoint records it, and a run going on from the checkpoint under another, or under
    /// none where it had one, is refused.
    /// </remarks>
    public GradientClip? GradientClip { get; init; }

    /// <summary>
    /// Where the arithmetic runs — the training passes, the looks at the validation rows and the report's measures; the light
    /// engine that ships with DeepSharp and needs nothing installed, unless said.
    /// </summary>
    /// <remarks>
    /// Which engine a run takes when none is named is said here and nowhere else: the options hold the light engine from the
    /// start, and one handed in as nothing leaves them holding it.
    /// </remarks>
    [AllowNull]
    public ITensorBackend Backend
    {
        get;
        init => field = value ?? new CpuBackend();
    } = new CpuBackend();

    /// <summary>Where a checkpoint goes at the end of an epoch, and which epochs keep one; none, unless said.</summary>
    public Checkpoints? Checkpoints { get; init; }

    /// <summary>What stops the run before its next batch, or before it looks at the validation rows; nothing stops it, unless said.</summary>
    /// <remarks>
    /// It is read before every batch and before every look at the validation rows, and a read draws nothing, so a run it never
    /// stops is the run without it, to the last bit. Once it is cancelled the run throws an
    /// <see cref="OperationCanceledException"/> carrying it, and returns nothing: no history, and no best epoch put back — the
    /// network holds what the last batch it finished left it, and the optimizer remembers the steps it took up to there, so
    /// fitting it again goes on from there, not from its start. A checkpoint taken before is as it was taken, and a run gone
    /// on from it is the run that never stopped.
    /// </remarks>
    public CancellationToken Cancellation { get; init; }

    /// <summary>
    /// What is handed every epoch as it ends — once it is judged and its checkpoint kept, before the next one begins — on the
    /// thread the run trains on; nothing, unless said.
    /// </summary>
    /// <remarks>
    /// It is called in order, once an epoch, and waited for: what it is handed never changes, and a callback that only looks
    /// at the run leaves every number as it is. One that throws abandons the run as a cancel does — nothing is returned, no
    /// best epoch put back — and its exception reaches the caller as it was thrown.
    /// </remarks>
    public Action<Epoch>? OnEpoch { get; init; }

    /// <summary>A checkpoint to go on from, as if the run had never stopped there; a run from its first epoch, unless said.</summary>
    /// <remarks>
    /// The run goes on under the seed, the batch size, the early stopping and the clip the checkpoint was taken under, on the
    /// engine it was taken on, and is refused under any other, each difference named, since it would be another run: the
    /// batch size decides which rows every step takes, the early stopping where the run ends, the clip how far every step
    /// moves, and the engine how every step's totals are rounded — the same engine, in the same version, on the same device, as far as the engine names them. It goes on to as
    /// many epochs as <see cref="Epochs"/> says, more than the checkpoint's run was given too. A checkpoint read from a file
    /// 0.4.0 wrote records only its seed, and goes on under whatever batch size and early stopping, and on whatever engine,
    /// it is handed.
    /// </remarks>
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

/// <summary>
/// Clips a step's gradients by their norm together, before the optimizer takes them: PyTorch's <c>clip_grad_norm_</c>,
/// in the norm of two.
/// </summary>
/// <remarks>
/// As PyTorch has it: the whole norm is the norm of every parameter's gradient's own norm, and when the most norm over the
/// whole norm plus a millionth is below one, every gradient is scaled by it, so their whole norm comes to just under the most.
/// Otherwise the gradients pass untouched, to the last bit, so a clip the gradients never reach leaves a run as it is without
/// one. The norm is worked out on the run's engine through its own arithmetic — each gradient's squares averaged and counted
/// back up — and added up beside it, so it runs on any engine. Only the norm is clipped: clipping each value on its own needs
/// an exact clamp that an engine is not asked for.
/// </remarks>
public sealed class GradientClip
{
    /// <summary>A clip of the gradients to the given norm together.</summary>
    /// <param name="maxNorm">The most norm the gradients may have together.</param>
    /// <exception cref="ArgumentOutOfRangeException">It is not a number above nothing.</exception>
    public GradientClip(double maxNorm) =>
        MaxNorm = double.IsFinite(maxNorm) && maxNorm > 0
            ? maxNorm
            : throw new ArgumentOutOfRangeException(nameof(maxNorm), maxNorm, "The most norm gradients are clipped to is a number above nothing.");

    /// <summary>The most norm the gradients may have together.</summary>
    public double MaxNorm { get; }

    /// <summary>The gradients a step takes: scaled down to the most norm when their whole norm is above it, and those handed otherwise.</summary>
    /// <param name="parameters">The parameters the step moves.</param>
    /// <param name="gradients">Their gradients, as the pass worked them out.</param>
    /// <param name="backend">The backend the arithmetic runs on, which records nothing.</param>
    /// <returns>The gradients the optimizer takes.</returns>
    internal Gradients Clipped(IReadOnlyList<Parameter> parameters, Gradients gradients, ITensorBackend backend)
    {
        // Each tensor once, as PyTorch's parameters are each counted once, however many places hold it.
        var tensors = parameters.Select(parameter => parameter.Value).Distinct().ToArray();
        var squares = 0d;

        foreach (var tensor in tensors.Where(tensor => tensor.Shape.Count > 0))
        {
            var gradient = gradients[tensor];

            squares += (double)backend.Mean(backend.Multiply(gradient, gradient)).Values[0] * gradient.Shape.Count;
        }

        var coefficient = MaxNorm / (Math.Sqrt(squares) + 1e-6);

        if (!(coefficient < 1))
        {
            return gradients;
        }

        var scale = backend.Fill(new Shape(), (float)coefficient);

        return new Gradients(tensors.ToDictionary(tensor => tensor, tensor => backend.Scale(gradients[tensor], scale)));
    }
}

/// <summary>Where a run's checkpoints go, and which epochs keep one: every epoch, or those that improved on the best.</summary>
/// <param name="keep">What is done with each checkpoint: kept in a list, written to a file.</param>
/// <remarks>
/// A checkpoint is taken at the end of an epoch and holds what the run needs to go on from there, as Keras's
/// <c>ModelCheckpoint</c> keeps the model and its optimizer, with the seed, the batch size, the early stopping and the clip
/// the run went under and the engine it was on, which a run going on from it is held to; every epoch keeps one, unless only the
/// best are asked for, which the validation rows judge.
/// </remarks>
public sealed class Checkpoints(Action<Checkpoint> keep)
{
    private readonly Action<Checkpoint> _keep = keep ?? throw new ArgumentNullException(nameof(keep));

    /// <summary>Whether only an epoch whose validation loss is the best so far keeps a checkpoint; every epoch, unless said.</summary>
    public bool BestOnly { get; init; }

    internal void Keep(Checkpoint checkpoint) => _keep(checkpoint);
}
