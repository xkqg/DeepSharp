// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>Whether a pass trains a network or only uses it.</summary>
public enum PassMode
{
    /// <summary>The pass trains: dropout drops, a normalisation measures its batch and moves its running statistics.</summary>
    Training,

    /// <summary>The pass only uses the network: dropout keeps everything, a normalisation uses what it measured in training.</summary>
    Evaluation,
}

/// <summary>
/// One run of a network forward: the backend its arithmetic runs on, whether it trains, and — when it does — where in the
/// run it stands and where its random draws come from.
/// </summary>
/// <remarks>
/// Handed to every layer by the layer above it, so what a layer does is said once, for the whole network, by whoever runs
/// it: the loop, for a training step or a look at the validation rows; a host, for rows it serves. Nothing is switched on
/// in a layer and left on, and nothing reaches for a shared random source.
/// </remarks>
public sealed class Pass
{
    private readonly RandomStream? _random;
    private readonly Recompute? _recompute;

    private Pass(ITensorBackend backend, PassMode mode, RandomStream? random, Recompute? recompute = null)
    {
        Backend = backend;
        Mode = mode;
        _random = random;
        _recompute = recompute;
    }

    /// <summary>The backend the pass's arithmetic runs on: a recording one, for a pass whose gradients are wanted.</summary>
    public ITensorBackend Backend { get; }

    /// <summary>Whether the pass trains or only uses the network.</summary>
    public PassMode Mode { get; }

    /// <summary>The epoch a training pass belongs to, counted from nought; nought for an evaluation pass.</summary>
    public int Epoch { get; private init; }

    /// <summary>The step of its epoch a training pass is, counted from nought; nought for an evaluation pass.</summary>
    public int Step { get; private init; }

    /// <summary>Whether a layer that measures while it trains moves what it measured: a pass that runs a layer again moves nothing, since the pass it repeats moved it.</summary>
    internal bool MovesStatistics { get; private init; } = true;

    /// <summary>The same pass told which layers to throw the workings of away and work out again.</summary>
    /// <param name="layers">The layers; nothing for none.</param>
    /// <returns>The pass; this one, when none is asked for.</returns>
    internal Pass Recomputing(Recompute? layers) =>
        layers is null ? this : new Pass(Backend, Mode, _random, layers) { Epoch = Epoch, Step = Step, MovesStatistics = MovesStatistics };

    /// <summary>The same place in the same run, on another backend, that records nothing and recomputes nothing.</summary>
    /// <param name="backend">Where its arithmetic runs.</param>
    /// <returns>The pass.</returns>
    internal Pass On(ITensorBackend backend) => new(backend, Mode, _random) { Epoch = Epoch, Step = Step, MovesStatistics = MovesStatistics };

    /// <summary>The same place in the same run, on a backend that records it again: it draws what the first pass drew and moves no statistic.</summary>
    /// <param name="backend">The recording the layer is worked out again on.</param>
    /// <returns>The pass.</returns>
    internal Pass Replaying(ITensorBackend backend) => new(backend, Mode, _random) { Epoch = Epoch, Step = Step, MovesStatistics = false };

    /// <summary>Whether a layer is to be thrown away and worked out again: this pass is recorded, and the layer's path is among those asked for.</summary>
    /// <param name="layer">The layer.</param>
    /// <returns>Whether it is.</returns>
    internal bool Recomputes(Layer layer) => _recompute is not null && Backend is RecordingBackend && _recompute.Includes(layer.Path);

    /// <summary>Runs a layer over what its steps made: kept as it is run, or thrown away and worked out again when the way back reaches it.</summary>
    /// <param name="layer">The layer.</param>
    /// <param name="inputs">What the steps it takes made, in order.</param>
    /// <returns>What it makes.</returns>
    internal Tensor Running(Layer layer, IReadOnlyList<Tensor> inputs) =>
        Recomputes(layer) ? ((RecordingBackend)Backend).Recomputed(layer, inputs, this) : layer.Taking(inputs, this);

    /// <summary>A pass that trains.</summary>
    /// <param name="backend">The backend the arithmetic runs on.</param>
    /// <param name="random">Where the run's random draws come from.</param>
    /// <param name="epoch">The epoch the pass belongs to, counted from nought.</param>
    /// <param name="step">The step of that epoch, counted from nought.</param>
    /// <returns>The pass.</returns>
    /// <exception cref="ArgumentOutOfRangeException">The epoch or the step is before the first.</exception>
    public static Pass Training(ITensorBackend backend, RandomStream random, int epoch, int step)
    {
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(random);
        ArgumentOutOfRangeException.ThrowIfNegative(epoch);
        ArgumentOutOfRangeException.ThrowIfNegative(step);

        return new Pass(backend, PassMode.Training, random) { Epoch = epoch, Step = step };
    }

    /// <summary>A pass that only uses the network.</summary>
    /// <param name="backend">The backend the arithmetic runs on.</param>
    /// <returns>The pass.</returns>
    public static Pass Evaluation(ITensorBackend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);

        return new Pass(backend, PassMode.Evaluation, random: null);
    }

    /// <summary>The draws for one purpose at the place in the run this pass stands.</summary>
    /// <param name="purpose">What they are for — a dropout layer's path, say — so no two uses share draws.</param>
    /// <returns>The draws the run's stream holds for that purpose, this epoch and this step.</returns>
    /// <exception cref="InvalidOperationException">The pass does not train, and an evaluation pass draws nothing.</exception>
    public Draws Draws(string purpose) =>
        _random is null
            ? throw new InvalidOperationException($"An evaluation pass draws nothing, and '{purpose}' asked it to.")
            : _random.Draw(purpose, Epoch, Step);
}
