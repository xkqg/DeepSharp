// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// Everything a run needs to go on from the end of an epoch as if it had never stopped: what every slot of the network
/// held, what the optimizer remembers of every parameter, how far early stopping had got, the history so far and the seed.
/// </summary>
/// <remarks>
/// Every part is kept by the path of the slot it belongs to, never by the objects of the network it was taken from, so a
/// checkpoint written to a file and read back into a freshly built network resumes the run exactly: the draws are worked
/// out from the seed, the epoch and the step, so nothing random needs keeping.
/// </remarks>
public sealed class Checkpoint
{
    internal Checkpoint(Resumable state) => State = state;

    /// <summary>The seed the run was worked out from; a resume is refused under another.</summary>
    public long Seed => State.Seed;

    /// <summary>How many epochs the run had finished.</summary>
    public int Epochs => State.History.Count;

    /// <summary>The epochs the run had gone through.</summary>
    public IReadOnlyList<Epoch> History => State.History;

    internal Resumable State { get; }
}

/// <summary>What a checkpoint holds, by path.</summary>
/// <param name="Seed">The run's seed.</param>
/// <param name="History">The epochs so far.</param>
/// <param name="Slots">What every slot of the network held.</param>
/// <param name="Memory">What the optimizer remembers of each parameter that it remembers anything of.</param>
internal sealed record Resumable(long Seed, IReadOnlyList<Epoch> History, IReadOnlyDictionary<string, Tensor> Slots, IReadOnlyDictionary<string, SlotMemory> Memory)
{
    /// <summary>How far early stopping had got; nothing when the run was not stopped early.</summary>
    public Judgement? Judgement { get; init; }
}

/// <summary>What an optimizer remembers of one parameter: how many steps it took, and the tensors it keeps, by PyTorch's names for them.</summary>
/// <param name="Steps">How many steps the parameter has taken, for an optimizer that counts them.</param>
/// <param name="Tensors">The tensors, by name: <c>momentum_buffer</c>; <c>exp_avg</c> and <c>exp_avg_sq</c>.</param>
internal readonly record struct SlotMemory(int Steps, IReadOnlyDictionary<string, Tensor> Tensors);

/// <summary>How far a run's early stopping, or its judging of the best epoch, has got.</summary>
/// <param name="Wait">How many epochs since the best.</param>
/// <param name="Best">The best validation loss so far.</param>
/// <param name="BestEpoch">The epoch it was reached in.</param>
/// <remarks>
/// There is one only once an epoch has been judged: the first epoch judged is the best so far whatever its loss, as it is
/// to Keras, whose best starts out as the largest loss there is, and a loss that is not a finite number is refused before
/// it is judged.
/// </remarks>
internal sealed record Judgement(int Wait, double Best, int BestEpoch)
{
    /// <summary>What every slot held at the end of the best epoch, when the run is to end holding it.</summary>
    public IReadOnlyDictionary<string, Tensor>? BestSlots { get; init; }

    /// <summary>Whether the run is to stop.</summary>
    public bool Stops { get; init; }

    /// <summary>The judgement of the first epoch judged: the best so far, with nothing to wait for.</summary>
    /// <param name="epoch">The epoch.</param>
    /// <param name="loss">Its validation loss.</param>
    /// <returns>The judgement from here on.</returns>
    public static Judgement First(int epoch, double loss) => new(0, loss, epoch);

    /// <summary>
    /// The judgement after an epoch, as Keras's <c>EarlyStopping</c> judges it: the wait grows by one; a validation loss below
    /// the best by more than the least fall that counts is the new best and starts the wait again; otherwise the run stops
    /// once the wait has reached the patience.
    /// </summary>
    /// <param name="epoch">The epoch just ended.</param>
    /// <param name="loss">Its validation loss.</param>
    /// <param name="stopping">Early stopping, when the run has it; without it the best epoch is still judged, and nothing stops.</param>
    /// <returns>The judgement from here on.</returns>
    public Judgement After(int epoch, double loss, EarlyStopping? stopping)
    {
        if (loss + (stopping?.MinDelta ?? 0) < Best)
        {
            return new Judgement(0, loss, epoch) { BestSlots = BestSlots };
        }

        var wait = Wait + 1;

        return this with { Wait = wait, Stops = stopping is not null && wait >= stopping.Patience };
    }
}
