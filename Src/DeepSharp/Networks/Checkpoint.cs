// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>
/// What a run needs to go on from the end of an epoch as if it had never stopped: what every slot of the network held, what
/// the optimizer remembers of every parameter, how far early stopping had got and the history so far — and what the run
/// went under, which a run going on from it is handed again or refused: the seed, the batch size, the early stopping and
/// the engine.
/// </summary>
/// <remarks>
/// Every part is kept by the path of the slot it belongs to, never by the objects of the network it was taken from, so a
/// checkpoint written to a file and read back into a freshly built network resumes the run exactly: the draws are worked
/// out from the seed, the epoch and the step, so nothing random needs keeping. The seed decides the draws, the batch size
/// which rows each step takes and so which draws each row meets, the early stopping where the run ends and which weights
/// it ends holding, and the engine how every step's totals are rounded — two engines that add up in another order drift
/// apart: a run handed another of them would be another run, and is refused before anything is put back. The engine is
/// recorded as it names itself, and, where it names them through <see cref="INamesItsVersionAndDevice"/>, with its version
/// and its device. The epochs are not among them, since going on to more of them is what a resume is for. A checkpoint read
/// from a file 0.4.0 wrote records neither the batch size nor the early stopping nor the engine, and goes on under whatever
/// it is handed, as it did then; the checkpoints of the run that goes on record what that run was handed.
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

    /// <summary>
    /// How the run went through its epochs beside its seed: its batch size and its early stopping; nothing when the checkpoint
    /// does not say, as one read from a file 0.4.0 wrote does not.
    /// </summary>
    public Pace? Pace { get; init; }

    /// <summary>The engine the run was on; nothing when the checkpoint does not say, as one read from a file 0.4.0 wrote does not.</summary>
    public Engine? Engine { get; init; }

    /// <summary>
    /// Why a run handed these options cannot go on from here as the run the checkpoint was taken of would have: each value it
    /// was taken under that they differ in — the seed, and whatever of its batch size, its early stopping and its engine it
    /// records — named with both; nothing when they differ in none.
    /// </summary>
    /// <param name="options">What the run that is to go on from here is handed.</param>
    /// <returns>The refusal, a sentence for each difference; nothing when the run can go on.</returns>
    public string? Unlike(FitOptions options)
    {
        var unlike = new List<string>();

        if (Seed != options.Seed)
        {
            unlike.Add(string.Create(
                CultureInfo.InvariantCulture, $"The checkpoint was taken of a run seeded {Seed}, and going on under {options.Seed} would draw other numbers."));
        }

        if (Pace is { } pace)
        {
            unlike.AddRange(pace.Unlike(options));
        }

        if (Engine?.Unlike(options.Backend) is { } otherwise)
        {
            unlike.Add(otherwise);
        }

        return unlike.Count == 0 ? null : string.Join(' ', unlike);
    }
}

/// <summary>
/// The engine a run was on, as it names itself: its name, and — for one that names them through
/// <see cref="INamesItsVersionAndDevice"/> — the version of what works its arithmetic out and the device it works it out
/// on; nothing of either for one that does not.
/// </summary>
/// <param name="Name">The engine's name.</param>
/// <param name="Version">The version of what works its arithmetic out; nothing when the engine does not name it.</param>
/// <param name="Device">Where it works it out; nothing when the engine does not name it.</param>
internal readonly record struct Engine(string Name, string? Version, string? Device)
{
    /// <summary>The engine as it names itself.</summary>
    /// <param name="backend">The engine.</param>
    /// <returns>Its name, and its version and its device where it names them.</returns>
    public static Engine Of(ITensorBackend backend) =>
        backend is INamesItsVersionAndDevice named ? new(backend.Name, named.Version, named.Device) : new(backend.Name, null, null);

    /// <summary>
    /// Why a run on this engine cannot go on on the one handed as the run it was would have: the two named, when the handed one
    /// names itself otherwise in anything — its name, its version, its device, or naming either where this one did not.
    /// </summary>
    /// <param name="backend">The engine the run that is to go on is handed.</param>
    /// <returns>The refusal; nothing when it is this engine.</returns>
    public string? Unlike(ITensorBackend backend) =>
        Of(backend) is var handed && handed != this
            ? $"The checkpoint was taken of a run on the engine {this}, and going on under the engine {handed} would round every step otherwise."
            : null;

    /// <summary>The engine as a refusal names it: <c>'torch' 2.10.0.0 on cuda:0</c>, or its name alone.</summary>
    /// <returns>Its name in quotes, its version and the device it works on, as far as it names them.</returns>
    public override string ToString() =>
        $"'{Name}'{(Version is null ? string.Empty : $" {Version}")}{(Device is null ? string.Empty : $" on {Device}")}";
}

/// <summary>
/// How a run went through its epochs beside its seed: how many rows a batch held, and the early stopping that judged where
/// it ends — what a run going on from its checkpoint is handed again, or is refused.
/// </summary>
/// <param name="BatchSize">How many rows a batch held, the last of an epoch holding what was left.</param>
/// <param name="EarlyStopping">The early stopping the run went under; nothing when it had none.</param>
internal readonly record struct Pace(int BatchSize, EarlyStopping? EarlyStopping)
{
    /// <summary>How a run handed these options goes through its epochs.</summary>
    /// <param name="options">What the run is handed.</param>
    /// <returns>Its batch size and its early stopping.</returns>
    public static Pace Of(FitOptions options) => new(options.BatchSize, options.EarlyStopping);

    /// <summary>
    /// Each way a run handed these options would go through its epochs otherwise than the run this was taken of, in the words
    /// a refusal names it with: its batch size, and its early stopping — whether it has any, its patience, its least fall that
    /// counts and whether it restores the best — each compared by what it says, never by which object says it.
    /// </summary>
    /// <param name="options">What the run that is to go on is handed.</param>
    /// <returns>A sentence for each difference, naming what the checkpoint was taken under and what the run is handed.</returns>
    public IEnumerable<string> Unlike(FitOptions options)
    {
        if (BatchSize != options.BatchSize)
        {
            yield return string.Create(
                CultureInfo.InvariantCulture,
                $"The checkpoint was taken of a run in batches of {BatchSize}, and going on in batches of {options.BatchSize} would take other rows into every step.");
        }

        if (EarlyStopping is not { } taken)
        {
            if (options.EarlyStopping is not null)
            {
                yield return "The checkpoint was taken of a run with no early stopping, and going on with early stopping would stop it by a rule it never ran under.";
            }

            yield break;
        }

        if (options.EarlyStopping is not { } handed)
        {
            yield return "The checkpoint was taken of a run with early stopping, and going on without it would never stop it early.";

            yield break;
        }

        if (taken.Patience != handed.Patience)
        {
            yield return string.Create(
                CultureInfo.InvariantCulture,
                $"The checkpoint was taken of a run whose early stopping had a patience of {taken.Patience}, and going on with a patience of {handed.Patience} would stop it by another rule.");
        }

        if (taken.MinDelta != handed.MinDelta)
        {
            yield return string.Create(
                CultureInfo.InvariantCulture,
                $"The checkpoint was taken of a run whose early stopping counted a fall of more than {taken.MinDelta} as better, and going on counting a fall of more than {handed.MinDelta} as better would stop it by another rule.");
        }

        if (taken.RestoreBest != handed.RestoreBest)
        {
            yield return taken.RestoreBest
                ? "The checkpoint was taken of a run that ends holding its best epoch's weights, and going on without restoring them would end it holding its last."
                : "The checkpoint was taken of a run that ends holding its last epoch's weights, and going on restoring the best would end it holding its best epoch's.";
        }
    }
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
