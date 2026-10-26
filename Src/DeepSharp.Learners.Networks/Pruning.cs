// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Learners.Networks;

/// <summary>
/// Where a trial's run stands at the end of an epoch, as a study's <see cref="Study.Pruner"/> hears it.
/// </summary>
/// <param name="Deal">The division of the rows the run is training on, from nought.</param>
/// <param name="SoFar">The validation loss of every epoch of this deal so far, the one just ended last.</param>
/// <param name="Earlier">The trials before this one, each with what it heard, those that were pruned among them.</param>
public readonly record struct Progress(int Deal, IReadOnlyList<double> SoFar, IReadOnlyList<Trial> Earlier);

/// <summary>
/// Where a trial's run stands at the end of a deal, as a study's <see cref="Study.DealStopper"/> hears it.
/// </summary>
/// <param name="Scores">
/// How this trial was judged on every deal that has finished so far, in deal order — the same number
/// <see cref="Study.Judge"/>, a <see cref="Study.Score"/> of the caller's, or the default validation loss already judges
/// a finished deal by, never a raw per-epoch loss.
/// </param>
/// <param name="Earlier">The trials before this one, each with how it was judged.</param>
/// <remarks>Not <see cref="Progress"/>: the two differ in more than their C# shape. A <see cref="Progress"/>'s <see cref="Progress.SoFar"/>
/// is one network's raw loss, once an epoch; this one's <see cref="Scores"/> is the whole trial's judged number, once a deal.</remarks>
public readonly record struct DealsJudged(IReadOnlyList<double> Scores, IReadOnlyList<Trial> Earlier)
{
    /// <summary>The deal that just finished, from nought.</summary>
    public int Deal => Scores.Count - 1;
}

/// <summary>
/// Optuna's median rule: a trial is stopped at an epoch when the best validation loss it has reached on a deal by then is
/// worse than the median of what the trials that finished reached at that same epoch of that same deal.
/// </summary>
/// <remarks>
/// A loss halfway through a run is not the loss it ends at, and a run that starts slowly can end best, so the rule waits: it
/// prunes nothing until <see cref="Startup"/> trials have finished, and nothing within a trial's first <see cref="WarmUp"/>
/// epochs. Neither has a default, since how long a slow starter needs is a fact of the run, not of the rule. A trial that
/// was pruned is no finished trial, and does not count towards the median.
/// </remarks>
public sealed class MedianPruner
{
    /// <summary>Declares the rule.</summary>
    /// <param name="startup">How many trials must have finished before any is pruned; nought or more.</param>
    /// <param name="warmUp">How many epochs of a deal a trial runs before it may be pruned; nought or more.</param>
    /// <exception cref="ArgumentOutOfRangeException">Either is below nought.</exception>
    public MedianPruner(int startup, int warmUp)
    {
        Startup = startup >= 0 ? startup : throw new ArgumentOutOfRangeException(nameof(startup), startup, "How many trials finish before any is pruned is a count, nought or more.");
        WarmUp = warmUp >= 0 ? warmUp : throw new ArgumentOutOfRangeException(nameof(warmUp), warmUp, "How many epochs a trial runs before it may be pruned is a count, nought or more.");
    }

    /// <summary>How many trials must have finished before any is pruned.</summary>
    public int Startup { get; }

    /// <summary>How many epochs of a deal a trial runs before it may be pruned.</summary>
    public int WarmUp { get; }

    /// <summary>Whether the trial whose run stands here is stopped.</summary>
    /// <param name="progress">Where the run stands: its deal, its losses so far, and the trials before it.</param>
    /// <returns>Whether its best loss so far is worse than the median of the finished trials' at this epoch of this deal.</returns>
    public bool Prunes(Progress progress)
    {
        ArgumentNullException.ThrowIfNull(progress.SoFar);
        ArgumentNullException.ThrowIfNull(progress.Earlier);

        var epoch = progress.SoFar.Count - 1;
        var finished = progress.Earlier.Where(trial => trial.Finished).ToArray();

        if (finished.Length < Startup || epoch < WarmUp)
        {
            return false;
        }

        var reached = finished
            .Where(trial => trial.Heard.Count > progress.Deal && trial.Heard[progress.Deal].Count > epoch)
            .Select(trial => trial.Heard[progress.Deal][epoch])
            .Order()
            .ToArray();

        if (reached.Length == 0)
        {
            return false;
        }

        var median = reached.Length % 2 == 1 ? reached[reached.Length / 2] : (reached[(reached.Length / 2) - 1] + reached[reached.Length / 2]) / 2;

        return progress.SoFar.Min() > median;
    }
}
