// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Learners.Networks;

// Choosing from what a study found: its best few, and its Pareto front. Both read the one verdict each trial is ranked by,
// the verdict the study itself chose its best by, so neither can rank a trial another way than the study did.
public static partial class EnsembleExtensions
{
    extension(StudyResult result)
    {
        /// <summary>The networks of the best few trials, best first, as declarations a pipeline trains again.</summary>
        /// <param name="count">How many: at least one, and no more than the trials that finished.</param>
        /// <returns>
        /// The declarations, best first by the verdict the study chose its <see cref="StudyResult.Best"/> by — the lowest score, or
        /// the lexicographic least of the mean objective vector for a study given <see cref="Study.Objectives"/> — the earlier
        /// of two judged alike first.
        /// </returns>
        /// <exception cref="ArgumentOutOfRangeException">The count is below one, or above the number of trials that finished.</exception>
        /// <exception cref="InvalidOperationException">One of them declared several networks, judged together.</exception>
        /// <remarks>
        /// A study chooses by the validation rows alone, so the few it hands on are chosen by them alone; train them with
        /// <c>prepared.TrainEnsemble(result.Top(3))</c>. Trials that were pruned, or judged by a number that is none, are not among them.
        /// </remarks>
        public IReadOnlyList<LearnNetworkStep> Top(int count)
        {
            ArgumentNullException.ThrowIfNull(result);

            Trial[] finished = [.. result.Trials.Where(trial => trial.Finished).OrderBy(trial => trial.Verdict)];

            if (count < 1 || count > finished.Length)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(count),
                    count,
                    $"A study hands on the best of the trials that finished, and {finished.Length} did: ask for between one and {finished.Length}.");
            }

            if (finished.Take(count).FirstOrDefault(trial => trial.Members.Count > 1) is { } several)
            {
                throw new InvalidOperationException(
                    $"Trial {several.Number} declared {several.Members.Count} networks, judged together: an ensemble blends networks that each answer the same rows, and these answer different ones. Train its networks from StudyResult.Members, or by their declarations in Trial.Members.");
            }

            return [.. finished.Take(count).Select(trial => trial.Declared!)];
        }

        /// <summary>The finished trials none of the others betters on every one of the study's objectives: its Pareto front.</summary>
        /// <returns>Those trials, in the order the study tried them.</returns>
        /// <exception cref="InvalidOperationException">The study was given no objectives, so no trial holds an objective score.</exception>
        /// <remarks>
        /// Where a single <see cref="StudyResult.Best"/> alone would hide that two or more trade-offs are really being weighed
        /// against each other, the front keeps every candidate: a trial on it is one no other trial is at least as good as on
        /// every objective, and strictly better on one.
        /// </remarks>
        public IReadOnlyList<Trial> ParetoFront()
        {
            ArgumentNullException.ThrowIfNull(result);

            Trial[] weighed = [.. result.Trials.Where(trial => trial.Finished && trial.ObjectiveScores.Count > 0)];

            if (weighed.Length == 0)
            {
                throw new InvalidOperationException("A Pareto front is of a study given objectives, and none of this study's finished trials holds an objective score: give it a Study.Objectives.");
            }

            return [.. weighed.Where(mine => !weighed.Any(other => other.Number != mine.Number && other.Verdict.Dominates(mine.Verdict)))];
        }
    }
}
