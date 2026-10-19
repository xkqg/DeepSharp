// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// Why a trained network, or an ensemble of them, answers as it does: the questions the pipeline asks of any model, put to the
/// model's own way of answering rows.
/// </summary>
/// <remarks>
/// Each question perturbs the rows handed in — columns shuffled among them, one column moved through its range, a row's values put
/// over another's one column after another — serves them through the pipeline the model was trained behind, and reads the answers it
/// gives in their own units. Nothing is fitted, so a category is one-hot and a log is undone exactly as they are when the model is asked
/// anything else; and every draw comes from a seed the caller names.
/// </remarks>
public static class AskWhyExtensions
{
    extension(TrainedNetwork trained)
    {
        /// <summary>Which columns the network's answers lean on: how far they move when a column's values are shuffled among the rows.</summary>
        /// <param name="rows">The rows to ask of, with the columns the pipeline reads; the answer's own column needs not be among them.</param>
        /// <param name="redraws">The seed the shuffles are worked out from and how many times each column is shuffled.</param>
        /// <returns>Each column, the one the answers move most with first.</returns>
        public Reliance Leaning(IRowSource rows, Redraws redraws)
        {
            ArgumentNullException.ThrowIfNull(trained);

            return trained.Prepared.RelianceOn(new RowsAnswered(rows, handed => trained.Predict(handed).Answers), redraws);
        }

        /// <summary>What the network's answers do as one column moves through its range, with the rest of each row left as it was.</summary>
        /// <param name="rows">The rows to ask of.</param>
        /// <param name="moving">The column, and at how many places a number is moved to.</param>
        /// <returns>The mean answer at each place.</returns>
        public Dependence Curve(IRowSource rows, Moving moving)
        {
            ArgumentNullException.ThrowIfNull(trained);

            return trained.Prepared.DependenceOn(new RowsAnswered(rows, handed => trained.Predict(handed).Answers), moving);
        }

        /// <summary>What one row's answer is made of, column by column.</summary>
        /// <param name="rows">The rows to ask of.</param>
        /// <param name="row">Which of them, from nought.</param>
        /// <param name="redraws">The seed the samples are worked out from and how many are drawn.</param>
        /// <returns>What each column adds to the row's answer.</returns>
        public RowMakeUp MakeUp(IRowSource rows, int row, Redraws redraws)
        {
            ArgumentNullException.ThrowIfNull(trained);

            return trained.Prepared.MakeUpOf(new RowsAnswered(rows, handed => trained.Predict(handed).Answers), row, redraws);
        }
    }

    extension(TrainedEnsemble trained)
    {
        /// <summary>Which columns the ensemble's combined answers lean on: how far they move when a column's values are shuffled among the rows.</summary>
        /// <param name="rows">The rows to ask of, with the columns the pipeline reads; the answer's own column needs not be among them.</param>
        /// <param name="redraws">The seed the shuffles are worked out from and how many times each column is shuffled.</param>
        /// <returns>Each column, the one the answers move most with first.</returns>
        public Reliance Leaning(IRowSource rows, Redraws redraws)
        {
            ArgumentNullException.ThrowIfNull(trained);

            return trained.Prepared.RelianceOn(new RowsAnswered(rows, handed => trained.Predict(handed).Answers), redraws);
        }

        /// <summary>What the ensemble's combined answers do as one column moves through its range, with the rest of each row left as it was.</summary>
        /// <param name="rows">The rows to ask of.</param>
        /// <param name="moving">The column, and at how many places a number is moved to.</param>
        /// <returns>The mean answer at each place.</returns>
        public Dependence Curve(IRowSource rows, Moving moving)
        {
            ArgumentNullException.ThrowIfNull(trained);

            return trained.Prepared.DependenceOn(new RowsAnswered(rows, handed => trained.Predict(handed).Answers), moving);
        }

        /// <summary>What one row's combined answer is made of, column by column.</summary>
        /// <param name="rows">The rows to ask of.</param>
        /// <param name="row">Which of them, from nought.</param>
        /// <param name="redraws">The seed the samples are worked out from and how many are drawn.</param>
        /// <returns>What each column adds to the row's answer.</returns>
        public RowMakeUp MakeUp(IRowSource rows, int row, Redraws redraws)
        {
            ArgumentNullException.ThrowIfNull(trained);

            return trained.Prepared.MakeUpOf(new RowsAnswered(rows, handed => trained.Predict(handed).Answers), row, redraws);
        }
    }
}
