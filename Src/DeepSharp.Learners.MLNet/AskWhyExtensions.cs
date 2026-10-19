// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Learners.MLNet;

/// <summary>
/// Why a trained tree answers as it does: the questions the pipeline asks of any model, put to the tree's own way of answering rows.
/// </summary>
/// <remarks>
/// The same questions, drawn the same way, as a network is asked — which columns the answers lean on, what they do as one column moves
/// through its range, and what one row's answer is made of — so a tree and a network are explained by one arithmetic and can be laid
/// side by side. Nothing is fitted, and every draw comes from a seed the caller names.
/// </remarks>
public static class AskWhyExtensions
{
    extension(TrainedMLModel trained)
    {
        /// <summary>Which columns the tree's answers lean on: how far they move when a column's values are shuffled among the rows.</summary>
        /// <param name="rows">The rows to ask of, with the columns the pipeline reads; the answer's own column needs not be among them.</param>
        /// <param name="redraws">The seed the shuffles are worked out from and how many times each column is shuffled.</param>
        /// <returns>Each column, the one the answers move most with first.</returns>
        public Reliance Leaning(IRowSource rows, Redraws redraws)
        {
            ArgumentNullException.ThrowIfNull(trained);

            return trained.Prepared.RelianceOn(new RowsAnswered(rows, trained.Predict), redraws);
        }

        /// <summary>What the tree's answers do as one column moves through its range, with the rest of each row left as it was.</summary>
        /// <param name="rows">The rows to ask of.</param>
        /// <param name="moving">The column, and at how many places a number is moved to.</param>
        /// <returns>The mean answer at each place.</returns>
        public Dependence Curve(IRowSource rows, Moving moving)
        {
            ArgumentNullException.ThrowIfNull(trained);

            return trained.Prepared.DependenceOn(new RowsAnswered(rows, trained.Predict), moving);
        }

        /// <summary>What one row's answer is made of, column by column.</summary>
        /// <param name="rows">The rows to ask of.</param>
        /// <param name="row">Which of them, from nought.</param>
        /// <param name="redraws">The seed the samples are worked out from and how many are drawn.</param>
        /// <returns>What each column adds to the row's answer.</returns>
        public RowMakeUp MakeUp(IRowSource rows, int row, Redraws redraws)
        {
            ArgumentNullException.ThrowIfNull(trained);

            return trained.Prepared.MakeUpOf(new RowsAnswered(rows, trained.Predict), row, redraws);
        }
    }
}
