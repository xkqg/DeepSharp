// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Learners.ML;

/// <summary>
/// The trainer a declaration names, written as a sentence.
/// </summary>
/// <remarks>
/// One trainer a declaration, so the line names it rather than collecting a list: the last one written is the one the
/// pipeline is declared for, and a second word would be a second learner, which the declaration refuses anyway.
/// </remarks>
public sealed class MLTrainerLine
{
    /// <summary>The trainer this line names, with the settings it was given.</summary>
    internal PartDeclaration Trainer { get; private set; } = MLWords.Default;

    /// <summary>Gradient-boosted trees: each tree corrects what the ones before it got wrong.</summary>
    /// <param name="leaves">How many leaves one tree may grow to.</param>
    /// <param name="trees">How many trees are grown.</param>
    /// <param name="leastRows">The fewest training rows a leaf may stand for.</param>
    /// <param name="rate">How much of each tree's correction is taken.</param>
    /// <returns>This line, so the sentence reads on.</returns>
    public MLTrainerLine FastTree(int leaves = 20, int trees = 100, int leastRows = 10, double rate = 0.2) =>
        Named("fastTree", [("leaves", PartValue.Of(leaves)), ("trees", PartValue.Of(trees)), ("leastRows", PartValue.Of(leastRows)), ("rate", PartValue.Of(rate))]);

    /// <summary>A forest of trees grown apart and averaged, which holds up better on few rows than boosting does.</summary>
    /// <param name="leaves">How many leaves one tree may grow to.</param>
    /// <param name="trees">How many trees are grown.</param>
    /// <param name="leastRows">The fewest training rows a leaf may stand for.</param>
    /// <returns>This line, so the sentence reads on.</returns>
    public MLTrainerLine FastForest(int leaves = 20, int trees = 100, int leastRows = 10) =>
        Named("fastForest", [("leaves", PartValue.Of(leaves)), ("trees", PartValue.Of(trees)), ("leastRows", PartValue.Of(leastRows))]);

    private MLTrainerLine Named(string kind, IReadOnlyList<(string Key, PartValue Value)> settings)
    {
        Trainer = new PartDeclaration(kind, [.. settings.Select(setting => new PartSetting(setting.Key, setting.Value))]);

        return this;
    }
}

/// <summary>
/// Declaring the trainer from ML.NET a pipeline's rows are prepared for.
/// </summary>
public static class MLChainExtensions
{
    extension(FittingBuilder fitting)
    {
        /// <summary>Declares the trainer from ML.NET this pipeline's rows train.</summary>
        /// <param name="trainer">The trainer, and the settings it takes.</param>
        /// <param name="seed">The number the trainer's own random draws are worked out from.</param>
        /// <returns>The chain, with the trainer declared.</returns>
        /// <exception cref="ArgumentNullException">There is no chain, or nothing names the trainer.</exception>
        /// <exception cref="DeclarationException">The declaration names no answer yet, or already names a learner.</exception>
        /// <remarks>
        /// Training what this names needs <c>DeepSharp.Learners.MLNet</c>; declaring it, writing it to a file and reading
        /// a model back do not.
        /// </remarks>
        public FittingBuilder WithML(Action<MLTrainerLine> trainer, int seed = 20260929)
        {
            ArgumentNullException.ThrowIfNull(fitting);
            ArgumentNullException.ThrowIfNull(trainer);

            var line = new MLTrainerLine();

            trainer(line);

            return fitting.Add(new LearnMLStep(line.Trainer) { Seed = seed });
        }
    }
}
