// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// A step that names the learner a pipeline is declared for, and changes no row.
/// </summary>
/// <remarks>
/// Pipeline-driven design is the course from raw data to a validated model, declared in advance: the learner belongs in
/// the declaration as much as the split does, so the file says what was trained behind these steps and a run can be
/// replayed whole. The step says one thing the pipeline itself uses — what the learner needs of the features it is
/// handed, so a run is made for the learner the declaration names without anybody having to say it twice. Everything
/// else about the learner is the step's own: which words describe it, which engine it runs on, how long it trains. The
/// pipeline reads none of that, and the package that brings the verb trains once the run is done — which is why this is
/// a step that acts on nothing, as an output that only names the answer and a report that only names its measures do.
/// <para>
/// A step that names a learner says which columns it leaves behind, through <see cref="IDescribesColumns"/>: a step
/// that says nothing of them leaves every column possible from there on, and a run for a learner would then be able to
/// leave nothing out.
/// </para>
/// </remarks>
public interface INamesTheLearner : IPipelineStep
{
    /// <summary>What this learner needs of the features it is handed.</summary>
    /// <remarks>
    /// The one thing the pipeline takes from a learner: <see cref="Pipeline.RunFor(Needs)"/> leaves out the steps a
    /// learner of this kind does without, and the run's file records which.
    /// </remarks>
    Needs Needs { get; }
}

/// <summary>
/// The learner a declaration names, for the package that trains it.
/// </summary>
internal static class DeclaredLearnerExtensions
{
    extension(PipelineDeclaration declaration)
    {
        /// <summary>The learner the declaration names, or the reason the package asking is the wrong one to train it.</summary>
        /// <typeparam name="TLearner">The learner step the package trains.</typeparam>
        /// <param name="kind">What the package trains, as the refusal says it of a learner it does not: a network this package trains.</param>
        /// <param name="hint">How a pipeline is declared for the package, as the refusal ends: declare one with .WithTorch(…).</param>
        /// <returns>The learner step.</returns>
        /// <exception cref="InvalidOperationException">The declaration names no learner, or names one the package does not train.</exception>
        public TLearner Learner<TLearner>(string kind, string hint)
            where TLearner : class, INamesTheLearner =>
            declaration.Learner as TLearner
            ?? throw new InvalidOperationException(
                declaration.Learner is null
                    ? $"This pipeline names no learner, so there is nothing here to train: {hint}"
                    : $"This pipeline names '{declaration.Learner.Verb}', which is not {kind}: {hint}");
    }
}
