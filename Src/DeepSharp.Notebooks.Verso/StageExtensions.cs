// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Notebooks.Verso;

/// <summary>
/// What a step acts through, and the stage of a pipeline that makes it, as a block shows it.
/// </summary>
/// <remarks>
/// Worked out from the capability the run asks a step for, and never stored beside the step, so it cannot say one
/// thing while the step does another. The capabilities a step can act through are a closed set; the verbs are the
/// open axis, and a verb another package brings has its capability the moment it acts. Two capabilities may share a
/// stage's name — leaving out rows and leaving out a column are both cleaning — and a verb is swapped only for one
/// that acts as it does.
/// </remarks>
internal static class StageExtensions
{
    // Every capability a step acts through, each with the name of its stage.
    private static readonly Dictionary<Type, string> Stages = new()
    {
        [typeof(IOpensRows)] = "read",
        [typeof(IBindsColumns)] = "columns",
        [typeof(IOrdersRows)] = "order",
        [typeof(IAddsColumns)] = "features",
        [typeof(IDropsRows)] = "clean",
        [typeof(IDropsColumns)] = "clean",
        [typeof(ISplitStep)] = "split",
        [typeof(IFittedStep)] = "learned from the training rows",
        [typeof(IProducesEvidence)] = "evidence",
    };

    /// <summary>The capability this step acts through: the one thing it does in a run.</summary>
    /// <param name="step">The step.</param>
    /// <returns>The capability; the step's own type for the target, which acts on nothing and only names the answer.</returns>
    public static Type ActingCapability(this IPipelineStep step) =>
        Stages.Keys.FirstOrDefault(capability => capability.IsInstanceOfType(step)) ?? step.GetType();

    /// <summary>The stage this step belongs to.</summary>
    /// <param name="step">The step.</param>
    /// <returns>Its stage, in a word or a few.</returns>
    public static string Stage(this IPipelineStep step) => Stages.GetValueOrDefault(step.ActingCapability(), "target");
}
