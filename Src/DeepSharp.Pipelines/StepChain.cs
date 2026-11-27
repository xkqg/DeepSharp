// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>The steps a pipeline being written holds, and the one way a step is added to them or the lot taken over.</summary>
/// <param name="rows">The rows handed in, when they are.</param>
/// <remarks>
/// The two halves of the builder hold the same steps — the half that learns goes on from where the split left the other —
/// and both add a step, and take saved decisions over, exactly alike: every step held to the rules a declaration keeps
/// where it is written. Held here once rather than written into each half.
/// </remarks>
internal sealed class StepChain(IRowSource? rows)
{
    private readonly List<IPipelineStep> _steps = [];

    /// <summary>The steps so far, in the order they were written.</summary>
    internal IReadOnlyList<IPipelineStep> Steps => _steps;

    /// <summary>The rows handed in, when they are.</summary>
    internal IRowSource? Rows { get; set; } = rows;

    /// <summary>What has been declared so far.</summary>
    internal PipelineDeclaration Declaration => new(_steps);

    /// <summary>Whether a split stands among the steps, which ends the half that does not learn.</summary>
    internal bool Split => _steps.FirstIndexOf<ISplitStep>() >= 0;

    /// <summary>Adds a step, refused where it is written when the steps would break a rule a declaration keeps.</summary>
    /// <param name="step">The step.</param>
    /// <exception cref="DeclarationException">The steps with this one break a rule.</exception>
    internal void Add(IPipelineStep step)
    {
        PipelineDeclaration.ThrowIfFaulty([.. _steps, step]);

        _steps.Add(step);
    }

    /// <summary>Takes the steps a take-over made in place of these, refused with every fault it found.</summary>
    /// <param name="steps">The steps the take-over made.</param>
    /// <param name="faults">The faults it found.</param>
    /// <exception cref="DeclarationException">There is a fault.</exception>
    internal void TakeOver(IReadOnlyList<IPipelineStep> steps, IReadOnlyList<DeclarationFault> faults)
    {
        if (faults.Count > 0)
        {
            throw new DeclarationException(faults);
        }

        _steps.Clear();
        _steps.AddRange(steps);
    }
}
