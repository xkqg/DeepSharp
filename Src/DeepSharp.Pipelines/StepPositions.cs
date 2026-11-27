// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// Where the first step of a kind stands among steps written one after another, and which it is.
/// </summary>
/// <remarks>
/// The one owner of the question every rule of a declaration, the declaration itself and the builder ask: where is the
/// split, the output, the schema, the order. The steps may not be a declaration yet — the rules judge them, and the
/// builder is still writing them — so it is asked of the steps and not of a declaration.
/// </remarks>
internal static class StepPositions
{
    extension(IReadOnlyList<IPipelineStep> steps)
    {
        /// <summary>Where the first step of a kind stands.</summary>
        /// <typeparam name="TStep">The kind: a step type, or a capability a step has.</typeparam>
        /// <returns>The place, counting from nought; minus one when no step is of the kind.</returns>
        public int FirstIndexOf<TStep>()
        {
            for (var at = 0; at < steps.Count; at++)
            {
                if (steps[at] is TStep)
                {
                    return at;
                }
            }

            return -1;
        }

        /// <summary>The first step of a kind.</summary>
        /// <typeparam name="TStep">The kind: a step type, or a capability a step has.</typeparam>
        /// <returns>The step; nothing when no step is of the kind.</returns>
        public TStep? FirstOf<TStep>() => steps.FirstIndexOf<TStep>() is var at and >= 0 ? (TStep)steps[at] : default;

        /// <summary>The column the schema says names each row, when the steps hold a schema that says one.</summary>
        /// <returns>The column's name; nothing when there is no schema or it names no id.</returns>
        public string? IdColumn() => steps.FirstOf<DeclareStep>()?.IdColumn;
    }
}
