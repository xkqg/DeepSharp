// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// What a line of a chain declares: the steps it collected, in the order they were written.
/// </summary>
/// <remarks>
/// A line names a kind and lists the columns it holds for, and hands back one step a column — which is what the
/// declaration, the file and a notebook's blocks have always held, so a line is a door and never a new shape. Said once
/// here, with the one fan-out beside it, so every such verb is written out by one piece of code rather than by a copy
/// of it per verb. Anything a column needs beyond the kind — a limit above which a fill is refused, a column to write
/// the result into — stays on the verb that takes one column, because it belongs to that column and not to the group.
/// </remarks>
internal interface IDeclaresSteps
{
    /// <summary>The steps the line declares.</summary>
    IReadOnlyList<IPipelineStep> Steps { get; }

    /// <summary>Where the pipeline declared its features land, handed to the line before it is written.</summary>
    /// <remarks>Only a line whose kinds land values in a range reads it; every other line leaves this as it is.</remarks>
    Form Features
    {
        set
        {
        }
    }
}

/// <summary>
/// The one fan-out every line of a chain goes through.
/// </summary>
internal static class LineExtensions
{
    /// <summary>A line of a chain, written by whoever called the verb it belongs to.</summary>
    /// <typeparam name="TLine">The kind of line.</typeparam>
    extension<TLine>(Action<TLine> line)
        where TLine : IDeclaresSteps, new()
    {
        /// <summary>Writes a line and hands back the steps it declared.</summary>
        /// <param name="features">Where the pipeline says its features land.</param>
        /// <param name="parameter">The name of the parameter the line came in as, for a refusal that names it.</param>
        /// <returns>The steps, in the order the columns were named.</returns>
        /// <exception cref="ArgumentNullException">There is no line.</exception>
        /// <exception cref="ArgumentException">The line names no column, so it declares nothing.</exception>
        internal IReadOnlyList<IPipelineStep> Declared(Form features, string parameter)
        {
            ArgumentNullException.ThrowIfNull(line);

            var declared = new TLine { Features = features };
            line(declared);

            // A line that declares nothing is a line somebody meant to finish, and it is refused where it is written
            // rather than read as a verb that does nothing — which the declaration's own rules would refuse later, at a
            // place that no longer says which line it was.
            return declared.Steps.Count > 0
                ? declared.Steps
                : throw new ArgumentException(
                    $"This line names no column, so it does nothing. Name the columns each kind holds for, as "
                    + $"{typeof(TLine).Name} says.",
                    parameter);
        }
    }
}
