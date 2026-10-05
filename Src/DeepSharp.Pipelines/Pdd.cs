// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Pipelines;

/// <summary>
/// Where a pipeline starts.
/// </summary>
/// <remarks>
/// Pipeline-driven design: the course from raw data to a validated model is declared in advance as one
/// artefact and then replayed, rather than performed again at each stage by whoever is writing that stage.
/// </remarks>
public static class Pdd
{
    /// <summary>Starts a new pipeline, which has said nothing yet.</summary>
    /// <returns>A builder that records what you declare and does none of it.</returns>
    public static PipelineBuilder Create() => new();

    /// <summary>Starts a pipeline from a course that has been filled in all the way.</summary>
    /// <param name="course">The course, with everything that names something of yours said.</param>
    /// <param name="catalog">The verbs the course may name.</param>
    /// <returns>The pipeline the course names, ready to run.</returns>
    /// <exception cref="DeclarationException">
    /// The course still waits for something, names a verb the catalog does not know, says a key or a value its verb
    /// refuses, or names steps that break a rule; every fault is said at its step.
    /// </exception>
    /// <remarks>
    /// The other way a pipeline starts from something already written: <see cref="Create"/> starts from nothing and
    /// <c>PipelineDeclaration.FromJson</c> from a pipeline's own file. A course is read as a file's steps are, so what it
    /// names is held to the rules every door keeps.
    /// </remarks>
    public static Pipeline From(PipelineCourse course, StepCatalog catalog)
    {
        ArgumentNullException.ThrowIfNull(course);
        ArgumentNullException.ThrowIfNull(catalog);

        return new Pipeline(course.ToDeclaration(catalog));
    }
}

/// <summary>
/// Hands out pipelines, for an application that resolves what it needs instead of constructing it.
/// </summary>
/// <remarks>
/// The same door as <see cref="Pdd.Create"/>, in the shape a host expects. What it hands back is a fresh
/// builder every time: the factory may be shared, what it produces never is.
/// </remarks>
public interface IPipelineFactory
{
    /// <summary>Starts a new pipeline, which has said nothing yet.</summary>
    /// <returns>A builder of its own, sharing nothing with any other.</returns>
    PipelineBuilder Create();
}

/// <summary>
/// The factory a host resolves, which owns nothing and remembers nothing.
/// </summary>
internal sealed class PipelineFactory : IPipelineFactory
{
    public PipelineBuilder Create() => Pdd.Create();
}
