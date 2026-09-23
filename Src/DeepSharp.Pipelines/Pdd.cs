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
