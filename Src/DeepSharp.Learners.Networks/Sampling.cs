// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// Decides which candidate the next trial of a study tries.
/// </summary>
/// <remarks>
/// The seam a smarter search plugs into: a sampler that learns from the trials that finished — a tree of Parzen
/// estimators, say — can live in any assembly and be handed to a <see cref="Study"/>. It is handed the space, the place of
/// the trial it suggests for, and the trials that finished before it, and must give a candidate with one setting for every
/// dimension of the space, in the order of the space. Anything it draws should be worked out from its own seed and the
/// trial's place, as <see cref="RandomSampler"/> does, so a study run again is the same study.
/// </remarks>
public interface ISampler
{
    /// <summary>Gives the candidate the trial at this place tries.</summary>
    /// <param name="space">What may vary.</param>
    /// <param name="trial">The place of the trial, from nought.</param>
    /// <param name="finished">The trials before it, each with how it was judged.</param>
    /// <returns>A value for every dimension of the space.</returns>
    Candidate Suggest(SearchSpace space, int trial, IReadOnlyList<Trial> finished);
}

/// <summary>
/// Draws every candidate from the seed and the trial's place alone.
/// </summary>
/// <remarks>
/// Random search is the baseline a cleverer search has to beat, and it needs no trial to have finished before the next
/// begins. Each dimension draws from a purpose of its own — its name — so a trial gets the same candidate whatever ran
/// before it, a study of twelve trials is the front of one of thirty, and a dimension added later leaves the values of the
/// others where they were.
/// </remarks>
public sealed class RandomSampler : ISampler
{
    private readonly RandomStream _stream;

    /// <summary>Starts a sampler.</summary>
    /// <param name="seed">The number every draw is worked out from.</param>
    public RandomSampler(long seed)
    {
        Seed = seed;
        _stream = new RandomStream(seed);
    }

    /// <summary>The number every draw is worked out from.</summary>
    public long Seed { get; }

    /// <inheritdoc />
    public Candidate Suggest(SearchSpace space, int trial, IReadOnlyList<Trial> finished)
    {
        ArgumentNullException.ThrowIfNull(space);
        ArgumentOutOfRangeException.ThrowIfNegative(trial);

        return new Candidate(space.Select(dimension => dimension.Draw(_stream.Draw($"search:{dimension.Name}", trial, 0))));
    }
}
