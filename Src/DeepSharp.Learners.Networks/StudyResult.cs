// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Learners.Networks;

/// <summary>
/// What a study found: every trial, the best of them, and the networks it trained.
/// </summary>
public sealed class StudyResult
{
    internal StudyResult(IReadOnlyList<Trial> trials, Trial best, IReadOnlyList<IReadOnlyList<TrainedNetwork>> members)
    {
        Trials = trials;
        Best = best;
        Members = members;
        Winner = [.. members.Select(deal => deal[0])];
    }

    /// <summary>Every trial, in the order they were tried.</summary>
    public IReadOnlyList<Trial> Trials { get; }

    /// <summary>The trial judged lowest, the earlier one when two were judged alike.</summary>
    public Trial Best { get; }

    /// <summary>
    /// The networks the best trial trained, one for each deal: the first of its networks, where it has several. They are the only
    /// networks a study keeps, and so the only ones whose report of the test rows anything can read: a trial reports nothing about them.
    /// </summary>
    public IReadOnlyList<TrainedNetwork> Winner { get; }

    /// <summary>The networks the best trial trained, every one of them, for each deal in order and in the order the trial declared them.</summary>
    public IReadOnlyList<IReadOnlyList<TrainedNetwork>> Members { get; }
}

/// <summary>What a caller-trained study found: every trial, the best of them, and the model trained for each of its deals.</summary>
public sealed class StudyResult<TModel>
{
    internal StudyResult(IReadOnlyList<Trial> trials, Trial best, IReadOnlyList<TModel> winner)
    {
        Trials = trials;
        Best = best;
        Winner = winner;
    }

    /// <summary>Every trial, in the order they were tried.</summary>
    public IReadOnlyList<Trial> Trials { get; }

    /// <summary>The trial judged lowest, the earlier one when two were judged alike.</summary>
    public Trial Best { get; }

    /// <summary>The models the best trial trained, one for each deal. They are the only models a study keeps.</summary>
    public IReadOnlyList<TModel> Winner { get; }
}
