// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Networks;

/// <summary>
/// What a network was trained on, as the file of a trained network records it beside the network's numbers: what each
/// value it takes and each answer it gives stands for, what named the answers, the digest of the preparation its rows came
/// through, and the seed and the epoch of the run.
/// </summary>
/// <remarks>
/// Names and numbers only, so the network's file says what a network was trained behind without the network knowing what
/// prepared its rows: whatever serves it checks the rows it is handed against these, and refuses a preparation fitted any
/// other way — the same names are not the same fit.
/// </remarks>
public sealed record TrainedOn
{
    /// <summary>What each value of an example stands for, in the order the network takes them.</summary>
    public required IReadOnlyList<string> Features { get; init; }

    /// <summary>What each answer stands for, in the order the network gives them.</summary>
    public required IReadOnlyList<string> Answers { get; init; }

    /// <summary>What named the answers: the verb of the step that made them.</summary>
    public required string Output { get; init; }

    /// <summary>The digest of the preparation the rows were trained behind, as it was fitted.</summary>
    public required string TrainedBehind { get; init; }

    /// <summary>The seed the run was worked out from.</summary>
    public required long Seed { get; init; }

    /// <summary>The epoch whose numbers the network holds, counted from nought.</summary>
    public required int Epoch { get; init; }
}
