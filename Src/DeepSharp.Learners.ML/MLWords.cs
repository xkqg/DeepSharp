// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Learners.ML;

/// <summary>
/// The words a declared trainer is written in: which trainer, and the settings it takes.
/// </summary>
/// <remarks>
/// <para>
/// Two words, and both of them are trees. That is not a first instalment waiting to be widened: a declaration is a
/// promise that running it again gives the same model, and a tree keeps that promise from a number written in the
/// declaration. The linear trainers do not. Measured on this library's own rows: at its defaults the stochastic dual
/// coordinate ascent trainer gave five different models in five runs, and pinning it to one thread makes it repeat on
/// one machine while the last bits still move when the processor takes another instruction path. A trainer that cannot
/// be replayed is refused by name here rather than offered with a warning nobody reads.
/// </para>
/// <para>
/// Every setting a word takes is written down and has a value of its own, because what is not written is decided by
/// whatever the library defaults to that month, and a declaration meant to outlive its library cannot rest on that.
/// </para>
/// </remarks>
public static class MLWords
{
    private static readonly WholeNumberParameter LeavesKey = new(
        "leaves", "How many leaves one tree may grow to: the larger it is, the finer the splits and the easier it is to learn the training rows by heart.", 20) { AtLeast = 2 };

    private static readonly WholeNumberParameter TreesKey = new(
        "trees", "How many trees are grown.", 100) { AtLeast = 1 };

    private static readonly WholeNumberParameter LeastRowsKey = new(
        "leastRows", "The fewest training rows a leaf may stand for, which is what keeps a split from being one passenger's story.", 10) { AtLeast = 1 };

    private static readonly NumberParameter StepKey = new(
        "rate", "How much of each tree's correction is taken: smaller is slower and steadier.", 0.2) { Above = 0 };

    /// <summary>The trainers a declaration may name, each with the settings it takes.</summary>
    public static IReadOnlyList<PartKind> Trainers { get; } =
    [
        new(
            "fastTree",
            "Gradient-boosted trees: each tree corrects what the ones before it got wrong. The usual first answer for a table.",
            [LeavesKey, TreesKey, LeastRowsKey, StepKey]),
        new(
            "fastForest",
            "A forest of trees grown apart and averaged, which holds up better on few rows than boosting does.",
            [LeavesKey, TreesKey, LeastRowsKey]),
    ];

    /// <summary>What a declaration starts with when it names no trainer of its own.</summary>
    public static PartDeclaration Default => Trainers[0].Declared();
}
