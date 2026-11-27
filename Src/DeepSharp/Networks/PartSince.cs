// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Networks;

/// <summary>
/// What a part of a network file says that an older library could not, and the version it is written from: the one table the
/// writer takes the version of a part from, and the reader takes its refusal of an earlier part from.
/// </summary>
/// <remarks>
/// A part is written as the oldest version that says everything in it, and a part of an earlier version that says one of these
/// things is refused by the version that has it. The two were once decided apart, in the writer and in the reader; both ask here,
/// so a row that is changed is changed for both, and the file a row makes is the file the reader reads.
/// </remarks>
/// <param name="Part">Which part says it, as a refusal names it: with its article.</param>
/// <param name="Says">What it says, as a refusal names it, starting in lower case.</param>
/// <param name="From">The version of the part that first says it.</param>
internal readonly record struct PartSince(string Part, string Says, int From)
{
    private const string NetworkPart = "a network's part";
    private const string TrainingPart = "a training part";

    /// <summary>What the learning does to a parameter's numbers: held still, a penalty, a bound.</summary>
    public static readonly PartSince Rules = new(NetworkPart, "what the learning does to a parameter's numbers", NetworkDocument.FifthVersion);

    /// <summary>The share of the run's rate a parameter's numbers move at; said of the same, and written from a version after it.</summary>
    public static readonly PartSince RateShare = Rules with { From = NetworkDocument.SeventhVersion };

    /// <summary>The norm the run's gradients were clipped to.</summary>
    public static readonly PartSince ClipNorm = new(TrainingPart, "the norm a run's gradients were clipped to", NetworkDocument.ThirdVersion);

    /// <summary>The bound each value of the run's gradients was clipped to.</summary>
    public static readonly PartSince ClipValue = new(TrainingPart, "the bound each value of the run's gradients was clipped to", NetworkDocument.FifthVersion);

    /// <summary>How many batches went into each step.</summary>
    public static readonly PartSince StepEvery = new(TrainingPart, "how many batches went into each step", NetworkDocument.FifthVersion);

    /// <summary>The share an average of the run's weights kept of itself.</summary>
    public static readonly PartSince Average = new(TrainingPart, "the share an average of the run's weights kept of itself", NetworkDocument.FifthVersion);

    /// <summary>That the run's epochs were judged by the average of its weights.</summary>
    public static readonly PartSince JudgedByAverage = new(TrainingPart, "that the run's epochs were judged by the average of its weights", NetworkDocument.SixthVersion);

    /// <summary>The loss scale a run went under.</summary>
    public static readonly PartSince LossScale = new(TrainingPart, "the loss scale a run went under", NetworkDocument.NinthVersion);

    /// <summary>The width an engine kept its work in.</summary>
    public static readonly PartSince EnginePrecision = new(TrainingPart, "the width an engine kept its work in", NetworkDocument.NinthVersion);

    /// <summary>How many steps an epoch skipped.</summary>
    public static readonly PartSince Skipped = new(TrainingPart, "how many steps an epoch skipped", NetworkDocument.NinthVersion);

    /// <summary>An option an optimizer is set with, which the second version of a training part could not say; the optimizer's own settings carry it.</summary>
    public static readonly PartSince OptimizerOption = new(TrainingPart, "an option an optimizer was set with", NetworkDocument.FifthVersion);

    /// <summary>Whether a part of the given version cannot say it.</summary>
    /// <param name="version">The version the part is of.</param>
    /// <returns>Whether the part is of an earlier version than the one that first says it.</returns>
    public bool Cannot(int version) => version < From;

    /// <summary>The refusal of a part of an earlier version that says it.</summary>
    /// <param name="key">The key it stands under.</param>
    /// <param name="version">The version the part is of.</param>
    /// <returns>The sentence that says what it is, where it stands, which version has it and which the part is.</returns>
    /// <exception cref="ArgumentOutOfRangeException">Either version is not one a network file has been written as.</exception>
    public string Refusal(string key, int version) =>
        $"{char.ToUpperInvariant(Says[0])}{Says[1..]}, under '{key}', is written from the {Ordinal(From)} version of {Part} on, and this one is of the {Ordinal(version)}.";

    private static string Ordinal(int version) => version switch
    {
        NetworkDocument.FirstVersion => "first",
        NetworkDocument.SecondVersion => "second",
        NetworkDocument.ThirdVersion => "third",
        NetworkDocument.FourthVersion => "fourth",
        NetworkDocument.FifthVersion => "fifth",
        NetworkDocument.SixthVersion => "sixth",
        NetworkDocument.SeventhVersion => "seventh",
        NetworkDocument.EighthVersion => "eighth",
        NetworkDocument.NinthVersion => "ninth",
        _ => throw new ArgumentOutOfRangeException(nameof(version), version, "No version of a network file is called that."),
    };
}
