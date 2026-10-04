// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Learners.ML;

/// <summary>
/// Names the trainer from ML.NET this pipeline's rows are prepared for.
/// </summary>
/// <remarks>
/// <para>
/// The same step a network has, for the other kind of learner: it changes no row and the pipeline reads one thing from
/// it — what the trainer needs of the features it is handed. A tree takes numbers of any size, so a run made for this
/// learner leaves the scalings out and writes down that it did, which is what lets a tree and a network be measured by
/// one report on the same rows.
/// </para>
/// <para>
/// It names a seed, and that is not decoration. The whole library rests on a declaration being replayable, so the
/// trainers it may name are the ones that repeat from a number written here. The threads a fit may use are not a
/// setting: a tree's model does not move with them, and for the trainers whose model does, the answer was to leave
/// those trainers out rather than to make a declaration carry the shape of the machine it ran on.
/// </para>
/// <para>
/// Training it needs <c>DeepSharp.Learners.MLNet</c>, which carries ML.NET. This package does not, so a notebook, a
/// server or an application that only reads a model never brings the library.
/// </para>
/// </remarks>
public sealed record LearnMLStep : IPipelineStep<LearnMLStep>, INamesTheLearner, IDescribesColumns
{
    private static readonly PartsParameter TrainerKey = new(
        "trainer",
        "The trainer these rows are prepared for, and the settings it takes.",
        [MLWords.Default],
        MLWords.Trainers)
    {
        Single = true,
    };

    private static readonly WholeNumberParameter SeedKey = new(
        "seed", "The number the trainer's own random draws are worked out from, so the same declaration gives the same model.", 20260929);

    /// <summary>Declares that this pipeline's rows train this trainer.</summary>
    /// <param name="trainer">The trainer, and the settings it takes.</param>
    /// <exception cref="ArgumentException">
    /// The part gives a name the words do not know — a trainer whose model does not repeat from a seed is not among
    /// them — or leaves out a setting that name takes.
    /// </exception>
    public LearnMLStep(PartDeclaration trainer) => Trainer = TrainerKey.Require([trainer])[0];

    /// <summary>The trainer these rows are prepared for, and the settings it takes.</summary>
    public PartDeclaration Trainer { get; }

    /// <summary>The number the trainer's own random draws are worked out from.</summary>
    public int Seed
    {
        get;
        init => field = SeedKey.Require(value);
    } = 20260929;

    /// <inheritdoc />
    public static string Name => "learn.ml";

    /// <inheritdoc />
    public static string Purpose =>
        "Names the trainer from ML.NET this pipeline is declared for: which trainer, the settings it takes and the seed it repeats from.";

    /// <inheritdoc />
    public static int Since => 7;

    /// <summary>What a trainer from ML.NET needs of the features it is handed: numbers of any size, as a tree takes them.</summary>
    /// <remarks>Said once here, so the step, the run made for it and whatever predicts from it cannot drift apart.</remarks>
    public static Needs FeatureNeeds => Needs.NoScale;

    /// <inheritdoc />
    public static StepParameters<LearnMLStep> Parameters { get; } = new StepParameters<LearnMLStep>()
        .With(TrainerKey, step => (IReadOnlyList<PartDeclaration>)[step.Trainer])
        .With(SeedKey, step => step.Seed);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    /// <remarks>Numbers of any size, as a tree takes them: a step that only scales a feature is left out of the run.</remarks>
    public Needs Needs => FeatureNeeds;

    /// <inheritdoc />
    public static LearnMLStep ReadFrom(JsonElement element) =>
        new(TrainerKey.Read(element)[0]) { Seed = SeedKey.Read(element) };

    /// <inheritdoc />
    /// <remarks>It changes no column: what it names happens once the pipeline has run.</remarks>
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    public bool Equals(LearnMLStep? other) => other is not null && Trainer == other.Trainer && Seed == other.Seed;

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(Trainer, Seed);
}
