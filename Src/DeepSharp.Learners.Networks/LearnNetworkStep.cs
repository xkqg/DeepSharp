// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Learners.Networks;

/// <summary>
/// The network a pipeline is declared for: its layers, the optimizer that moves them, the loss they are judged by, when
/// the run stops, and the engine it runs on.
/// </summary>
/// <remarks>
/// Pipeline-driven design is the whole course from raw data to a validated model, declared in advance and replayed. This
/// is the last step of that course: the file says which network was trained behind these steps, and training it again
/// from the file gives the same network. The step changes no column and runs nothing itself — the package that brings it
/// trains once the pipeline has run, through <see cref="TrainExtensions.Train(Pipeline, Engines)"/> — so it stands in
/// the declaration the way the report does, below the output it learns to answer.
/// </remarks>
public sealed record LearnNetworkStep : IPipelineStep<LearnNetworkStep>, INamesTheLearner, IDescribesColumns
{
    /// <summary>The engine a declaration names when it names none of its own: the light one DeepSharp ships.</summary>
    public const string LightEngine = "light";

    private static readonly PartsParameter LayersKey = new(
        "layers",
        "The layers, from the one the prepared rows reach first.",
        [new PartDeclaration("dense", [new PartSetting("units", PartValue.Of(16))]), new PartDeclaration("relu", [])],
        NetworkWords.Layers);

    private static readonly PartsParameter OptimizerKey = new(
        "optimizer", "What moves the network's numbers at every step.", [NetworkWords.Optimizers[1].Declared()], NetworkWords.Optimizers)
    {
        Single = true,
    };

    private static readonly PartsParameter LossKey = new(
        "loss", "What the network's answers are judged by while it trains.", [NetworkWords.Losses[0].Declared()], NetworkWords.Losses)
    {
        Single = true,
    };

    private static readonly PartsParameter StoppingKey = new(
        "stopping", "When the run stops: at its last pass, or once the validation loss stops falling.", [NetworkWords.Stopping[0].Declared()], NetworkWords.Stopping)
    {
        Single = true,
    };

    private static readonly TextParameter EngineKey = new(
        "engine",
        "The name of the engine the run's arithmetic happens on; which engine it stands for — and which device it works on — is the application's to say.",
        LightEngine);

    private static readonly WholeNumberParameter SeedKey = new(
        "seed", "The number every random draw of the run is worked out from, so the same declaration gives the same run.", 20260929);

    private static readonly WholeNumberParameter EpochsKey = new("epochs", "How many times the run goes over the training rows, at most.", 100) { AtLeast = 1 };

    private static readonly WholeNumberParameter BatchKey = new("batch", "How many rows one step of the run is worked out from.", 32) { AtLeast = 1 };

    /// <summary>Declares that this pipeline's rows train this network.</summary>
    /// <param name="layers">The layers, from the one the prepared rows reach first.</param>
    /// <param name="optimizer">What moves the network's numbers.</param>
    /// <param name="loss">What its answers are judged by.</param>
    /// <exception cref="ArgumentException">A part gives a name the words do not know, or leaves out a setting that name takes.</exception>
    public LearnNetworkStep(IReadOnlyList<PartDeclaration> layers, PartDeclaration optimizer, PartDeclaration loss)
    {
        Layers = LayersKey.Require(layers);
        Optimizer = OptimizerKey.Require([optimizer])[0];
        Loss = LossKey.Require([loss])[0];
    }

    /// <summary>The layers, from the one the prepared rows reach first.</summary>
    public IReadOnlyList<PartDeclaration> Layers { get; }

    /// <summary>What moves the network's numbers at every step.</summary>
    public PartDeclaration Optimizer { get; }

    /// <summary>What the network's answers are judged by while it trains.</summary>
    public PartDeclaration Loss { get; }

    /// <summary>When the run stops; at its last pass, unless said.</summary>
    public PartDeclaration Stopping
    {
        get => field.Kind is null ? NetworkWords.Stopping[0].Declared() : field;
        init => field = StoppingKey.Require([value])[0];
    }

    /// <summary>The name of the engine the run happens on; the light one, unless said.</summary>
    public string Engine
    {
        get => field ?? LightEngine;
        init => field = EngineKey.Require(value);
    }

    /// <summary>The number every random draw of the run is worked out from.</summary>
    public int Seed
    {
        get;
        init => field = SeedKey.Require(value);
    } = 20260929;

    /// <summary>How many times the run goes over the training rows, at most.</summary>
    public int Epochs
    {
        get;
        init => field = EpochsKey.Require(value);
    } = 100;

    /// <summary>How many rows one step of the run is worked out from.</summary>
    public int Batch
    {
        get;
        init => field = BatchKey.Require(value);
    } = 32;

    /// <inheritdoc />
    public static string Name => "learn.network";

    /// <inheritdoc />
    public static string Purpose =>
        "Names the network this pipeline is declared for: its layers, what moves them, what judges them, when the run stops and the engine it runs on.";

    /// <inheritdoc />
    public static int Since => 5;

    /// <inheritdoc />
    public static StepParameters<LearnNetworkStep> Parameters { get; } = new StepParameters<LearnNetworkStep>()
        .With(LayersKey, step => step.Layers)
        .With(OptimizerKey, step => (IReadOnlyList<PartDeclaration>)[step.Optimizer])
        .With(LossKey, step => (IReadOnlyList<PartDeclaration>)[step.Loss])
        .With(StoppingKey, step => (IReadOnlyList<PartDeclaration>)[step.Stopping])
        .With(EngineKey, step => step.Engine)
        .With(SeedKey, step => step.Seed)
        .With(EpochsKey, step => step.Epochs)
        .With(BatchKey, step => step.Batch);

    /// <inheritdoc />
    public string Verb => Name;

    /// <inheritdoc />
    /// <remarks>Every feature between minus one and one, as a network takes them.</remarks>
    public Needs Needs => TrainedNetwork.FeatureNeeds;

    /// <inheritdoc />
    public static LearnNetworkStep ReadFrom(JsonElement element) =>
        new(LayersKey.Read(element), OptimizerKey.Read(element)[0], LossKey.Read(element)[0])
        {
            Stopping = StoppingKey.Read(element)[0],
            Engine = EngineKey.Read(element),
            Seed = SeedKey.Read(element),
            Epochs = EpochsKey.Read(element),
            Batch = BatchKey.Read(element),
        };

    /// <inheritdoc />
    /// <remarks>It changes no column: what it names happens once the pipeline has run.</remarks>
    public ColumnState After(ColumnState before) => before;

    /// <inheritdoc />
    public bool Equals(LearnNetworkStep? other) =>
        other is not null
        && Layers.SequenceEqual(other.Layers)
        && Optimizer == other.Optimizer
        && Loss == other.Loss
        && Stopping == other.Stopping
        && Engine == other.Engine
        && Seed == other.Seed
        && Epochs == other.Epochs
        && Batch == other.Batch;

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();

        foreach (var layer in Layers)
        {
            hash.Add(layer);
        }

        hash.Add(Optimizer);
        hash.Add(Loss);
        hash.Add(Stopping);
        hash.Add(Engine);
        hash.Add(Seed);
        hash.Add(Epochs);
        hash.Add(Batch);

        return hash.ToHashCode();
    }
}
