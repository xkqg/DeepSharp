// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json.Nodes;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A rate that follows the training loss, said in a declaration: the word is written with every setting it takes, read back as
/// the schedule the same words written as code make, and trains a network the way that schedule does. A file written without
/// it is the file 0.13.0 wrote, but for the number of the version a pipeline file names.
/// </summary>
public class DeclaredPlateauTests
{
    private static FittingBuilder Passengers() => Pdd.Create()
        .ReadCsv(Repository.Data("titanic.csv"))
        .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("sex").Optional("age", ColumnKind.Number).Number("fare"))
        .SplitStratified("survived", 0.70, 0.15)
        .FillMissing("age", With.Median)
        .EncodeCategories()
        .Normalise("age", Scale.MidRange)
        .Normalise("fare", Scale.MidRange)
        .Normalise("sibsp", Scale.MidRange)
        .Normalise("parch", Scale.MidRange)
        .Target("survived");

    private static PipelineDeclaration Declared(Func<NetworkDeclaration, NetworkDeclaration> run) =>
        Passengers().WithTorch(network => run(network.Dense(1).Adam(0.01).BinaryCrossEntropy().Run(seed: 20260929, epochs: 6))).Build().Declaration;

    [Fact]
    public void APlateau_IsWrittenWithEverySettingItTakes_AndReadBackAsTheScheduleTheSameSettingsMakeAsCode()
    {
        var declaration = Declared(network => network.Plateau(patience: 2, factor: 0.5, minimum: 0.001));
        var written = JsonNode.Parse(declaration.ToJson())!["declaration"]!.AsArray().Single(each => (string)each!["step"]! == "learn.network")!.AsObject();
        var step = Assert.IsType<LearnNetworkStep>(PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn().WithNetworks()).Learner);
        var schedule = Assert.IsType<TrainingPlateau>(step.Schedule.Scheduled());

        Assert.Equal("""{"kind":"trainingPlateau","patience":2,"factor":0.5,"threshold":0.0001,"cooldown":0,"minimum":0.001}""", written["schedule"]!.ToJsonString());
        Assert.Equal((2, 0.5, 0.0001, 0, 0.001), (schedule.Patience, schedule.Factor, schedule.Threshold, schedule.Cooldown, schedule.Minimum));
    }

    [Fact]
    public void APlateauWithOnlyTheDefaults_IsPyTorchsOwn()
    {
        var schedule = Assert.IsType<TrainingPlateau>(
            Assert.IsType<LearnNetworkStep>(Declared(network => network.Plateau()).Learner).Schedule.Scheduled());

        Assert.Equal((10, 0.1, 1e-4, 0, 0.0), (schedule.Patience, schedule.Factor, schedule.Threshold, schedule.Cooldown, schedule.Minimum));
    }

    [Fact]
    public void ALaterScheduleReplacesAPlateau_ForADeclarationHoldsOne()
    {
        var step = Assert.IsType<LearnNetworkStep>(Declared(network => network.Plateau(3).CosineDecay(epochs: 5)).Learner);

        Assert.IsType<CosineDecay>(step.Schedule.Scheduled());
    }

    [Fact]
    public void APatienceBelowNought_IsRefusedWhereTheScheduleIsMade()
    {
        var declaration = Declared(network => network.Plateau(patience: -1));

        Assert.ThrowsAny<Exception>(() => Assert.IsType<LearnNetworkStep>(declaration.Learner).Schedule.Scheduled());
    }

    [Fact]
    public void AFactorThatDoesNotLowerTheRate_IsRefusedWhereTheScheduleIsMade()
    {
        var declaration = Declared(network => network.Plateau(factor: 1.5));

        Assert.ThrowsAny<Exception>(() => Assert.IsType<LearnNetworkStep>(declaration.Learner).Schedule.Scheduled());
    }
}
