// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json.Nodes;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// How a declared run's rate changes from epoch to epoch, and the norm its gradients are clipped to: two keys a network's
/// declaration gained after files were written without them. Left out, each means what a declaration meant before it had
/// it — the optimizer's own rate every epoch, no clipping — and is not written, so such a declaration writes the bytes it
/// always wrote; said, each is written, read back, and trains the network the same words written as code train.
/// </summary>
public class DeclaredTrainingControlsTests
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

    private static LearnNetworkStep Learner(Func<NetworkDeclaration, NetworkDeclaration> run) =>
        Assert.IsType<LearnNetworkStep>(Passengers()
            .WithTorch(network => run(network.Dense(1).Adam(0.01).BinaryCrossEntropy().Run(seed: 20260929, epochs: 3)))
            .Build().Declaration.Learner);

    [Fact]
    public void ADeclarationThatSaysNoScheduleAndNoClip_WritesNeitherKey_AndReadsAsTheOptimizersRateUnclipped()
    {
        var step = Learner(network => network);
        var written = JsonNode.Parse(Passengers().WithTorch(network => network.Dense(1).Adam(0.01).BinaryCrossEntropy().Run(1)).Build().Declaration.ToJson())!;
        var learner = written["declaration"]!.AsArray().Single(each => (string)each!["step"]! == "learn.network")!.AsObject();

        Assert.Equal("constant", step.Schedule.Kind);
        Assert.Equal(0, step.Clip);
        Assert.False(learner.ContainsKey("schedule"));
        Assert.False(learner.ContainsKey("clip"));
        Assert.Equal(["step", "layers", "optimizer", "loss", "stopping", "engine", "seed", "epochs", "batch"], learner.Select(key => key.Key));
    }

    [Fact]
    public void AScheduleAndAClip_AreWrittenUnderTheirKeys_AndReadBackTheSame()
    {
        var declaration = Passengers()
            .WithTorch(network => network.Dense(1).Adam(0.01).BinaryCrossEntropy().Run(1).WarmUp(epochs: 4, start: 0.25).ClipGradients(0.5))
            .Build().Declaration;
        var learner = JsonNode.Parse(declaration.ToJson())!["declaration"]!.AsArray().Single(each => (string)each!["step"]! == "learn.network")!.AsObject();
        var again = PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn().WithNetworks());
        var step = Assert.IsType<LearnNetworkStep>(again.Learner);

        Assert.Equal("""{"kind":"linearWarmup","epochs":4,"start":0.25}""", learner["schedule"]!.ToJsonString());
        Assert.Equal(0.5, (double)learner["clip"]!);
        Assert.Equal(declaration, again);
        Assert.Equal(declaration.ToJson(), again.ToJson());
        Assert.Equal("linearWarmup", step.Schedule.Kind);
        Assert.Equal(0.5, step.Clip);
    }

    [Fact]
    public void EveryScheduleDoor_WritesItsWord_AndEveryScheduleWord_BecomesTheScheduleTheCoreHas()
    {
        var constant = Learner(network => network).Schedule.Scheduled();
        var steps = Assert.IsType<StepDecay>(Learner(network => network.StepDecay(every: 3)).Schedule.Scheduled());
        var exponential = Assert.IsType<ExponentialDecay>(Learner(network => network.ExponentialDecay(0.9)).Schedule.Scheduled());
        var cosine = Assert.IsType<CosineDecay>(Learner(network => network.CosineDecay(epochs: 20, minimum: 0.001)).Schedule.Scheduled());
        var warm = Assert.IsType<LinearWarmup>(Learner(network => network.WarmUp(epochs: 5)).Schedule.Scheduled());

        Assert.IsType<ConstantRate>(constant);
        Assert.Equal(3, steps.Every);
        Assert.Equal(0.1, steps.Factor);
        Assert.Equal(0.9, exponential.Factor);
        Assert.Equal(20, cosine.Epochs);
        Assert.Equal(0.001, cosine.Minimum);
        Assert.Equal(5, warm.Epochs);
        Assert.Equal(1.0 / 3, warm.Start);
        Assert.Null(warm.Then);
        Assert.Equal(0.25, Assert.IsType<StepDecay>(Learner(network => network.StepDecay(every: 2, factor: 0.25)).Schedule.Scheduled()).Factor);
        Assert.Contains("'linearWarmup'", Assert.Throws<NotSupportedException>(() => new PartDeclaration("plateau", []).Scheduled()).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheScheduleWords_AreTheOnesTheNetworksFileWritesTheSchedulesUnder_EachSettingUnderTheSameKey()
    {
        foreach (var kind in NetworkWords.Schedules)
        {
            var schedule = (ISaved)kind.Declared().Scheduled();
            var settings = JsonNode.Parse(Settings(schedule))!.AsObject().Select(property => property.Key);

            Assert.Equal(kind.Name, schedule.Kind);
            Assert.Equal(kind.Settings.Select(setting => setting.Key).Order(StringComparer.Ordinal), settings.Order(StringComparer.Ordinal));
        }
    }

    [Fact]
    public void BothVocabularies_DeclareTheSchedulesAndTheClip_WithTheNumbersACallerNames()
    {
        var keras = Assert.IsType<LearnNetworkStep>(Passengers()
            .WithTensorflow(network => network.Dense(1).Adam().BinaryCrossEntropy().Run(1).CosineDecay(epochs: 10).ClipGradients(1))
            .Build().Declaration.Learner);

        Assert.Equal("cosineDecay", keras.Schedule.Kind);
        Assert.Equal(10, keras.Schedule.Whole("epochs"));
        Assert.Equal(0, keras.Schedule.Number("minimum"));
        Assert.Equal(1, keras.Clip);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(double.NaN)]
    public void AClipToANormOfNothing_OrBelowIt_IsRefusedAtTheDoor_AndAStepWrittenByHandIsHeldToTheKeysRule(double norm)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Learner(network => network.ClipGradients(norm)));

        var step = Learner(network => network);

        Assert.Equal(0, (step with { Clip = 0 }).Clip);
        Assert.Throws<ArgumentOutOfRangeException>(() => step with { Clip = -1 });
        Assert.Throws<ArgumentException>(() => step with { Schedule = new PartDeclaration("plateau", []) });
        Assert.NotEqual(step, step with { Clip = 0.5 });
        Assert.NotEqual(step, step with { Schedule = NetworkWords.Schedules.Single(kind => kind.Name == "linearWarmup").Declared() });
        Assert.NotEqual(step.GetHashCode(), (step with { Clip = 0.5 }).GetHashCode());
    }

    [Fact]
    public void TrainingADeclaredRun_WarmsItsRateUpAndClipsItsGradients_AsTheSameWordsWrittenAsCodeDo_NumberForNumber()
    {
        var declared = Passengers()
            .WithTorch(network => network
                .Dense(4).Relu().Dense(1)
                .Adam(0.01)
                .BinaryCrossEntropy()
                .Run(seed: 20260929, epochs: 4)
                .WarmUp(epochs: 2, start: 0.5)
                .ClipGradients(0.1))
            .Build()
            .Train();
        var written = new Sequential().Dense(4).Relu().Dense(1)
            .Compile(new Adam(0.01), new BinaryCrossEntropy(), new LinearWarmup(2, 0.5))
            .Fit(Passengers().Build().RunFor(Needs.OneScale), new FitOptions(20260929) { Epochs = 4, GradientClip = new GradientClip(0.1) });
        var unclipped = new Sequential().Dense(4).Relu().Dense(1)
            .Compile(new Adam(0.01), new BinaryCrossEntropy(), new LinearWarmup(2, 0.5))
            .Fit(Passengers().Build().RunFor(Needs.OneScale), new FitOptions(20260929) { Epochs = 4 });

        Assert.Equal([0.005, 0.0075, 0.01, 0.01], declared.History!.Epochs.Select(epoch => epoch.LearningRate), new Near());
        Assert.Equal(written.History!.Epochs, declared.History.Epochs);
        Assert.Equal(
            written.Network.Parameters().Select(parameter => parameter.Value.Values.ToArray()),
            declared.Network.Parameters().Select(parameter => parameter.Value.Values.ToArray()));
        Assert.NotEqual(unclipped.History!.Epochs, declared.History.Epochs);
    }

    [Fact]
    public void TheTrainDoorWithACallbackAndAToken_HandsOverEveryEpochInOrder_AndTrainsTheNetworkTheDoorWithoutTrains()
    {
        var pipeline = Passengers().WithTorch(network => network.Dense(4).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy().Run(seed: 20260929, epochs: 3)).Build();
        var handed = new List<Epoch>();
        var prepared = new List<Epoch>();

        var watched = pipeline.Train(null, handed.Add, TestContext.Current.CancellationToken);
        var plain = pipeline.Train();
        var fromPrepared = pipeline.RunFor(Needs.OneScale).Train(new Engines(), prepared.Add, TestContext.Current.CancellationToken);

        Assert.Equal(plain.History!.Epochs, watched.History!.Epochs);
        Assert.Equal(watched.History.Epochs, handed);
        Assert.Equal(watched.History.Epochs, prepared);
        Assert.Equal(plain.History.Epochs, fromPrepared.History!.Epochs);
        Assert.Equal(
            plain.Network.Parameters().Select(parameter => parameter.Value.Values.ToArray()),
            watched.Network.Parameters().Select(parameter => parameter.Value.Values.ToArray()));
        Assert.NotNull(pipeline.Train(null, onEpoch: null, TestContext.Current.CancellationToken).History);
    }

    [Fact]
    public void ACancelledTrain_ThrowsTheTokensCancel_ThroughEitherDoor_AndReturnsNothing()
    {
        var pipeline = Passengers().WithTorch(network => network.Dense(4).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy().Run(seed: 20260929, epochs: 5)).Build();
        using var fromPipeline = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        using var fromPrepared = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);
        var handed = 0;

        var one = Assert.Throws<OperationCanceledException>(() => pipeline.Train(null, epoch => Cancel(fromPipeline, epoch.Number == 1), fromPipeline.Token));
        var other = Assert.Throws<OperationCanceledException>(() => pipeline.RunFor(Needs.OneScale).Train(null, _ => Cancel(fromPrepared, ++handed == 1), fromPrepared.Token));

        Assert.Equal(fromPipeline.Token, one.CancellationToken);
        Assert.Equal(fromPrepared.Token, other.CancellationToken);
        Assert.Equal(1, handed);
    }

    [Fact]
    public void TheTrainDoorWithACallbackAndAToken_NeedsAPipeline()
    {
        Pipeline pipeline = null!;
        PreparedData prepared = null!;

        Assert.Throws<ArgumentNullException>(() => pipeline.Train(null, null, TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentNullException>(() => prepared.Train((Engines?)null, null, TestContext.Current.CancellationToken));
    }

    private static void Cancel(CancellationTokenSource source, bool now)
    {
        if (now)
        {
            source.Cancel();
        }
    }

    // Two rates that are the same to within the last bits a double's arithmetic leaves.
    private sealed class Near : IEqualityComparer<double>
    {
        public bool Equals(double one, double other) => Math.Abs(one - other) <= 1e-15;

        public int GetHashCode(double value) => 0;
    }

    private static string Settings(ISaved saved)
    {
        using var stream = new MemoryStream();

        using (var writer = new System.Text.Json.Utf8JsonWriter(stream))
        {
            writer.WriteStartObject();
            saved.WriteSettings(writer);
            writer.WriteEndObject();
        }

        return System.Text.Encoding.UTF8.GetString(stream.ToArray());
    }
}
