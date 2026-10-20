// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A network that is not a straight line, declared: every layer may be given a name and the names of the steps it takes, three
/// more words join and cut steps, and the declaration lowers onto the stack the same words written as code give — written to the
/// file only when something is said, so a declaration that says none of it is the file it always was.
/// </summary>
public class DeclaredBranchTests
{
    private static PartKind Word(string name) => NetworkWords.Layers.Single(kind => kind.Name == name);

    private static LearnNetworkStep Learner(Func<NetworkDeclaration, NetworkDeclaration> layers) =>
        Assert.IsType<LearnNetworkStep>(DeclaredNetworkTests.Passengers()
            .WithTorch(network => layers(network).Adam(0.01).BinaryCrossEntropy().Run(20260929, epochs: 2))
            .Build().Declaration.Learner);

    [Theory]
    [InlineData("dense")]
    [InlineData("relu")]
    [InlineData("batchNorm")]
    [InlineData("flatten")]
    [InlineData("tokenEmbedding")]
    [InlineData("add")]
    [InlineData("concatenate")]
    [InlineData("columns")]
    public void EveryWordOfALayer_MayBeNamedAndMayTakeSteps_AsSettingsLeavingOutMeansNoneOf(string name)
    {
        var kind = Word(name);

        Assert.Equal(string.Empty, kind.Declared().Text("name"));
        Assert.Equal(string.Empty, kind.Declared().Text("takes"));
        Assert.Empty(kind.Settings.Single(setting => setting.Key == "name").RequiredKeys);
        Assert.Empty(kind.Settings.Single(setting => setting.Key == "takes").RequiredKeys);
    }

    [Fact]
    public void TheEmbeddingWord_ReadsTheRowAsThePipelineHandsItOver_SoItIsNeverNamedNorGivenSteps()
    {
        Assert.DoesNotContain(Word("embedding").Settings, setting => setting.Key is "name" or "takes");
    }

    [Fact]
    public void TheColumnsWord_SaysWhereItsBlockStartsAndHowWideItIs()
    {
        Assert.Equal(["first", "count", "name", "takes"], Word("columns").Settings.Select(setting => setting.Key));
        Assert.Equal(["name", "takes"], Word("add").Settings.Select(setting => setting.Key));
    }

    [Fact]
    public void AResidualDeclaration_LowersOntoTheStackTheSameWordsWrittenAsCodeGive_NumberForNumber()
    {
        var step = Learner(network => network.Dense(6).Named("first").Relu().Dense(6).Named("second").Add("input", "second"));
        var declared = step.Layers.Described().Lower(new Shape(6), new RandomStream(7));
        var written = new Sequential().Dense(6).Named("first").Relu().Dense(6).Named("second").Add("input", "second").Lower(new Shape(6), new RandomStream(7));

        Assert.Equal(written.Slots().Select(slot => slot.Path), declared.Slots().Select(slot => slot.Path));
        Assert.Equal(["first.weight", "first.bias", "second.weight", "second.bias"], declared.Slots().Select(slot => slot.Path));
        Assert.Equal(
            written.Slots().Select(slot => slot.Slot.Value.Values.ToArray()),
            declared.Slots().Select(slot => slot.Slot.Value.Values.ToArray()));
    }

    [Fact]
    public void ADenseBlockAndATwoInputNetwork_AreDeclaredInTheSameWords()
    {
        var dense = Learner(network => network
            .Dense(4).Named("a").Relu().Named("ra")
            .Concatenate("input", "ra").Named("joined")
            .Dense(2).Named("out")).Layers.Described().Lower(new Shape(5), new RandomStream(1));
        var halves = Learner(network => network
            .Columns(0, 3).Named("left").Takes("input")
            .Dense(4).Named("leftOut")
            .Columns(3, 3).Named("right").Takes("input")
            .Dense(4).Named("rightOut")
            .Add("leftOut", "rightOut")).Layers.Described().Lower(new Shape(6), new RandomStream(1));

        Assert.Equal(new Shape(3, 2), dense.Forward(Tensor.Zeros(new Shape(3, 5)), Pass.Evaluation(new CpuBackend())).Shape);
        Assert.Equal(new Shape(3, 4), halves.Forward(Tensor.Zeros(new Shape(3, 6)), Pass.Evaluation(new CpuBackend())).Shape);
    }

    [Fact]
    public void TheDoors_SayTheNameAndTheStepsOfTheLayerWrittenLast_AndWriteThemOnlyWhenSaid()
    {
        var said = Learner(network => network.Dense(4).Named("hidden").Relu().Takes("hidden").Dense(1));
        var plain = Learner(network => network.Dense(4).Relu().Dense(1));
        var text = DeclaredNetworkTests.Passengers().WithTorch(network => network.Dense(4).Relu().Dense(1).Adam(0.01).MeanSquaredError().Run(1)).Build().Declaration.ToJson();

        Assert.Equal("hidden", said.Layers[0].Text("name"));
        Assert.Equal("hidden", said.Layers[1].Text("takes"));
        Assert.Equal(string.Empty, plain.Layers[0].Text("name"));
        var layers = System.Text.Json.JsonDocument.Parse(text).RootElement.GetProperty("declaration").EnumerateArray()
            .Single(step => step.GetProperty("step").GetString() == "learn.network").GetProperty("layers").EnumerateArray().ToArray();

        Assert.All(layers, layer => Assert.False(layer.TryGetProperty("name", out _) || layer.TryGetProperty("takes", out _)));
    }

    [Fact]
    public void ADeclarationOfSteps_GoesThroughTheFileAndComesBackTheSame_AndTheStepsStayInTheOrderTheyWereTaken()
    {
        var declaration = DeclaredNetworkTests.Passengers()
            .WithTorch(network => network.Dense(4).Named("a").Relu().Named("ra").Concatenate("ra", "input").Named("joined").Dense(1).Adam(0.01).BinaryCrossEntropy().Run(1))
            .Build().Declaration;
        var again = PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn().WithNetworks());

        Assert.Equal(declaration, again);
        Assert.Equal(declaration.ToJson(), again.ToJson());
        Assert.Contains("\"takes\": \"ra,input\"", declaration.ToJson(), StringComparison.Ordinal);
    }

    [Fact]
    public void ADeclaredResidualNetwork_IsTrainedBehindItsPipeline_AndLearns()
    {
        var trained = DeclaredNetworkTests.Passengers()
            .WithTorch(network => network.Dense(4).Named("a").Relu().Dense(4).Named("b").Add("a", "b").Dense(1).Adam(0.01).BinaryCrossEntropy().Run(20260929, epochs: 3))
            .Build().Train();

        Assert.Equal(3, trained.History!.Epochs.Count);
        Assert.Contains("\"add\"", trained.ToJson(), StringComparison.Ordinal);
    }

    [Fact]
    public void ANameOrStepsForALayerThatTakesNone_OrBeforeAnyLayer_AreRefusedByName()
    {
        Assert.Contains("Named", Assert.Throws<InvalidOperationException>(() => Learner(network => network.Named("early").Dense(1))).Message, StringComparison.Ordinal);
        Assert.Throws<InvalidOperationException>(() => Learner(network => network.Takes("input").Dense(1)));
        Assert.Throws<ArgumentException>(() => Learner(network => network.Dense(1).Named("a.b")));
        Assert.Throws<ArgumentException>(() => Learner(network => network.Dense(1).Named("a,b")));
        Assert.Throws<ArgumentException>(() => Learner(network => network.Dense(1).Named("3")));
        Assert.Throws<ArgumentException>(() => Learner(network => network.Dense(1).Named("input")));
        Assert.Throws<ArgumentException>(() => Learner(network => network.Dense(1).Takes()));
        Assert.Throws<ArgumentException>(() => Learner(network => network.Dense(1).Takes("a,b")));
        Assert.Throws<ArgumentException>(() => Learner(network => network.Add()));
        Assert.Throws<ArgumentException>(() => Learner(network => network.Concatenate("input", "input,x")));
        Assert.Throws<ArgumentOutOfRangeException>(() => Learner(network => network.Columns(-1, 2)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Learner(network => network.Columns(0, 0)));
    }

    [Fact]
    public void AJoinWrittenByHandWithNoSteps_IsRefusedWhereItIsLowered_NamingTheWord()
    {
        var join = new[] { Word("add").Declared() };

        Assert.Contains("add", Assert.Throws<ArgumentException>(() => join.Described()).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheStepsOfAJoin_AreTheTakesOfItsWord_SoAnAddOfOneStepIsRefusedWhereItIsLowered()
    {
        var one = new[] { Word("dense").Declared().With("units", PartValue.Of(3)).With("name", PartValue.Of("a")), Word("add").Declared().With("takes", PartValue.Of("a")) };

        Assert.Throws<ArgumentException>(() => one.Described().Lower(new Shape(3), new RandomStream(1)));
    }
}
