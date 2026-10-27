// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A network that is not a straight line, written in Keras's words: a word is a step, it may be named, it may take other steps than
/// the one before it, and words join what several steps made. Lowered, it is the stack a network written as code is, layer for layer
/// and bit for bit, each layer starting from the draws of the place it stands at, so naming a step moves no layer's start.
/// </summary>
public class SequentialBranchTests
{
    private static readonly Shape Six = new(6);

    [Fact]
    public void AResidualBlockInWords_IsTheStackWrittenAsCode_LayerForLayerAndBitForBit()
    {
        var described = new Sequential().Dense(6).Named("first").Relu().Dense(6).Named("second").Add("input", "second").Lower(Six, new RandomStream(7));
        var stream = new RandomStream(7);
        var written = new LayerStack(
            [
                new Step(new Dense(6, 6, stream.Draw("initialise:0", 0, 0)), "first"),
                new Step(new Relu()),
                new Step(new Dense(6, 6, stream.Draw("initialise:2", 0, 0)), "second"),
                new Step(new Sum(), Takes: ["input", "second"]),
            ]);
        var rows = Tensor.From(new Shape(3, 6), [.. Enumerable.Range(0, 18).Select(at => (float)(Math.Sin(at) / 2))]);
        var backend = new CpuBackend();

        Assert.Equal(written.Slots().Select(slot => slot.Path), described.Slots().Select(slot => slot.Path));
        Assert.Equal(["first.weight", "first.bias", "second.weight", "second.bias"], described.Slots().Select(slot => slot.Path));
        Assert.Equal(
            written.Slots().Select(slot => slot.Slot.Value.Values.ToArray()),
            described.Slots().Select(slot => slot.Slot.Value.Values.ToArray()));
        Assert.Equal(written.Forward(rows, Pass.Evaluation(backend)).Values.ToArray(), described.Forward(rows, Pass.Evaluation(backend)).Values.ToArray());
    }

    [Fact]
    public void AGateInWords_IsTheStackWrittenAsCode_LayerForLayerAndBitForBit()
    {
        var described = new Sequential().Dense(6).Named("gate").Relu().Named("bent").Dense(6).Named("value").Multiply("bent", "value").Lower(Six, new RandomStream(7));
        var stream = new RandomStream(7);
        var written = new LayerStack(
            [
                new Step(new Dense(6, 6, stream.Draw("initialise:0", 0, 0)), "gate"),
                new Step(new Relu(), "bent"),
                new Step(new Dense(6, 6, stream.Draw("initialise:2", 0, 0)), "value"),
                new Step(new Product(), Takes: ["bent", "value"]),
            ]);
        var rows = Tensor.From(new Shape(3, 6), [.. Enumerable.Range(0, 18).Select(at => (float)(Math.Sin(at) / 2))]);
        var backend = new CpuBackend();

        Assert.Equal(written.Slots().Select(slot => slot.Path), described.Slots().Select(slot => slot.Path));
        Assert.Equal(
            written.Slots().Select(slot => slot.Slot.Value.Values.ToArray()),
            described.Slots().Select(slot => slot.Slot.Value.Values.ToArray()));
        Assert.Equal(written.Forward(rows, Pass.Evaluation(backend)).Values.ToArray(), described.Forward(rows, Pass.Evaluation(backend)).Values.ToArray());
    }

    [Fact]
    public void NamingAStep_MovesNoLayersStart()
    {
        var named = new Sequential().Dense(4).Named("hidden").Relu().Dense(1).Lower(Six, new RandomStream(3));
        var unnamed = new Sequential().Dense(4).Relu().Dense(1).Lower(Six, new RandomStream(3));

        Assert.Equal(
            unnamed.Slots().Select(slot => slot.Slot.Value.Values.ToArray()),
            named.Slots().Select(slot => slot.Slot.Value.Values.ToArray()));
    }

    [Fact]
    public void ADenseBlock_ReadsEverythingMadeBeforeIt_AndEveryWordKnowsHowWideWhatItTakesIs()
    {
        var described = new Sequential()
            .Dense(4).Named("a").Relu().Named("ra")
            .Concatenate("input", "ra").Named("c1")
            .Dense(4).Named("b").Relu().Named("rb")
            .Concatenate("input", "ra", "rb").Named("c2")
            .Dense(2).Named("out")
            .Lower(new Shape(5), new RandomStream(1));

        Assert.Equal(new Shape(5, 4), described.Steps.Single(step => step.Name == "a").Layer.Slots().First().Slot.Value.Shape);
        Assert.Equal(new Shape(9, 4), described.Steps.Single(step => step.Name == "b").Layer.Slots().First().Slot.Value.Shape);
        Assert.Equal(new Shape(13, 2), described.Steps.Single(step => step.Name == "out").Layer.Slots().First().Slot.Value.Shape);
        Assert.Equal(new Shape(3, 2), described.Forward(Tensor.Zeros(new Shape(3, 5)), Pass.Evaluation(new CpuBackend())).Shape);
    }

    [Fact]
    public void ANetworkOfTwoInputs_ReadsEachFromTheColumnsOfTheOneRow()
    {
        var described = new Sequential()
            .Columns(0, 3).Named("leftColumns").Takes("input")
            .Dense(4).Named("left")
            .Columns(3, 3).Named("rightColumns").Takes("input")
            .Dense(4).Named("right")
            .Add("left", "right")
            .Lower(Six, new RandomStream(2));

        Assert.Equal(["left.weight", "left.bias", "right.weight", "right.bias"], described.Slots().Select(slot => slot.Path));
        Assert.Equal(new Shape(3, 4), described.Forward(Tensor.Zeros(new Shape(3, 6)), Pass.Evaluation(new CpuBackend())).Shape);
    }

    [Fact]
    public void AWordThatTakesAStepThatIsNotThere_IsRefused_NamingTheWord()
    {
        var refused = Assert.Throws<ArgumentException>(() => new Sequential().Dense(4).Relu().Takes("missing").Lower(Six, new RandomStream(1)));

        Assert.Contains("word 2", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'missing'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void StepsAddedThatAreNotOfOneShape_AreRefused_NamingBoth()
    {
        var refused = Assert.Throws<ArgumentException>(() => new Sequential().Dense(4).Named("wide").Add("input", "wide").Lower(Six, new RandomStream(1)));

        Assert.Contains("6", refused.Message, StringComparison.Ordinal);
        Assert.Contains("4", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void StepsLaidSideBySide_MustBeAlikeButInTheLastAxis()
    {
        var images = new Shape(4, 4, 2);

        Assert.Throws<ArgumentException>(() => new Sequential().Flatten().Named("flat").Concatenate("input", "flat").Lower(images, new RandomStream(1)));
        var joined = new Sequential().Columns(0, 1).Takes("input").Named("x").Concatenate("input", "x", "x").Lower(new Shape(2, 2), new RandomStream(1));

        Assert.Equal([1, 2, 4], joined.Forward(Tensor.Zeros(new Shape(1, 2, 2)), Pass.Evaluation(new CpuBackend())).Shape.Axes.ToArray());
    }

    [Fact]
    public void AColumnBlockThatReachesBeyondTheRow_IsRefused_NamingTheWord()
    {
        var refused = Assert.Throws<ArgumentException>(() => new Sequential().Columns(4, 3).Lower(Six, new RandomStream(1)));

        Assert.Contains("word 1", refused.Message, StringComparison.Ordinal);
        Assert.Contains("columns", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnAddOrAConcatenationOfOneStep_IsRefused_ForItJoinsNothing()
    {
        Assert.Throws<ArgumentException>(() => new Sequential().Add("input").Lower(Six, new RandomStream(1)));
        Assert.Throws<ArgumentException>(() => new Sequential().Concatenate("input").Lower(Six, new RandomStream(1)));
        Assert.Throws<ArgumentException>(() => new Sequential().Add());
        Assert.Throws<ArgumentException>(() => new Sequential().Dense(1).Takes());
        Assert.Throws<ArgumentNullException>(() => new Sequential().Dense(1).Takes(null!));
    }

    [Fact]
    public void ANameOrATakesBeforeAnyWord_IsRefused_ForThereIsNoStepToGiveThemTo()
    {
        Assert.Throws<InvalidOperationException>(() => new Sequential().Named("early"));
        Assert.Throws<InvalidOperationException>(() => new Sequential().Takes("input"));
        Assert.Throws<ArgumentException>(() => new Sequential().Dense(1).Named("a.b"));
        Assert.Throws<ArgumentException>(() => new Sequential().Dense(1).Named("3"));
    }

    [Fact]
    public void ADescriptionWhoseStepsBreakTheRules_IsRefusedWhereItIsCompiled_WhenItsInputIsStated()
    {
        var dangling = new Sequential().Input(Six).Dense(4).Named("unused").Relu().Takes("input").Dense(1);

        Assert.Contains("'unused'", Assert.Throws<ArgumentException>(() => dangling.Compile(new Sgd(0.1), new MeanSquaredError())).Message, StringComparison.Ordinal);
        Assert.NotNull(new Sequential().Input(Six).Dense(1).Compile(new Sgd(0.1), new MeanSquaredError()));
    }

    [Fact]
    public void ABlock_IsALayerOfItsOwn_ItsNumbersFoundUnderItsNameAndThenTheNamesInsideIt_AndNamingItMovesNoStart()
    {
        var inner = new Sequential().Dense(6).Named("project").Relu();
        var named = new Sequential().Block(inner).Named("encoder").Dense(1).Lower(Six, new RandomStream(5));
        var unnamed = new Sequential().Block(inner).Dense(1).Lower(Six, new RandomStream(5));

        Assert.Equal(["encoder.project.weight", "encoder.project.bias", "1.weight", "1.bias"], named.Slots().Select(slot => slot.Path));
        Assert.Equal(["0.project.weight", "0.project.bias", "1.weight", "1.bias"], unnamed.Slots().Select(slot => slot.Path));
        Assert.Equal(
            unnamed.Slots().Select(slot => slot.Slot.Value.Values.ToArray()),
            named.Slots().Select(slot => slot.Slot.Value.Values.ToArray()));
    }

    [Fact]
    public void ABlockOfTheSameWordsAtTwoPlaces_StartsEachFromTheDrawsOfItsOwnPlace()
    {
        var inner = new Sequential().Dense(6).Relu();
        var lowered = new Sequential().Block(inner).Block(inner).Lower(Six, new RandomStream(5));
        var slots = lowered.Slots().Select(slot => slot.Slot.Value.Values.ToArray()).ToArray();

        Assert.NotEqual(slots[0], slots[2]);
    }

    [Fact]
    public void ABlockThatHoldsNoWord_OrCannotTakeWhatReachesIt_IsRefused()
    {
        Assert.Throws<InvalidOperationException>(() => new Sequential().Block(new Sequential()));
        Assert.Throws<ArgumentNullException>(() => new Sequential().Block(null!));
        Assert.Contains("block", Assert.Throws<ArgumentException>(() => new Sequential().Block(new Sequential().Columns(0, 9)).Lower(Six, new RandomStream(1))).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ALastActivationTheLossApplies_IsLiftedOutOfABranchingDescriptionToo()
    {
        var compiled = new Sequential().Input(Six).Dense(1).Named("a").Sigmoid().Compile(new Sgd(0.1), new BinaryCrossEntropy());
        var data = new TrainingData(Tensor.Zeros(new Shape(4, 6)), Tensor.Zeros(new Shape(4, 1)));

        compiled.Fit(data, null, new FitOptions(seed: 1) { Epochs = 1 });

        Assert.Equal(["a.weight", "a.bias"], compiled.Network.Slots().Select(slot => slot.Path));
    }

    [Fact]
    public void ADescriptionLoweredTwice_GivesTheSameBits_AndOneThatWentOnBeingWritten_IsNotChangedByIt()
    {
        var description = new Sequential().Dense(4).Named("a").Relu();
        var first = description.Lower(Six, new RandomStream(9));

        description.Dense(1);

        var again = new Sequential().Dense(4).Named("a").Relu().Lower(Six, new RandomStream(9));

        Assert.Equal(first.Slots().Select(slot => slot.Slot.Value.Values.ToArray()), again.Slots().Select(slot => slot.Slot.Value.Values.ToArray()));
    }
}
