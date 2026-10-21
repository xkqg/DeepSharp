// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A description says why it cannot be lowered where it cannot: the word, and the shape of what reaches it. The words that take the work of
/// several steps — joins, blocks, attention, recurrent layers, a layer run over every step — each refuse what they cannot take, by name,
/// and a rule goes only on a word that makes a layer of its own.
/// </summary>
public class WordRefusalTests
{
    private static RandomStream Stream => new(1);

    private static string Refused(Func<LayerStack> lower) => Assert.Throws<ArgumentException>(lower).Message;

    [Fact]
    public void Attention_ReadingSomethingThatIsNotASeries_IsRefusedWhereItIsLowered() =>
        Assert.Contains("takes each example as a series of steps and channels", Refused(() => new Sequential().Attention(2).Lower(new Shape(8), Stream)), StringComparison.Ordinal);

    [Fact]
    public void AttentionOfHeadsThatDoNotDivideTheStep_IsRefusedWhereItIsLowered() =>
        Assert.Contains("splits a step of 8 channels between 3 heads", Refused(() => new Sequential().MultiHeadAttention(3).Lower(new Shape(5, 8), Stream)), StringComparison.Ordinal);

    [Fact]
    public void Attention_GivenFiveStepsOrMultiHeadAttentionGivenThree_IsRefusedByTheNumberItWasGiven()
    {
        var five = Refused(() => Fan(5).Attention(2).Takes("a", "b", "c", "d", "e").Lower(new Shape(5, 4), Stream));
        var three = Refused(() => Fan(3).MultiHeadAttention(2).Takes("a", "b", "c").Lower(new Shape(5, 4), Stream));

        Assert.Contains("was given the work of 5 steps", five, StringComparison.Ordinal);
        Assert.Contains("was given the work of 3 steps", three, StringComparison.Ordinal);
    }

    [Fact]
    public void AttentionOverKeysOfAnotherWidthThanTheQueries_IsRefused_NamingTheThreeShapes()
    {
        var description = new Sequential().Dense(4).Named("q").Dense(6).Named("k").Takes(Step.Input).Dense(6).Named("v").Takes("k").Attention(2).Takes("q", "k", "v");

        Assert.Contains("keys of as wide a step as its queries", Refused(() => description.Lower(new Shape(5, 3), Stream)), StringComparison.Ordinal);
    }

    [Fact]
    public void AttentionGivenQueriesKeysAndValuesOfAnotherWidth_IsRefusedAtTheQueriesWhenTheyAreNotASeries()
    {
        var description = new Sequential().Dense(4).Named("q").Flatten().Named("k").Takes("q").Dense(4).Named("v").Takes("q").Attention(2).Takes("q", "k", "v");

        Assert.Contains("takes each example as a series", Refused(() => description.Lower(new Shape(5, 3), Stream)), StringComparison.Ordinal);
    }

    [Fact]
    public void AMaskOfAnotherLengthThanTheSteps_IsRefusedByAttentionAndByARecurrentLayer()
    {
        var attention = Refused(() => new Sequential().Attention(2).Takes(Step.Input, Step.Mask).Lower(new Shape(5, 4), new Shape(4), Stream));
        var recurrent = Refused(() => new Sequential().Lstm(3).Takes(Step.Input, Step.Mask).Lower(new Shape(5, 4), new Shape(4), Stream));

        Assert.Contains("a one or a nought for each of the 5 steps", attention, StringComparison.Ordinal);
        Assert.Contains("a one or a nought for each of the 5 steps", recurrent, StringComparison.Ordinal);
    }

    [Fact]
    public void ARecurrentLayerGivenThreeSteps_IsRefused_ForItTakesASeriesAndItsMask() =>
        Assert.Contains("was given the work of 3 steps", Refused(() => Fan(3).Gru(3).Takes("a", "b", "c").Lower(new Shape(5, 4), Stream)), StringComparison.Ordinal);

    [Fact]
    public void ThePositionsOfSomethingThatIsNotASeries_AndTheMaskOfSomethingThatIsNotARowOfIds_AreRefusedWhereTheyAreLowered()
    {
        Assert.Contains("takes each example as a series of steps and channels", Refused(() => new Sequential().SinePositions().Lower(new Shape(8), Stream)), StringComparison.Ordinal);
        Assert.Contains("takes each example as a row of ids", Refused(() => new Sequential().Unpadded().Lower(new Shape(2, 3), Stream)), StringComparison.Ordinal);
    }

    [Fact]
    public void AJoinGivenOneStep_IsRefused_ForItTakesTheWorkOfTwoOrMore() =>
        Assert.Contains("takes the work of two steps or more, and was given 1", Refused(() => new Sequential().Dense(2).Named("a").Add("a").Lower(new Shape(3), Stream)), StringComparison.Ordinal);

    [Fact]
    public void ABlockGivenThreeSteps_OrASeriesItsLayersCannotTake_IsRefusedByName()
    {
        var three = Refused(() => Fan(3).Block(new Sequential().Dense(2)).Takes("a", "b", "c").Lower(new Shape(5, 4), Stream));
        var inside = Refused(() => new Sequential().Block(new Sequential().MultiHeadAttention(3)).Lower(new Shape(5, 8), Stream));

        Assert.Contains("block, takes a series and, besides, the mask of its steps, and was given the work of 3 steps", three, StringComparison.Ordinal);
        Assert.Contains("block, ", inside, StringComparison.Ordinal);
        Assert.Contains("3 heads", inside, StringComparison.Ordinal);
    }

    [Fact]
    public void ALayerRunOverEveryStep_IsRefusedForWhatIsNotASeries_ForWhatItsLayersCannotTake_AndForMoreThanOneStep()
    {
        var flat = Refused(() => new Sequential().EveryStep(new Sequential().Dense(2)).Lower(new Shape(8), Stream));
        var inside = Refused(() => new Sequential().EveryStep(new Sequential().Attention(2)).Lower(new Shape(5, 4), Stream));
        var two = Refused(() => Fan(2).EveryStep(new Sequential().Dense(2)).Takes("a", "b").Lower(new Shape(5, 4), Stream));

        Assert.Contains("everyStep, takes each example as a series of steps", flat, StringComparison.Ordinal);
        Assert.Contains("everyStep, ", inside, StringComparison.Ordinal);
        Assert.Contains("takes the work of one step, and was given 2", two, StringComparison.Ordinal);
    }

    [Fact]
    public void ALayerThatWorksOnOneStep_GivenTwo_IsRefusedByTheNumberItWasGiven() =>
        Assert.Contains("takes the work of one step, and was given 2", Refused(() => Fan(2).Dense(3).Takes("a", "b").Lower(new Shape(4), Stream)), StringComparison.Ordinal);

    [Fact]
    public void AnAttentionLayerHandedNoStepsOrFive_IsRefusedWhenTheStackIsMade()
    {
        var steps = Enumerable.Range(0, 5).Select(at => new Step(new Dense(4, 4, new Beginning(Stream.Draw("a", at, 0))), ((char)('a' + at)).ToString(), [Step.Input])).ToList();
        var none = Assert.Throws<ArgumentException>(() => new LayerStack([.. steps, new Step(new Attention(new Attending(2)), "x", [])]));
        var five = Assert.Throws<ArgumentException>(() => new LayerStack([.. steps, new Step(new Attention(new Attending(2)), "x", ["a", "b", "c", "d", "e"])]));

        Assert.Contains("takes the work of 0 steps", none.Message, StringComparison.Ordinal);
        Assert.Contains("takes the work of 5 steps", five.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ARuleOnAJoin_AABlockOrALayerRunOverEveryStep_IsRefusedWhereItIsSaid()
    {
        var join = Fan(2).Add("a", "b");
        var block = new Sequential().Block(new Sequential().Dense(2));
        var every = new Sequential().EveryStep(new Sequential().Dense(2));

        Assert.Contains("makes no layer of its own", Assert.Throws<InvalidOperationException>(() => join.Held()).Message, StringComparison.Ordinal);
        Assert.Contains("makes no layer of its own", Assert.Throws<InvalidOperationException>(() => block.RateShare(0.5)).Message, StringComparison.Ordinal);
        Assert.Contains("makes no layer of its own", Assert.Throws<InvalidOperationException>(() => every.Ruled(default)).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ARuleOnAttentionThatTakesAMask_StillTakesIt_AndHoldsTheNumbers()
    {
        var stack = new Sequential().MultiHeadAttention(2).Takes(Step.Input, Step.Mask).Held().Lower(new Shape(5, 4), new Shape(5), Stream);

        Assert.All(stack.Parameters(), parameter => Assert.True(parameter.Rules.Held));
    }

    private static Sequential Fan(int count)
    {
        var description = new Sequential();

        for (var at = 0; at < count; at++)
        {
            description.Dense(4).Named(((char)('a' + at)).ToString()).Takes(Step.Input);
        }

        return description;
    }
}
