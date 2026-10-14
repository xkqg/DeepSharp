// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// The words of a description that put rules on the numbers of the layer the word before them makes — a penalty, a bound, a hold
/// — as Keras puts them on a layer it is handed: the rules on the weights or the bias, applied when the layer is made, and a
/// word for a layer without the number a rule is for refused when the description is lowered.
/// </summary>
public class RulesWordsTests
{
    [Fact]
    public void TheRulesWords_PutTheRulesOnTheLayerTheWordBeforeThemMakes_AndOnNoOther()
    {
        var stack = new Sequential()
            .Dense(4).Ruled(new SlotRules { L2 = 0.1, MaxNorm = 3 }, new SlotRules { L1 = 0.2 })
            .Relu()
            .Dense(2)
            .Dense(1).Held()
            .Lower(new Shape(3), new RandomStream(1));
        var dense = stack.Layers.OfType<Dense>().ToArray();

        Assert.Equal(new SlotRules { L2 = 0.1, MaxNorm = 3 }, dense[0].Weight.Rules);
        Assert.Equal(new SlotRules { L1 = 0.2 }, dense[0].Bias!.Rules);
        Assert.Equal(default, dense[1].Weight.Rules);
        Assert.Equal(default, dense[1].Bias!.Rules);
        Assert.Equal(new SlotRules { Held = true }, dense[2].Weight.Rules);
        Assert.Equal(new SlotRules { Held = true }, dense[2].Bias!.Rules);
    }

    [Fact]
    public void TheRulesWords_CanBeSaidInEitherOrder_AndAHeldLayerKeepsItsPenalty()
    {
        var stack = new Sequential()
            .Dense(2).Held().Ruled(new SlotRules { L2 = 0.1 })
            .Dense(1).Ruled(new SlotRules { L1 = 0.5 }).Held()
            .Lower(new Shape(3), new RandomStream(1));
        var dense = stack.Layers.OfType<Dense>().ToArray();

        Assert.Equal(new SlotRules { L2 = 0.1, Held = true }, dense[0].Weight.Rules);
        Assert.Equal(new SlotRules { Held = true }, dense[0].Bias!.Rules);
        Assert.Equal(new SlotRules { L1 = 0.5, Held = true }, dense[1].Weight.Rules);
    }

    [Fact]
    public void TheRulesWords_ReachANormalisationsScaleAndShift_AndAConvolutionsKernel()
    {
        var stack = new Sequential()
            .Conv2D(4, new Window(3, 3)).Ruled(new SlotRules { MaxNorm = 2 })
            .BatchNorm().Ruled(new SlotRules { L2 = 0.1 }, new SlotRules { L2 = 0.2 })
            .Flatten()
            .Dense(1)
            .Lower(new Shape(6, 6, 1), new RandomStream(1));

        Assert.Equal(2, stack.Layers.OfType<Conv2D>().Single().Weight.Rules.MaxNorm);
        Assert.Equal(0.1, stack.Layers.OfType<BatchNorm>().Single().Weight!.Rules.L2);
        Assert.Equal(0.2, stack.Layers.OfType<BatchNorm>().Single().Bias!.Rules.L2);
    }

    [Fact]
    public void ARuleForANumberTheLayerHasNot_IsRefusedWhenTheDescriptionIsLowered_NamingTheLayerAndTheNumber()
    {
        var withoutBias = new Sequential().Dense(2, new Beginning { WithoutBias = true }).Ruled(default, new SlotRules { L2 = 0.1 });
        var refused = Assert.Throws<ArgumentException>(() => withoutBias.Lower(new Shape(3), new RandomStream(1)));

        Assert.Contains("Dense", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'bias'", refused.Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentException>(() => new Sequential().Relu().Ruled(new SlotRules { L2 = 0.1 }).Lower(new Shape(3), new RandomStream(1)));
    }

    [Fact]
    public void AHeldWordWhoseLayerHoldsNoNumber_IsMadeAsItWas()
    {
        var stack = new Sequential().Dense(2).Relu().Held().Dense(1).Lower(new Shape(3), new RandomStream(1));

        Assert.IsType<Relu>(stack.Layers[1]);
    }

    [Fact]
    public void ARuleBeforeAnyWord_IsRefused()
    {
        Assert.Throws<InvalidOperationException>(() => new Sequential().Ruled(new SlotRules { L2 = 0.1 }));
        Assert.Throws<InvalidOperationException>(() => new Sequential().Held());
    }

    [Fact]
    public void ADescriptionWithRules_IsFittedUnderThem_AndWritesThemIntoTheFile()
    {
        var compiled = new Sequential().Dense(3).Ruled(new SlotRules { L2 = 0.05 }).Relu().Dense(1).Held()
            .Compile(new Sgd(0.1), new MeanSquaredError());
        var rows = new TrainingData(
            Tensor.From(new Shape(4, 2), [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f]),
            Tensor.From(new Shape(4, 1), [0.2f, 0.4f, 0.6f, 0.8f]));

        compiled.Fit(rows, validation: null, new FitOptions(seed: 3) { Epochs = 2, BatchSize = 2 });

        var dense = Assert.IsType<LayerStack>(compiled.Network).Layers.OfType<Dense>().ToArray();

        Assert.Equal(0.05, dense[0].Weight.Rules.L2);
        Assert.False(dense[1].Weight.Learns);
    }
}
