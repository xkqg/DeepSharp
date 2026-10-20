// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// Attention in words, and a block that is handed the mask of the padding as its second take: the steps inside it name the mask as
/// <c>input.mask</c>, which no name a person gives can be, so attention looks at no padding and a series of ids of any length is read as the
/// steps that are steps.
/// </summary>
public class BlockMaskTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static Sequential Text() =>
        new Sequential().TokenEmbedding(7, 8, keepSteps: true).Named("steps")
            .Unpadded().Named("mask").Takes("input")
            .Block(new Sequential().MultiHeadAttention(2).Named("self_attn").Takes(Step.Input, Step.Mask)).Named("encoder").Takes("steps", "mask");

    private static Tensor Ids(int count, params float[] ids) => Tensor.From(new Shape(count, ids.Length / count), ids);

    [Fact]
    public void PaddingAtTheEnd_ChangesNothingOfTheStepsThatTakePart_WhereTheBlockIsGivenTheMask()
    {
        var long5 = Text().Lower(new Shape(5), new RandomStream(3)).Predict(Ids(2, 3, 4, 0, 0, 0, 5, 2, 6, 0, 0), new MeanSquaredError(), Engine).Values.ToArray();
        var cut3 = Text().Lower(new Shape(3), new RandomStream(3)).Predict(Ids(2, 3, 4, 0, 5, 2, 6), new MeanSquaredError(), Engine).Values.ToArray();

        Assert.Equal(80, long5.Length);
        Assert.Equal(48, cut3.Length);
        Assert.All(cut3[..16].Zip(long5[..16], (one, other) => Math.Abs(one - other)), difference => Assert.InRange(difference, 0, 1e-6));
        Assert.All(cut3[24..48].Zip(long5[40..64], (one, other) => Math.Abs(one - other)), difference => Assert.InRange(difference, 0, 1e-6));
    }

    [Fact]
    public void ABlockThatNamesTheMask_IsRefusedWhereItIsGivenNone()
    {
        var withoutMask = new Sequential().TokenEmbedding(7, 8, keepSteps: true).Named("steps")
            .Block(new Sequential().MultiHeadAttention(2).Named("self_attn").Takes(Step.Input, Step.Mask)).Named("encoder").Takes("steps");

        var refused = Assert.Throws<ArgumentException>(() => withoutMask.Lower(new Shape(5), new RandomStream(1)));

        Assert.Contains(Step.Mask, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheMask_IsNoStepOfANetworkThatIsNotABlock()
    {
        var refused = Assert.Throws<ArgumentException>(() => new Sequential().TokenEmbedding(7, 8, keepSteps: true).MultiHeadAttention(2).Takes(Step.Input, Step.Mask).Lower(new Shape(5), new RandomStream(1)));

        Assert.Contains(Step.Mask, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ANameCannotBeTheMasksOrHoldADot()
    {
        Assert.Throws<ArgumentException>(() => new Sequential().Dense(2).Named(Step.Mask));
    }

    [Fact]
    public void AStackThatTakesAMask_IsWrittenDownAsTheEighthVersion_AndReadBackAsItWas()
    {
        var stack = Text().Lower(new Shape(5), new RandomStream(3));
        var written = Written(stack);
        var read = NetworkDocument.ReadNetwork(written, "network", NetworkCatalog.BuiltIn());

        Assert.Contains("input.mask", written, StringComparison.Ordinal);
        Assert.Equal(written, Written(read.Network));
        Assert.Equal(stack.Slots().Select(slot => slot.Path), read.Network.Slots().Select(slot => slot.Path));
        Assert.Contains("encoder.self_attn.in_proj_weight", stack.Slots().Select(slot => slot.Path));
    }

    [Fact]
    public void AStackThatTakesAMask_IsNotRunAsANetworkWithoutOne()
    {
        var block = (LayerStack)Text().Lower(new Shape(5), new RandomStream(3)).Layers[2];

        Assert.Throws<InvalidOperationException>(() => block.Forward(Tensor.Zeros(new Shape(2, 5, 8)), Pass.Evaluation(Engine)));
        Assert.Equal(new Shape(2, 5, 8), block.Forward(Tensor.Zeros(new Shape(2, 5, 8)), Tensor.Zeros(new Shape(2, 5)), Pass.Evaluation(Engine)).Shape);
    }

    private static string Written(Network network)
    {
        using var stream = new MemoryStream();

        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, network, new MeanSquaredError());
            writer.WriteEndObject();
        }

        return System.Text.Encoding.UTF8.GetString(stream.ToArray()).Replace("\r\n", "\n", StringComparison.Ordinal);
    }
}
