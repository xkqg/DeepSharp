// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A gated feed-forward block: the hidden layer split into a gate and a value, the gate bending and then weighing the
/// value, as the newer transformer recipes write it (Shazeer, "GLU Variants Improve Transformer") — PyTorch's own
/// <c>TransformerEncoderLayer</c> has no such form, so the gate and value projections are named beyond its own state.
/// </summary>
public class GatedFeedForwardTests
{
    private static readonly Shape Series = new(3, 2);

    [Fact]
    public void AGateBesideAGelu_IsRefused_ForBothSayTheFeedForwardsBend()
    {
        var shape = new EncoderShape(1, 2) { Gelu = true, Gate = FeedForwardGate.ReGlu };

        var block = Assert.Throws<ArgumentException>(() => new Sequential().EncoderBlock(shape));
        var words = Assert.Throws<ArgumentException>(() => Sequential.EncoderWords(shape, 2));

        Assert.Contains("one or the other", block.Message, StringComparison.Ordinal);
        Assert.Contains("one or the other", words.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(FeedForwardGate.ReGlu)]
    [InlineData(FeedForwardGate.GeGlu)]
    [InlineData(FeedForwardGate.SwiGlu)]
    public void AGatedFeedForward_NamesAGateAndAValueProjection_MultipliedIntoLinear1(FeedForwardGate gate)
    {
        var shape = new EncoderShape(1, 2) { Gate = gate };
        var described = Sequential.EncoderWords(shape, 2).Lower(Series, new RandomStream(7));

        Assert.Contains("linear1_gate", described.Steps.Select(step => step.Name));
        Assert.Contains("linear1_value", described.Steps.Select(step => step.Name));
        Assert.Contains("linear1", described.Steps.Select(step => step.Name));
        Assert.IsType<Product>(described.Steps.Single(step => step.Name == "linear1").Layer);
    }

    [Fact]
    public void AGatedFeedForward_IsTheStackWrittenAsCode_LayerForLayerAndBitForBit()
    {
        var shape = new EncoderShape(1, 2) { Gate = FeedForwardGate.ReGlu };
        var described = Sequential.EncoderWords(shape, 2).Lower(Series, new RandomStream(7));

        var byHand = new Sequential()
            .MultiHeadAttention(new Attending(1)).Named("self_attn").Takes("input")
            .Add("input", "self_attn").Named("sum1")
            .LayerNorm().Named("norm1")
            .Dense(2).Named("linear1_gate_proj").Takes("norm1").Relu().Named("linear1_gate")
            .Dense(2).Named("linear1_value").Takes("norm1")
            .Multiply("linear1_gate", "linear1_value").Named("linear1")
            .Dense(2).Named("linear2")
            .Add("norm1", "linear2").Named("sum2")
            .LayerNorm().Named("norm2")
            .Lower(Series, new RandomStream(7));

        var rows = Tensor.From(new Shape(2, 3, 2), [.. Enumerable.Range(0, 12).Select(at => (float)(Math.Sin(at) / 2))]);
        var backend = new CpuBackend();

        Assert.Equal(byHand.Slots().Select(slot => slot.Path), described.Slots().Select(slot => slot.Path));
        Assert.Equal(
            byHand.Slots().Select(slot => slot.Slot.Value.Values.ToArray()),
            described.Slots().Select(slot => slot.Slot.Value.Values.ToArray()));
        Assert.Equal(
            byHand.Forward(rows, Pass.Evaluation(backend)).Values.ToArray(),
            described.Forward(rows, Pass.Evaluation(backend)).Values.ToArray());
    }

    [Fact]
    public void AnUngatedFeedForward_IsUnchanged_WhereNoGateIsSaid()
    {
        var shape = new EncoderShape(1, 2);
        var described = Sequential.EncoderWords(shape, 2).Lower(Series, new RandomStream(7));

        Assert.DoesNotContain("linear1_gate", described.Steps.Select(step => step.Name));
        Assert.DoesNotContain("linear1_value", described.Steps.Select(step => step.Name));
        Assert.Contains("linear1", described.Steps.Select(step => step.Name));
    }
}
