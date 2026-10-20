// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Import.Keras;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Import;

/// <summary>
/// Keras functional models that look across the steps of a series (Fixtures/keras-attention-fixtures.py made them), in the two files Keras
/// saves a model to: multi-head attention over the series itself, with each step looking at every step or only at the steps up to its own,
/// and an encoder block of attention, sums, normalisations and a feed-forward part. Keras keeps a projection as one kernel with an axis for
/// the heads, and names the four of them otherwise in the two files; they are put into the layer's one projection and its output layer.
/// </summary>
public class KerasAttentionFileTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static readonly JsonElement Keras = JsonDocument.Parse(File.ReadAllText(Path.Join(KerasFixtures.Folder, "keras-attention.json"))).RootElement;

    private static float[] Values(JsonElement numbers) => [.. numbers.EnumerateArray().Select(value => (float)value.GetDouble())];

    private static Tensor Examples(string name)
    {
        var written = Keras.GetProperty(name).GetProperty("examples");

        return Tensor.From(new Shape([.. written.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]), Values(written.GetProperty("values")));
    }

    public static TheoryData<string, string> EveryFile
    {
        get
        {
            var files = new TheoryData<string, string>();

            foreach (var name in new[] { "attention", "attention-causal", "attention-block" })
            {
                files.Add(name, "keras");
                files.Add(name, "h5");
            }

            return files;
        }
    }

    [Theory]
    [MemberData(nameof(EveryFile))]
    public void AModelThatLooksAcrossASeries_AnswersAsKerasDoes_InEitherFile(string name, string kind)
    {
        var read = new KerasFile().Read(KerasFixtures.Open($"keras-{name}.{kind}"));
        var here = read.Network.Predict(Examples(name), read.Loss, Engine).Values.ToArray();
        var keras = Values(Keras.GetProperty(name).GetProperty("outputs"));

        Assert.Equal(keras.Length, here.Length);
        Assert.InRange(keras.Zip(here, (was, now) => Math.Abs(was - now)).Max(), 0, 1e-5);
    }

    [Fact]
    public void TheAttention_IsReadAsTheLayerItIs_WithItsHeadsAndItsMask()
    {
        var plain = Assert.IsType<LayerStack>(new KerasFile().Read(KerasFixtures.Open("keras-attention.keras")).Network);
        var causal = Assert.IsType<LayerStack>(new KerasFile().Read(KerasFixtures.Open("keras-attention-causal.h5")).Network);
        var block = Assert.IsType<LayerStack>(new KerasFile().Read(KerasFixtures.Open("keras-attention-block.keras")).Network);

        Assert.Equal(new Attending(2), Assert.IsType<MultiHeadAttention>(plain.Layers[0]).Attending);
        Assert.Equal(new Attending(2) { Causal = true }, Assert.IsType<MultiHeadAttention>(causal.Layers[0]).Attending);
        Assert.Equal(8, Assert.IsType<MultiHeadAttention>(block.Layers[0]).Embed);
        Assert.Equal(["0.in_proj_weight", "0.in_proj_bias", "0.out_proj.weight", "0.out_proj.bias"], plain.Slots().Select(slot => slot.Path).Take(4));
    }

    [Theory]
    [InlineData("keras-attention.keras", "config.layers.1.config.use_bias=false", "it keeps no bias, use_bias false")]
    [InlineData("keras-attention.keras", "config.layers.1.config.value_dim=3", "its value_dim is 3 where its key_dim is 4")]
    [InlineData("keras-attention.keras", "config.layers.1.config.output_shape=[8]", "it gives an output_shape")]
    [InlineData("keras-attention.keras", "config.layers.1.config.attention_axes=[2]", "it looks along axes [2]")]
    [InlineData("keras-attention.keras", "config.layers.1.config.use_gate=true", "it gates its heads, use_gate")]
    [InlineData("keras-attention.keras", "config.layers.1.inbound_nodes.0.kwargs.attention_mask=true", "it is called with 'attention_mask'")]
    [InlineData("keras-attention.keras", "config.layers.1.inbound_nodes.0.kwargs.return_attention_scores=true", "it is called with 'return_attention_scores'")]
    public void AnAttentionThatSaysWhatNoLayerHereDoes_IsRefusedAtTheLayer(string file, string edit, string refusal)
    {
        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Edited(file, edit)));

        Assert.Contains(refusal, refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("keras-attention-cross.keras")]
    [InlineData("keras-attention-cross.h5")]
    public void AttentionOverTwoSeries_IsRefused_ForNetworkHereLooksAcrossOne(string file)
    {
        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Open(file)));

        Assert.Contains("two different tensors", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ANumberOfAnotherShapeThanTheLayerKeeps_IsRefusedAtTheSlot()
    {
        var refused = Assert.Throws<SlotLoadException>(() => new KerasFile().Read(KerasFixtures.Edited("keras-attention.keras", "config.layers.1.config.num_heads=4")));

        Assert.Contains(refused.Faults, fault => fault.Slot.StartsWith("0.", StringComparison.Ordinal));
    }
}
