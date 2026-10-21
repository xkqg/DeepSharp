// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Import.Keras;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Import;

/// <summary>
/// Keras models that read a series a step at a time (Fixtures/keras-recurrent-fixtures.py made them), in the two files Keras saves a model
/// to: an LSTM and a GRU, stacked, read at every step or to their last state, and read in both directions by a Bidirectional layer. A
/// layer's numbers sit one group deeper than a dense layer's, under its cell, and a GRU's gates stand in another order than PyTorch's and
/// the library's, so they are put in theirs.
/// </summary>
public class KerasRecurrentFileTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static readonly JsonElement Keras = JsonDocument.Parse(File.ReadAllText(Path.Join(KerasFixtures.Folder, "keras-recurrent.json"))).RootElement;

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

            foreach (var name in new[] { "lstm", "gru", "stacked", "both", "both-seq" })
            {
                files.Add(name, "keras");
                files.Add(name, "h5");
            }

            return files;
        }
    }

    [Theory]
    [MemberData(nameof(EveryFile))]
    public void AModelThatReadsASeriesAStepAtATime_AnswersAsKerasDoes_InEitherFile(string name, string kind)
    {
        var read = new KerasFile().Read(KerasFixtures.Open($"keras-{name}.{kind}"));
        var here = read.Network.Predict(Examples(name), read.Loss, Engine).Values.ToArray();
        var keras = Values(Keras.GetProperty(name).GetProperty("outputs"));

        Assert.Equal(keras.Length, here.Length);
        Assert.InRange(keras.Zip(here, (was, now) => Math.Abs(was - now)).Max(), 0, 1e-5);
    }

    [Fact]
    public void TheLayersAreReadAsTheLayersTheyAre_WithTheirSettings()
    {
        var lstm = Assert.IsType<LayerStack>(new KerasFile().Read(KerasFixtures.Open("keras-lstm.keras")).Network);
        var stacked = Assert.IsType<LayerStack>(new KerasFile().Read(KerasFixtures.Open("keras-stacked.keras")).Network);
        var both = Assert.IsType<LayerStack>(new KerasFile().Read(KerasFixtures.Open("keras-both-seq.h5")).Network);

        Assert.Equal(new Recurrence(3, 4), Assert.IsType<Lstm>(lstm.Layers[0]).Recurrence);
        Assert.Equal(new Recurrence(3, 4) { Sequences = true }, Assert.IsType<Lstm>(stacked.Layers[0]).Recurrence);
        Assert.Equal(new Recurrence(4, 3), Assert.IsType<Gru>(stacked.Layers[1]).Recurrence);
        Assert.Equal(new Recurrence(3, 3) { BothWays = true, Sequences = true }, Assert.IsType<Gru>(both.Layers[0]).Recurrence);
    }

    [Fact]
    public void AnLstmsBias_IsTheInputBias_AndTheRecurrentBiasKerasNeverKeptIsNought()
    {
        var lstm = Assert.IsType<Lstm>(Assert.IsType<LayerStack>(new KerasFile().Read(KerasFixtures.Open("keras-lstm.keras")).Network).Layers[0]);
        var recurrent = lstm.Slots().Single(slot => slot.Path == "bias_hh_l0");

        Assert.All(recurrent.Slot.Value.Values.ToArray(), value => Assert.Equal(0f, value));
    }

    [Theory]
    [InlineData("keras-lstm.keras", "config.layers.1.config.go_backwards=true", "it reads each series backwards, go_backwards")]
    [InlineData("keras-lstm.keras", "config.layers.1.config.dropout=0.2", "it drops numbers out inside the layer, dropout 0.2")]
    [InlineData("keras-lstm.keras", "config.layers.1.config.recurrent_dropout=0.1", "it drops numbers out inside the layer, recurrent_dropout 0.1")]
    [InlineData("keras-lstm.keras", "config.layers.1.config.activation=\"relu\"", "its activation, 'relu'")]
    [InlineData("keras-lstm.keras", "config.layers.1.config.recurrent_activation=\"hard_sigmoid\"", "its recurrent_activation, 'hard_sigmoid'")]
    [InlineData("keras-lstm.keras", "config.layers.1.config.return_state=true", "it hands back its state as well, return_state")]
    [InlineData("keras-lstm.keras", "config.layers.1.config.stateful=true", "it carries its state from one batch to the next, stateful")]
    [InlineData("keras-lstm.keras", "config.layers.1.config.use_bias=false", "it keeps no bias, use_bias false")]
    [InlineData("keras-gru-old.keras", "config.layers.1.config.units=4", "its reset_after is false")]
    [InlineData("keras-both.keras", "config.layers.1.config.merge_mode=\"sum\"", "it merges the two directions by 'sum'")]
    [InlineData("keras-both.keras", "config.layers.1.config.backward_layer.config.units=3", "it is given a backward layer of its own")]
    [InlineData("keras-both.keras", "config.layers.1.config.backward_layer.config.go_backwards=false", "it is given a backward layer of its own")]
    [InlineData("keras-both.keras", "config.layers.1.config.backward_layer.config.return_sequences=true", "it is given a backward layer of its own")]
    [InlineData("keras-both.keras", "config.layers.1.config.backward_layer.class_name=\"GRU\"", "it is given a backward layer of its own")]
    public void ARecurrentLayerThatSaysWhatNoLayerHereDoes_IsRefusedAtTheLayer(string file, string edit, string refusal)
    {
        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Edited(file, edit)));

        Assert.Contains(refusal, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ABidirectionalLayerThatWrapsNeitherAnLstmNorAGru_IsRefusedAtTheLayer()
    {
        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Edited("keras-both.keras", "config.layers.1.config.backward_layer=-", "config.layers.1.config.layer.class_name=\"SimpleRNN\"")));

        Assert.Contains("it wraps a SimpleRNN, and a bidirectional layer here wraps an LSTM or a GRU", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ANumberOfAnotherShapeThanTheLayerKeeps_IsRefusedAtTheSlot()
    {
        var refused = Assert.Throws<SlotLoadException>(() => new KerasFile().Read(KerasFixtures.Edited("keras-lstm.keras", "config.layers.1.config.units=5")));

        Assert.Contains(refused.Faults, fault => fault.Slot == "0.weight_ih_l0");
    }
}
