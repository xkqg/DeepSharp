// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Import.Keras;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Import;

/// <summary>
/// A Keras model whose first dense layer has a low-rank pair beside its held kernel (Fixtures/keras-lora-fixtures.py made it), in
/// the two files Keras saves a model to, read into a dense layer with a pair that answers as Keras answers through it. The
/// archive Keras 3 saves holds the kernel merged — the held kernel plus the pair's product scaled by alpha over the rank — and
/// neither matrix, so the layer read from it holds that kernel and begins a pair of its own whose second matrix is nothing; the
/// HDF5 file holds the kernel, the bias and both matrices as they are, and the layer read from it holds them all.
/// </summary>
public class KerasLoraFileTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static readonly JsonElement Keras = JsonDocument.Parse(File.ReadAllText(Path.Join(KerasFixtures.Folder, "keras-lora.json"))).RootElement;

    [Theory]
    [InlineData("keras-lora.keras")]
    [InlineData("keras-lora.h5")]
    public void AModelWithAPair_AnswersAsKerasDoes_InEitherFile(string file)
    {
        var written = Keras.GetProperty("examples");
        var examples = Tensor.From(new Shape([.. written.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]), Values(written.GetProperty("values")));
        var read = new KerasFile().Read(KerasFixtures.Open(file));
        var here = read.Network.Predict(examples, read.Loss, Engine).Values.ToArray();
        var keras = Values(Keras.GetProperty("outputs"));

        Assert.InRange(keras.Zip(here, (was, now) => Math.Abs(was - now)).Max(), 0, 1e-5);
    }

    [Fact]
    public void TheArchive_IsReadAsTheMergedKernelHeld_BesideAPairWhoseSecondMatrixIsNothing()
    {
        var dense = First("keras-lora.keras");

        Assert.Equal(new LowRank(Keras.GetProperty("rank").GetInt32(), Keras.GetProperty("alpha").GetDouble()), dense.LowRank);
        Assert.False(dense.Weight.Learns);
        Assert.True(dense.Bias!.Learns);
        Near(Values(Keras.GetProperty("merged")), dense.Weight.Value.Values.ToArray());
        Near(Values(Keras.GetProperty("bias")), dense.Bias.Value.Values.ToArray());
        Assert.All(dense.B!.Value.Values.ToArray(), value => Assert.Equal(0f, value));
        Assert.Contains(dense.A!.Value.Values.ToArray(), value => value != 0f);
    }

    [Fact]
    public void TheHdf5File_IsReadAsTheKernelTheBiasAndTheTwoMatricesAsTheyAre()
    {
        var dense = First("keras-lora.h5");

        Assert.Equal(new LowRank(Keras.GetProperty("rank").GetInt32(), Keras.GetProperty("alpha").GetDouble()), dense.LowRank);
        Assert.False(dense.Weight.Learns);
        Near(Values(Keras.GetProperty("kernel")), dense.Weight.Value.Values.ToArray());
        Near(Values(Keras.GetProperty("bias")), dense.Bias!.Value.Values.ToArray());
        Near(Values(Keras.GetProperty("a")), dense.A!.Value.Values.ToArray());
        Near(Values(Keras.GetProperty("b")), dense.B!.Value.Values.ToArray());
    }

    [Theory]
    [InlineData("keras-lora.keras")]
    [InlineData("keras-lora.h5")]
    public void AModelReadWithAPair_IsTrainedFurther_TheKernelHeldStillAndThePairMoving(string file)
    {
        var read = new KerasFile().Read(KerasFixtures.Open(file));
        var dense = Assert.IsType<Dense>(Assert.IsType<LayerStack>(read.Network).Layers[0]);
        var kernel = dense.Weight.Value.Values.ToArray();
        var pair = dense.B!.Value.Values.ToArray();
        var rows = new TrainingData(
            Tensor.From(new Shape(6, 3), Values(Keras.GetProperty("examples").GetProperty("values"))),
            Tensor.From(new Shape(6, 1), [1f, -1f, 0.5f, 0f, 2f, -0.5f]));

        read.Network.Compile(new Adam(0.05), read.Loss).Fit(rows, validation: null, new FitOptions(seed: 1) { Epochs = 5, BatchSize = 3 });

        Assert.Equal(kernel, dense.Weight.Value.Values.ToArray());
        Assert.NotEqual(pair, dense.B.Value.Values.ToArray());
    }

    [Fact]
    public void ADenseLayerWithoutAPair_IsReadAsItWas_WithNoPairAndAKernelThatLearns()
    {
        var plain = new KerasFile().Read(KerasFixtures.Open("keras-free-dense.keras"));
        var dense = plain.Network.HeldLayers().Select(held => held.Layer).OfType<Dense>().First();

        Assert.Null(dense.LowRank);
        Assert.True(dense.Weight.Learns);
    }

    private static Dense First(string file) =>
        new KerasFile().Read(KerasFixtures.Open(file)).Network.HeldLayers().Select(held => held.Layer).OfType<Dense>().First();

    private static float[] Values(JsonElement numbers) => [.. numbers.EnumerateArray().Select(value => (float)value.GetDouble())];

    private static void Near(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], actual[at], 5);
        }
    }
}
