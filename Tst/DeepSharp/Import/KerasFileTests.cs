// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Import.Keras;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;
using DeepSharp.Tests.Learners;

namespace DeepSharp.Tests.Import;

/// <summary>
/// A model Keras 3 trained and saved — in its own <c>.keras</c> archive, or in the HDF5 file it saved before — read into a
/// network here through the importer seam: the layers its description names, built as Keras's words build them, holding its
/// numbers as Keras laid them out, and answering as Keras answered, to within a few roundings of a single-precision number.
/// </summary>
public class KerasFileTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static readonly JsonElement Titanic = KerasFixtures.Answers.GetProperty("titanic");

    private static readonly JsonElement Images = KerasFixtures.Answers.GetProperty("images");

    [Theory]
    [InlineData("keras-titanic.keras")]
    [InlineData("keras-titanic.h5")]
    public void TheTitanicNetworkKerasTrained_AnswersEveryTestPassengerAsKerasDid_WithinEightRoundings(string file)
    {
        var test = WikiTitanic.In(WikiTitanic.DataFolder).Run().Batch(Part.Test, Needs.OneScale);
        var keras = Titanic.GetProperty("test");

        // The rows Keras answered are the rows the pipeline hands over, number for number.
        Assert.Equal(135, test.RowCount);
        Assert.Equal(Rows(keras), [.. test.Features.Select(Single)]);

        IImporter importer = new KerasFile();
        var saved = importer.Read(KerasFixtures.Open(file));

        Assert.InRange(RoundingsApart(Values(keras.GetProperty("chances")), saved.Network.Predict(Batch(test.Features), saved.Loss, Engine)), 0, 8);
    }

    [Theory]
    [InlineData("keras-titanic.keras")]
    [InlineData("keras-titanic.h5")]
    public void TheTwoPassengersTheSampleServes_AreGivenTheChancesKerasGaveThem(string file)
    {
        var prepared = WikiTitanic.In(WikiTitanic.DataFolder).Run();
        var passengers = new InMemoryRowSource(
            ["pclass", "sex", "age", "sibsp", "parch", "fare"], [["3", "male", "22", "1", "0", "7.25"], ["1", "female", "38", "1", "0", "71.2833"]]);
        var served = prepared.Served(passengers, Needs.OneScale);
        var keras = Titanic.GetProperty("served");

        Assert.Equal(Rows(keras), [.. served.Features.Select(Single)]);

        var saved = new KerasFile().Read(KerasFixtures.Open(file));
        var chances = saved.Network.Predict(Batch(served.Features), saved.Loss, Engine);

        Assert.InRange(RoundingsApart(Values(keras.GetProperty("chances")), chances), 0, 8);
    }

    [Fact]
    public void AModelKerasTrained_FittedOneEpochBehindThePipeline_IsWrittenAsTheOneFile_AndReadBackAnswersAsItDid()
    {
        var prepared = WikiTitanic.In(WikiTitanic.DataFolder).Run();
        var saved = new KerasFile().Read(KerasFixtures.Open("keras-titanic.keras"));
        var kerasNumbers = Bits(saved.Network);

        // An import is a network and its loss, trained behind nothing here: it stands behind a pipeline once a fit here has
        // trained it on the rows that pipeline prepares, and that fit is what the one file records.
        Assert.Null(saved.TrainedOn);

        var trained = saved.Network.Compile(new Adam(0.001), saved.Loss).Fit(prepared, new FitOptions(seed: 7) { Epochs = 1 });
        var json = trained.ToJson();
        var read = TrainedNetwork.FromJson(json, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn());
        var passengers = new InMemoryRowSource(
            ["pclass", "sex", "age", "sibsp", "parch", "fare"], [["3", "male", "22", "1", "0", "7.25"], ["1", "female", "38", "1", "0", "71.2833"]]);

        Assert.Equal(0, trained.TrainedOn.Epoch);
        Assert.Equal(PipelineText.Of(prepared).Digest, read.TrainedOn.TrainedBehind);
        Assert.NotEqual(kerasNumbers, Bits(trained.Network));
        Assert.Equal(Bits(trained.Network), Bits(read.Network));
        Assert.Equal(trained.Predict(passengers).Answers, read.Predict(passengers).Answers);
        Assert.Equal(json, read.ToJson());
    }

    [Fact]
    public void TheTitanicModel_IsBuiltAsItsDescriptionSays_ItsActivationsLayersOfTheirOwn_AndItsLastSigmoidLiftedIntoItsLoss()
    {
        var saved = new KerasFile().Read(KerasFixtures.Open("keras-titanic.keras"));
        var stack = Assert.IsType<LayerStack>(saved.Network);

        Assert.Collection(
            stack.Layers,
            layer => Assert.Equal(new Shape(14, 16), Assert.IsType<Dense>(layer).Weight.Value.Shape),
            layer => Assert.IsType<Relu>(layer),
            layer => Assert.Equal(new Shape(16, 1), Assert.IsType<Dense>(layer).Weight.Value.Shape));
        Assert.Equal(["0.weight", "0.bias", "2.weight", "2.bias"], stack.Slots().Select(named => named.Path));
        Assert.IsType<BinaryCrossEntropy>(saved.Loss);
        Assert.Null(saved.TrainedOn);

        // The network ends where its loss takes over, so it compiles with that loss.
        Assert.Same(stack, stack.Compile(new Adam(0.01), saved.Loss).Network);
    }

    [Theory]
    [InlineData("keras-images.keras")]
    [InlineData("keras-images.h5")]
    public void AModelOfEveryKindThisReaderBuilds_GivesEachImageTheSharesKerasGaveIt(string file)
    {
        var saved = new KerasFile().Read(KerasFixtures.Open(file));
        var images = Rows(Images);

        var shares = saved.Network.Predict(Tensor.From(new Shape(images.Length, 84), [.. images.SelectMany(image => image)]), saved.Loss, Engine);

        Assert.InRange(RoundingsApart(Values(Images.GetProperty("shares")), shares), 0, 8);
    }

    [Fact]
    public void EachKindIsBuiltWithItsSettings_KerasEpsilons_TheComplementOfItsMomentum_AndItsWindows_AndTheSoftmaxLiftedIntoTheCrossEntropy()
    {
        var saved = new KerasFile().Read(KerasFixtures.Open("keras-images.keras"));
        var stack = Assert.IsType<LayerStack>(saved.Network);

        Assert.Collection(
            stack.Layers,
            layer => Assert.Equal(new Shape(6, 7, 2), Assert.IsType<Reshape>(layer).Each),
            layer => AssertConvolution(new Window(3, 2) { Stride = 2, PaddingMode = PaddingMode.Same }, 2, 4, layer),
            layer => Assert.IsType<Relu>(layer),
            layer =>
            {
                var norm = Assert.IsType<BatchNorm>(layer);
                Assert.Equal(1 - 0.9, norm.Momentum);
                Assert.Equal(0.001, norm.Epsilon);
                Assert.Equal(4, norm.Features);
            },
            layer => AssertConvolution(new Window(2, 2), 4, 3, layer),
            layer => Assert.IsType<Relu>(layer),
            layer => Assert.IsType<Flatten>(layer),
            layer => Assert.Equal(0.25, Assert.IsType<Dropout>(layer).Rate),
            layer => Assert.Equal(new Shape(18, 6), Assert.IsType<Dense>(layer).Weight.Value.Shape),
            layer => Assert.IsType<Tanh>(layer),
            layer =>
            {
                var norm = Assert.IsType<LayerNorm>(layer);
                Assert.Equal(0.001, norm.Epsilon);
                Assert.Equal(6, norm.Features);
            },
            layer => Assert.Equal(new Shape(6, 3), Assert.IsType<Dense>(layer).Weight.Value.Shape));
        Assert.IsType<CrossEntropy>(saved.Loss);
    }

    [Theory]
    [InlineData("keras-titanic")]
    [InlineData("keras-images")]
    public void AModelsArchive_AndTheHdf5FileKerasSavedItToBefore_PutTheSameNumbersIntoTheSameSlots(string model)
    {
        var archive = new KerasFile().Read(KerasFixtures.Open($"{model}.keras")).Network;
        var legacy = new KerasFile().Read(KerasFixtures.Open($"{model}.h5")).Network;

        Assert.Equal(archive.Slots().Select(named => named.Path), legacy.Slots().Select(named => named.Path));
        Assert.Equal(Bits(archive), Bits(legacy));
    }

    [Fact]
    public void AModelThatStatesItsInputByItsBuildRatherThanAnInputLayer_IsReadTheSame()
    {
        var saved = new KerasFile().Read(KerasFixtures.Edited("keras-titanic.keras", "config.layers.0=-"));

        Assert.Equal(Bits(new KerasFile().Read(KerasFixtures.Open("keras-titanic.keras")).Network), Bits(saved.Network));
    }

    [Theory]
    [InlineData("""{"module": "keras.losses", "class_name": "function", "config": "binary_crossentropy", "registered_name": "function"}""")]
    [InlineData("""{"module": "keras.losses", "class_name": "BinaryCrossentropy", "config": {"name": "binary_crossentropy", "reduction": "sum_over_batch_size", "from_logits": false, "label_smoothing": 0.0, "axis": -1}}""")]
    public void TheLossIsReadAsKerasWritesIt_ByItsName_AsTheFunction_OrAsTheObject(string loss)
    {
        var saved = new KerasFile().Read(KerasFixtures.Edited("keras-titanic.keras", $"compile_config.loss={loss}"));

        Assert.IsType<BinaryCrossEntropy>(saved.Loss);
        Assert.Equal(3, Assert.IsType<LayerStack>(saved.Network).Layers.Count);
    }

    [Theory]
    [InlineData("keras-titanic.keras", "\"binary_crossentropy\"", typeof(BinaryCrossEntropy))]
    [InlineData("keras-titanic.keras", "{\"class_name\": \"BinaryCrossentropy\"}", typeof(BinaryCrossEntropy))]
    [InlineData("keras-images.keras", "\"categorical_crossentropy\"", typeof(CrossEntropy))]
    [InlineData("keras-images.keras", "{\"class_name\": \"CategoricalCrossentropy\"}", typeof(CrossEntropy))]
    [InlineData("keras-titanic.keras", "\"mean_squared_error\"", typeof(MeanSquaredError))]
    [InlineData("keras-titanic.keras", "\"mse\"", typeof(MeanSquaredError))]
    [InlineData("keras-titanic.keras", "{\"class_name\": \"MeanSquaredError\"}", typeof(MeanSquaredError))]
    public void EveryNameKerasWritesALossUnder_IsReadAsThatLossHere(string file, string loss, Type read) =>
        Assert.IsType(read, new KerasFile().Read(KerasFixtures.Edited(file, $"compile_config.loss={loss}")).Loss);

    [Fact]
    public void AModelEndingInItsLogits_WhoseLossTakesLogits_KeepsItsLastLayer_AndAnswersThroughTheLossesSigmoid()
    {
        // The same numbers, described as Keras describes a network trained on its logits: the last layer gives them as they
        // are, and the loss applies the sigmoid — so the chances are the ones the model ending in its sigmoid gave.
        var saved = new KerasFile().Read(KerasFixtures.Edited(
            "keras-titanic.keras",
            "config.layers.2.config.activation=\"linear\"",
            """compile_config.loss={"class_name": "BinaryCrossentropy", "config": {"from_logits": true}}"""));
        var stack = Assert.IsType<LayerStack>(saved.Network);
        var keras = Titanic.GetProperty("served");

        Assert.IsType<Dense>(stack.Layers[^1]);
        Assert.InRange(RoundingsApart(Values(keras.GetProperty("chances")), saved.Network.Predict(Batch(Rows(keras)), saved.Loss, Engine)), 0, 8);
    }

    [Fact]
    public void ASigmoidBeforeALossThatAppliesNone_StaysALayer()
    {
        var saved = new KerasFile().Read(KerasFixtures.Edited("keras-titanic.keras", "compile_config.loss=\"mse\""));

        Assert.IsType<MeanSquaredError>(saved.Loss);
        Assert.IsType<Sigmoid>(Assert.IsType<LayerStack>(saved.Network).Layers[^1]);
    }

    private static void AssertConvolution(Window window, int inChannels, int outChannels, Layer layer)
    {
        var convolution = Assert.IsType<Conv2D>(layer);

        Assert.Equal(window, convolution.Window);
        Assert.Equal(inChannels, convolution.InChannels);
        Assert.Equal(outChannels, convolution.OutChannels);
    }

    // The most roundings of a single-precision number any value lies from Keras's: the distance between their bit patterns,
    // which for numbers of one sign counts the floats between them.
    private static long RoundingsApart(float[] keras, Tensor here)
    {
        Assert.Equal(keras.Length, here.Values.Length);

        return keras.Zip(here.Values.ToArray(), (expected, actual) => Math.Abs((long)BitConverter.SingleToInt32Bits(expected) - BitConverter.SingleToInt32Bits(actual))).Max();
    }

    private static float[][] Rows(JsonElement answered) => [.. answered.GetProperty("rows").EnumerateArray().Select(Values)];

    private static float[] Values(JsonElement numbers) => [.. numbers.EnumerateArray().Select(value => (float)value.GetDouble())];

    private static float[] Single(double[] row) => [.. row.Select(value => (float)value)];

    private static Tensor Batch(IReadOnlyList<double[]> rows) => Batch([.. rows.Select(Single)]);

    private static Tensor Batch(float[][] rows) => Tensor.From(new Shape(rows.Length, rows[0].Length), [.. rows.SelectMany(row => row)]);

    private static int[][] Bits(Network network) =>
        [.. network.Slots().Select(named => named.Slot.Value.Values.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray())];
}
