// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Import.Keras;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Import;

/// <summary>
/// Keras functional models (Fixtures/keras-functional-fixtures.py made them), in the two files Keras saves a model to: one with a
/// residual step and a dense block, one with two inputs, each read into a stack of steps that takes the steps the model's layers
/// take, so the network answers as Keras answers through the model. The model's inputs are handed to a network here as the
/// columns of one row, in the order of the model's inputs.
/// </summary>
public class KerasFunctionalFileTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static readonly JsonElement Keras = JsonDocument.Parse(File.ReadAllText(Path.Join(KerasFixtures.Folder, "keras-functional.json"))).RootElement;

    private static float[] Values(JsonElement numbers) => [.. numbers.EnumerateArray().Select(value => (float)value.GetDouble())];

    private static Tensor Examples(JsonElement written) =>
        Tensor.From(new Shape([.. written.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]), Values(written.GetProperty("values")));

    // The two inputs of the second model, handed as the columns of one row: every left row beside the right one.
    private static Tensor Together()
    {
        var model = Keras.GetProperty("twoInputs");
        var lefts = Examples(model.GetProperty("lefts"));
        var rights = Examples(model.GetProperty("rights"));
        var rows = lefts.Shape[0];

        return Tensor.From(
            new Shape(rows, lefts.Shape[1] + rights.Shape[1]),
            [.. Enumerable.Range(0, rows).SelectMany(row => lefts.Values.Slice(row * lefts.Shape[1], lefts.Shape[1]).ToArray().Concat(rights.Values.Slice(row * rights.Shape[1], rights.Shape[1]).ToArray()))]);
    }

    private static LayerStack Stack(string file) => Assert.IsType<LayerStack>(new KerasFile().Read(KerasFixtures.Open(file)).Network);

    private static void Near(float[] keras, float[] here) => Assert.InRange(keras.Zip(here, (was, now) => Math.Abs(was - now)).Max(), 0, 1e-5);

    [Theory]
    [InlineData("keras-residual.keras")]
    [InlineData("keras-residual.h5")]
    public void AResidualModelWithADenseBlock_AnswersAsKerasDoes_InEitherFile(string file)
    {
        var model = Keras.GetProperty("residual");
        var read = new KerasFile().Read(KerasFixtures.Open(file));

        Near(Values(model.GetProperty("outputs")), read.Network.Predict(Examples(model.GetProperty("examples")), read.Loss, Engine).Values.ToArray());
    }

    [Theory]
    [InlineData("keras-residual.keras")]
    [InlineData("keras-residual.h5")]
    public void TheResidualModel_IsReadAsSteps_EachTakingTheStepsTheModelsLayerTakes(string file)
    {
        var stack = Stack(file);

        Assert.Equal(["Dense", "Relu", "Dense", "Sum", "Concatenation", "Dense"], stack.Steps.Select(step => step.Layer.GetType().Name));
        Assert.Equal(
            [["input"], ["0"], ["1"], ["input", "2"], ["input", "3"], ["4"]],
            stack.Steps.Select(step => (IReadOnlyList<string>)[.. step.Takes]));
        Assert.Equal(["0.weight", "0.bias", "2.weight", "2.bias", "5.weight", "5.bias"], stack.Slots().Select(slot => slot.Path));
    }

    [Theory]
    [InlineData("keras-two-inputs.keras")]
    [InlineData("keras-two-inputs.h5")]
    public void AModelOfTwoInputs_AnswersAsKerasDoes_HandedItsInputsAsTheColumnsOfOneRow(string file)
    {
        var read = new KerasFile().Read(KerasFixtures.Open(file));

        Near(Values(Keras.GetProperty("twoInputs").GetProperty("outputs")), read.Network.Predict(Together(), read.Loss, Engine).Values.ToArray());
    }

    [Theory]
    [InlineData("keras-two-inputs.keras")]
    [InlineData("keras-two-inputs.h5")]
    public void TheTwoInputs_AreReadAsTheColumnsOfOneRow_EachInTheOrderOfTheModelsInputs(string file)
    {
        var stack = Stack(file);

        Assert.Equal(["Columns", "Columns", "Dense", "Relu", "Dense", "Sum", "Dense"], stack.Steps.Select(step => step.Layer.GetType().Name));
        Assert.Equal(new Columns(0, 3).First + new Columns(0, 3).Count, Assert.IsType<Columns>(stack.Steps[0].Layer).Count);
        Assert.Equal((3, 4), (Assert.IsType<Columns>(stack.Steps[1].Layer).First, Assert.IsType<Columns>(stack.Steps[1].Layer).Count));
        Assert.Equal(
            [["input"], ["input"], ["0"], ["2"], ["1"], ["3", "4"], ["5"]],
            stack.Steps.Select(step => (IReadOnlyList<string>)[.. step.Takes]));
    }

    [Fact]
    public void AModelRead_IsTrainedFurther_AsAnyNetworkIs()
    {
        var read = new KerasFile().Read(KerasFixtures.Open("keras-residual.keras"));
        var compiled = ((LayerStack)read.Network).Compile(new Sgd(0.01), new MeanSquaredError());
        var rows = Examples(Keras.GetProperty("residual").GetProperty("examples"));

        var history = compiled.Fit(new TrainingData(rows, Tensor.Zeros(new Shape(rows.Shape[0], 2))), null, new FitOptions(seed: 1) { Epochs = 2 });

        Assert.Equal(2, history.Epochs.Count);
    }

    [Theory]
    [InlineData("config.layers.1.inbound_nodes=[[[\"rows\", 0, 0, {}]]]", "layer 'first' (Dense): it takes its layers in the form Keras 2 wrote, and a Keras 3 model states them as arguments")]
    [InlineData("config.layers.2.inbound_nodes=[]", "layer 'second' (Dense): it takes no layer, and a layer of a functional model takes the layers it names")]
    [InlineData("config.layers.2.inbound_nodes.0.args.0.config.keras_history.0=\"ghost\"", "layer 'second' (Dense): it takes 'ghost', which is no layer before it")]
    [InlineData("config.layers.2.inbound_nodes.0.args.0.config.keras_history.1=1", "layer 'second' (Dense): it takes the layer 'first' at its node 1, tensor 0, and a layer is read here when it is called once, giving one tensor")]
    [InlineData("config.layers.2.inbound_nodes.0.kwargs={\"training\": true}", "layer 'second' (Dense): it is called with 'training', and a layer is read here when it is called with the tensors it takes alone")]
    [InlineData("config.layers.2.inbound_nodes=[1, 2]", "layer 'second' (Dense): it is called 2 times, and a layer is read here when it is called once")]
    [InlineData("config.layers.2.inbound_nodes.0.args=[]", "layer 'second' (Dense): it is called with other than the tensors of layers, and a layer of a functional model is called with the tensors it takes")]
    [InlineData("config.layers.2.inbound_nodes.0.args.0=5", "layer 'second' (Dense): it is called with other than the tensors of layers, and a layer of a functional model is called with the tensors it takes")]
    [InlineData("config.input_layers=-", "the model lists no input layer under 'input_layers'")]
    [InlineData("config.input_layers=[\"ghost\", 0, 0]", "the model lists the input 'ghost', and no input layer of that name")]
    [InlineData("config.layers.0.config.batch_shape=[null, null]", "does not state every length of an example")]
    [InlineData("config.layers.4.inbound_nodes.0.args.0=[{\"class_name\": \"__keras_tensor__\", \"config\": {\"shape\": [null, 6], \"dtype\": \"float32\", \"keras_history\": [\"rows\", 0, 0]}}]", "it joins 1 tensor, and a layer that adds or lays side by side is read here when it takes two or more")]
    [InlineData("config.layers.4.config.axis=0", "layer 'joined' (Concatenate): it lays its tensors side by side along axis 0, and a network here lays them side by side along their last axis, -1")]
    public void AFunctionalModelThatSaysWhatNoNetworkHereIsBuiltOf_IsRefusedAtTheLayerThatSaysIt(string edit, string refusal)
    {
        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Edited("keras-residual.keras", edit)));

        Assert.Contains(refusal, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ASwishActivation_IsReadAsTheSigmoidLinearUnit()
    {
        var read = new KerasFile().Read(KerasFixtures.Edited("keras-residual.keras", "config.layers.1.config.activation=\"swish\""));

        Assert.Contains(((LayerStack)read.Network).Layers, layer => layer is Silu);
    }

    [Fact]
    public void AMergeLayerThatIsNoneOfAddOrConcatenate_IsRefusedByName()
    {
        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Edited("keras-residual.keras", "config.layers.3.class_name=\"Multiply\"")));

        Assert.Contains("'Multiply' is no kind a network here is built of", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AModelThatGivesMoreThanOneTensor_OrEndsInALayerThatIsNotItsLast_IsRefused()
    {
        var two = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Edited("keras-residual.keras", "config.output_layers=[[\"head\", 0, 0], [\"joined\", 0, 0]]")));
        var other = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Edited("keras-residual.keras", "config.output_layers=[\"joined\", 0, 0]")));

        Assert.Contains("gives 2 tensors, and a network here gives one", two.Message, StringComparison.Ordinal);
        Assert.Contains("gives the output of 'joined', and a network here gives what its last layer, 'head', makes", other.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AModelOfTwoInputsThatAreNotRows_IsRefused()
    {
        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Edited("keras-two-inputs.keras", "config.layers.0.config.batch_shape=[null, 3, 2]")));

        Assert.Contains("layer 'left' (InputLayer): it is an input of 3x2 values", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AStepThatNothingTakes_IsRefusedAsALayerWhoseWorkWouldBeMadeAndNeverUsed()
    {
        var refused = Assert.Throws<FormatException>(() => new KerasFile().Read(KerasFixtures.Edited("keras-residual.keras", "config.layers.5.inbound_nodes.0.args.0.config.keras_history.0=\"residual\"")));

        Assert.Contains("taken by no later step", refused.Message, StringComparison.Ordinal);
    }
}
