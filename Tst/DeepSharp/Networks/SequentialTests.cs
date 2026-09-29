// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A network described in Keras's words is read once and lowered onto the same stack of layers a network written as code
/// is made of: every width worked out from the shape of an example, every layer drawing what it starts from out of a
/// stream by its place — so nothing downstream can tell which door a network came through, down to its file.
/// </summary>
public class SequentialTests
{
    private readonly ITensorBackend _backend = new CpuBackend();

    [Fact]
    public void TheSameNetwork_ThroughEitherDoor_GivesPyTorchsLogitsBitForBit_AndTheSameFile()
    {
        var lowered = new Sequential().Dense(4).Relu().Dense(1).Lower(new Shape(14), new RandomStream(1));
        var written = new LayerStack(
            new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias()), new Relu(), new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias()));

        foreach (var (slot, value) in lowered.Slots().Zip(written.Slots()))
        {
            slot.Slot.Replace(value.Slot.Value);
        }

        var fromWords = lowered.Forward(WalkedRows.Passengers(), Pass.Evaluation(_backend)).Values.ToArray();
        var fromCode = written.Forward(WalkedRows.Passengers(), Pass.Evaluation(_backend)).Values.ToArray();

        Assert.Equal(fromCode.Select(BitConverter.SingleToInt32Bits), fromWords.Select(BitConverter.SingleToInt32Bits));
        Assert.Equal([0.2f, 0.2f, 0.22550064f, 0.2f], fromWords);
        Assert.Equal(Written(written), Written(lowered));
    }

    [Fact]
    public void TheSameStream_GivesTheSameParameters_AndEachLayerDrawsByItsPlace()
    {
        var description = new Sequential().Dense(4).Relu().Dense(1);

        var one = Bits(description.Lower(new Shape(14), new RandomStream(7)));
        var again = Bits(description.Lower(new Shape(14), new RandomStream(7)));
        var other = Bits(description.Lower(new Shape(14), new RandomStream(8)));

        Assert.Equal(one, again);
        Assert.NotEqual(one, other);

        // As a network written as code draws them, each layer from the stream by the place it stands at.
        var stream = new RandomStream(7);
        var byHand = new LayerStack(new Dense(14, 4, stream.Draw("initialise:0", 0, 0)), new Relu(), new Dense(4, 1, stream.Draw("initialise:2", 0, 0)));

        Assert.Equal(Bits(byHand), one);
    }

    [Fact]
    public void EveryWord_LowersToTheLayerItNames_WithTheWidthsItsShapeGives()
    {
        var stack = new Sequential()
            .Input(new Shape(36))
            .Reshape(new Shape(6, 6, 1))
            .Conv2D(4, new Window(3, 3))
            .BatchNorm()
            .Relu()
            .Flatten()
            .Dense(8)
            .LayerNorm()
            .Tanh()
            .Dropout(0.25)
            .Dense(3)
            .Sigmoid()
            .Lower(new Shape(36), new RandomStream(3));

        Assert.Equal(
            [typeof(Reshape), typeof(Conv2D), typeof(BatchNorm), typeof(Relu), typeof(Flatten), typeof(Dense), typeof(LayerNorm), typeof(Tanh), typeof(Dropout), typeof(Dense), typeof(Sigmoid)],
            stack.Layers.Select(layer => layer.GetType()));
        var convolution = (Conv2D)stack.Layers[1];
        var hidden = (Dense)stack.Layers[5];
        var output = (Dense)stack.Layers[9];

        Assert.Equal(new Shape(6, 6, 1), ((Reshape)stack.Layers[0]).Each);
        Assert.Equal(1, convolution.InChannels);
        Assert.Equal(4, convolution.OutChannels);
        Assert.Equal(new Window(3, 3), convolution.Window);
        Assert.Equal(4, ((BatchNorm)stack.Layers[2]).Features);
        Assert.Equal(64, hidden.Inputs);
        Assert.Equal(8, hidden.Outputs);
        Assert.Equal(8, ((LayerNorm)stack.Layers[6]).Features);
        Assert.Equal(0.25, ((Dropout)stack.Layers[8]).Rate);
        Assert.Equal(8, output.Inputs);
        Assert.Equal(3, output.Outputs);

        var rows = Tensor.From(new Shape(2, 36), [.. Enumerable.Range(0, 72).Select(at => MathF.Cos(at))]);

        Assert.Equal(new Shape(2, 3), stack.Forward(rows, Pass.Evaluation(_backend)).Shape);
        Assert.Equal(new Shape(36), new Sequential().Input(new Shape(36)).InputShape);
        Assert.Null(new Sequential().InputShape);
    }

    [Theory]
    [InlineData("dense", "word 2, dense, takes each example as a row of numbers, and each reaching it is 6x6x1: flatten it first.")]
    [InlineData("conv2d", "word 1, conv2d, takes each example as an image of rows, columns and channels, and each reaching it is 36.")]
    [InlineData("reshape", "word 1, reshape, makes each example 5x7, which holds 35 values, and each reaching it holds 36.")]
    [InlineData("window", "word 2, conv2d, slides a window 7x7 (stride 1, padding 0) over each example, and an example 6x6x1 is smaller than it.")]
    public void AWordAnExampleCannotReach_IsRefused_NamingTheWordAndWhatReachesIt(string wrong, string message)
    {
        var description = wrong switch
        {
            "dense" => new Sequential().Reshape(new Shape(6, 6, 1)).Dense(3),
            "conv2d" => new Sequential().Conv2D(2, new Window(3, 3)),
            "reshape" => new Sequential().Reshape(new Shape(5, 7)),
            _ => new Sequential().Reshape(new Shape(6, 6, 1)).Conv2D(2, new Window(7, 7)),
        };

        var refused = Assert.Throws<ArgumentException>(() => description.Lower(new Shape(36), new RandomStream(1)));

        Assert.StartsWith("This network cannot be lowered at 36: " + message, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AWordWrittenWrongly_IsRefusedWhereItIsWritten()
    {
        var description = new Sequential();

        Assert.Throws<ArgumentOutOfRangeException>(() => description.Dense(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => description.Dropout(1));
        Assert.Throws<ArgumentOutOfRangeException>(() => description.Conv2D(0, new Window(3, 3)));
        Assert.Throws<ArgumentException>(() => description.Conv2D(2, new Window(0, 3)));
        Assert.Throws<ArgumentException>(() => description.Reshape(new Shape()));
        Assert.Throws<ArgumentException>(() => description.Input(new Shape()));
        Assert.Throws<InvalidOperationException>(() => description.Dense(2).Input(new Shape(3)));
        Assert.Throws<InvalidOperationException>(() => new Sequential().Input(new Shape(3)).Input(new Shape(3)));
    }

    [Fact]
    public void ALoweringAtAnotherShapeThanItsInputSays_OrOfNoWords_IsRefused()
    {
        var stated = new Sequential().Input(new Shape(14)).Dense(1);

        Assert.Contains("takes each example as 14", Assert.Throws<ArgumentException>(() => stated.Lower(new Shape(12), new RandomStream(1))).Message, StringComparison.Ordinal);
        Assert.Contains("no layers", Assert.Throws<InvalidOperationException>(() => new Sequential().Lower(new Shape(3), new RandomStream(1))).Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentException>(() => stated.Lower(new Shape(), new RandomStream(1)));
        Assert.Throws<ArgumentNullException>(() => stated.Lower(new Shape(14), null!));
    }

    [Fact]
    public void ACompiledDescription_IsLoweredAtItsFirstFit_FromTheRowsShapeAndTheRunsSeed_AsIfLoweredByHand()
    {
        var description = new Sequential().Dense(4).Tanh().Dense(1);
        var options = new FitOptions(seed: 11) { Epochs = 3, BatchSize = 8 };

        var described = description.Compile(new Adam(0.01), new MeanSquaredError());
        var history = described.Fit(Rows(32, 0), Rows(8, 32), options);

        var lowered = description.Lower(new Shape(2), new RandomStream(11)).Compile(new Adam(0.01), new MeanSquaredError());
        var byHand = lowered.Fit(Rows(32, 0), Rows(8, 32), options);

        var stream = new RandomStream(11);
        var written = new LayerStack(new Dense(2, 4, stream.Draw("initialise:0", 0, 0)), new Tanh(), new Dense(4, 1, stream.Draw("initialise:2", 0, 0)))
            .Compile(new Adam(0.01), new MeanSquaredError());
        var asCode = written.Fit(Rows(32, 0), Rows(8, 32), options);

        Assert.Equal(Bits(lowered.Network), Bits(described.Network));
        Assert.Equal(Bits(written.Network), Bits(described.Network));
        Assert.Equal(byHand.Epochs, history.Epochs);
        Assert.Equal(asCode.Epochs, history.Epochs);
        Assert.IsType<ConstantRate>(described.Schedule);
    }

    [Fact]
    public void BeforeItsFirstFit_ACompiledDescriptionHasNoNetwork_AndSaysHowToHaveOneNow()
    {
        var compiled = new Sequential().Dense(1).Compile(new Sgd(), new MeanSquaredError());

        var wrong = Assert.Throws<InvalidOperationException>(() => compiled.Network);

        Assert.Contains("first Fit", wrong.Message, StringComparison.Ordinal);
        Assert.Contains("Lower(", wrong.Message, StringComparison.Ordinal);
        Assert.Equal(wrong.Message, Assert.Throws<InvalidOperationException>(() => compiled.Predict(Tensor.Zeros(new Shape(1, 2)), _backend)).Message);
    }

    [Fact]
    public void AWordWrittenAfterCompile_ChangesNothingThatWasCompiled()
    {
        var description = new Sequential().Dense(3);
        var compiled = description.Compile(new Sgd(), new MeanSquaredError());

        description.Relu().Dense(1);
        compiled.Fit(Rows(8, 0, answers: 3), validation: null, new FitOptions(seed: 1));

        Assert.IsType<Dense>(Assert.Single(Assert.IsType<LayerStack>(compiled.Network).Layers));
    }

    [Fact]
    public void ADescriptionWhoseInputIsStated_IsCheckedWhereItIsCompiled()
    {
        var images = new Sequential().Input(new Shape(6, 6, 1)).Dense(3);

        var wrong = Assert.Throws<ArgumentException>(() => images.Compile(new Sgd(), new MeanSquaredError()));

        Assert.StartsWith("This network cannot be lowered at 6x6x1: word 1, dense,", wrong.Message, StringComparison.Ordinal);
        Assert.Contains("no layers", Assert.Throws<InvalidOperationException>(() => new Sequential().Compile(new Sgd(), new MeanSquaredError())).Message, StringComparison.Ordinal);
        Assert.Throws<ArgumentNullException>(() => new Sequential().Dense(1).Compile(null!, new MeanSquaredError()));
        Assert.Throws<ArgumentNullException>(() => new Sequential().Dense(1).Compile(new Sgd(), null!));
    }

    [Fact]
    public void RowsOfAnotherShapeThanTheNetworkTakes_AreRefused_TrainingAndValidationRowsAlike()
    {
        var stated = new Sequential().Input(new Shape(2)).Dense(1).Compile(new Sgd(), new MeanSquaredError());
        var taken = new Sequential().Dense(1).Compile(new Sgd(), new MeanSquaredError());

        Assert.Equal(
            "This network takes each example as 2, and was handed examples of 3. (Parameter 'train')",
            Assert.Throws<ArgumentException>(() => stated.Fit(Rows(8, 0, width: 3), validation: null, new FitOptions(seed: 1))).Message);
        Assert.Equal(
            "validation",
            Assert.Throws<ArgumentException>(() => stated.Fit(Rows(8, 0), Rows(4, 8, width: 3), new FitOptions(seed: 1) { EarlyStopping = new EarlyStopping() })).ParamName);

        // A network whose first fit took its shape from the rows keeps it: later rows are held to it.
        taken.Fit(Rows(8, 0), validation: null, new FitOptions(seed: 1));

        Assert.Contains(
            "takes each example as 2",
            Assert.Throws<ArgumentException>(() => taken.Fit(Rows(8, 0, width: 3), validation: null, new FitOptions(seed: 1))).Message,
            StringComparison.Ordinal);
    }

    [Fact]
    public void AFirstFitThatIsRefused_BuildsNothing_AndTheNextOneStartsFromItsOwnSeed()
    {
        var kept = new List<Checkpoint>();
        new Sequential().Dense(1).Compile(new Sgd(), new MeanSquaredError())
            .Fit(Rows(8, 0), validation: null, new FitOptions(seed: 1) { Checkpoints = new Checkpoints(kept.Add) });
        var compiled = new Sequential().Dense(1).Compile(new Sgd(), new MeanSquaredError());

        Assert.Throws<ArgumentException>(() => compiled.Fit(Rows(8, 0), validation: null, new FitOptions(seed: 2) { Epochs = 2, ResumeFrom = kept[0] }));
        Assert.Throws<InvalidOperationException>(() => compiled.Network);

        var fresh = new Sequential().Dense(1).Compile(new Sgd(), new MeanSquaredError());

        compiled.Fit(Rows(8, 0), validation: null, new FitOptions(seed: 2));
        fresh.Fit(Rows(8, 0), validation: null, new FitOptions(seed: 2));

        Assert.Equal(Bits(fresh.Network), Bits(compiled.Network));
    }

    // Rows of so many values whose answers are their sum, as many answers as asked for.
    private static TrainingData Rows(int count, int from, int width = 2, int answers = 1)
    {
        var features = Enumerable.Range(width * from, width * count).Select(at => MathF.Sin(at) / 2).ToArray();
        var sums = Enumerable.Range(0, count).SelectMany(row => Enumerable.Repeat(features.Skip(row * width).Take(width).Sum(), answers)).ToArray();

        return new TrainingData(Tensor.From(new Shape(count, width), features), Tensor.From(new Shape(count, answers), sums));
    }

    private static string Written(Network network)
    {
        using var stream = new MemoryStream();

        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            NetworkDocument.WriteNetwork(writer, network, new BinaryCrossEntropy());
        }

        return Encoding.UTF8.GetString(stream.ToArray());
    }

    private static int[][] Bits(Layer network) =>
        [.. network.Slots().Select(slot => slot.Slot.Value.Values.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray())];
}
