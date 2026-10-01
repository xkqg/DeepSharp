// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// A network described in Keras's words is read once and lowered onto the same stack of layers a network written as code
/// is made of: every width worked out from the shape of an example, every layer drawing what it starts from out of a
/// stream by its place — so nothing downstream can tell which door a network came through, down to its file.
/// </summary>
public abstract class SequentialContract(ITensorBackend engine)
{
    private readonly ITensorBackend _backend = engine;

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
    public void ANetworkInKerasWords_HoldingKerasNumbers_AnswersAsKerasDoes_WithinAHundredThousandth()
    {
        // Keras described and ran it (Fixtures/keras-same.py): windows padded as 'same' whose border no stated padding gives,
        // a batch normalisation and a layer normalisation with Keras's settings, and a last sigmoid. Its numbers go in as Keras
        // keeps them — a kernel's values in Keras's own order — and it answers as Keras did.
        var compiled = KerasDescription().Compile(new Adam(), new BinaryCrossEntropy());
        var images = Keras("images");
        compiled.Fit(new TrainingData(images, Keras("predictions")), validation: null, new FitOptions(seed: 1) { Backend = _backend, BatchSize = 6 });

        compiled.Network.Load(KerasSlots(compiled.Network));

        Assert.Equal(9, Assert.IsType<LayerStack>(compiled.Network).Layers.Count);
        AssertWithin(1e-5, Keras("logits"), compiled.Network.Forward(images, Pass.Evaluation(_backend)));
        AssertWithin(1e-5, Keras("predictions"), compiled.Predict(images, _backend));
    }

    [Fact]
    public void ABatchNormalisationInKerasWords_MovesItsRunningMeanAsKerasDoes_AndItsVarianceByTheBatchsCountedOverOneRowFewer()
    {
        // Keras's momentum is the share the running statistics keep, and the layer keeps its complement, PyTorch's. Keras moves
        // the running variance towards the batch's variance over all its rows; this layer, as PyTorch's does, towards the
        // batch's variance over one row fewer: 96 places of six images, so the batch's share is 96/95 of Keras's.
        var compiled = KerasDescription().Compile(new Adam(), new BinaryCrossEntropy());
        var images = Keras("images");
        compiled.Fit(new TrainingData(images, Keras("predictions")), validation: null, new FitOptions(seed: 1) { Backend = _backend, BatchSize = 6 });
        compiled.Network.Load(KerasSlots(compiled.Network));
        var norm = Assert.IsType<BatchNorm>(Assert.IsType<LayerStack>(compiled.Network).Layers[1]);
        var before = norm.RunningVariance.Value.Values.ToArray();

        compiled.Network.Forward(images, Pass.Training(_backend, new RandomStream(1), 0, 0));

        var moved = Trained("1.running_var");
        var counted = before.Select((kept, at) => (0.9 * kept) + ((moved[at] - (0.9 * kept)) * 96 / 95)).ToArray();

        Assert.Equal(1 - 0.9, norm.Momentum, 15);
        Assert.Equal(1e-3, norm.Epsilon);
        AssertWithin(1e-6, Trained("1.running_mean"), norm.RunningMean.Value);
        AssertWithin(1e-6, counted, norm.RunningVariance.Value);
    }

    [Fact]
    public void ALastSigmoidItsLossAppliesItself_IsLiftedIntoTheLoss_SoEveryPredictionGoesThroughOneSigmoid()
    {
        // Keras's habit: the activation on the last layer, and a loss that takes what it gives. Here the loss takes the logits
        // and applies the sigmoid itself, so the words leave the last one out of the network; 0.4.0 kept it, and every
        // prediction went through the sigmoid twice — a logit of minus twenty came out a half.
        var rows = Tensor.From(new Shape(4, 1), [-20f, -1f, 1f, 20f]);
        var compiled = new Sequential().Dense(1).Sigmoid().Compile(new Adam(), new BinaryCrossEntropy());
        compiled.Fit(new TrainingData(rows, Tensor.From(new Shape(4, 1), [0f, 0f, 1f, 1f])), validation: null, new FitOptions(seed: 7) { Backend = _backend });
        compiled.Network.Load([new SlotEntry("0.weight", Tensor.From(new Shape(1, 1), [1f]), "one"), new SlotEntry("0.bias", Tensor.Zeros(new Shape(1)), "none")]);

        var predictions = compiled.Predict(rows, _backend);

        Assert.IsType<Dense>(Assert.Single(Assert.IsType<LayerStack>(compiled.Network).Layers));
        AssertWithin(1e-6, [2.0611536e-9, 0.2689414213699951, 0.7310585786300049, 0.9999999979388464], predictions);
    }

    [Fact]
    public void ALastSigmoid_IsKeptBeforeALossThatDoesNotApplyIt_AndAloneBeforeOneThatDoes_IsRefused()
    {
        var rows = new TrainingData(Tensor.From(new Shape(4, 2), [0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f]), Tensor.From(new Shape(4, 2), [0.5f, 0.5f, 0.25f, 0.75f, 1f, 0f, 0f, 1f]));
        var squared = new Sequential().Dense(2).Sigmoid().Compile(new Adam(), new MeanSquaredError());
        var shares = new Sequential().Dense(2).Sigmoid().Compile(new Adam(), new CrossEntropy());

        squared.Fit(rows, validation: null, new FitOptions(seed: 1) { Backend = _backend });
        shares.Fit(rows, validation: null, new FitOptions(seed: 1) { Backend = _backend });

        Assert.IsType<Sigmoid>(Assert.IsType<LayerStack>(squared.Network).Layers[^1]);
        Assert.IsType<Sigmoid>(Assert.IsType<LayerStack>(shares.Network).Layers[^1]);
        Assert.Contains(
            "nothing but the sigmoid its loss applies itself",
            Assert.Throws<InvalidOperationException>(() => new Sequential().Sigmoid().Compile(new Adam(), new BinaryCrossEntropy())).Message,
            StringComparison.Ordinal);
    }

    [Fact]
    public void TheNormalisationWords_TakeKerasMomentumAndEpsilon_AndWithoutThemArePyTorchs()
    {
        var keras = new Sequential().Dense(4).BatchNorm(0.99, 1e-3).LayerNorm(1e-3).Lower(new Shape(3), new RandomStream(1));
        var plain = new Sequential().Dense(4).BatchNorm().LayerNorm().Lower(new Shape(3), new RandomStream(1));

        Assert.Equal(1 - 0.99, Assert.IsType<BatchNorm>(keras.Layers[1]).Momentum, 15);
        Assert.Equal(1e-3, ((BatchNorm)keras.Layers[1]).Epsilon);
        Assert.Equal(1e-3, Assert.IsType<LayerNorm>(keras.Layers[2]).Epsilon);
        Assert.Equal(0.1, Assert.IsType<BatchNorm>(plain.Layers[1]).Momentum);
        Assert.Equal(1e-5, ((BatchNorm)plain.Layers[1]).Epsilon);
        Assert.Equal(1e-5, Assert.IsType<LayerNorm>(plain.Layers[2]).Epsilon);
        Assert.Equal(0, new Sequential().Dense(4).BatchNorm(1, 1e-3).Lower(new Shape(3), new RandomStream(1)).Layers.OfType<BatchNorm>().Single().Momentum);
    }

    [Fact]
    public void AConvolutionWordPaddedAsSame_KeepsAsManyPlacesAsTheStrideFits_EvenThroughAWindowLargerThanTheImage()
    {
        var stack = new Sequential().Reshape(new Shape(2, 3, 1)).Conv2D(2, new Window(5, 5) { Stride = 2, PaddingMode = PaddingMode.Same }).Flatten()
            .Lower(new Shape(6), new RandomStream(1));

        Assert.Equal(new Shape(1, 4), stack.Forward(Tensor.Zeros(new Shape(1, 6)), Pass.Evaluation(_backend)).Shape);
        Assert.Equal(PaddingMode.Same, Assert.IsType<Conv2D>(stack.Layers[1]).Window.PaddingMode);
    }

    [Fact]
    public void AWordWrittenWrongly_IsRefusedWhereItIsWritten()
    {
        var description = new Sequential();

        Assert.Throws<ArgumentOutOfRangeException>(() => description.Dense(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => description.Dropout(1));
        Assert.Equal("momentum", Assert.Throws<ArgumentOutOfRangeException>(() => description.BatchNorm(1.5, 1e-3)).ParamName);
        Assert.Equal("momentum", Assert.Throws<ArgumentOutOfRangeException>(() => description.BatchNorm(-0.1, 1e-3)).ParamName);
        Assert.Equal("momentum", Assert.Throws<ArgumentOutOfRangeException>(() => description.BatchNorm(double.NaN, 1e-3)).ParamName);
        Assert.Equal("epsilon", Assert.Throws<ArgumentOutOfRangeException>(() => description.BatchNorm(0.99, 0)).ParamName);
        Assert.Equal("epsilon", Assert.Throws<ArgumentOutOfRangeException>(() => description.BatchNorm(0.99, double.PositiveInfinity)).ParamName);
        Assert.Equal("epsilon", Assert.Throws<ArgumentOutOfRangeException>(() => description.LayerNorm(-1e-3)).ParamName);
        Assert.Equal("epsilon", Assert.Throws<ArgumentOutOfRangeException>(() => description.LayerNorm(double.NaN)).ParamName);
        Assert.Throws<ArgumentException>(() => description.Conv2D(2, new Window(3, 3) { Padding = 1, PaddingMode = PaddingMode.Same }));
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
        var options = new FitOptions(seed: 11) { Backend = _backend, Epochs = 3, BatchSize = 8 };

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
        Assert.Contains("without a last activation its loss applies itself, which compiling leaves out", wrong.Message, StringComparison.Ordinal);
        Assert.Equal(wrong.Message, Assert.Throws<InvalidOperationException>(() => compiled.Predict(Tensor.Zeros(new Shape(1, 2)), _backend)).Message);

        // Handed nothing to predict for, or nowhere to run it, it says so before anything else, as it always has.
        Assert.Equal("features", Assert.Throws<ArgumentNullException>(() => compiled.Predict(null!, _backend)).ParamName);
        Assert.Equal("backend", Assert.Throws<ArgumentNullException>(() => compiled.Predict(Tensor.Zeros(new Shape(1, 2)), null!)).ParamName);
    }

    [Fact]
    public void AWordWrittenAfterCompile_ChangesNothingThatWasCompiled()
    {
        var description = new Sequential().Dense(3);
        var compiled = description.Compile(new Sgd(), new MeanSquaredError());

        description.Relu().Dense(1);
        compiled.Fit(Rows(8, 0, answers: 3), validation: null, new FitOptions(seed: 1) { Backend = _backend });

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
            Assert.Throws<ArgumentException>(() => stated.Fit(Rows(8, 0, width: 3), validation: null, new FitOptions(seed: 1) { Backend = _backend })).Message);
        Assert.Equal(
            "validation",
            Assert.Throws<ArgumentException>(() => stated.Fit(Rows(8, 0), Rows(4, 8, width: 3), new FitOptions(seed: 1) { Backend = _backend, EarlyStopping = new EarlyStopping() })).ParamName);

        // A network whose first fit took its shape from the rows keeps it: later rows are held to it.
        taken.Fit(Rows(8, 0), validation: null, new FitOptions(seed: 1) { Backend = _backend });

        Assert.Contains(
            "takes each example as 2",
            Assert.Throws<ArgumentException>(() => taken.Fit(Rows(8, 0, width: 3), validation: null, new FitOptions(seed: 1) { Backend = _backend })).Message,
            StringComparison.Ordinal);
    }

    [Fact]
    public void AFirstFitThatIsRefused_BuildsNothing_AndTheNextOneStartsFromItsOwnSeed()
    {
        var kept = new List<Checkpoint>();
        new Sequential().Dense(1).Compile(new Sgd(), new MeanSquaredError())
            .Fit(Rows(8, 0), validation: null, new FitOptions(seed: 1) { Backend = _backend, Checkpoints = new Checkpoints(kept.Add) });
        var compiled = new Sequential().Dense(1).Compile(new Sgd(), new MeanSquaredError());

        Assert.Throws<ArgumentException>(() => compiled.Fit(Rows(8, 0), validation: null, new FitOptions(seed: 2) { Backend = _backend, Epochs = 2, ResumeFrom = kept[0] }));
        Assert.Throws<InvalidOperationException>(() => compiled.Network);

        var fresh = new Sequential().Dense(1).Compile(new Sgd(), new MeanSquaredError());

        compiled.Fit(Rows(8, 0), validation: null, new FitOptions(seed: 2) { Backend = _backend });
        fresh.Fit(Rows(8, 0), validation: null, new FitOptions(seed: 2) { Backend = _backend });

        Assert.Equal(Bits(fresh.Network), Bits(compiled.Network));
    }

    // The network Fixtures/keras-same.py describes to Keras, in Keras's words here: its ten layers, the last sigmoid among them.
    private static Sequential KerasDescription() =>
        new Sequential()
            .Input(new Shape(7, 8, 2))
            .Conv2D(4, new Window(4, 3) { Stride = 2, PaddingMode = PaddingMode.Same })
            .BatchNorm(0.9, 1e-3)
            .Relu()
            .Conv2D(3, new Window(2, 2) { PaddingMode = PaddingMode.Same })
            .Flatten()
            .Dense(5)
            .LayerNorm(1e-3)
            .Tanh()
            .Dense(1)
            .Sigmoid();

    private static JsonElement KerasFixture { get; } = JsonDocument.Parse(
        File.ReadAllText(Path.Join(Repository.Root, "Tst", "DeepSharp.Backends.Contract", "Networks", "Fixtures", "keras-same.json"))).RootElement;

    // A tensor the fixture holds, in the shape Keras gave it.
    private static Tensor Keras(string name)
    {
        var written = KerasFixture.GetProperty(name);

        return Tensor.From(new Shape([.. written.GetProperty("shape").EnumerateArray().Select(length => length.GetInt32())]), Values(written));
    }

    private static float[] Values(JsonElement written) => [.. written.GetProperty("values").EnumerateArray().Select(value => value.GetSingle())];

    private static float[] Trained(string path) => [.. KerasFixture.GetProperty("trained").GetProperty(path).EnumerateArray().Select(value => value.GetSingle())];

    // Every number Keras keeps, as Keras keeps it, into the slot of the network in its words that stands where it stood: its
    // values in the order Keras lays them out, under the slot's shape, which holds as many.
    private static IEnumerable<SlotEntry> KerasSlots(Network network) =>
        network.Slots().Select(named =>
        {
            var written = KerasFixture.GetProperty("slots").GetProperty(named.Path);
            var values = Values(written);

            Assert.Equal(named.Slot.Value.Shape.Count, values.Length);

            return new SlotEntry(named.Path, Tensor.From(named.Slot.Value.Shape, values), written.GetProperty("keras").GetString()!);
        });

    private static void AssertWithin(double tolerance, Tensor expected, Tensor actual) =>
        AssertWithin(tolerance, [.. expected.Values.ToArray().Select(value => (double)value)], actual);

    private static void AssertWithin(double tolerance, float[] expected, Tensor actual) =>
        AssertWithin(tolerance, [.. expected.Select(value => (double)value)], actual);

    private static void AssertWithin(double tolerance, double[] expected, Tensor actual)
    {
        Assert.Equal(expected.Length, actual.Values.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.True(
                Math.Abs(expected[at] - actual.Values[at]) <= tolerance,
                string.Create(CultureInfo.InvariantCulture, $"[{at}]: {actual.Values[at]} against {expected[at]}, {Math.Abs(expected[at] - actual.Values[at]):G3} apart"));
        }
    }

    // Rows of so many values whose answers are their sum, as many answers as asked for.
    private static TrainingData Rows(int count, int from, int width = 2, int answers = 1)
    {
        var features = Enumerable.Range(width * from, width * count).Select(at => MathF.Sin(at) / 2).ToArray();
        var sums = Enumerable.Range(0, count).SelectMany(row => Enumerable.Repeat(features.Skip(row * width).Take(width).Sum(), answers)).ToArray();

        return new TrainingData(Tensor.From(new Shape(count, width), features), Tensor.From(new Shape(count, answers), sums));
    }

    private static int[][] Bits(Layer network) =>
        [.. network.Slots().Select(slot => slot.Slot.Value.Values.ToArray().Select(BitConverter.SingleToInt32Bits).ToArray())];
}
