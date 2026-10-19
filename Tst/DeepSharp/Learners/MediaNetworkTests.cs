// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;
using DeepSharp.Tests.Import;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A network trained on rows that are images takes its example as the shape the pipeline hands over beside them, instead of the image's
/// size being stated twice — once to the reader and once to the network. A convolution reads an image whole, a network that is told another
/// size than the rows hold is refused, and a network trained further from Keras's words is handed the same rows.
/// </summary>
public class MediaNetworkTests
{
    private const int Side = 6;

    // Images of 6 by 6, the answer being whether the left half is brighter than the right.
    private static InMemoryRowSource Images(int count, int offset = 0)
    {
        var random = new Random(41 + offset);

        return new InMemoryRowSource(
            ["image", "bright"],
            [.. Enumerable.Range(0, count).Select(_ =>
            {
                var leftBright = random.NextDouble() < 0.5;
                var pixels = new byte[Side * Side];

                for (var row = 0; row < Side; row++)
                {
                    for (var column = 0; column < Side; column++)
                    {
                        var bright = (column < Side / 2) == leftBright;

                        pixels[(row * Side) + column] = (byte)((bright ? 170 : 60) + random.Next(40));
                    }
                }

                return (IReadOnlyList<string?>)[MediaCell.Pixels([Side, Side, 1], pixels), leftBright ? "1" : "0"];
            })]);
    }

    private static PreparedData Prepared(Func<NetworkDeclaration, NetworkDeclaration>? declared = null) =>
        Pdd.Create()
            .Read(Images(300), "three hundred images")
            .Declare(schema => schema.Media("image").Integer("bright"))
            .SplitAtRandom(0.60, 0.20, 3)
            .Target("bright")
            .Report(report => report.Measure(Metric.Accuracy).On(Part.Validation, Part.Test).As(Shown.Numbers))
            .WithTorch(network => (declared ?? (net => net.Flatten().Dense(8).Relu().Dense(1)))(network).Adam(0.02).BinaryCrossEntropy().Run(seed: 1, epochs: 25))
            .Build()
            .Run();

    [Fact]
    public void AConvolution_TakesTheImageAsItIs_FromTheShapeTheRowsCarry()
    {
        var prepared = Prepared();
        var compiled = new Sequential()
            .Conv2D(4, new Window(3, 3))
            .Relu()
            .Flatten()
            .Dense(1)
            .Compile(new Adam(0.02), new BinaryCrossEntropy());
        var trained = compiled.Fit(prepared, new FitOptions(seed: 1) { Epochs = 25 });
        var accuracy = trained.Measures!.Parts.Single(part => part.Part == Part.Test).Values.Single(value => value.Metric == Metric.Accuracy);

        Assert.True(accuracy.Value > 0.9, $"accuracy {accuracy.Value}");
        Assert.Equal(Side * Side, trained.TrainedOn.Features.Count);
    }

    [Fact]
    public void ARowServedLater_IsAnsweredAsATrainingRowIs_ThroughTheSameShape()
    {
        var prepared = Prepared();
        var trained = new Sequential().Conv2D(4, new Window(3, 3)).Relu().Flatten().Dense(1).Compile(new Adam(0.02), new BinaryCrossEntropy())
            .Fit(prepared, new FitOptions(seed: 1) { Epochs = 25 });
        var predicted = trained.Predict(Images(40, offset: 5));
        var held = Images(40, offset: 5).Rows.Select(row => row[1] == "1" ? 1.0 : 0.0).ToArray();
        var right = Enumerable.Range(0, 40).Count(at => (predicted.Answers[at][0] >= 0.5 ? 1.0 : 0.0) == held[at]);

        Assert.Equal(40, predicted.Answers.Count);
        Assert.True(right >= 34, $"{right} of 40");
    }

    [Fact]
    public void ANetworkThatIsToldAnotherSizeThanTheRowsHold_IsRefused()
    {
        var prepared = Prepared();
        var compiled = new Sequential().Input(new Shape(8, 8, 1)).Conv2D(4, new Window(3, 3)).Flatten().Dense(1).Compile(new Adam(0.02), new BinaryCrossEntropy());
        var refused = Assert.Throws<ArgumentException>(() => compiled.Fit(prepared, new FitOptions(seed: 1) { Epochs = 2 }));

        Assert.Contains("8x8x1", refused.Message.Replace(", ", "x").Replace("(", string.Empty).Replace(")", string.Empty), StringComparison.Ordinal);
    }

    [Fact]
    public void ADeclaredNetwork_FlattensWhatItIsHanded_VariesItWhileTraining_AndTrainsBehindThePipeline()
    {
        var trained = Prepared(net => net.RandomFlip(seed: 7).RandomCrop(pad: 1, seed: 8).GaussianNoise(spread: 0.05, seed: 9).Flatten().Dense(12).Relu().Dense(1)).Train();
        var accuracy = trained.Measures!.Parts.Single(part => part.Part == Part.Test).Values.Single(value => value.Metric == Metric.Accuracy);

        Assert.InRange(accuracy.Value, 0.5, 1.0);
        Assert.Equal(Side * Side, trained.TrainedOn.Features.Count);
        Assert.Equal(["image.0", "image.1"], trained.TrainedOn.Features.Take(2));
    }

    [Fact]
    public void TheVariationsOfADeclaredNetwork_AreHeldFirst_AndTheFileSaysThem()
    {
        var declared = Assert.IsType<LearnNetworkStep>(
            Prepared(net => net.RandomFlip(seed: 7, horizontal: true, vertical: true).GaussianNoise(spread: 0.1, seed: 9).Flatten().Dense(1)).Declaration.Learner);

        Assert.Equal(["randomFlip", "gaussianNoise", "flatten", "dense"], declared.Layers.Select(layer => layer.Kind));
        Assert.Throws<ArgumentException>(() => Prepared(net => net.Flatten().RandomFlip(seed: 1).Dense(1)));
    }

    [Fact]
    public void VariedRows_AreNeverTheRowsOfTheReport_ThePipelineAnswersAsItAlwaysDid()
    {
        var trained = Prepared(net => net.RandomFlip(seed: 7).Flatten().Dense(1)).Train();
        var again = trained.Predict(Images(5, offset: 9)).Answers;
        var twice = trained.Predict(Images(5, offset: 9)).Answers;

        Assert.Equal(again, twice);
    }

    [Fact]
    public void AConvolutionalModelKerasTrained_IsTrainedFurtherOnMediaRows_WithoutTheImagesSizeStatedTwice()
    {
        // The model reads an image of 9 rows, 8 columns and 2 channels and answers three classes; the rows say each as the reader wrote it.
        var random = new Random(77);
        var rows = Enumerable.Range(0, 90).Select(at =>
        {
            var pixels = new byte[9 * 8 * 2];
            var kind = at % 3;

            for (var pixel = 0; pixel < pixels.Length; pixel++)
            {
                pixels[pixel] = (byte)(random.Next(80) + (kind * 60));
            }

            return (IReadOnlyList<string?>)[MediaCell.Pixels([9, 8, 2], pixels), kind == 0 ? "1" : "0", kind == 1 ? "1" : "0", kind == 2 ? "1" : "0"];
        });
        var prepared = Pdd.Create()
            .Read(new InMemoryRowSource(["image", "a", "b", "c"], rows), "ninety images")
            .Declare(schema => schema.Media("image").Integer("a", "b", "c"))
            .SplitAtRandom(0.60, 0.20, 3)
            .Labels(["a", "b", "c"], ones: 1)
            .Report(report => report.Measure(Metric.Accuracy).On(Part.Validation, Part.Test).As(Shown.Numbers))
            .Build()
            .Run();
        var saved = new DeepSharp.Import.Keras.KerasFile().Read(KerasFixtures.Open("keras-spatial-image.keras"));
        var trained = saved.Network.Compile(new Adam(0.01), saved.Loss).Fit(prepared, new FitOptions(seed: 1) { Epochs = 4 });

        Assert.Equal(4, trained.History!.Epochs.Count);
        Assert.Equal(9 * 8 * 2, trained.TrainedOn.Features.Count);
        Assert.NotNull(trained.Measures);
    }
}
