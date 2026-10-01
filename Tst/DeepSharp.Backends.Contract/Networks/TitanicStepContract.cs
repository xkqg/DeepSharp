// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// One training step of the published passenger list as the loop takes it — the first thirty-two rows of the run's first
/// shuffle, fourteen features on one scale, a dense layer of sixteen, a rectifier, one output, a binary cross-entropy and
/// Adam at a hundredth — on this engine beside the light one. Every total of the step, its outputs, its loss and its four
/// gradients, lies within three roundings of the size of the terms it adds up; every parameter Adam moves lies within
/// PyTorch's float tolerance. A step, not a run: engines that round in another order drift apart over many steps, as any
/// two float engines do.
/// </summary>
/// <param name="engine">The engine held to the light one.</param>
public abstract class TitanicStepContract(ITensorBackend engine)
{
    private const long Seed = 20260929;

    // The network's slots, in their order.
    private static readonly string[] Slots = ["0.weight", "0.bias", "2.weight", "2.bias"];

    private readonly ITensorBackend _backend = engine;

    [Fact]
    public void OneTrainingStep_ComesToTheLightEngines_EveryTotalWithinThreeRoundingsOfItsTerms_EveryParameterWithinPyTorchsTolerance()
    {
        var batch = FirstBatch();
        var light = Stepped(new CpuBackend(), batch);
        var own = Stepped(_backend, batch);
        var sizes = TitanicStepSizes.Of(batch, light.Start);

        WithinRoundings("outputs", light.Outputs, own.Outputs, sizes.Outputs);
        WithinRoundings("loss", [light.Loss], [own.Loss], [sizes.Loss]);

        for (var slot = 0; slot < light.Gradients.Length; slot++)
        {
            WithinRoundings($"gradient of {Slots[slot]}", light.Gradients[slot], own.Gradients[slot], sizes.Gradients[slot]);
            WithinTolerance($"{Slots[slot]} after Adam", light.After[slot], own.After[slot]);
        }
    }

    // The loop's first batch of the first epoch: the run's shuffle of the training rows, the first thirty-two of it.
    private static TitanicBatch FirstBatch()
    {
        var train = Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("pclass", "sex").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            .FillMissing("age", With.Median)
            .EncodeCategories()
            .Normalise("age", Scale.MidRange)
            .Normalise("fare", Scale.MidRange)
            .Normalise("sibsp", Scale.MidRange)
            .Normalise("parch", Scale.MidRange)
            .Target("survived")
            .Build()
            .Run()
            .Batch(Part.Train, Needs.OneScale);
        var places = new RandomStream(Seed).Draw("shuffle", 0, 0).Permutation(train.RowCount)[..32];

        return new TitanicBatch(
            [.. places.SelectMany(place => train.Features[place].Select(value => (float)value))],
            [.. places.Select(place => (float)train.Answers![place][0])],
            train.Width);
    }

    // The step as the loop takes it, on one engine: the forward pass through a recording, the loss, the gradients worked
    // back, and Adam's step.
    private static TakenStep Stepped(ITensorBackend engine, TitanicBatch batch)
    {
        var network = new Sequential().Dense(16).Relu().Dense(1).Lower(new Shape(batch.Width), new RandomStream(Seed));
        var parameters = network.Parameters().ToArray();
        var start = parameters.Select(parameter => parameter.Value.Values.ToArray()).ToArray();
        var recording = new RecordingBackend(engine);

        var outputs = network.Forward(Tensor.From(new Shape(batch.Rows, batch.Width), batch.Features), Pass.Training(recording, new RandomStream(Seed), 0, 0));
        var loss = new BinaryCrossEntropy().Of(outputs, Tensor.From(new Shape(batch.Rows, 1), batch.Answers), recording);
        var gradients = recording.GradientsOf(loss, parameters.Select(parameter => parameter.Value));
        var taken = parameters.Select(parameter => gradients[parameter.Value].Values.ToArray()).ToArray();

        new Adam(0.01).Step(parameters, gradients, 0.01, engine);

        return new TakenStep(start, outputs.Values.ToArray(), loss.Values[0], taken, [.. parameters.Select(parameter => parameter.Value.Values.ToArray())]);
    }

    private static void WithinRoundings(string what, float[] light, float[] own, double[] sizes)
    {
        Assert.Equal(light.Length, own.Length);

        for (var at = 0; at < light.Length; at++)
        {
            var apart = Math.Abs((double)own[at] - light[at]);

            Assert.True(
                apart <= AgreementContract.Bound(sizes[at]),
                string.Create(CultureInfo.InvariantCulture, $"{what}[{at}]: {own[at]} against {light[at]}, {apart / (AgreementContract.Roundoff * sizes[at]):0.###} roundings of the terms' size {sizes[at]}"));
        }
    }

    private static void WithinTolerance(string what, float[] light, float[] own)
    {
        Assert.Equal(light.Length, own.Length);

        for (var at = 0; at < light.Length; at++)
        {
            Assert.True(
                Math.Abs((double)own[at] - light[at]) <= AgreementContract.AbsoluteTolerance + (AgreementContract.RelativeTolerance * Math.Abs(light[at])),
                string.Create(CultureInfo.InvariantCulture, $"{what}[{at}]: {own[at]} against {light[at]}"));
        }
    }

    // What one engine's step came to: the parameters it started from, its outputs, its loss, the four gradients and the
    // four parameters after Adam's step.
    private sealed record TakenStep(float[][] Start, float[] Outputs, float Loss, float[][] Gradients, float[][] After);
}

/// <summary>So many rows of features, row-major, and the answer of each.</summary>
/// <param name="Features">The rows, row after row.</param>
/// <param name="Answers">Whether each passenger survived.</param>
/// <param name="Width">How many features a row holds.</param>
internal sealed record TitanicBatch(float[] Features, float[] Answers, int Width)
{
    /// <summary>How many rows there are.</summary>
    public int Rows => Answers.Length;
}
