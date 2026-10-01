// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Backends.TorchSharp;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;
using TorchSharp;

namespace DeepSharp.Tests.Backends.Torch;

/// <summary>
/// What libtorch keeps for the whole process is the application's, and the engine leaves it alone: how many threads libtorch
/// works with, and its random generator — every draw a run makes is DeepSharp's own, counted from the run's seed. How many
/// threads there are changes where the time goes, not what a run comes to.
/// </summary>
[Collection(nameof(LibtorchCounted))]
public class ProcessStateTests
{
    [Fact]
    public void ARunOnTheEngine_LeavesLibtorchsThreadsAndItsGenerator_AsItFoundThem()
    {
        var threads = torch.get_num_threads();
        var interop = torch.get_num_interop_threads();
        var generator = torch.random.get_rng_state().bytes.ToArray();

        // Dropout draws a mask every step; the engine is handed it, drawn from the run's own stream.
        new Sequential().Dense(16).Relu().Dropout(0.25).Dense(1)
            .Compile(new Adam(0.01), new BinaryCrossEntropy())
            .Fit(Handed(Passengers.Prepared(), rows: 623), validation: null, new FitOptions(seed: 3) { Backend = TorchBackend.OnCpu(), Epochs = 2 });

        Assert.Equal(threads, torch.get_num_threads());
        Assert.Equal(interop, torch.get_num_interop_threads());
        Assert.Equal(generator, torch.random.get_rng_state().bytes.ToArray());
    }

    [Fact]
    public void TwoHundredTitanicStepsAtOneThread_ComeToTheBitsTheyComeToAtTheApplicationsThreadCount()
    {
        var threads = torch.get_num_threads();
        var atTheApplications = Stepped();
        float[][] atOne;

        try
        {
            torch.set_num_threads(1);
            atOne = Stepped();
        }
        finally
        {
            torch.set_num_threads(threads);
        }

        Assert.Equal(atTheApplications, atOne);
    }

    // Two hundred training steps of the Titanic network on the first batch of the passenger list, on a fresh engine: the
    // parameters they end at.
    private static float[][] Stepped()
    {
        var engine = TorchBackend.OnCpu();
        var batch = Handed(Passengers.Prepared(), rows: 32);
        var stream = new RandomStream(20260929);
        var network = new LayerStack(new Dense(14, 16, stream.Draw("initialise:0", 0, 0)), new Relu(), new Dense(16, 1, stream.Draw("initialise:2", 0, 0)));
        var parameters = network.Parameters().ToArray();
        var adam = new Adam(0.01);

        for (var step = 0; step < 200; step++)
        {
            var recording = new RecordingBackend(engine);
            var loss = new BinaryCrossEntropy().Of(network.Forward(batch.Features, Pass.Training(recording, stream, 0, step)), batch.Answers, recording);

            adam.Step(parameters, recording.GradientsOf(loss, parameters.Select(parameter => parameter.Value)), 0.01, engine);
        }

        return [.. parameters.Select(parameter => parameter.Value.Values.ToArray())];
    }

    // The first so many training rows of a run, as the loop takes them.
    private static TrainingData Handed(PreparedData prepared, int rows)
    {
        var batch = prepared.Batch(Part.Train, Needs.OneScale);

        return new TrainingData(
            Tensor.From(new Shape(rows, batch.Width), [.. batch.Features.Take(rows).SelectMany(row => row.Select(value => (float)value))]),
            Tensor.From(new Shape(rows, 1), [.. batch.Answers!.Take(rows).Select(row => (float)row[0])]));
    }
}
