// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using DeepSharp.Backends.TorchSharp;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Torch;

/// <summary>
/// The once-seen libtorch error, hunted where it was seen: whole trainings on libtorch, five at a time in one process, each on an
/// engine of its own, run by hand until a count of them has gone by. Explicit: a soak is read, not gated.
/// </summary>
public class ParallelTrainingSoakTests
{
    [Fact(Explicit = true)]
    public void WholeTrainingsOnLibtorch_FiveAtATimeInOneProcess_NeverRaiseWhatOneRaisedOnce()
    {
        var seconds = int.Parse(Environment.GetEnvironmentVariable("DEEPSHARP_SOAK_SECONDS") ?? "60", CultureInfo.InvariantCulture);
        var threads = int.Parse(Environment.GetEnvironmentVariable("DEEPSHARP_SOAK_THREADS") ?? "5", CultureInfo.InvariantCulture);
        var shared = Environment.GetEnvironmentVariable("DEEPSHARP_SOAK_SHARED") == "1";
        var data = Rows();
        var trainings = 0L;
        var faults = new ConcurrentQueue<string>();
        var clock = Stopwatch.StartNew();
        var workers = Enumerable.Range(0, threads).Select(worker => new Thread(() =>
        {
            for (var seed = worker * 100_000L; clock.Elapsed.TotalSeconds < seconds; seed++)
            {
                try
                {
                    new Sequential().Dense(16).Relu().Dense(8).Relu().Dense(1).Compile(new Adam(0.01), new BinaryCrossEntropy())
                        .Fit(shared ? data : Rows(), null, new FitOptions(seed) { Epochs = 8, BatchSize = 32, Backend = TorchBackend.OnCpu() });
                    Interlocked.Increment(ref trainings);
                }
                catch (Exception fault)
                {
                    faults.Enqueue($"{fault.GetType().Name}: {fault.Message}");
                }
            }
        })).ToArray();

        foreach (var worker in workers)
        {
            worker.Start();
        }

        foreach (var worker in workers)
        {
            worker.Join();
        }

        var report = $"{trainings} trainings on {threads} threads in {seconds} s, {faults.Count} faults{(faults.IsEmpty ? string.Empty : ": " + string.Join(" | ", faults.Distinct().Take(5)))}";

        File.WriteAllText(Path.Join(Path.GetTempPath(), "training-soak.txt"), report);
        TestContext.Current.SendDiagnosticMessage(report);
        Assert.True(faults.IsEmpty, report);
    }

    private static TrainingData Rows()
    {
        var random = new Random(3);
        var features = new float[400 * 10];
        var answers = new float[400];

        for (var at = 0; at < features.Length; at++)
        {
            features[at] = (float)((random.NextDouble() * 2) - 1);
        }

        for (var row = 0; row < 400; row++)
        {
            answers[row] = features[row * 10] > 0 ? 1f : 0f;
        }

        return new TrainingData(Tensor.From(new Shape(400, 10), features), Tensor.From(new Shape(400, 1), answers));
    }
}
