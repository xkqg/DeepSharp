// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Globalization;
using System.Text;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Measure;

/// <summary>A network and the rows it is trained on, made afresh for every run so no run is handed what the one before it learned.</summary>
/// <param name="Name">What the run stands for.</param>
/// <param name="Epochs">How many epochs a run trains.</param>
/// <param name="Make">Makes the network, compiled.</param>
/// <param name="Rows">Makes the rows, features and answers.</param>
internal readonly record struct TrainingWorkload(string Name, int Epochs, Func<CompiledNetwork> Make, Func<TrainingData> Rows);

/// <summary>
/// The runs a team makes of the light engine: a small table, a distribution over seventy bands, a few images — each trained as a
/// whole, as many epochs as it takes to count, alone and five at a time in one process, with what the run allocates and how
/// often the collector ran.
/// </summary>
internal sealed class TrainingWorkloads
{
    private static readonly TrainingWorkload[] Runs =
    [
        new("a table of 623 rows, 14 -> 16 -> 1, Adam", 20,
            () => new Sequential().Dense(16).Relu().Dense(1).Compile(new Adam(0.01), new BinaryCrossEntropy()),
            () => Rows(623, [14], 1, labels: 2)),
        new("640 flocks, 30 features -> 128 -> 70 bands, Adam", 5,
            () => new Sequential().Dense(128).Relu().Dense(70).Compile(new Adam(0.001), new CrossEntropy()),
            () => Rows(640, [30], 70, labels: 70)),
        new("64 images of 12 x 12, 3 x 3 convolution of 8, 10 classes", 2,
            () => new Sequential().Conv2D(8, new Window(3, 3)).Relu().Flatten().Dense(10).Compile(new Adam(0.001), new CrossEntropy()),
            () => Rows(64, [12, 12, 1], 10, labels: 10)),
    ];

    private readonly int _repeats;

    /// <summary>Runs measured, each trained this many times, the median kept.</summary>
    /// <param name="repeats">How many runs of each.</param>
    public TrainingWorkloads(int repeats) => _repeats = repeats;

    /// <summary>Trains every workload alone and then five at a time.</summary>
    /// <returns>A table, in Markdown.</returns>
    public string Table()
    {
        var table = new StringBuilder();

        table.AppendLine("| run | engine | threads | epochs | ms per epoch | bytes per epoch | gen-0 collections | wall ms of the whole |");
        table.AppendLine("|---|---|---:|---:|---:|---:|---:|---:|");

        foreach (var run in Runs)
        {
            foreach (var (engine, plain) in new[] { ("light", false), ("light without its shortcuts", true) })
            {
                foreach (var threads in new[] { 1, 5 })
                {
                    var cost = Median(run, threads, plain);

                    table.AppendLine(string.Create(
                        CultureInfo.InvariantCulture,
                        $"| {run.Name} | {engine} | {threads} | {run.Epochs} | {cost.Wall / run.Epochs:F2} | {cost.Bytes / (run.Epochs * threads)} | {cost.Collections} | {cost.Wall:F0} |"));
                }
            }
        }

        return table.ToString();
    }

    private Cost Median(TrainingWorkload run, int threads, bool plain)
    {
        Train(run, 1, plain);

        var costs = new List<Cost>();

        for (var repeat = 0; repeat < _repeats; repeat++)
        {
            costs.Add(Train(run, threads, plain));
        }

        return costs.OrderBy(cost => cost.Wall).ElementAt(costs.Count / 2);
    }

    // Trains a workload on this many threads of its own at once — dedicated ones, since a thread pool starved by a run is a
    // measurement of the pool — and says what it cost: the wall time of all, what was allocated, how often the collector ran.
    private static Cost Train(TrainingWorkload run, int threads, bool plain)
    {
        var bytes = GC.GetTotalAllocatedBytes(precise: true);
        var collections = GC.CollectionCount(0);
        var watch = Stopwatch.StartNew();
        var workers = Enumerable.Range(0, threads).Select(_ => new Thread(() => run.Make().Fit(run.Rows(), null, new FitOptions(seed: 7) { Epochs = run.Epochs, Backend = plain ? new PlainBackend(new CpuBackend()) : new CpuBackend() }))).ToArray();

        foreach (var worker in workers)
        {
            worker.Start();
        }

        foreach (var worker in workers)
        {
            worker.Join();
        }

        return new Cost(watch.Elapsed.TotalMilliseconds, GC.GetTotalAllocatedBytes(precise: true) - bytes, GC.CollectionCount(0) - collections);
    }

    private static TrainingData Rows(int count, int[] each, int answers, int labels)
    {
        var random = new Random(11);
        var width = each.Aggregate(1, (total, length) => total * length);
        var features = new float[count * width];
        var given = new float[count * answers];

        for (var at = 0; at < features.Length; at++)
        {
            features[at] = (float)((random.NextDouble() * 2) - 1);
        }

        for (var row = 0; row < count; row++)
        {
            var label = random.Next(labels);

            given[(row * answers) + (answers == 1 ? 0 : label)] = answers == 1 ? label : 1f;
        }

        return new TrainingData(Tensor.From(new Shape([count, .. each]), features), Tensor.From(new Shape(count, answers), given));
    }

    private readonly record struct Cost(double Wall, long Bytes, int Collections);
}
