// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;
using DeepSharp.Tensors;
using DeepSharp.Tests.Tensors;

namespace DeepSharp.Measure;

/// <summary>
/// The log-softmax of the rows a network ends in, and a large product spread over threads: 0.13.0's arithmetic against this build's,
/// then one thread against several, alone and with five at a time asking for the same.
/// </summary>
internal sealed class OtherWorkloads(Measurement measurement)
{
    /// <summary>Times the log-softmax of a few shapes, 0.13.0's arithmetic against this build's.</summary>
    /// <returns>A table, in Markdown.</returns>
    public string LogSoftmax()
    {
        var table = new StringBuilder();
        var backend = new CpuBackend();
        var random = new Random(5);

        table.AppendLine("| shape | 0.13.0 ns | this build ns | speed-up | 0.13.0 bytes | this build bytes | same bits |");
        table.AppendLine("|---|---:|---:|---:|---:|---:|---|");

        foreach (var (rows, columns) in new[] { (32, 10), (32, 70), (32, 1000), (256, 128) })
        {
            var matrix = Tensor.From(new Shape(rows, columns), [.. Enumerable.Range(0, rows * columns).Select(_ => (float)((random.NextDouble() * 20) - 10))]);
            var comparison = measurement.Between(() => ReleasedCpuArithmetic.LogSoftmax(matrix.Values, columns), () => backend.LogSoftmax(matrix));
            var same = ReleasedCpuArithmetic.LogSoftmax(matrix.Values, columns).AsSpan().SequenceEqual(backend.LogSoftmax(matrix).Values);

            table.AppendLine(string.Create(
                CultureInfo.InvariantCulture,
                $"| {rows}x{columns} | {comparison.Before.Nanoseconds:F0} | {comparison.After.Nanoseconds:F0} | {comparison.Speedup:F2} | {comparison.Before.Bytes} | {comparison.After.Bytes} | {(same ? "yes" : "NO")} |"));
        }

        return table.ToString();
    }

    /// <summary>Times large products on one thread and on several, and several of them asked for at once.</summary>
    /// <returns>A table, in Markdown.</returns>
    public string Spread()
    {
        var table = new StringBuilder();
        var random = new Random(9);

        table.AppendLine("| product | multiply-adds | threads | asked at once | ns per product | against one thread |");
        table.AppendLine("|---|---:|---:|---:|---:|---:|");

        foreach (var (rows, inner, columns) in new[] { (256, 256, 256), (512, 512, 512), (1024, 512, 512), (2048, 1024, 128) })
        {
            var left = Tensor.From(new Shape(rows, inner), [.. Enumerable.Range(0, rows * inner).Select(_ => (float)((random.NextDouble() * 2) - 1))]);
            var right = Tensor.From(new Shape(inner, columns), [.. Enumerable.Range(0, inner * columns).Select(_ => (float)((random.NextDouble() * 2) - 1))]);
            var one = new CpuBackend();
            double baseline = 0;

            foreach (var (threads, together) in new[] { (1, 1), (2, 1), (4, 1), (8, 1), (1, 5), (8, 5) })
            {
                var engine = new CpuBackend { Threads = threads };
                var ns = Timed(() => engine.MatMul(left, right), together);

                if (threads == 1 && together == 1)
                {
                    baseline = ns;
                }

                var against = together == 1 ? baseline : Timed(() => one.MatMul(left, right), together);

                table.AppendLine(string.Create(
                    CultureInfo.InvariantCulture,
                    $"| {rows}x{inner}·{inner}x{columns} | {(long)rows * inner * columns} | {threads} | {together} | {ns:F0} | {ns / against:F2} |"));
            }
        }

        return table.ToString();
    }

    // The time of one product when this many threads of their own ask for it at once, as the median of a few rounds.
    private static double Timed(Action product, int together)
    {
        var times = new List<double>();

        for (var round = 0; round < 5; round++)
        {
            var watch = System.Diagnostics.Stopwatch.StartNew();
            var workers = Enumerable.Range(0, together).Select(_ => new Thread(() =>
            {
                for (var call = 0; call < 4; call++)
                {
                    product();
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

            times.Add(watch.Elapsed.TotalMilliseconds * 1e6 / 4);
        }

        times.Sort();

        return times[times.Count / 2];
    }
}
