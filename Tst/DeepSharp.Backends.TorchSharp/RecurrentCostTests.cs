// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Globalization;
using System.Text;
using DeepSharp.Backends.TorchSharp;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Torch;

/// <summary>
/// What a recurrent layer costs, measured by hand on the two engines: an LSTM of a batch of 32 series, 16 channels and 64 hidden numbers, read
/// forwards and trained back through, at 16, 64 and 256 steps. The time of each is split by what the products are: a product with a vector, which
/// picks a step out of the series or places one into the answer, and the rest. It is the evidence for the question whether the seam should learn
/// to cut a row out of a matrix. Explicit: a measurement is read, not gated; it writes <c>recurrent-cost.md</c> in the temporary folder.
/// </summary>
public class RecurrentCostTests
{
    private const int Series = 32;
    private const int Channels = 16;
    private const int Hidden = 64;

    [Fact(Explicit = true)]
    public void TheCostOfAnLstmAtSixteenSixtyFourAndTwoHundredFiftySixSteps_IsWrittenDown()
    {
        var report = new StringBuilder();

        report.AppendLine("| engine | steps | ms for the forward and the backward pass | picks and places, ms | share | other products, ms | transposes and reshapes, ms | bytes allocated |");
        report.AppendLine("|---|---:|---:|---:|---:|---:|---:|---:|");

        foreach (var engine in new ITensorBackend[] { new CpuBackend(), TorchBackend.OnCpu() })
        {
            foreach (var steps in new[] { 16, 64, 256 })
            {
                Line(report, engine, steps);
            }
        }

        File.WriteAllText(Path.Join(Path.GetTempPath(), "recurrent-cost.md"), report.ToString());
        TestContext.Current.SendDiagnosticMessage(report.ToString());
    }

    private static void Line(StringBuilder report, ITensorBackend engine, int steps)
    {
        var layer = new Lstm(new Recurrence(Channels, Hidden), new RandomStream(1).Draw("cost", 0, 0));
        var series = Tensor.From(new Shape(Series, steps, Channels), [.. Enumerable.Range(0, Series * steps * Channels).Select(at => (float)Math.Sin(at * 0.01))]);
        Cost best = default;

        for (var repeat = 0; repeat < 4; repeat++)
        {
            var timed = new TimedBackend(engine);
            var recording = new RecordingBackend(timed);
            var before = GC.GetAllocatedBytesForCurrentThread();
            var wall = Stopwatch.StartNew();
            var made = layer.Forward(series, Pass.Training(recording, new RandomStream(1), 0, 0));
            var loss = recording.Mean(recording.Multiply(made, made));

            _ = recording.GradientsOf(loss, layer.Parameters().Select(parameter => parameter.Value));
            wall.Stop();

            var cost = new Cost(wall.Elapsed.TotalMilliseconds, timed.Picks, timed.Products, timed.Moves, GC.GetAllocatedBytesForCurrentThread() - before);

            // The first run warms the engine; the quickest of the others is kept, so a slow patch of the machine is not charged to the layer.
            if (repeat == 1 || (repeat > 1 && cost.Wall < best.Wall))
            {
                best = cost;
            }
        }

        report.AppendLine(string.Create(
            CultureInfo.InvariantCulture,
            $"| {engine.Name} | {steps} | {best.Wall:F1} | {best.Picks:F1} | {best.Picks / Math.Max(best.Picks + best.Products + best.Moves, 1e-9):P0} | {best.Products:F1} | {best.Moves:F1} | {best.Bytes:N0} |"));
    }

    private readonly record struct Cost(double Wall, double Picks, double Products, double Moves, long Bytes);
}
