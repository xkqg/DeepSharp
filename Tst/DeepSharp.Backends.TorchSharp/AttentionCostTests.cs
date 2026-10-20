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
/// What multi-head attention costs, measured by hand on the two engines: a batch of 8 and of 32 series of 16 and of 128 steps, 128 channels and 4
/// and 8 heads, read forwards and trained back through. The attention is made one series and one head at a time of the products the seam has; this
/// is the evidence for the question whether the seam should learn a product over a batch. Explicit: a measurement is read, not gated; it writes
/// <c>attention-cost.md</c> in the temporary folder.
/// </summary>
public class AttentionCostTests
{
    private const int Channels = 128;

    [Fact(Explicit = true)]
    public void TheCostOfMultiHeadAttentionAtEightAndThirtyTwoSeries_SixteenAndAHundredTwentyEightSteps_IsWrittenDown()
    {
        var report = new StringBuilder();

        report.AppendLine("| engine | series | steps | heads | ms for the forward and the backward pass | products asked for | ms in them | ms turning and laying out | bytes allocated |");
        report.AppendLine("|---|---:|---:|---:|---:|---:|---:|---:|---:|");

        foreach (var engine in new ITensorBackend[] { new CpuBackend(), TorchBackend.OnCpu() })
        {
            foreach (var series in new[] { 8, 32 })
            {
                foreach (var steps in new[] { 16, 128 })
                {
                    foreach (var heads in new[] { 4, 8 })
                    {
                        Line(report, engine, new Looking(series, steps, heads));
                    }
                }
            }
        }

        File.WriteAllText(Path.Join(Path.GetTempPath(), "attention-cost.md"), report.ToString());
        TestContext.Current.SendDiagnosticMessage(report.ToString());
    }

    private static void Line(StringBuilder report, ITensorBackend engine, Looking looking)
    {
        var layer = new MultiHeadAttention(Channels, new Attending(looking.Heads), new RandomStream(1).Draw("cost", 0, 0));
        var series = Tensor.From(
            new Shape(looking.Series, looking.Steps, Channels),
            [.. Enumerable.Range(0, looking.Series * looking.Steps * Channels).Select(at => (float)Math.Sin(at * 0.01))]);
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

            var cost = new Cost(wall.Elapsed.TotalMilliseconds, timed.Multiplications, timed.Picks + timed.Products, timed.Moves, GC.GetAllocatedBytesForCurrentThread() - before);

            // The first run warms the engine; the quickest of the others is kept, so a slow patch of the machine is not charged to the layer.
            if (repeat == 1 || (repeat > 1 && cost.Wall < best.Wall))
            {
                best = cost;
            }
        }

        report.AppendLine(string.Create(
            CultureInfo.InvariantCulture,
            $"| {engine.Name} | {looking.Series} | {looking.Steps} | {looking.Heads} | {best.Wall:F1} | {best.Products} | {best.InProducts:F1} | {best.Moves:F1} | {best.Bytes:N0} |"));
    }

    private readonly record struct Looking(int Series, int Steps, int Heads);

    private readonly record struct Cost(double Wall, int Products, double InProducts, double Moves, long Bytes);
}
