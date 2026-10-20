// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using TorchSharp;

namespace DeepSharp.Tests.Backends.Torch;

/// <summary>
/// The once-seen libtorch error, hunted by hand: five threads add a row to a matrix as the engine does, while a sixth makes the
/// collector run as often as it will. TorchSharp's <c>add(Tensor)</c> makes the scalar one it adds by itself, a finalizable
/// object nothing holds once its handle has been read, so a collection that falls while libtorch is still adding can let the
/// scalar go under it. Explicit: a soak is read, not gated.
/// </summary>
public class ScalarSoakTests
{
    private const int Rows = 8;
    private const int Columns = 8;
    private const int Batch = 512;

    [Fact(Explicit = true)]
    public void TheScalarTorchSharpMakesForAnAdd_IsLetGoUnderLibtorch_WhenTheCollectorRunsAtTheWrongMoment()
    {
        var seconds = int.Parse(Environment.GetEnvironmentVariable("DEEPSHARP_SOAK_SECONDS") ?? "20", System.Globalization.CultureInfo.InvariantCulture);
        var arms = Environment.GetEnvironmentVariable("DEEPSHARP_SOAK_ARMS") ?? "both";
        var implicitScalar = arms is "both" or "implicit" ? Soak(seconds, rooted: false) : default;
        var rootedScalar = arms is "both" or "rooted" ? Soak(seconds, rooted: true) : default;
        var report = $"implicit: {implicitScalar}; rooted: {rootedScalar}";

        File.WriteAllText(Path.Join(Path.GetTempPath(), "scalar-soak.txt"), report);
        TestContext.Current.SendDiagnosticMessage(report);
        Assert.True(rootedScalar.Faults == 0, report);
    }

    private static SoakResult Soak(int seconds, bool rooted)
    {
        var deadline = Stopwatch.StartNew();
        var calls = 0L;
        var faults = 0L;
        var wrong = 0L;
        var stop = false;
        var pause = int.Parse(Environment.GetEnvironmentVariable("DEEPSHARP_SOAK_PAUSE") ?? "0", System.Globalization.CultureInfo.InvariantCulture);
        var hammer = new Thread(() =>
        {
            while (!Volatile.Read(ref stop))
            {
                if (pause >= 0)
                {
                    GC.Collect(0, GCCollectionMode.Forced, blocking: true);
                }

                Thread.Sleep(Math.Max(pause, 0));
            }
        });
        var workers = Enumerable.Range(0, 5).Select(_ => new Thread(() =>
        {
            using var keep = torch.NewDisposeScope();
            var matrix = torch.ones(Rows, Columns);
            var row = torch.full([Columns], 0.5f);
            var one = 1.ToScalar();

            while (deadline.Elapsed.TotalSeconds < seconds)
            {
                try
                {
                    using (torch.NewDisposeScope())
                    {
                        torch.Tensor added = matrix;

                        for (var at = 0; at < Batch; at++)
                        {
                            added = rooted ? matrix.add(row, one) : matrix.add(row);
                        }

                        GC.KeepAlive(one);

                        if (Math.Abs(added.mean().item<float>() - 1.5f) > 1e-6f)
                        {
                            Interlocked.Increment(ref wrong);
                        }
                    }

                    Interlocked.Add(ref calls, Batch);
                }
                catch (Exception)
                {
                    Interlocked.Increment(ref faults);
                }
            }
        })).ToArray();

        hammer.Start();

        foreach (var worker in workers)
        {
            worker.Start();
        }

        foreach (var worker in workers)
        {
            worker.Join();
        }

        Volatile.Write(ref stop, true);
        hammer.Join();

        return new SoakResult(rooted, calls, faults, wrong);
    }

    private readonly record struct SoakResult(bool Rooted, long Calls, long Faults, long Wrong)
    {
        public override string ToString() => $"{Calls} calls, {Faults} exceptions, {Wrong} wrong answers";
    }
}
