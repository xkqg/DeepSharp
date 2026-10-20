// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Diagnostics;

namespace DeepSharp.Measure;

/// <summary>What one piece of work costs: how long a call takes, and how many bytes it allocates.</summary>
/// <param name="Nanoseconds">The median time of a call, over the rounds.</param>
/// <param name="Bytes">What a call allocates on the thread that makes it.</param>
internal readonly record struct Timing(double Nanoseconds, long Bytes);

/// <summary>Two pieces of work timed back to back, round after round, so the machine's mood is the same for both.</summary>
/// <param name="Before">The first piece of work.</param>
/// <param name="After">The second.</param>
internal readonly record struct Comparison(Timing Before, Timing After)
{
    /// <summary>How many times faster the second is than the first; below one it is slower.</summary>
    public double Speedup => Before.Nanoseconds / After.Nanoseconds;
}

/// <summary>Times work until the clock can be trusted: a few rounds, each long enough to hold a call many times over.</summary>
internal sealed class Measurement(int rounds, TimeSpan roundLength)
{
    /// <summary>The usual: seven rounds of a tenth of a second.</summary>
    public Measurement()
        : this(7, TimeSpan.FromMilliseconds(100))
    {
    }

    /// <summary>Times two pieces of work alternately, the first then the second, so a slow patch of the machine falls on both.</summary>
    /// <param name="before">The first.</param>
    /// <param name="after">The second.</param>
    /// <returns>Each one's median call and what it allocates.</returns>
    public Comparison Between(Action before, Action after)
    {
        var calls = CallsInARound(before, after);
        var first = new List<double>();
        var second = new List<double>();

        for (var round = 0; round < rounds; round++)
        {
            first.Add(Round(before, calls));
            second.Add(Round(after, calls));
        }

        return new Comparison(new Timing(Median(first), Allocated(before)), new Timing(Median(second), Allocated(after)));
    }

    // How many calls fill a round: enough that the slower of the two takes as long as a round is meant to, found by doubling.
    private int CallsInARound(Action before, Action after)
    {
        before();
        after();

        var calls = 1;

        while (Math.Max(Time(before, calls), Time(after, calls)) < roundLength.TotalMilliseconds * 1e6 && calls < 1 << 24)
        {
            calls *= 2;
        }

        return calls;
    }

    private static double Round(Action work, int calls) => Time(work, calls) / calls;

    private static double Time(Action work, int calls)
    {
        var watch = Stopwatch.StartNew();

        for (var call = 0; call < calls; call++)
        {
            work();
        }

        return watch.Elapsed.TotalMilliseconds * 1e6;
    }

    private static long Allocated(Action work)
    {
        work();

        var before = GC.GetAllocatedBytesForCurrentThread();

        work();

        return GC.GetAllocatedBytesForCurrentThread() - before;
    }

    private static double Median(List<double> times)
    {
        times.Sort();

        return times[times.Count / 2];
    }
}
