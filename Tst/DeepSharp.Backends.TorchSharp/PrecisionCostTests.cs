// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Globalization;
using System.Text;
using DeepSharp.Backends.TorchSharp;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using DeepSharp.Tests.Backends.TorchGpu;

namespace DeepSharp.Tests.Backends.Torch;

/// <summary>
/// What a training step costs on a graphics card, measured by hand: for every number format the engine can keep, with and without its
/// layers worked out again, on a stack of dense layers or of encoder blocks; and on two cards against one card taking two batches to a
/// step. Every measurement is made in a process of its own, because libtorch keeps the memory it has asked of the card for the next run
/// and a second run in the same process would show none of its own: the time of a step is the quickest epoch of the last three, the
/// memory is what the card's driver says the process held beside what it held once libtorch had its context. Where two ways are compared
/// the cells are run in turn, one after the other, and the median of the runs is taken, because a single run of one cell differs from
/// another by as much as a third. Explicit: a measurement is read, not gated; each writes a file in the temporary folder. The card is the
/// second when there are two (the first carries the display), or the one named by <c>DEEPSHARP_CARD</c>.
/// </summary>
public class PrecisionCostTests
{
    private const int Epochs = 4;
    private const int StepsPerEpoch = 4;
    private const int EncoderSteps = 128;
    private const int EncoderStepsPerEpoch = 2;
    private const string CellVariable = "DEEPSHARP_CELL";
    private const string OutVariable = "DEEPSHARP_CELL_OUT";
    private const string Dense = "dense";
    private const string Encoder = "encoder";

    [Fact(Explicit = true)]
    public void TheCostOfATrainingStepOnTheCard_ForEveryFormat_AtThreeBatchesAndAtTheLargestThatFits_IsWrittenDown()
    {
        Assert.SkipUnless(Card.Present, Card.Absent);

        var card = TheCard();
        var formats = Environment.GetEnvironmentVariable("DEEPSHARP_FORMATS")?.Split(',') ?? Formats();
        var widths = Ints("DEEPSHARP_WIDTHS", [512, 2048]);
        var depths = Ints("DEEPSHARP_DEPTHS", [3, 8]);
        var report = new StringBuilder();

        report.AppendLine($"card {card}, {TorchSharp.torch.__version__}");
        report.AppendLine();
        report.AppendLine("| format | width | depth | batch | ms a step | peak MiB over the idle process | outcome |");
        report.AppendLine("|---|---:|---:|---:|---:|---:|---|");

        foreach (var format in formats)
        {
            foreach (var width in widths)
            {
                foreach (var depth in depths)
                {
                    foreach (var batch in new[] { 1024, 2048, 4096 })
                    {
                        var cost = InAProcessOfItsOwn(new Cell(format, new Shaped(Dense, width, depth, batch), card));

                        report.AppendLine(string.Create(CultureInfo.InvariantCulture, $"| {format} | {width} | {depth} | {batch} | {cost.Step:F2} | {cost.PeakMiB:F0} | {cost.Outcome} |"));
                    }
                }
            }
        }

        report.AppendLine();
        report.AppendLine("The largest batch that fits, width 2048 and depth 8: the batch doubled from 1024 until the card runs out, then the last two bisected to a multiple of 1024.");
        report.AppendLine();
        report.AppendLine("| format | largest batch that fits | ms a step there | peak MiB over the idle process |");
        report.AppendLine("|---|---:|---:|---:|");

        foreach (var format in formats)
        {
            var (low, there) = Largest(format, new Shaped(Dense, 2048, 8, 0), card, start: 2048, grain: 1024);

            report.AppendLine(string.Create(CultureInfo.InvariantCulture, $"| {format} | {low} | {there.Step:F2} | {there.PeakMiB:F0} |"));
        }

        File.WriteAllText(Path.Join(Path.GetTempPath(), "precision-cost.md"), report.ToString());
        TestContext.Current.SendDiagnosticMessage(report.ToString());
    }

    [Fact(Explicit = true)]
    public void TheStepTimeAtWidth2048_OfEveryFormat_IsTheMedianOfRunsTakenInTurn_AndWrittenDown()
    {
        Assert.SkipUnless(Card.Present, Card.Absent);

        var shapes = Ints("DEEPSHARP_DEPTHS", [3, 8]).SelectMany(depth => Ints("DEEPSHARP_BATCHES", [2048, 4096]).Select(batch => new Shaped(Dense, 2048, depth, batch))).ToArray();
        var report = Compared(["full", "bfloat16", "half"], shapes, Repeats(), TheCard());

        File.WriteAllText(Path.Join(Path.GetTempPath(), "precision-median.md"), report);
        TestContext.Current.SendDiagnosticMessage(report);
    }

    [Fact(Explicit = true)]
    public void TheCostOfWorkingLayersOutAgain_OnAStackOfEncoderBlocks_IsWrittenDown()
    {
        Assert.SkipUnless(Card.Present, Card.Absent);

        var card = TheCard();
        var report = new StringBuilder();

        report.AppendLine($"card {card}, {TorchSharp.torch.__version__}; 4 encoder blocks, {EncoderSteps} steps of 128 channels, 4 heads, feed-forward 512; the batch counted in series");
        report.AppendLine();
        report.AppendLine("The largest batch that fits: the batch doubled from 64 until the card runs out, then the last two bisected to a multiple of 16.");
        report.AppendLine();
        report.AppendLine("| format | largest batch that fits | ms a step there | peak MiB over the idle process |");
        report.AppendLine("|---|---:|---:|---:|");

        var fits = new Dictionary<string, int>();

        foreach (var format in new[] { "full", "full+recompute" })
        {
            var (low, there) = Largest(format, new Shaped(Encoder, 128, 4, 0), card, start: 128, grain: 16);

            fits[format] = low;
            report.AppendLine(string.Create(CultureInfo.InvariantCulture, $"| {format} | {low} | {there.Step:F2} | {there.PeakMiB:F0} |"));
        }

        // The batch both fit, at the largest of the two: the step time and the peak beside each other, and the runs taken in turn.
        var equal = fits.Values.Min();

        report.AppendLine();
        report.AppendLine(string.Create(CultureInfo.InvariantCulture, $"At the batch both fit, {equal} series, and at half of it:"));
        report.AppendLine();
        report.AppendLine(Compared(["full", "full+recompute"], [new Shaped(Encoder, 128, 4, equal), new Shaped(Encoder, 128, 4, equal / 16 * 8)], Repeats(), card));

        File.WriteAllText(Path.Join(Path.GetTempPath(), "recompute-encoder.md"), report.ToString());
        TestContext.Current.SendDiagnosticMessage(report.ToString());
    }

    [Fact(Explicit = true)]
    public void OneCellOfThatCost_IsMeasuredInTheProcessThatWasStartedForIt()
    {
        var cell = Cell.Parse(Environment.GetEnvironmentVariable(CellVariable) ?? throw new InvalidOperationException($"{CellVariable} names the cell this process measures."));

        Assert.SkipUnless(Card.Present, Card.Absent);

        var cost = Measure(cell);

        File.WriteAllText(Environment.GetEnvironmentVariable(OutVariable)!, string.Create(CultureInfo.InvariantCulture, $"{cost.Step:R};{cost.PeakMiB:R};{cost.Outcome}"));
    }

    private static int TheCard() =>
        Environment.GetEnvironmentVariable("DEEPSHARP_CARD") is { } named
            ? int.Parse(named, CultureInfo.InvariantCulture)
            : TorchSharp.torch.cuda.device_count() > 1 ? 1 : 0;

    private static int Repeats() => Ints("DEEPSHARP_REPEATS", [5])[0];

    // The number formats an engine can keep its numbers in, by the name the measurement gives them, each as it is and with its layers
    // worked out again.
    private static string[] Formats() => ["full", "full+recompute", "bfloat16", "bfloat16+recompute", "half", "half+recompute"];

    // The engine that keeps its numbers in a format, on a card; a name ends with +recompute where the layers are worked out again.
    private static TorchBackend Engine(string format, int card) =>
        format.Split('+')[0] switch
        {
            "full" => TorchBackend.OnGpu(card),
            "bfloat16" => TorchBackend.OnGpu(card, Precision.BFloat16),
            "half" => TorchBackend.OnGpu(card, Precision.Half),
            _ => throw new ArgumentOutOfRangeException(nameof(format), format, "No engine keeps its numbers in that format."),
        };

    private static int[] Ints(string variable, int[] otherwise) =>
        Environment.GetEnvironmentVariable(variable)?.Split(',').Select(each => int.Parse(each, CultureInfo.InvariantCulture)).ToArray() ?? otherwise;

    // The largest batch of a shape that fits: doubled from the start until the card runs out, then the last two bisected to a multiple of the grain.
    private static Reach Largest(string format, Shaped shape, int card, int start, int grain)
    {
        var low = start / 2;
        var high = 0;

        for (var batch = start; batch <= 1048576 && high == 0; batch *= 2)
        {
            if (InAProcessOfItsOwn(new Cell(format, shape with { Batch = batch }, card)).Outcome == "ok")
            {
                low = batch;
            }
            else
            {
                high = batch;
            }
        }

        while (high - low > grain)
        {
            var middle = (low + high) / 2 / grain * grain;

            if (InAProcessOfItsOwn(new Cell(format, shape with { Batch = middle }, card)).Outcome == "ok")
            {
                low = middle;
            }
            else
            {
                high = middle;
            }
        }

        return new Reach(low, InAProcessOfItsOwn(new Cell(format, shape with { Batch = low }, card)));
    }

    // Each shape at each format, the runs taken in turn — every format of a shape once before any is taken twice — and the median of the
    // runs of each: the first format is the one the others are held against, per shape and as the geometric mean over the shapes.
    private static string Compared(string[] formats, Shaped[] shapes, int repeats, int card)
    {
        var steps = new Dictionary<Sample, List<double>>();
        var peaks = new Dictionary<Sample, List<double>>();

        for (var run = 0; run < repeats; run++)
        {
            foreach (var shape in shapes)
            {
                foreach (var format in formats)
                {
                    var cost = InAProcessOfItsOwn(new Cell(format, shape, card));
                    var sample = new Sample(format, shape);

                    if (!steps.TryAdd(sample, [cost.Step]))
                    {
                        steps[sample].Add(cost.Step);
                    }

                    if (!peaks.TryAdd(sample, [cost.PeakMiB]))
                    {
                        peaks[sample].Add(cost.PeakMiB);
                    }
                }
            }
        }

        var report = new StringBuilder();

        report.AppendLine($"card {card}, {TorchSharp.torch.__version__}; the median of {repeats} runs taken in turn, each in a process of its own; the ratio is to {formats[0]}");
        report.AppendLine();
        report.AppendLine("| kind | width | depth | batch | " + string.Join(" | ", formats.Select(format => $"{format} ms (runs)")) + " | " + string.Join(" | ", formats.Skip(1).Select(format => $"{format} / {formats[0]}")) + " | " + string.Join(" | ", formats.Select(format => $"{format} peak MiB")) + " |");
        report.AppendLine("|---|---:|---:|---:|" + string.Concat(formats.Select(_ => "---:|")) + string.Concat(formats.Skip(1).Select(_ => "---:|")) + string.Concat(formats.Select(_ => "---:|")));

        var ratios = formats.Skip(1).ToDictionary(format => format, _ => new List<double>());

        foreach (var shape in shapes)
        {
            var medians = formats.ToDictionary(format => format, format => Median(steps[new Sample(format, shape)]));
            var runs = formats.Select(format => string.Create(CultureInfo.InvariantCulture, $"{medians[format]:F2} ({string.Join(", ", steps[new Sample(format, shape)].Select(each => each.ToString("F1", CultureInfo.InvariantCulture)))})"));
            var versus = formats.Skip(1).Select(format =>
            {
                var ratio = medians[format] / medians[formats[0]];

                ratios[format].Add(ratio);

                return ratio.ToString("F3", CultureInfo.InvariantCulture);
            });
            var memory = formats.Select(format => Median(peaks[new Sample(format, shape)]).ToString("F0", CultureInfo.InvariantCulture));

            report.AppendLine($"| {shape.Kind} | {shape.Width} | {shape.Depth} | {shape.Batch} | {string.Join(" | ", runs)} | {string.Join(" | ", versus)} | {string.Join(" | ", memory)} |");
        }

        report.AppendLine();

        foreach (var (format, each) in ratios)
        {
            report.AppendLine(string.Create(
                CultureInfo.InvariantCulture,
                $"{format} / {formats[0]}: geometric mean {Math.Exp(each.Average(Math.Log)):F3}, largest cell {each.Max():F3}, smallest {each.Min():F3}"));
        }

        return report.ToString();
    }

    private static double Median(List<double> values)
    {
        var ordered = values.Where(double.IsFinite).Order().ToArray();

        return ordered.Length == 0 ? double.NaN : ordered[ordered.Length / 2];
    }

    // A cell measured by a process started for it alone, so what libtorch kept of an earlier run is not mistaken for this one's.
    private static Cost InAProcessOfItsOwn(Cell cell)
    {
        var output = Path.Join(Path.GetTempPath(), $"cell-{Guid.NewGuid():N}.txt");
        var start = new ProcessStartInfo(Environment.ProcessPath!, "-explicit only -method DeepSharp.Tests.Backends.Torch.PrecisionCostTests.OneCellOfThatCost_IsMeasuredInTheProcessThatWasStartedForIt")
        {
            RedirectStandardOutput = true,
            UseShellExecute = false,
            CreateNoWindow = true,
        };

        start.Environment[CellVariable] = cell.ToString();
        start.Environment[OutVariable] = output;

        using var child = Process.Start(start)!;

        _ = child.StandardOutput.ReadToEnd();
        child.WaitForExit();

        if (!File.Exists(output))
        {
            return new Cost(double.NaN, double.NaN, "no result");
        }

        var parts = File.ReadAllText(output).Split(';');

        File.Delete(output);

        return new Cost(double.Parse(parts[0], CultureInfo.InvariantCulture), double.Parse(parts[1], CultureInfo.InvariantCulture), parts[2]);
    }

    private static Cost Measure(Cell cell)
    {
        using var memory = new CardMemory(cell.Card);
        var engine = Engine(cell.Format, cell.Card);

        // libtorch has its context once it has made a tensor; what the process holds then is what this run is measured over.
        _ = engine.Fill(new Shape(1), 1f).Values[0];
        memory.Reset();

        var shape = cell.Shape;
        var perStep = shape.Batch;
        var stepsPerEpoch = shape.Kind == Encoder ? EncoderStepsPerEpoch : StepsPerEpoch;
        var rows = perStep * stepsPerEpoch;
        var features = Features(shape, rows);
        var answers = Tensor.From(new Shape(rows, 1), Fill(rows, at => (float)Math.Cos(at * 0.01)));
        var compiled = Described(shape).Compile(new Adam(0.0001), new MeanSquaredError());
        var stamps = new List<TimeSpan>();
        var clock = Stopwatch.StartNew();
        var options = new FitOptions(seed: 1)
        {
            Epochs = Epochs,
            BatchSize = shape.Batch,
            Backend = engine,
            Recompute = cell.Format.EndsWith("+recompute", StringComparison.Ordinal) ? Recompute.Everything : null,
            OnEpoch = _ => stamps.Add(clock.Elapsed),
        };

        try
        {
            _ = compiled.Fit(new TrainingData(features, answers), null, options);

            // The quickest epoch after the first, which warms the engine, per step.
            var epoch = stamps.Zip(stamps.Skip(1), (before, after) => (after - before).TotalMilliseconds).Min();

            return new Cost(epoch / stepsPerEpoch, memory.PeakMiB, "ok");
        }
        catch (Exception fault) when (fault.Message.Contains("out of memory", StringComparison.OrdinalIgnoreCase))
        {
            return new Cost(double.NaN, memory.PeakMiB, "out of memory");
        }
    }

    // The network of a shape: dense layers with a rectifier after each, or encoder blocks over series, a single value at the end.
    private static Sequential Described(Shaped shape)
    {
        var description = new Sequential();

        if (shape.Kind == Encoder)
        {
            for (var block = 0; block < shape.Depth; block++)
            {
                description = description.EncoderBlock(new EncoderShape(4, 512));
            }

            return description.GlobalAvgPool1D().Dense(1);
        }

        for (var layer = 0; layer < shape.Depth; layer++)
        {
            description = description.Dense(shape.Width).Relu();
        }

        return description.Dense(1);
    }

    private static Tensor Features(Shaped shape, int rows) =>
        shape.Kind == Encoder
            ? Tensor.From(new Shape(rows, EncoderSteps, shape.Width), Fill(rows * EncoderSteps * shape.Width, at => (float)Math.Sin(at * 0.001)))
            : Tensor.From(new Shape(rows, shape.Width), Fill(rows * shape.Width, at => (float)Math.Sin(at * 0.001)));

    private static float[] Fill(int count, Func<int, float> value)
    {
        var made = new float[count];

        for (var at = 0; at < made.Length; at++)
        {
            made[at] = value(at);
        }

        return made;
    }

    // What is measured: a kind of network, its width and depth, and the batch a step takes.
    private readonly record struct Shaped(string Kind, int Width, int Depth, int Batch);

    // A shape at a format on a card.
    private readonly record struct Cell(string Format, Shaped Shape, int Card)
    {
        public static Cell Parse(string text)
        {
            var parts = text.Split(';');

            return new Cell(
                parts[0],
                new Shaped(parts[1], int.Parse(parts[2], CultureInfo.InvariantCulture), int.Parse(parts[3], CultureInfo.InvariantCulture), int.Parse(parts[4], CultureInfo.InvariantCulture)),
                int.Parse(parts[5], CultureInfo.InvariantCulture));
        }

        public override string ToString() => string.Create(CultureInfo.InvariantCulture, $"{Format};{Shape.Kind};{Shape.Width};{Shape.Depth};{Shape.Batch};{Card}");
    }

    // One run's shape at one format.
    private readonly record struct Sample(string Format, Shaped Shape);

    private readonly record struct Cost(double Step, double PeakMiB, string Outcome);

    // The largest batch found that fits, and what a step there costs.
    private readonly record struct Reach(int Batch, Cost There);

    /// <summary>What the driver says a card's memory holds, sampled while a run goes on, kept as the most it held over what it held when asked to start.</summary>
    private sealed class CardMemory : IDisposable
    {
        private readonly Process _query;
        private readonly Thread _reader;
        private long _idle;
        private long _peak;
        private long _last;

        public CardMemory(int card)
        {
            _query = Process.Start(new ProcessStartInfo("nvidia-smi", $"-lms 50 -i {card} --query-gpu=memory.used --format=csv,noheader,nounits")
            {
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            })!;
            _reader = new Thread(() =>
            {
                while (_query.StandardOutput.ReadLine() is { } line)
                {
                    var used = long.Parse(line.Trim(), CultureInfo.InvariantCulture);

                    Volatile.Write(ref _last, used);

                    long seen;

                    while ((seen = Volatile.Read(ref _peak)) < used && Interlocked.CompareExchange(ref _peak, used, seen) != seen)
                    {
                        // The compare-exchange is the whole of the work: another thread raised the peak first, so it is read again.
                    }
                }
            })
            { IsBackground = true };
            _reader.Start();
            Thread.Sleep(500);
        }

        public double PeakMiB => Math.Max(0, Volatile.Read(ref _peak) - Volatile.Read(ref _idle));

        public void Reset()
        {
            Thread.Sleep(300);

            var now = Volatile.Read(ref _last);

            Volatile.Write(ref _idle, now);
            Volatile.Write(ref _peak, now);
        }

        public void Dispose()
        {
            _query.Kill();
            _query.Dispose();
        }
    }
}
