// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Globalization;
using System.Text;
using DeepSharp.Backends.TorchSharp;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using DeepSharp.Tests.Backends.TorchGpu;
using TorchSharp;

namespace DeepSharp.Tests.Backends.Torch;

/// <summary>
/// Where the time of a training step goes on a graphics card, operation by operation: every operation the engine is asked for is waited for
/// to the end, and its time and its count kept by its name, for the engine at full width and at the widths it can keep its work in. The sum
/// is longer than a step runs, since a card is not waited for between operations in a step; the difference between two widths is what is read.
/// Explicit: a measurement is read, not gated; it writes <c>step-profile.md</c> in the temporary folder.
/// </summary>
public class StepProfileTests
{
    [Fact(Explicit = true)]
    public void TheOperationsOfATrainingStepOnTheCard_AtEveryWidth_AreTimedOneByOne()
    {
        Assert.SkipUnless(Card.Present, Card.Absent);

        var card = torch.cuda.device_count() > 1 ? 1 : 0;
        var names = new[] { "full", "bfloat16", "half" };
        var profiles = names.ToDictionary(name => name, name => Collected(() => Profile(Engine(name, card), card)));
        var operations = profiles.Values.SelectMany(each => each.Keys).Distinct().Order(StringComparer.Ordinal).ToArray();
        var report = new StringBuilder();

        report.AppendLine("Dense(2048) x3 with a rectifier after each, then one value, batch 2048, Adam, mean squared error: one step after warm-up, every operation waited for.");
        report.AppendLine();
        report.AppendLine("| operation | full: calls | full: ms | bfloat16: calls | bfloat16: ms | half: calls | half: ms |");
        report.AppendLine("|---|---:|---:|---:|---:|---:|---:|");

        foreach (var operation in operations)
        {
            var cells = names.Select(name => profiles[name].TryGetValue(operation, out var spent) ? string.Create(CultureInfo.InvariantCulture, $"{spent.Calls} | {spent.Milliseconds:F3}") : "0 | 0.000");

            report.AppendLine($"| {operation} | {string.Join(" | ", cells)} |");
        }

        report.AppendLine($"| total | {string.Join(" | ", names.Select(name => string.Create(CultureInfo.InvariantCulture, $"{profiles[name].Values.Sum(spent => spent.Calls)} | {profiles[name].Values.Sum(spent => spent.Milliseconds):F3}")))} |");

        File.WriteAllText(Path.Join(Path.GetTempPath(), "step-profile.md"), report.ToString());
        TestContext.Current.SendDiagnosticMessage(report.ToString());
    }

    // What one engine profiled, with the garbage of the runs before it let go of first and its own after, so that no engine is timed while libtorch frees another's.
    private static Dictionary<string, Spent> Collected(Func<Dictionary<string, Spent>> profile)
    {
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
        GC.WaitForPendingFinalizers();

        var spent = profile();

        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true);
        GC.WaitForPendingFinalizers();

        return spent;
    }

    private static TorchBackend Engine(string name, int card) =>
        name switch
        {
            "bfloat16" => TorchBackend.OnGpu(card, Precision.BFloat16),
            "half" => TorchBackend.OnGpu(card, Precision.Half),
            _ => TorchBackend.OnGpu(card),
        };

    private static Dictionary<string, Spent> Profile(TorchBackend engine, int card)
    {
        const int Rows = 2048;
        const int Width = 2048;

        var timed = new SyncedBackend(engine, card);
        var sines = Enumerable.Range(0, Rows * Width).Select(at => (float)Math.Sin(at * 0.001)).ToArray();
        var features = Tensor.From(new Shape(Rows * 2, Width), [.. sines, .. sines]);
        var cosines = Enumerable.Range(0, Rows).Select(at => (float)Math.Cos(at * 0.01)).ToArray();
        var answers = Tensor.From(new Shape(Rows * 2, 1), [.. cosines, .. cosines]);
        var compiled = new Sequential().Dense(Width).Relu().Dense(Width).Relu().Dense(Width).Relu().Dense(1).Compile(new Adam(0.0001), new MeanSquaredError());

        // Three epochs of two steps: the first warms the engine, and the last is the one that is read.
        var seen = new List<Dictionary<string, Spent>>();
        var options = new FitOptions(seed: 1) { Epochs = 3, BatchSize = Rows, Backend = timed, OnEpoch = _ => seen.Add(timed.TakeProfile()) };

        _ = compiled.Fit(new TrainingData(features, answers), null, options);

        return seen[^1].ToDictionary(each => each.Key, each => new Spent(each.Value.Calls / 2, each.Value.Milliseconds / 2));
    }

    private readonly record struct Spent(int Calls, double Milliseconds);

    // An engine asked for everything, each operation waited for to the end and its time kept by its name.
    private sealed class SyncedBackend(TorchBackend inner, int card) : ITensorBackend, IMultipliesInBlocks, IWidens
    {
        private Dictionary<string, Spent> _spent = [];

        public string Name => inner.Name;

        public Dictionary<string, Spent> TakeProfile()
        {
            var taken = _spent;

            _spent = [];

            return taken;
        }

        public Tensor Widened(Tensor values) => Timed("Widened", () => inner.Widened(values));

        public Tensor Add(Tensor left, Tensor right) => Timed("Add", () => inner.Add(left, right));

        public Tensor Subtract(Tensor left, Tensor right) => Timed("Subtract", () => inner.Subtract(left, right));

        public Tensor Multiply(Tensor left, Tensor right) => Timed("Multiply", () => inner.Multiply(left, right));

        public Tensor MatMul(Tensor left, Tensor right) => Timed("MatMul", () => inner.MatMul(left, right));

        Tensor IMultipliesInBlocks.MatMulInBlocks(Tensor left, Tensor right, BlockProduct how) => Timed("MatMulInBlocks", () => inner.MatMulInBlocks(left, right, how));

        public Tensor Transpose(Tensor matrix) => Timed("Transpose", () => inner.Transpose(matrix));

        public Tensor AddRow(Tensor matrix, Tensor row) => Timed("AddRow", () => inner.AddRow(matrix, row));

        public Tensor SumRows(Tensor matrix) => Timed("SumRows", () => inner.SumRows(matrix));

        public Tensor Mean(Tensor values) => Timed("Mean", () => inner.Mean(values));

        public Tensor Scale(Tensor values, Tensor factor) => Timed("Scale", () => inner.Scale(values, factor));

        public Tensor Fill(Shape shape, float value) => Timed("Fill", () => inner.Fill(shape, value));

        public Tensor Relu(Tensor values) => Timed("Relu", () => inner.Relu(values));

        public Tensor Positive(Tensor values) => Timed("Positive", () => inner.Positive(values));

        public Tensor FirstLargest(Tensor matrix) => Timed("FirstLargest", () => inner.FirstLargest(matrix));

        public Tensor Tanh(Tensor values) => Timed("Tanh", () => inner.Tanh(values));

        public Tensor Sigmoid(Tensor values) => Timed("Sigmoid", () => inner.Sigmoid(values));

        public Tensor Exp(Tensor values) => Timed("Exp", () => inner.Exp(values));

        public Tensor Log(Tensor values) => Timed("Log", () => inner.Log(values));

        public Tensor Sqrt(Tensor values) => Timed("Sqrt", () => inner.Sqrt(values));

        public Tensor Softplus(Tensor values) => Timed("Softplus", () => inner.Softplus(values));

        public Tensor Sin(Tensor values) => Timed("Sin", () => inner.Sin(values));

        public Tensor Erf(Tensor values) => Timed("Erf", () => inner.Erf(values));

        public Tensor Divide(Tensor left, Tensor right) => Timed("Divide", () => inner.Divide(left, right));

        public Tensor LogSoftmax(Tensor matrix) => Timed("LogSoftmax", () => inner.LogSoftmax(matrix));

        public Tensor Reshape(Tensor values, Shape shape) => Timed("Reshape", () => inner.Reshape(values, shape));

        public Tensor Unfold(Tensor images, Window window) => Timed("Unfold", () => inner.Unfold(images, window));

        public Tensor Fold(Tensor patches, Shape images, Window window) => Timed("Fold", () => inner.Fold(patches, images, window));

        private Tensor Timed(string name, Func<Tensor> operation)
        {
            torch.cuda.synchronize(new torch.Device(DeviceType.CUDA, card));

            var started = Stopwatch.GetTimestamp();
            var made = operation();

            torch.cuda.synchronize(new torch.Device(DeviceType.CUDA, card));

            var took = Stopwatch.GetElapsedTime(started).TotalMilliseconds;

            _spent[name] = _spent.TryGetValue(name, out var before) ? new Spent(before.Calls + 1, before.Milliseconds + took) : new Spent(1, took);

            return made;
        }
    }
}
