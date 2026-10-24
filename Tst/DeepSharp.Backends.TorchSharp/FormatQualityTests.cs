// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;
using DeepSharp.Backends.TorchSharp;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;
using DeepSharp.Tests.Backends.TorchGpu;

namespace DeepSharp.Tests.Backends.Torch;

/// <summary>
/// What a number format does to what a network learns, measured by hand: the published passenger list, the same network trained for
/// thirty epochs from three seeds at every format the engine can keep its work in, and the validation loss the last epoch came to,
/// beside full width's. A format is kept only if its loss is within two in a hundred of full width's. Explicit: a measurement is read, not
/// gated; it writes <c>format-quality.md</c> in the temporary folder.
/// </summary>
public class FormatQualityTests
{
    [Fact(Explicit = true)]
    public void TheValidationLossOfThePassengers_AtEveryFormat_FromThreeSeeds_IsWrittenDown()
    {
        var report = new StringBuilder();
        var prepared = Passengers.Prepared();
        var engines = new List<(string Name, Func<ITensorBackend> Make, LossScale? Scale)>
        {
            ("full on the processor", () => TorchBackend.OnCpu(), null),
            ("bfloat16 on the processor", () => TorchBackend.OnCpu(Precision.BFloat16), null),
        };

        if (Card.Present)
        {
            var card = TorchSharp.torch.cuda.device_count() > 1 ? 1 : 0;

            engines.Add(("full on the card", () => TorchBackend.OnGpu(card), null));
            engines.Add(("bfloat16 on the card", () => TorchBackend.OnGpu(card, Precision.BFloat16), null));
            engines.Add(("bfloat16 on the card, loss scale 2^16", () => TorchBackend.OnGpu(card, Precision.BFloat16), LossScale.Dynamic()));
            engines.Add(("half on the card", () => TorchBackend.OnGpu(card, Precision.Half), null));
            engines.Add(("half on the card, loss scale 2^16", () => TorchBackend.OnGpu(card, Precision.Half), LossScale.Dynamic()));
        }

        report.AppendLine("| engine | seed 1 | seed 2 | seed 3 | mean | against full on the same device | steps skipped |");
        report.AppendLine("|---|---:|---:|---:|---:|---:|---:|");

        var baselines = new Dictionary<string, double>();

        foreach (var (name, make, scale) in engines)
        {
            var runs = new[] { 1, 2, 3 }.Select(seed => Run(prepared, make(), scale, seed)).ToArray();
            var losses = runs.Select(run => run.Loss).ToArray();
            var mean = losses.Average();
            var device = name.Contains("card", StringComparison.Ordinal) ? "card" : "processor";

            baselines.TryAdd(device, mean);

            report.AppendLine(string.Create(
                CultureInfo.InvariantCulture,
                $"| {name} | {losses[0]:F4} | {losses[1]:F4} | {losses[2]:F4} | {mean:F4} | {100 * ((mean / baselines[device]) - 1):+0.00;-0.00;0.00} % | {runs.Sum(run => run.Skipped)} |"));
        }

        File.WriteAllText(Path.Join(Path.GetTempPath(), "format-quality.md"), report.ToString());
        TestContext.Current.SendDiagnosticMessage(report.ToString());
    }

    // The validation loss the last epoch of thirty came to, and how many steps the run skipped.
    private static Outcome Run(PreparedData prepared, ITensorBackend engine, LossScale? scale, int seed)
    {
        var compiled = new Sequential().Dense(32).Relu().Dense(32).Relu().Dense(1).Compile(new Adam(0.01), new BinaryCrossEntropy());
        var history = compiled.Fit(prepared, new FitOptions(seed) { Epochs = 30, BatchSize = 32, Backend = engine, LossScale = scale }).History!;

        return new Outcome(history.Epochs[^1].ValidationLoss!.Value, history.Epochs.Sum(epoch => epoch.Skipped));
    }

    private readonly record struct Outcome(double Loss, int Skipped);
}
