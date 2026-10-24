// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tests.Backends.Parts;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// The scale a declared run multiplies its loss by, so that small gradients survive an engine that keeps its work in a narrow width:
/// a key a network's declaration gained after files were written without it. Left out, it means what a declaration meant before it
/// had it — the loss is not scaled — and is not written, so such a declaration writes the bytes it always wrote; said, it is written,
/// read back, and scales the loss of the run as the same words written as code do. It names the scale a dynamic scale begins at, with
/// the growth, the interval and the backoff PyTorch's scaler has; another is code.
/// </summary>
public class DeclaredLossScaleTests
{
    private static readonly string Folder = Path.Join(Path.GetTempPath(), "deepsharp-lossscale-" + Guid.NewGuid().ToString("N"));

    private static FittingBuilder Far()
    {
        Directory.CreateDirectory(Folder);

        var rows = new StringBuilder("x,y\n");

        for (var row = 0; row < 60; row++)
        {
            rows.Append(CultureInfo.InvariantCulture, $"{1 + (row % 4)},{1000 + (row % 5)}\n");
        }

        File.WriteAllText(Path.Join(Folder, "far.csv"), rows.ToString());

        return Pdd.Create().ReadCsv(Path.Join(Folder, "far.csv")).Declare(schema => schema.Number("x", "y")).SplitAtRandom(0.70, 0.15, seed: 5).Normalise("x", Scale.MidRange).Target("y");
    }

    [Fact]
    public void ADeclarationThatSaysNoLossScale_WritesNoKey_AndReadsAsAnUnscaledRun()
    {
        var written = JsonNode.Parse(Far().WithTorch(network => network.Dense(1).Adam(0.01).MeanSquaredError().Run(1)).Build().Declaration.ToJson())!;
        var learner = written["declaration"]!.AsArray().Single(each => (string)each!["step"]! == "learn.network")!.AsObject();

        Assert.False(learner.ContainsKey("lossScale"));
        Assert.Equal(["step", "layers", "optimizer", "loss", "stopping", "engine", "seed", "epochs", "batch"], learner.Select(key => key.Key));
    }

    [Fact]
    public void ALossScale_IsWrittenUnderItsKey_ReadBack_AndWrittenAgainAsItWas()
    {
        var declaration = Far().WithTorch(network => network.Dense(1).Adam(0.01).MeanSquaredError().LossScale(1024).Run(1)).Build().Declaration;
        var learner = JsonNode.Parse(declaration.ToJson())!["declaration"]!.AsArray().Single(each => (string)each!["step"]! == "learn.network")!.AsObject();
        var again = PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn().WithNetworks());

        Assert.Equal(1024, (double)learner["lossScale"]!);
        Assert.Equal(declaration, again);
        Assert.Equal(declaration.ToJson(), again.ToJson());
        Assert.Equal(1024, Assert.IsType<LearnNetworkStep>(again.Learner).LossScaleStart);
    }

    [Fact]
    public void ALossScaleThatIsNotAPowerOfTwo_IsRefusedWhereItIsDeclared()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Far().WithTorch(network => network.Dense(1).Adam(0.01).MeanSquaredError().LossScale(1000)));
        Assert.Throws<ArgumentOutOfRangeException>(() => Far().WithTorch(network => network.Dense(1).Adam(0.01).MeanSquaredError().LossScale(-2)));
    }

    [Fact]
    public void ADeclaredLossScale_ReachesTheLoop_SoAnEngineThatOverflowsSkipsSteps()
    {
        // 2^30 times a gradient of the size of the rows' own overflows what half holds on the very first step.
        var engines = new Engines().Use("rounded", new RoundedBackend(RoundedBackend.Width.Half));
        var scaled = Far().WithTorch(network => network.Dense(1).Adam(0.01).MeanSquaredError().LossScale(1073741824).On("rounded").Run(1, epochs: 2)).Build();
        var plain = Far().WithTorch(network => network.Dense(1).Adam(0.01).MeanSquaredError().On("rounded").Run(1, epochs: 2)).Build();

        Assert.True(scaled.Train(engines).History!.Epochs.Sum(epoch => epoch.Skipped) > 0);
        Assert.Equal(0, plain.Train(engines).History!.Epochs.Sum(epoch => epoch.Skipped));
    }
}
