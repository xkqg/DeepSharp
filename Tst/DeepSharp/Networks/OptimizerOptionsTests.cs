// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// The options an optimizer has beyond its defaults — a weight decay added to the gradient, dampening and Nesterov's momentum,
/// AMSGrad and the centred form — and what they leave behind: the tensors a checkpoint keeps under PyTorch's own names, a run
/// that goes on from them bit for bit as if it had never stopped, a memory that is not this optimizer's refused by the name of
/// what is missing or of what is more, and a file that says an option only when it is one, as the oldest version that holds it.
/// Where each option moves the parameters, step by step, is held to PyTorch's own on every engine by the contract.
/// </summary>
public class OptimizerOptionsTests
{
    private static readonly ITensorBackend Backend = new CpuBackend();

    // Fixtures/optimizers-options-pytorch.py: the same walks the contract holds every engine to.
    private static readonly JsonElement Walks = JsonDocument
        .Parse(File.ReadAllText(Path.Join(Repository.Root, "Tst", "DeepSharp.Backends.Contract", "Networks", "Fixtures", "optimizers-options-pytorch.json")))
        .RootElement.GetProperty("walks");

    public static TheoryData<string> EveryWalk { get; } = [.. Walks.EnumerateArray().Select(each => each.GetProperty("name").GetString()!)];

    [Fact]
    public void ADecayAddedToTheGradient_ShrinksEvenAParameterNothingReaches()
    {
        var parameter = new Dense(Tensor.From(new Shape(1, 2), [1f, -2f]), Tensor.Zeros(new Shape(2))).Weight;
        var gradients = new Gradients(new Dictionary<Tensor, Tensor>(ReferenceEqualityComparer.Instance) { [parameter.Value] = Tensor.Zeros(new Shape(1, 2)) });

        new Sgd { WeightDecay = 0.5 }.Step([parameter], gradients, 0.5, Backend);

        Assert.Equal<float[]>([0.75f, -1.5f], parameter.Value.Values.ToArray());
    }

    [Fact]
    public void NesterovsMomentumWithoutAMomentumOrWithDampening_IsRefusedByTheStepToo()
    {
        var parameter = new Dense(1, 1, new RandomStream(1).Draw("initialise:test", 0, 0)).Weight;
        var gradients = new Gradients(new Dictionary<Tensor, Tensor>(ReferenceEqualityComparer.Instance));

        Assert.Contains("Nesterov", Assert.Throws<ArgumentException>(() => new Sgd(0.1) { Nesterov = true }.Step([parameter], gradients, 0.1, Backend)).Message, StringComparison.Ordinal);
        Assert.Contains("Nesterov", Assert.Throws<ArgumentException>(() => new Sgd(0.1) { Momentum = 0.9, Dampening = 0.5, Nesterov = true }.Step([parameter], gradients, 0.1, Backend)).Message, StringComparison.Ordinal);
    }

    [Theory]
    [MemberData(nameof(EveryWalk))]
    public void WhatAnOptimizerRemembersAfterAStep_IsKeptUnderTheNamesPyTorchKeepsItUnder(string name)
    {
        var walk = Walk(name);
        var (network, optimizer) = (Network(), OptimizerOf(walk));

        Stepped(network, optimizer);

        var kept = optimizer.KeptOf(network.Parameters().First());
        var pytorch = walk.GetProperty("state").EnumerateArray().Select(key => key.GetString()!).Where(key => key is not ("step" or "mu_product")).Order(StringComparer.Ordinal);

        Assert.Equal(pytorch, kept?.Tensors.Keys.Order(StringComparer.Ordinal) ?? Enumerable.Empty<string>());
    }

    [Theory]
    [MemberData(nameof(EveryWalk))]
    public void ARunThatGoesOnFromWhatWasRemembered_StepsAsIfItHadNeverStopped_BitForBit(string name)
    {
        var walk = Walk(name);
        var steps = walk.GetProperty("steps").GetInt32();

        // One run of every step; and another that stops after the first, puts back what was remembered, and goes on.
        var straight = Network();
        var optimizer = OptimizerOf(walk);

        for (var step = 0; step < steps; step++)
        {
            Stepped(straight, optimizer);
        }

        var stopped = Network();
        var first = OptimizerOf(walk);

        Stepped(stopped, first);

        var memory = stopped.Parameters().ToDictionary(parameter => parameter, parameter => first.KeptOf(parameter));
        var resumed = OptimizerOf(walk);

        foreach (var (parameter, kept) in memory.Where(each => each.Value is not null))
        {
            resumed.PutBack(parameter, kept!.Value);
        }

        for (var step = 1; step < steps; step++)
        {
            Stepped(stopped, resumed);
        }

        Assert.Equal(Bits(straight), Bits(stopped));
    }

    [Fact]
    public void AMemoryThatLacksATensorTheOptimizerKeeps_IsRefusedByItsName()
    {
        var weight = new Dense(1, 2, new RandomStream(1).Draw("initialise:test", 0, 0)).Weight;
        var shape = weight.Value.Shape;
        Tensor Some() => Tensor.Zeros(shape);

        Assert.Contains("max_exp_avg_sq", Assert.Throws<ArgumentException>(() => new Adam { AmsGrad = true }.PutBack(weight, Memory(3, ("exp_avg", Some()), ("exp_avg_sq", Some())))).Message, StringComparison.Ordinal);
        Assert.Contains("max_exp_avg_sq", Assert.Throws<ArgumentException>(() => new AdamW { AmsGrad = true }.PutBack(weight, Memory(3, ("exp_avg", Some()), ("exp_avg_sq", Some())))).Message, StringComparison.Ordinal);
        Assert.Contains("grad_avg", Assert.Throws<ArgumentException>(() => new RmsProp { Centered = true }.PutBack(weight, Memory(3, ("square_avg", Some())))).Message, StringComparison.Ordinal);
        Assert.Contains("momentum_buffer", Assert.Throws<ArgumentException>(() => new RmsProp { Momentum = 0.9 }.PutBack(weight, Memory(3, ("square_avg", Some())))).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AMemoryWithATensorTheOptimizerDoesNotKeep_IsRefusedByItsName_NotLeftOutQuietly()
    {
        var weight = new Dense(1, 2, new RandomStream(1).Draw("initialise:test", 0, 0)).Weight;
        var shape = weight.Value.Shape;
        Tensor Some() => Tensor.Zeros(shape);

        Assert.Contains("max_exp_avg_sq", Assert.Throws<ArgumentException>(() => new Adam().PutBack(weight, Memory(3, ("exp_avg", Some()), ("exp_avg_sq", Some()), ("max_exp_avg_sq", Some())))).Message, StringComparison.Ordinal);
        Assert.Contains("max_exp_avg_sq", Assert.Throws<ArgumentException>(() => new Nadam().PutBack(weight, Memory(3, ("exp_avg", Some()), ("exp_avg_sq", Some()), ("max_exp_avg_sq", Some())))).Message, StringComparison.Ordinal);
        Assert.Contains("grad_avg", Assert.Throws<ArgumentException>(() => new RmsProp().PutBack(weight, Memory(3, ("square_avg", Some()), ("grad_avg", Some())))).Message, StringComparison.Ordinal);
        Assert.Contains("momentum_buffer", Assert.Throws<ArgumentException>(() => new RmsProp().PutBack(weight, Memory(3, ("square_avg", Some()), ("momentum_buffer", Some())))).Message, StringComparison.Ordinal);
        Assert.Contains("exp_avg", Assert.Throws<ArgumentException>(() => new Sgd { Momentum = 0.9 }.PutBack(weight, Memory(0, ("momentum_buffer", Some()), ("exp_avg", Some())))).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnOptimizerThatSaysNothingOfTheNewOptions_IsWrittenAsItAlwaysWas_AndTheSecondVersionHoldsIt()
    {
        foreach (var optimizer in new Optimizer[] { new Sgd(0.1) { Momentum = 0.9 }, new Adam(0.01), new AdamW(0.01), new RmsProp(0.001) { Momentum = 0.5 }, new Nadam(0.01) })
        {
            var text = SettingsOf(optimizer);


            if (optimizer is not AdamW)
            {
                Assert.DoesNotContain("weightDecay", text, StringComparison.Ordinal);
            }

            Assert.DoesNotContain("amsgrad", text, StringComparison.Ordinal);
            Assert.DoesNotContain("centered", text, StringComparison.Ordinal);
            Assert.DoesNotContain("nesterov", text, StringComparison.Ordinal);
            Assert.DoesNotContain("dampening", text, StringComparison.Ordinal);
            Assert.Equal(2, ((ISaved)optimizer).FileVersion);
        }
    }

    [Fact]
    public void AnOptimizerThatSaysAnOption_WritesIt_AndNeedsTheFifthVersion()
    {
        Assert.Equal(
            "{\"rate\":0.1,\"momentum\":0.9,\"weightDecay\":0.01,\"dampening\":0.5}",
            SettingsOf(new Sgd(0.1) { Momentum = 0.9, WeightDecay = 0.01, Dampening = 0.5 }));
        Assert.Contains("\"nesterov\":true", SettingsOf(new Sgd(0.1) { Momentum = 0.9, Nesterov = true }), StringComparison.Ordinal);
        Assert.Contains("\"amsgrad\":true", SettingsOf(new Adam { AmsGrad = true }), StringComparison.Ordinal);
        Assert.Contains("\"amsgrad\":true", SettingsOf(new AdamW { AmsGrad = true }), StringComparison.Ordinal);
        Assert.Contains("\"centered\":true", SettingsOf(new RmsProp { Centered = true }), StringComparison.Ordinal);
        Assert.Contains("\"weightDecay\":0.1", SettingsOf(new Adam { WeightDecay = 0.1 }), StringComparison.Ordinal);
        Assert.Contains("\"weightDecay\":0.1", SettingsOf(new RmsProp { WeightDecay = 0.1 }), StringComparison.Ordinal);
        Assert.Contains("\"weightDecay\":0.1", SettingsOf(new Nadam { WeightDecay = 0.1 }), StringComparison.Ordinal);

        foreach (var optimizer in new Optimizer[]
        {
            new Sgd { WeightDecay = 0.1 }, new Sgd { Dampening = 0.1 }, new Sgd { Momentum = 0.9, Nesterov = true },
            new Adam { WeightDecay = 0.1 }, new Adam { AmsGrad = true }, new AdamW { AmsGrad = true },
            new RmsProp { WeightDecay = 0.1 }, new RmsProp { Centered = true }, new Nadam { WeightDecay = 0.1 },
        })
        {
            Assert.Equal(5, ((ISaved)optimizer).FileVersion);
        }
    }

    [Fact]
    public void ARunWhoseOptimizerSaysAnOption_IsWrittenAsTheFifthVersionOfTheTrainingPart_AndReadBackAsItself()
    {
        var compiled = new LayerStack(new Dense(1, 1, new RandomStream(5).Draw("initialise:0", 0, 0)))
            .Compile(new Adam(0.01) { WeightDecay = 0.1, AmsGrad = true }, new MeanSquaredError());
        Checkpoint? kept = null;
        var rows = new TrainingData(Tensor.From(new Shape(4, 1), [0.1f, 0.2f, 0.3f, 0.4f]), Tensor.From(new Shape(4, 1), [0.2f, 0.4f, 0.6f, 0.8f]));

        compiled.Fit(rows, validation: null, new FitOptions(seed: 3) { Epochs = 2, BatchSize = 2, Checkpoints = new Checkpoints(checkpoint => kept = checkpoint) });

        var written = Written(compiled, kept!);
        var read = NetworkDocument.ReadCheckpoint(written, "network", "training", NetworkCatalog.BuiltIn());

        using var document = JsonDocument.Parse(written);

        Assert.Equal(5, document.RootElement.GetProperty("training").GetProperty("version").GetInt32());
        Assert.Equal(2, document.RootElement.GetProperty("network").GetProperty("version").GetInt32());

        var adam = Assert.IsType<Adam>(read.Run.Compiled.Optimizer);

        Assert.Equal(0.1, adam.WeightDecay);
        Assert.True(adam.AmsGrad);
        Assert.Equal(written, Written(read.Run.Compiled, read.Run.Checkpoint));
    }

    [Fact]
    public void ARunWhoseOptimizerSaysNoOption_IsWrittenAsTheSecondVersionOfTheTrainingPart()
    {
        var compiled = new LayerStack(new Dense(1, 1, new RandomStream(5).Draw("initialise:0", 0, 0))).Compile(new Adam(0.01), new MeanSquaredError());
        Checkpoint? kept = null;
        var rows = new TrainingData(Tensor.From(new Shape(4, 1), [0.1f, 0.2f, 0.3f, 0.4f]), Tensor.From(new Shape(4, 1), [0.2f, 0.4f, 0.6f, 0.8f]));

        compiled.Fit(rows, validation: null, new FitOptions(seed: 3) { Epochs = 1, BatchSize = 2, Checkpoints = new Checkpoints(checkpoint => kept = checkpoint) });

        using var document = JsonDocument.Parse(Written(compiled, kept!));

        Assert.Equal(2, document.RootElement.GetProperty("training").GetProperty("version").GetInt32());
    }

    private static JsonElement Walk(string name) => Walks.EnumerateArray().Single(each => each.GetProperty("name").GetString() == name);

    private static LayerStack Network() => new(
        new Dense(WalkedRows.HiddenWeights(), WalkedRows.HiddenBias()), new Relu(), new Dense(WalkedRows.OutputWeights(), WalkedRows.OutputBias()));

    private static void Stepped(Network network, Optimizer optimizer)
    {
        var recording = new RecordingBackend(Backend);
        var value = new BinaryCrossEntropy().Of(network.Forward(WalkedRows.Passengers(), Pass.Training(recording, new RandomStream(1), 0, 0)), WalkedRows.Survived(), recording);
        var gradients = recording.GradientsOf(value, network.Parameters().Select(parameter => parameter.Value));

        optimizer.Step(network.Parameters(), gradients, optimizer.Rate, Backend);
    }

    private static int[] Bits(Network network) =>
        [.. network.Parameters().SelectMany(parameter => parameter.Value.Values.ToArray()).Select(BitConverter.SingleToInt32Bits)];

    private static Optimizer OptimizerOf(JsonElement walk)
    {
        var rate = walk.GetProperty("rate").GetDouble();
        var settings = walk.GetProperty("settings");
        double Number(string key) => settings.TryGetProperty(key, out var written) ? written.GetDouble() : 0.0;
        bool Yes(string key) => settings.TryGetProperty(key, out var written) && written.GetBoolean();
        var betas = settings.TryGetProperty("betas", out var pair) ? new Betas(pair[0].GetDouble(), pair[1].GetDouble()) : new Betas(0.9, 0.999);

        return walk.GetProperty("optimizer").GetString() switch
        {
            "sgd" => new Sgd(rate) { Momentum = Number("momentum"), WeightDecay = Number("weightDecay"), Dampening = Number("dampening"), Nesterov = Yes("nesterov") },
            "adam" => new Adam(rate) { Betas = betas, WeightDecay = Number("weightDecay"), AmsGrad = Yes("amsgrad") },
            "adamw" => new AdamW(rate) { Betas = betas, WeightDecay = Number("weightDecay"), AmsGrad = Yes("amsgrad") },
            "rmsprop" => new RmsProp(rate) { Momentum = Number("momentum"), WeightDecay = Number("weightDecay"), Centered = Yes("centered") },
            _ => new Nadam(rate) { WeightDecay = Number("weightDecay") },
        };
    }

    private static SlotMemory Memory(int steps, params (string Name, Tensor Tensor)[] tensors) =>
        new(steps, tensors.ToDictionary(each => each.Name, each => each.Tensor));

    private static string SettingsOf(Optimizer optimizer)
    {
        var buffer = new System.Buffers.ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            ((ISaved)optimizer).WriteSettings(writer);
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }

    private static string Written(CompiledNetwork compiled, Checkpoint checkpoint)
    {
        var buffer = new System.Buffers.ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, compiled.Network, compiled.Loss);
            writer.WritePropertyName("training");
            NetworkDocument.WriteTraining(writer, compiled, checkpoint);
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }
}
