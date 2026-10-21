// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A layer that works on whole examples, run over every step of a series: the steps are laid out as rows, the layer called once, and the
/// answers put back as the series they came from — Keras's <c>TimeDistributed</c>.
/// </summary>
public class EveryStepTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static Tensor Series(int count, int steps, int width) =>
        Tensor.From(new Shape(count, steps, width), [.. Enumerable.Range(0, count * steps * width).Select(at => (float)Math.Sin((at * 0.37) + 0.3))]);

    [Fact]
    public void ALayerRunOverEveryStep_IsTheLayerRunOnceOverTheStepsLaidOutAsRows_AndThePlacesAreKept()
    {
        var network = new Sequential().EveryStep(new Sequential().Periodic(new Periodicity(2, 0.3, 3))).Lower(new Shape(5, 4), new RandomStream(1));
        var series = Series(2, 5, 4);

        var made = network.Forward(series, Pass.Evaluation(Engine));
        var periodic = (Periodic)((LayerStack)((EveryStep)network.Layers[0]).Layer).Layers[0];
        var alone = periodic.Forward(Engine.Reshape(series, new Shape(10, 4)), Pass.Evaluation(Engine));

        Assert.Equal(new Shape(2, 5, 12), made.Shape);
        Assert.Equal(alone.Values.ToArray(), made.Values.ToArray());
        Assert.Equal(["0.layer.0.frequencies", "0.layer.0.weight", "0.layer.0.bias"], network.Slots().Select(slot => slot.Path));
    }

    [Fact]
    public void AGradientReachesTheLayerThroughEveryStep()
    {
        var network = new Sequential().EveryStep(new Sequential().Dense(3)).Lower(new Shape(4, 2), new RandomStream(1));
        var recording = new RecordingBackend(Engine);
        var series = Series(2, 4, 2);
        var made = network.Forward(series, Pass.Evaluation(recording));
        var weight = network.Slots().First().Slot.Value;
        var gradients = recording.GradientsOf(recording.Mean(made), [weight]);

        Assert.Contains(gradients[weight].Values.ToArray(), value => value is not 0f);
    }

    [Fact]
    public void ALayerRunOverEveryStep_IsWrittenDownAndReadBackAsItWas()
    {
        var network = new Sequential().EveryStep(new Sequential().Dense(3).Relu()).Dense(1).Lower(new Shape(4, 2), new RandomStream(1));
        var written = Written(network);
        var read = NetworkDocument.ReadNetwork(written, "network", NetworkCatalog.BuiltIn());
        var series = Series(2, 4, 2);

        Assert.Equal(network.Slots().Select(slot => slot.Path), read.Network.Slots().Select(slot => slot.Path));
        Assert.Equal(written, Written(read.Network));
        Assert.Equal(network.Forward(series, Pass.Evaluation(Engine)).Values.ToArray(), read.Network.Forward(series, Pass.Evaluation(Engine)).Values.ToArray());
        Assert.Equal(NetworkDocument.EighthVersion, ((ISaved)network.Layers[0]).FileVersion);
    }

    [Fact]
    public void AWordThatRunsOverEveryStep_IsRefusedWhereAnExampleHasNoSteps_OrTheLayerCannotTakeAStep()
    {
        var row = Assert.Throws<ArgumentException>(() => new Sequential().EveryStep(new Sequential().Dense(2)).Lower(new Shape(4), new RandomStream(1)));
        var wrong = Assert.Throws<ArgumentException>(() => new Sequential().EveryStep(new Sequential().Conv2D(2, new Window(1, 1))).Lower(new Shape(4, 3), new RandomStream(1)));

        Assert.Contains("everyStep", row.Message, StringComparison.Ordinal);
        Assert.Contains("everyStep", wrong.Message, StringComparison.Ordinal);
        Assert.Throws<InvalidOperationException>(() => new Sequential().EveryStep(new Sequential()));
        Assert.Throws<ArgumentNullException>(() => new Sequential().EveryStep(null!));
        Assert.Throws<ArgumentException>(() => new EveryStep(new Dense(2, 1, new RandomStream(1).Draw("d", 0, 0))).Forward(Series(1, 2, 3), Pass.Evaluation(Engine)));
    }

    private static string Written(Network network)
    {
        using var stream = new MemoryStream();

        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, network, new MeanSquaredError());
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(stream.ToArray()).Replace("\r\n", "\n", StringComparison.Ordinal);
    }
}
