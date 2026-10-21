// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json.Nodes;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>What the layers that read a series, or its ids, refuse to be handed, and what a file that describes one wrongly is refused for.</summary>
public class SeriesLayerRefusalTests
{
    private static readonly Pass Evaluation = Pass.Evaluation(new CpuBackend());

    [Fact]
    public void TheMaskOfAnythingButARowOfIds_IsRefused() =>
        Assert.Contains("rows of ids", Assert.Throws<ArgumentException>(() => new Unpadded().Forward(Tensor.Zeros(new Shape(2, 3, 4)), Evaluation)).Message, StringComparison.Ordinal);

    [Theory]
    [InlineData(2, 8)]
    [InlineData(2, 0, 4)]
    public void ALayerRunOverEveryStep_IsRefusedWhatHoldsNoSeriesOfAStep(params int[] axes) =>
        Assert.Throws<ArgumentException>(() => new EveryStep(new Dense(4, 2, new Beginning(new RandomStream(1).Draw("a", 0, 0)))).Forward(Tensor.Zeros(new Shape(axes)), Evaluation));

    [Fact]
    public void ALayerRunOverEveryStep_IsReadBackFromAFileOfOneLayer_AndRefusedOneOfTwo()
    {
        var layer = new EveryStep(new Dense(4, 2, new Beginning(new RandomStream(1).Draw("a", 0, 0))));
        var written = Written(new LayerStack([new Step(layer, "every")]));

        Assert.IsType<EveryStep>(((LayerStack)NetworkDocument.ReadNetwork(written, "network", NetworkCatalog.BuiltIn()).Network).Layers[0]);

        var document = JsonNode.Parse(written)!;
        var layers = FindLayersOf(document, "everyStep");

        layers.Add(JsonNode.Parse(layers[0]!.ToJsonString()));

        Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(document.ToJsonString(), "network", NetworkCatalog.BuiltIn()));
    }

    // The array of layers inside the settings of the layer of that kind.
    private static JsonArray FindLayersOf(JsonNode node, string kind)
    {
        if (node is JsonObject obj)
        {
            if (obj["kind"]?.GetValue<string>() == kind && obj["layers"] is JsonArray layers)
            {
                return layers;
            }

            foreach (var child in obj)
            {
                if (child.Value is { } inside && FindLayersOf(inside, kind) is { } found)
                {
                    return found;
                }
            }
        }
        else if (node is JsonArray array)
        {
            foreach (var child in array)
            {
                if (child is { } inside && FindLayersOf(inside, kind) is { } found)
                {
                    return found;
                }
            }
        }

        return null!;
    }

    private static string Written(Network network)
    {
        using var stream = new MemoryStream();

        using (var writer = new System.Text.Json.Utf8JsonWriter(stream))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, network, new MeanSquaredError());
            writer.WriteEndObject();
        }

        return System.Text.Encoding.UTF8.GetString(stream.ToArray());
    }
}
