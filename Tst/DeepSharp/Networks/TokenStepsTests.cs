// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// The ids of a text kept a step at a time — each id its own vector, the padding nothing — and the mask the padding makes, so a recurrent
/// layer reads a batch of texts of unequal lengths each as it would be read alone, learning nothing of the padding.
/// </summary>
public class TokenStepsTests
{
    private static readonly ITensorBackend Engine = new CpuBackend();

    private static Tensor Ids(int count, params float[] ids) => Tensor.From(new Shape(count, ids.Length / count), ids);

    private static Draws Draws() => new RandomStream(1).Draw("tokens", 0, 0);

    [Fact]
    public void AnEmbeddingThatKeepsItsSteps_MakesTheVectorOfEachIdAtItsStep_AndNothingAtThePadding()
    {
        var embedding = new TokenEmbedding(4, 6, 3, Draws()) { KeepsSteps = true };
        var made = embedding.Forward(Ids(2, 1, 2, 5, 0, 3, 0, 0, 0), Pass.Evaluation(Engine));
        var weight = embedding.Weight.Value.Values.ToArray();

        Assert.Equal(new Shape(2, 4, 3), made.Shape);
        Assert.Equal(weight[3..6], made.Values[..3].ToArray());
        Assert.Equal(weight[6..9], made.Values[3..6].ToArray());
        Assert.Equal(weight[15..18], made.Values[6..9].ToArray());
        Assert.All(made.Values[9..12].ToArray(), value => Assert.Equal(0f, value));
        Assert.Equal(weight[9..12], made.Values[12..15].ToArray());
        Assert.All(made.Values[15..].ToArray(), value => Assert.Equal(0f, value));
    }

    [Fact]
    public void AnEmbeddingThatKeepsItsSteps_ReleasesTheGradientToTheVectorsOfTheIdsTheRowsHold()
    {
        var embedding = new TokenEmbedding(3, 5, 2, Draws()) { KeepsSteps = true };
        var recording = new RecordingBackend(Engine);
        var made = embedding.Forward(Ids(1, 2, 0, 2), Pass.Evaluation(recording));
        var gradients = recording.GradientsOf(recording.Mean(made), [embedding.Weight.Value]);
        var gradient = gradients[embedding.Weight.Value].Values.ToArray();

        Assert.All(gradient[..4], value => Assert.Equal(0f, value));
        Assert.All(gradient[4..6], value => Assert.NotEqual(0f, value));
        Assert.All(gradient[6..], value => Assert.Equal(0f, value));
    }

    [Fact]
    public void AnEmbeddingThatKeepsNoSteps_IsTheAverageItAlwaysWas_AndWritesNothingMore()
    {
        var plain = new TokenEmbedding(4, 6, 3, Draws());
        var kept = new TokenEmbedding(4, 6, 3, Draws()) { KeepsSteps = true };

        Assert.Equal(new Shape(2, 3), plain.Forward(Ids(2, 1, 2, 5, 0, 3, 0, 0, 0), Pass.Evaluation(Engine)).Shape);
        Assert.Equal(NetworkDocument.SecondVersion, ((ISaved)plain).FileVersion);
        Assert.Equal(NetworkDocument.EighthVersion, ((ISaved)kept).FileVersion);
        Assert.DoesNotContain("keepSteps", Written(new LayerStack(plain)), StringComparison.Ordinal);
    }

    [Fact]
    public void AnEmbeddingThatKeepsItsSteps_IsWrittenDownAndReadBackAsItWas()
    {
        var kept = new TokenEmbedding(4, 6, 3, Draws()) { KeepsSteps = true };
        var written = Written(new LayerStack(kept));
        var read = NetworkDocument.ReadNetwork(written, "network", NetworkCatalog.BuiltIn());
        var again = Assert.IsType<TokenEmbedding>(((LayerStack)read.Network).Layers[0]);

        Assert.True(again.KeepsSteps);
        Assert.Contains("\"keepSteps\": true", written, StringComparison.Ordinal);
        Assert.Equal(written, Written(read.Network));
    }

    [Fact]
    public void TheMaskOfARowOfIds_IsOneWhereAnIdIsAWordAndNoughtWhereItIsPadding_AndLearnsNothing()
    {
        var unpadded = new Unpadded();
        var mask = unpadded.Forward(Ids(2, 1, 2, 5, 0, 3, 0, 0, 0), Pass.Evaluation(Engine));

        Assert.Equal(new Shape(2, 4), mask.Shape);
        Assert.Equal([1f, 1f, 1f, 0f, 1f, 0f, 0f, 0f], mask.Values.ToArray());
        Assert.Empty(unpadded.Slots());
        Assert.Equal("unpadded", ((ISaved)unpadded).Kind);
        Assert.Equal(NetworkDocument.EighthVersion, ((ISaved)unpadded).FileVersion);
    }

    [Fact]
    public void AMaskIsWrittenDownAndReadBack()
    {
        var written = Written(new LayerStack(new Unpadded()));

        Assert.IsType<Unpadded>(((LayerStack)NetworkDocument.ReadNetwork(written, "network", NetworkCatalog.BuiltIn()).Network).Layers[0]);
    }

    [Fact]
    public void ABatchOfTextsOfUnequalLengths_IsReadEachAsItWouldBeReadAlone_TheMaskGivenToTheLayerOnTheWords()
    {
        // The same table and the same layer, once over texts padded to five ids and once over the shortest text on its own.
        var words = new Sequential().TokenEmbedding(7, 3, keepSteps: true).Named("steps").Unpadded().Named("mask").Takes("input")
            .Lstm(4).Takes("steps", "mask").Dense(1);
        var padded = Fill(words.Lower(new Shape(5), new RandomStream(2)));
        var alone = Fill(new Sequential().TokenEmbedding(7, 3, keepSteps: true).Named("steps").Unpadded().Named("mask").Takes("input")
            .Lstm(4).Takes("steps", "mask").Dense(1).Lower(new Shape(2), new RandomStream(2)));

        var together = padded.Forward(Ids(2, 3, 4, 0, 0, 0, 1, 2, 5, 6, 3), Pass.Evaluation(Engine)).Values.ToArray();
        var shortest = alone.Forward(Ids(1, 3, 4), Pass.Evaluation(Engine)).Values.ToArray();

        Assert.InRange(Math.Abs(together[0] - shortest[0]), 0, 1e-6);
    }

    // The network with the numbers of its slots worked out from their places, so two networks of the same words hold the same numbers.
    private static LayerStack Fill(LayerStack network)
    {
        network.Load([.. network.Slots().Select(named => new SlotEntry(
            named.Path,
            Tensor.From(named.Slot.Value.Shape, [.. Enumerable.Range(0, named.Slot.Value.Shape.Count).Select(at => (float)Math.Sin((at * 0.9) + 0.4))]),
            "worked out"))]);

        return network;
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
