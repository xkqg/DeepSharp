// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A network of four million parameters written to its file and read back through each door that reads a network's file:
/// what the read allocates on the thread that reads — never how long it takes, which a busy machine moves — stays under a
/// hundred bytes a parameter, and the network read is the one written, to the last bit.
/// </summary>
/// <remarks>
/// The numbers are Kaiming's start, as a network here starts, so the file spells them as a trained network's file does. Each
/// test is one read of the file on one thread, so what that thread allocated is what the read allocated; memory a pool
/// already held and hands back again is not counted, which only ever lowers the count.
/// </remarks>
public class LargeNetworkFileTests
{
    // Two thousand by two thousand: the network M16 read a network's part of, four million and two thousand parameters.
    private const int Width = 2000;

    private const double BytesAParameter = 100;

    [Fact]
    public void ANetworksPart_OfFourMillionParameters_IsReadBack_AllocatingUnderAHundredBytesAParameter()
    {
        var network = new LayerStack(new Dense(Width, Width, Start(1)));
        var json = Part(network);
        var catalog = NetworkCatalog.BuiltIn();
        SavedNetwork read = default;

        var allocated = Allocated(() => read = NetworkDocument.ReadNetwork(json, "network", catalog));

        Assert.Equal(4_002_000, Parameters(network));
        Assert.InRange(allocated / (double)Parameters(network), 0, BytesAParameter);
        Assert.Equal(Bits(network), Bits(read.Network));
    }

    [Fact]
    public void TheOneFile_OfANetworkOfFourMillionParameters_IsReadBack_AllocatingUnderAHundredBytesAParameter()
    {
        var prepared = WikiTitanic.In(WikiTitanic.DataFolder).Run();
        var network = new LayerStack(
            new Dense(14, Width, Start(1)), new Relu(), new Dense(Width, Width, Start(2)), new Relu(), new Dense(Width, 1, Start(3)));
        var json = TrainedNetwork.Of(network, new BinaryCrossEntropy(), prepared, seed: 1).ToJson();
        var networks = NetworkCatalog.BuiltIn();
        var steps = StepCatalog.BuiltIn();
        TrainedNetwork? read = null;

        var allocated = Allocated(() => read = TrainedNetwork.FromJson(json, networks, steps));

        Assert.Equal(4_034_001, Parameters(network));
        Assert.InRange(allocated / (double)Parameters(network), 0, BytesAParameter);
        Assert.Equal(Bits(network), Bits(read!.Network));

        // What the largest file stands on: .NET makes no text longer than 1,073,741,791 characters, and a parameter takes
        // between 23.6 and 24 of them, so the one file holds a network of up to about 45 million parameters.
        Assert.InRange(json.Length / (double)Parameters(network), 23.6, 24);
    }

    [Theory]
    [InlineData(false, 75.5, 77.5)]
    [InlineData(true, 103, 105)]
    public void ACheckpointUnderAdam_TakesAbout76CharactersAParameter_And104WhenItsRunKeepsTheBestEpochs(bool keepsTheBest, double least, double most)
    {
        // What the largest checkpoint stands on: beside each parameter, the two numbers Adam remembers of it, and the best
        // epoch's when the run keeps it — about 14 million parameters, or about 10 million, in a text .NET can make.
        var prepared = WikiTitanic.In(WikiTitanic.DataFolder).Run();
        var network = new LayerStack(new Dense(14, 256, Start(1)), new Relu(), new Dense(256, 256, Start(2)), new Relu(), new Dense(256, 1, Start(3)));
        var compiled = network.Compile(new Adam(0.001), new BinaryCrossEntropy());
        string? file = null;

        compiled.Fit(prepared, new FitOptions(seed: 7)
        {
            Epochs = 1,
            EarlyStopping = keepsTheBest ? new EarlyStopping { RestoreBest = true } : null,
            Checkpoints = new Checkpoints(checkpoint => file = CheckpointFile.Write(compiled, prepared, checkpoint)),
        });

        Assert.InRange(file!.Length / (double)Parameters(network), least, most);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ACheckpointOfFourMillionParametersUnderAdam_IsReadBack_AllocatingNoMoreThanItsTextOnceBesidesItsOwnCopy(bool keepsTheBest)
    {
        // A checkpoint is read once: its network and its run from one reading of its text, so what the read allocates is
        // the text's own copy in UTF-8 and the numbers it holds — never the whole text again.
        var prepared = Unreported();
        var network = new LayerStack(
            new Dense(14, Width, Start(1)), new Relu(), new Dense(Width, Width, Start(2)), new Relu(), new Dense(Width, 1, Start(3)));
        var compiled = network.Compile(new Adam(0.001), new BinaryCrossEntropy());
        string? file = null;

        compiled.Fit(prepared, new FitOptions(seed: 7)
        {
            Epochs = 1,
            BatchSize = 1024,
            EarlyStopping = keepsTheBest ? new EarlyStopping { RestoreBest = true } : null,
            Checkpoints = new Checkpoints(checkpoint => file = CheckpointFile.Write(compiled, prepared, checkpoint)),
        });

        var networks = NetworkCatalog.BuiltIn();
        ResumedRun read = default;

        var allocated = Allocated(() => read = CheckpointFile.Read(file!, networks, prepared));

        Assert.Equal(4_034_001, Parameters(network));
        Assert.InRange(allocated / (double)Parameters(network), 0, 2.0 * file!.Length / Parameters(network));
        Assert.Equal(Bits(network), Bits(read.Compiled.Network));
    }

    // What a read allocates on this thread, the only one it runs on.
    private static long Allocated(Action read)
    {
        var before = GC.GetAllocatedBytesForCurrentThread();

        read();

        return GC.GetAllocatedBytesForCurrentThread() - before;
    }

    // A network's part on its own, under the key it is read from, laid out as the one file lays it out.
    private static string Part(Network network)
    {
        using var stream = new MemoryStream();

        using (var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true }))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, network, new MeanSquaredError());
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(stream.ToArray()).ReplaceLineEndings("\n");
    }

    private static Draws Start(int seed) => new RandomStream(seed).Draw("initialise:test", 0, 0);

    // The wiki's Titanic pipeline without its report, which would measure the network of four million parameters on every
    // part once the run is over: what is read here is the checkpoint, not the measures.
    private static PreparedData Unreported() =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("pclass", "sex").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            .FillMissing("age", With.Median)
            .EncodeCategories()
            .Normalise("age", Scale.MidRange)
            .Normalise("fare", Scale.MidRange)
            .Normalise("sibsp", Scale.MidRange)
            .Normalise("parch", Scale.MidRange)
            .Target("survived")
            .Build()
            .Run();

    private static long Parameters(Network network) => network.Slots().Sum(named => (long)named.Slot.Value.Shape.Count);

    private static int[] Bits(Network network) =>
        [.. network.Slots().SelectMany(named => named.Slot.Value.Values.ToArray()).Select(BitConverter.SingleToInt32Bits)];
}
