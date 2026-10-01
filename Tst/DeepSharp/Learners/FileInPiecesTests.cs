// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// The one file is walked a piece of its UTF-8 at a time — by the survey of its top and by the reader of the pipeline it
/// carries — so neither holds a copy of the file: what stands across two pieces, or is longer than a piece, is read as it
/// is read in one, and a fault many pieces in is placed on its own line.
/// </summary>
/// <remarks>A piece is 64 KiB; the files here put what they test across one, or make it longer than one.</remarks>
public class FileInPiecesTests
{
    private const int Piece = 1 << 16;

    private const string PipelineKey = "\"pipeline\": ";

    [Fact]
    public void TheOneFile_ItsPipelineStandingAcrossTwoPiecesOfTheWalk_IsReadAsTheSameFileInOnePiece()
    {
        var trained = Trained();
        var json = trained.ToJson();
        var at = json.IndexOf(PipelineKey, StringComparison.Ordinal);

        // Spaces before the pipeline's key, so its object starts fifty bytes before the end of the first piece.
        var padded = json[..at] + new string(' ', Piece - 50 - (at + PipelineKey.Length)) + json[at..];
        var read = TrainedNetwork.FromJson(padded, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn());

        Assert.Equal(Piece - 50, Encoding.UTF8.GetByteCount(padded[..padded.IndexOf(PipelineKey, StringComparison.Ordinal)]) + PipelineKey.Length);
        Assert.Equal(json, read.ToJson());
        Assert.Equal(trained.Prepared.ToJson(), PreparedData.FromJson(padded, StepCatalog.BuiltIn(), "pipeline").ToJson());
    }

    [Fact]
    public void AWordLongerThanAPiece_IsReadPast_ByTheSurveyOfTheFile_AndByTheReaderOfThePipelineInIt()
    {
        var trained = Trained();
        var json = trained.ToJson();
        var at = json.IndexOf(PipelineKey, StringComparison.Ordinal);
        var noted = json[..at] + "\"note\": \"" + new string('ü', Piece) + "\",\n  " + json[at..];

        var fault = Assert.Single(Assert.Throws<NetworkFileException>(() => TrainedNetwork.FromJson(noted, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn())).Faults);

        Assert.Equal(Place(noted, "\"note\""), new Spot(fault.Line, fault.Column));
        Assert.StartsWith("'note' is no part of a trained network's file", fault.Message, StringComparison.Ordinal);
        Assert.Equal(trained.Prepared.ToJson(), PreparedData.FromJson(noted, StepCatalog.BuiltIn(), "pipeline").ToJson());
    }

    [Theory]
    [InlineData("pipeline")]
    [InlineData("network")]
    [InlineData("trained network")]
    public void TextThatStopsBeingJsonManyPiecesIntoTheFile_IsPlacedWhereItStops_ByEachReader(string reader)
    {
        // Thirty thousand lines of letters of one, two, three and four bytes before the text stops being JSON.
        var lines = string.Join(",\n", Enumerable.Range(0, 30_000).Select(line => $"    \"größe {line} €\": \"😀\""));
        var json = "{\n  \"version\": 1,\n  \"network\": {\n" + lines + "\n  },\n  \"pipeline\": {\"declaration\": []},\n  \"über\": 😀\n}\n";

        var fault = reader switch
        {
            "pipeline" => Said.Of(Assert.Single(Assert.Throws<PipelineFileException>(() => PreparedData.FromJson(json, StepCatalog.BuiltIn(), "pipeline")).Faults)),
            "network" => Said.Of(Assert.Single(Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(json, "network", NetworkCatalog.BuiltIn())).Faults)),
            _ => Said.Of(Assert.Single(Assert.Throws<NetworkFileException>(() => TrainedNetwork.FromJson(json, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn())).Faults)),
        };

        Assert.True(Encoding.UTF8.GetByteCount(json) > 10 * Piece);
        Assert.Equal(Place(json, "😀\n}"), fault.Spot);
        Assert.StartsWith("The text stops being JSON here: ", fault.Message, StringComparison.Ordinal);
    }

    // A small network behind the wiki's Titanic pipeline.
    private static TrainedNetwork Trained() =>
        TrainedNetwork.Of(
            new LayerStack(new Dense(14, 1, new RandomStream(3).Draw("initialise:test", 0, 0))), new BinaryCrossEntropy(), WikiTitanic.In(WikiTitanic.DataFolder).Run(), seed: 1);

    // Where a piece of a text starts, as an editor shows it: its line and its column, both from one, the column counted in
    // the text's own characters.
    private static Spot Place(string text, string piece)
    {
        var at = text.IndexOf(piece, StringComparison.Ordinal);

        Assert.Equal(at, text.LastIndexOf(piece, StringComparison.Ordinal));

        return new Spot(text[..at].Count(letter => letter == '\n') + 1, at - (text.LastIndexOf('\n', Math.Max(at - 1, 0)) + 1) + 1);
    }

    private readonly record struct Spot(int Line, int Column);

    // A fault a reader placed: where, and what it says.
    private readonly record struct Said(Spot Spot, string Message)
    {
        public static Said Of(NetworkFileFault fault) => new(new Spot(fault.Line, fault.Column), fault.Message);

        public static Said Of(PipelineFileFault fault) => new(new Spot(fault.Line, fault.Column), fault.Message);
    }
}
