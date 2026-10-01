// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// Every file DeepSharp writes places a fault where an editor shows it: its line, and its column counted in characters
/// rather than in the bytes a word takes.
/// </summary>
/// <remarks>
/// The pipeline's file, a network's part and a trained network's one file are each read by the package that owns them,
/// and the packages do not reference one another, so each places its faults with a copy of the same rule. A rule kept
/// three times holds only while the three are compared: this is that comparison, one fault after a word written in
/// letters of more than one byte, through each reader's own door.
/// </remarks>
public class FaultPlaceTests
{
    [Fact]
    public void APipelineFile_PlacesAFaultAfterAWordOfWideLetters_ByCharacters()
    {
        const string json = """{"version":2,"declaration":[{"step":"read.csv","path":"Größe.csv"},{"step":"zürich"}]}""";

        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));
        var fault = Assert.Single(refused.Faults);

        Assert.Equal((1, CharacterColumn(json, "{\"step\":\"zürich\"")), (fault.Line, fault.Column));
    }

    [Fact]
    public void ANetworksPart_PlacesAFaultAfterAWordOfWideLetters_ByCharacters()
    {
        const string json = """{"network": {"version": 1, "layers": {"kind": "stack", "layers": [{"kind": "relu", "Größe": 1}, {"kind": "zürich"}]}, "parameters": {}, "state": {}, "loss": {"kind": "meanSquaredError"}}}""";

        var refused = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(json, "network", NetworkCatalog.BuiltIn()));
        var fault = refused.Faults.Single(each => each.Message.Contains("zürich", StringComparison.Ordinal));

        Assert.Equal(1, fault.Line);
        Assert.Contains(fault.Column, Columns(json, "zürich"));
    }

    [Fact]
    public void ATrainedNetworksFile_PlacesAFaultAfterAWordOfWideLetters_ByCharacters()
    {
        const string json = """{"Größe": 0, "zürich": 1}""";

        var refused = Assert.Throws<NetworkFileException>(() => TrainedNetwork.FromJson(json, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn()));
        var fault = refused.Faults.Single(each => each.Message.Contains("'zürich'", StringComparison.Ordinal));

        Assert.Equal((1, CharacterColumn(json, "\"zürich\"")), (fault.Line, fault.Column));
    }

    // Lines before the fault carry letters of two bytes, of three, and a pair of surrogates — one character of four bytes,
    // which a column counts as two, as an editor that counts UTF-16 does — and the fault's own line carries them before it.
    private const string Wide = "\"Größe €\": \"😀 über\"";

    [Fact]
    public void APipelineInTheOneFile_PlacesAFaultAfterLinesOfWideLetters_ByCharacters()
    {
        var json = "{\n  \"version\": 1,\n  \"network\": {" + Wide + "},\n  \"pipeline\": {\"version\": 3, \"declaration\": [\n    {\"step\": \"read.csv\", \"path\": \"😀.csv\"}, {\"step\": \"zürich\"}]}\n}\n";

        var fault = Assert.Single(Assert.Throws<PipelineFileException>(() => PreparedData.FromJson(json, StepCatalog.BuiltIn(), "pipeline")).Faults);

        Assert.Equal(Place(json, "{\"step\": \"zürich\"}"), new Spot(fault.Line, fault.Column));
    }

    [Fact]
    public void ANetworksPart_PlacesAFaultAtAValueOfAListAfterLinesOfWideLetters_ByCharacters()
    {
        var json = "{\"network\": {\"version\": 1, \"layers\": {\"kind\": \"stack\", \"layers\": [{\"kind\": \"dense\", \"inputs\": 1, \"outputs\": 3}]},\n  "
            + "\"parameters\": {\"0.weight\": {\"shape\": [1, 3], \"values\": [0.5,\n    \"Größe😀\", \"x\"]}, \"0.bias\": {\"shape\": [3], \"values\": [0, 0, 0]}},\n  "
            + "\"state\": {}, \"loss\": {\"kind\": \"meanSquaredError\"}}}";

        var faults = Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(json, "network", NetworkCatalog.BuiltIn())).Faults;

        Assert.Equal([Place(json, "\"Größe😀\""), Place(json, "\"x\"")], faults.Select(fault => new Spot(fault.Line, fault.Column)));
    }

    [Fact]
    public void ATrainedNetworksFile_PlacesAFaultAtItsTopAfterLinesOfWideLetters_ByCharacters()
    {
        var json = "{\n  \"version\": 1,\n  \"network\": {" + Wide + "},\n  \"😀\": 0, \"zürich\": 1\n}\n";

        var faults = Assert.Throws<NetworkFileException>(() => TrainedNetwork.FromJson(json, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn())).Faults;

        Assert.Equal([Place(json, "\"😀\""), Place(json, "\"zürich\"")], faults.Select(fault => new Spot(fault.Line, fault.Column)));
    }

    [Theory]
    [InlineData("pipeline")]
    [InlineData("network")]
    [InlineData("trained network")]
    public void TextThatStopsBeingJsonAfterLinesOfWideLetters_IsPlacedWhereItStops_ByEachReader(string reader)
    {
        var json = "{\n  \"version\": 1,\n  \"network\": {" + Wide + "},\n  \"pipeline\": {\"declaration\": []},\n  \"über\": 😀\n}\n";

        var fault = reader switch
        {
            "pipeline" => Said.Of(Assert.Single(Assert.Throws<PipelineFileException>(() => PreparedData.FromJson(json, StepCatalog.BuiltIn(), "pipeline")).Faults)),
            "network" => Said.Of(Assert.Single(Assert.Throws<NetworkFileException>(() => NetworkDocument.ReadNetwork(json, "network", NetworkCatalog.BuiltIn())).Faults)),
            _ => Said.Of(Assert.Single(Assert.Throws<NetworkFileException>(() => TrainedNetwork.FromJson(json, NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn())).Faults)),
        };

        Assert.Equal(Place(json, "😀\n}"), fault.Spot);
        Assert.StartsWith("The text stops being JSON here: ", fault.Message, StringComparison.Ordinal);
    }

    // Where a piece of a text starts, as an editor shows it: its line and its column, both from one, the column counted in
    // the text's own characters.
    private static Spot Place(string text, string piece)
    {
        var at = text.IndexOf(piece, StringComparison.Ordinal);

        Assert.Equal(at, text.LastIndexOf(piece, StringComparison.Ordinal));

        return new Spot(text[..at].Count(letter => letter == '\n') + 1, at - (text.LastIndexOf('\n', Math.Max(at - 1, 0)) + 1) + 1);
    }

    // The column, from one, a piece of a one-line text starts at, counted in characters.
    private static int CharacterColumn(string text, string piece)
    {
        Assert.Equal(text.IndexOf(piece, StringComparison.Ordinal), text.LastIndexOf(piece, StringComparison.Ordinal));

        return text.IndexOf(piece, StringComparison.Ordinal) + 1;
    }

    // The columns, from one and in characters, of the object and the value a word is written in: a network's part places
    // a fault of a layer at one of them.
    private static int[] Columns(string text, string word)
    {
        var value = text.IndexOf($"\"{word}\"", StringComparison.Ordinal);
        var layer = text.LastIndexOf('{', value);

        return [layer + 1, text.LastIndexOf("\"kind\"", value, StringComparison.Ordinal) + 1, value + 1];
    }

    private readonly record struct Spot(int Line, int Column);

    // A fault a reader placed: where, and what it says.
    private readonly record struct Said(Spot Spot, string Message)
    {
        public static Said Of(NetworkFileFault fault) => new(new Spot(fault.Line, fault.Column), fault.Message);

        public static Said Of(PipelineFileFault fault) => new(new Spot(fault.Line, fault.Column), fault.Message);
    }
}
