// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A pipeline's text as one writer writes it — two spaces, a line feed, the escapes System.Text.Json writes — so however
/// another writer spaced it, ended its lines or escaped its strings, it is the text its own writer wrote, and a record of it
/// still holds. Two pipelines are one fit when that writer writes them alike but for the version of the pipeline file each
/// names, and only between versions in which every step means what it means now: the one rule a network's file, a
/// checkpoint and predictions handed back as text are all held to a fit by.
/// </summary>
public class PipelineTextTests
{
    // The version of the pipeline file every verb this library ships first appears in. A row is added when a verb is; a
    // row never changes, because a verb that came to mean something else would have to.
    private static readonly (string Verb, int Since)[] Versions =
    [
        ("declare", 1),
        ("drop.columns", 2),
        ("drop.gaps", 2),
        ("drop.warmup", 2),
        ("encode", 1),
        ("encode.categories", 2),
        ("evidence.correlation", 2),
        ("evidence.profile", 2),
        ("evidence.report", 3),
        ("feature.add", 1),
        ("feature.cycle", 8),
        ("feature.cyclical", 1),
        ("feature.indicator", 1),
        ("feature.timeParts", 1),
        ("fill.missing", 1),
        ("fill.nan", 1),
        ("learn.ml", 7),
        ("learn.network", 5),
        ("maths", 1),
        ("normalise", 1),
        ("normalise.row", 1),
        ("order.by", 2),
        ("outliers.clip", 1),
        ("read.csv", 1),
        ("read.excel", 1),
        ("read.join", 8),
        ("read.json", 1),
        ("read.parquet", 1),
        ("read.rows", 1),
        ("scale.given", 7),
        ("settle.gaps", 6),
        ("shuffle", 7),
        ("split.atRandom", 2),
        ("split.byTime", 2),
        ("split.stratified", 2),
        ("target", 1),
        ("target.ahead", 2),
        ("target.distribution", 2),
        ("target.labels", 2),
        ("target.numbers", 8),
    ];

    private static readonly JsonSerializerOptions Indented = new() { WriteIndented = true };

    [Fact]
    public void APipelineThisLibraryWrites_IsCarriedAsItsOwnFile_ByteForByte()
    {
        var prepared = Rows(20);

        var carried = PipelineText.Of(prepared);

        Assert.Equal(prepared.ToJson(), carried.Text);
        Assert.Equal(Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(prepared.ToJson()))).ToLowerInvariant(), carried.Digest);
        Assert.Equal(PipelineDeclaration.Version, carried.Version);
    }

    [Theory]
    [InlineData("compact")]
    [InlineData("tabs")]
    [InlineData("carriage returns")]
    [InlineData("escaped")]
    public void ItsText_SpacedOrEscapedOtherwise_IsTheTextItsWriterWrote(string how)
    {
        var written = Rows(20).ToJson();

        var carried = PipelineText.Of(Encoding.UTF8.GetBytes(Otherwise(written, how)));

        Assert.Equal(written, carried.Text);
    }

    [Fact]
    public void KeysInAnotherOrder_OrANumberSpelledOtherwise_AreAnotherText_ForOnlyTheWriterThatChoseThemCouldSayThemBack()
    {
        var written = Rows(20).ToJson();
        var reordered = JsonNode.Parse(written)!.AsObject();
        var declaration = reordered["declaration"];
        reordered.Remove("declaration");
        reordered["declaration"] = declaration;

        Assert.NotEqual(written, PipelineText.Of(Encoding.UTF8.GetBytes(reordered.ToJsonString(Indented))).Text);
        Assert.NotEqual(written, PipelineText.Of(Encoding.UTF8.GetBytes(Once(written, "\"train\": 0.5,").Replace("\"train\": 0.5,", "\"train\": 0.50,", StringComparison.Ordinal))).Text);
    }

    [Fact]
    public void TwoPipelinesWrittenAlikeButForTheVersionTheyName_AreOneFit_AndAnotherFitIsNot()
    {
        var prepared = Rows(20);
        var handed = PipelineText.Of(prepared);
        var newer = PipelineText.Of(Encoding.UTF8.GetBytes(Restamped(prepared.ToJson(), PipelineDeclaration.Version + 1)));
        var grown = PipelineText.Of(Rows(24));

        Assert.True(newer.IsTheFitOf(handed));
        Assert.NotEqual(handed.Digest, newer.Digest);
        Assert.Equal(PipelineDeclaration.Version + 1, newer.Version);
        Assert.False(grown.IsTheFitOf(handed));
        Assert.False(PipelineText.Of(Encoding.UTF8.GetBytes(Restamped(Rows(24).ToJson(), PipelineDeclaration.Version + 1))).IsTheFitOf(handed));
    }

    [Fact]
    public void TheDigestAModelIsBoundBy_NamesTheVersion_AndTheFitsDigestIsTheOneThatDoesNot()
    {
        // Why a model trained behind an older pipeline keeps loading: its file carries that pipeline's text exactly as it
        // was written, so the version the text names never changes and neither does the digest taken over it. The digest
        // that leaves the version out is the other one, which is how two pipelines are compared fit for fit.
        var handed = PipelineText.Of(Rows(20));
        var newer = PipelineText.Of(Encoding.UTF8.GetBytes(Restamped(Rows(20).ToJson(), PipelineDeclaration.Version + 1)));

        Assert.Contains($"\"version\": {PipelineDeclaration.Version}", handed.Text, StringComparison.Ordinal);
        Assert.NotEqual(handed.Digest, newer.Digest);
        Assert.Equal(handed.FitDigest, newer.FitDigest);
    }

    [Fact]
    public void TheFitsDigest_IsTheDigestOfTheTextWithoutItsVersion_AndTwoPipelinesAreOneFitExactlyWhenTheirsAreEqual()
    {
        // What a comparison of two pipelines carries when the texts themselves cannot travel: the SHA-256 of the text the one
        // writer writes with the version left out.
        var prepared = Rows(20);
        var handed = PipelineText.Of(prepared);
        var unversioned = JsonNode.Parse(prepared.ToJson())!.AsObject();

        unversioned.Remove("version");

        Assert.Equal(
            Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(unversioned.ToJsonString(Indented).ReplaceLineEndings("\n")))).ToLowerInvariant(),
            handed.FitDigest);
        Assert.Equal(handed.FitDigest, PipelineText.Of(Encoding.UTF8.GetBytes(Restamped(prepared.ToJson(), PipelineDeclaration.Version + 1))).FitDigest);
        Assert.Equal(handed.FitDigest, PipelineText.Of(Rows(20)).FitDigest);
        Assert.NotEqual(handed.FitDigest, PipelineText.Of(Rows(24)).FitDigest);
        Assert.NotEqual(handed.Digest, handed.FitDigest);
        Assert.Matches("^[0-9a-f]{64}$", handed.FitDigest);
    }

    [Theory]
    [InlineData(0, false)]
    [InlineData(2, false)]
    [InlineData(3, true)]
    [InlineData(4, true)]
    public void TwoTexts_AreComparedOnlyBetweenVersionsInWhichEveryStepMeansWhatItMeansNow(int version, bool comparable)
    {
        Assert.Equal(comparable, PipelineText.IsComparable(version));
        Assert.True(PipelineText.IsComparable(PipelineDeclaration.Version));
        Assert.False(PipelineText.IsComparable(PipelineDeclaration.Version + 1));
        Assert.Equal(3, PipelineText.FirstComparable);
    }

    [Theory]
    [InlineData("\"three\"")]
    [InlineData("-3")]
    [InlineData(null)]
    public void AVersionThatIsNoWholeNumber_ANumberBelowNought_OrNone_IsReadAsNought(string? version)
    {
        var node = JsonNode.Parse(Rows(20).ToJson())!.AsObject();

        if (version is null)
        {
            node.Remove("version");
        }
        else
        {
            node["version"] = JsonNode.Parse(version);
        }

        Assert.Equal(0, PipelineText.Of(Encoding.UTF8.GetBytes(node.ToJsonString(Indented))).Version);
    }

    [Fact]
    public void WhatIsNoPipelinesText_IsRefused()
    {
        Assert.Contains("one JSON object", Assert.Throws<ArgumentException>(() => PipelineText.Of(Encoding.UTF8.GetBytes("[1, 2]"))).Message, StringComparison.Ordinal);
        Assert.ThrowsAny<JsonException>(() => PipelineText.Of(Encoding.UTF8.GetBytes("not JSON")));
        Assert.Throws<ArgumentNullException>(() => PipelineText.Of((PreparedData)null!));
        Assert.Throws<ArgumentNullException>(() => PipelineText.Of(Rows(20)).IsTheFitOf(null!));
    }

    [Fact]
    public void NoStepMeansAnythingElseThanInTheFirstVersionANetworksFileWasWrittenWith()
    {
        // Two pipelines are compared, never read, so the version each names is left out only while every step means now what
        // it meant when the first network's file was written — the version 0.4.0's files name. A step that comes to mean
        // something else from a later version fails here, and the comparison has to learn its old meaning before that version
        // ships.
        var catalog = Shipped.Catalog();
        var first = JsonNode.Parse(File.ReadAllText(Path.Join(AppContext.BaseDirectory, "Learners", "Fixtures", "titanic-0.4.0.network.json")))!["pipeline"]!["version"]!.GetValue<int>();

        Assert.Equal(PipelineText.FirstComparable, first);
        Assert.All(StepCatalog.VerbsOtherPackagesBring.Keys, verb => Assert.True(catalog.Knows(verb), $"The catalog here does not know '{verb}'."));

        // The version a verb first appears in, written down once for every verb this library ships. A verb may be added —
        // a file written before it cannot name it, so no text's meaning moves — but a verb already here may never be given
        // another first version: that is what a changed meaning would need, and it would make two texts of one fit differ.
        Assert.Equal(
            Versions,
            catalog.Descriptions.Select(description => (description.Verb, description.Since)).OrderBy(each => each.Verb, StringComparer.Ordinal));
    }

    // Rows t = 1…count of x = sin t, whose answer is whether x is above nought: the same declaration, whatever the count.
    private static PreparedData Rows(int count) =>
        Pdd.Create()
            .Read(
                CsvRowSource.FromText("t,x,y\n" + string.Join('\n', Enumerable.Range(1, count).Select(t =>
                    string.Create(CultureInfo.InvariantCulture, $"{t},{Math.Sin(t)},{(Math.Sin(t) > 0 ? 1 : 0)}"))) + "\n"),
                "rows of a sine")
            .Declare(schema => schema.Integer("t", "y").Number("x"))
            .SplitByTime("t", 0.5, 0.25)
            .Normalise("x", Scale.MidRange)
            .Drop("t")
            .Target("y")
            .Build()
            .Run();

    // The text as another writer would have written it.
    private static string Otherwise(string text, string how) => how switch
    {
        "compact" => JsonNode.Parse(text)!.ToJsonString(),
        "tabs" => string.Join('\n', text.Split('\n').Select(line => new string('\t', (line.Length - line.TrimStart(' ').Length) / 2) + line.TrimStart(' '))),
        "carriage returns" => text.ReplaceLineEndings("\r\n"),
        _ => Once(text, "\"rows of a sine\"").Replace("\"rows of a sine\"", "\"\\u0072ows of a sine\"", StringComparison.Ordinal),
    };

    private static string Restamped(string text, int version)
    {
        var node = JsonNode.Parse(text)!.AsObject();
        node["version"] = version;

        return node.ToJsonString(Indented).ReplaceLineEndings("\n");
    }

    private static string Once(string text, string piece)
    {
        Assert.Contains(piece, text, StringComparison.Ordinal);
        Assert.Equal(text.IndexOf(piece, StringComparison.Ordinal), text.LastIndexOf(piece, StringComparison.Ordinal));

        return text;
    }
}
