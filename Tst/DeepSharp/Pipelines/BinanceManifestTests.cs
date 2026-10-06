// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What a landing says of itself, beside the file: what was asked, when, of whom and under whose terms, what came back page
/// by page, and the fingerprint of the bytes that landed. It is written and read as the library's other versioned documents
/// are — a version first, a newer one refused whole — and it holds nothing that says the landing is complete, because a
/// marker written next to the work is a promise nobody checked: the file's own move is what completes it.
/// </summary>
public sealed class BinanceManifestTests
{
    private static readonly string Fingerprint = string.Concat(Enumerable.Repeat("ab", 32));

    private static DateTime Utc(int year, int month, int day, int hour = 0, int minute = 0, int second = 0, int millisecond = 0) =>
        new(year, month, day, hour, minute, second, millisecond, DateTimeKind.Utc);

    private static LandingManifest Sample() => new(
        Host: "https://data-api.binance.vision",
        Symbol: "BTCEUR",
        Interval: "1d",
        From: Utc(2024, 1, 1),
        To: Utc(2024, 1, 4),
        AskedAt: Utc(2026, 10, 6, 8, 0, 0, 123),
        ServerTime: Utc(2026, 10, 6, 8, 0, 0, 456),
        SettleAfterCloseSeconds: 60,
        PagesEstimated: 1,
        Pages: [new PageReceipt(Utc(2024, 1, 1), 2, new string('c', 64), "Tue, 06 Oct 2026 08:00:01 GMT", "5e7d0c6a-3b2f-4a8e-9d11-0a1b2c3d4e5f")],
        Rows: 2,
        Gaps: 1,
        First: Utc(2024, 1, 1),
        Last: Utc(2024, 1, 3),
        File: "BTCEUR-1d-20240101T000000Z-20240104T000000Z.csv",
        Fingerprint: Fingerprint);

    private static JsonElement Parsed(string json) => JsonDocument.Parse(json).RootElement.Clone();

    private static IEnumerable<string> Keys(JsonElement element) => element.ValueKind switch
    {
        JsonValueKind.Object => element.EnumerateObject().SelectMany(property => new[] { property.Name }.Concat(Keys(property.Value))),
        JsonValueKind.Array => element.EnumerateArray().SelectMany(Keys),
        _ => [],
    };

    [Fact]
    public void TheManifest_IsOneJsonObject_OfVersionOne_FirstOfAll()
    {
        var json = Sample().ToJson();

        Assert.StartsWith("{\n  \"version\": 1,\n", json, StringComparison.Ordinal);
        Assert.Equal(1, Parsed(json).GetProperty("version").GetInt32());
        Assert.Equal(1, LandingManifest.Version);
    }

    [Fact]
    public void TheManifest_SaysNothingOfBeingComplete_ForTheFileItsMoveIsWhatCompletesIt()
    {
        var keys = Keys(Parsed(Sample().ToJson())).Select(key => key.ToLowerInvariant()).ToArray();

        Assert.DoesNotContain(keys, key => key.Contains("complete", StringComparison.Ordinal) || key.Contains("done", StringComparison.Ordinal));
    }

    [Fact]
    public void TheManifestNamesTheHostAndTheTermsItWasFetchedUnder()
    {
        var root = Parsed(Sample().ToJson());

        Assert.Equal("Binance", root.GetProperty("venue").GetString());
        Assert.Equal("https://data-api.binance.vision", root.GetProperty("host").GetString());
        Assert.Equal("/api/v3/klines", root.GetProperty("endpoint").GetString());
        Assert.Equal("https://data.binance.vision/Binance_Vision-Terms_of_Use.pdf", root.GetProperty("terms").GetString());
    }

    [Fact]
    public void TheManifestRecordsWhatWasAskedAndWhen_AndEachPageAsItCame()
    {
        var root = Parsed(Sample().ToJson());
        var asked = root.GetProperty("asked");
        var page = root.GetProperty("pages")[0];

        Assert.Equal("BTCEUR", asked.GetProperty("symbol").GetString());
        Assert.Equal("1d", asked.GetProperty("interval").GetString());
        Assert.Equal("2024-01-01T00:00:00Z", asked.GetProperty("from").GetString());
        Assert.Equal("2024-01-04T00:00:00Z", asked.GetProperty("to").GetString());
        Assert.Equal("2026-10-06T08:00:00.123Z", root.GetProperty("askedAt").GetString());
        Assert.Equal("2026-10-06T08:00:00.456Z", root.GetProperty("serverTime").GetString());
        Assert.Equal(60, root.GetProperty("settleAfterCloseSeconds").GetInt32());
        Assert.Equal(1, root.GetProperty("pagesEstimated").GetInt32());
        Assert.Equal("2024-01-01T00:00:00Z", page.GetProperty("from").GetString());
        Assert.Equal(2, page.GetProperty("rows").GetInt32());
        Assert.Equal(new string('c', 64), page.GetProperty("sha256").GetString());
        Assert.Equal("Tue, 06 Oct 2026 08:00:01 GMT", page.GetProperty("date").GetString());
        Assert.Equal("5e7d0c6a-3b2f-4a8e-9d11-0a1b2c3d4e5f", page.GetProperty("uuid").GetString());
    }

    [Fact]
    public void TheManifestCountsTheCandlesTheVenueDidNotAnswerWith_AndFillsNone()
    {
        var root = Parsed(Sample().ToJson());

        Assert.Equal(2, root.GetProperty("rows").GetInt32());
        Assert.Equal(1, root.GetProperty("gaps").GetInt32());
        Assert.Equal("2024-01-01T00:00:00Z", root.GetProperty("first").GetString());
        Assert.Equal("2024-01-03T00:00:00Z", root.GetProperty("last").GetString());
    }

    [Fact]
    public void TheManifestNamesTheFileAndTheFingerprintOfItsBytes()
    {
        var root = Parsed(Sample().ToJson());

        Assert.Equal("BTCEUR-1d-20240101T000000Z-20240104T000000Z.csv", root.GetProperty("file").GetString());
        Assert.Equal(Fingerprint, root.GetProperty("fingerprint").GetString());
    }

    [Fact]
    public void ThePageThatCameWithoutAnIdentifier_IsWrittenAsNothing_NotAsAnEmptyOne()
    {
        var bare = Sample() with { Pages = [new PageReceipt(Utc(2024, 1, 1), 2, new string('c', 64), null, null)] };
        var page = Parsed(bare.ToJson()).GetProperty("pages")[0];

        Assert.False(page.TryGetProperty("date", out _));
        Assert.False(page.TryGetProperty("uuid", out _));
        Assert.Null(LandingManifest.FromJson(bare.ToJson()).Pages[0].Uuid);
    }

    [Fact]
    public void TheManifest_IsWrittenWithLineFeedsAndTwoSpaces_WhateverTheMachine()
    {
        var json = Sample().ToJson();

        Assert.DoesNotContain('\r', json);
        Assert.EndsWith("}\n", json, StringComparison.Ordinal);
        Assert.Contains("\n  \"asked\": {\n    \"symbol\": \"BTCEUR\",", json, StringComparison.Ordinal);
    }

    [Fact]
    public void TheManifest_ReadsBackAsItWasWritten()
    {
        var written = Sample();
        var read = LandingManifest.FromJson(written.ToJson());

        Assert.Equal(written.Host, read.Host);
        Assert.Equal(written.Symbol, read.Symbol);
        Assert.Equal(written.Interval, read.Interval);
        Assert.Equal(written.From, read.From);
        Assert.Equal(written.To, read.To);
        Assert.Equal(written.AskedAt, read.AskedAt);
        Assert.Equal(written.ServerTime, read.ServerTime);
        Assert.Equal(written.SettleAfterCloseSeconds, read.SettleAfterCloseSeconds);
        Assert.Equal(written.PagesEstimated, read.PagesEstimated);
        Assert.Equal(written.Pages, read.Pages);
        Assert.Equal(written.Rows, read.Rows);
        Assert.Equal(written.Gaps, read.Gaps);
        Assert.Equal(written.First, read.First);
        Assert.Equal(written.Last, read.Last);
        Assert.Equal(written.File, read.File);
        Assert.Equal(written.Fingerprint, read.Fingerprint);
        Assert.Equal(DateTimeKind.Utc, read.AskedAt.Kind);
        Assert.Equal(written.ToJson(), read.ToJson());
    }

    [Fact]
    public void ANewerManifest_IsRefusedWhole_SayingWhichVersionWroteIt()
    {
        var newer = Sample().ToJson().Replace("\"version\": 1", "\"version\": 2", StringComparison.Ordinal);
        var refused = Assert.Throws<FormatException>(() => LandingManifest.FromJson(newer));

        Assert.Contains("version 2", refused.Message, StringComparison.Ordinal);
        Assert.Contains("up to version 1", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("")]
    [InlineData("not json")]
    [InlineData("[]")]
    [InlineData("{}")]
    [InlineData("{\"version\": 0}")]
    [InlineData("{\"version\": \"1\"}")]
    public void WhatIsNotAManifest_IsRefused_SayingWhy(string text)
    {
        Assert.Throws<FormatException>(() => LandingManifest.FromJson(text));
    }

    [Theory]
    [InlineData("host")]
    [InlineData("asked")]
    [InlineData("askedAt")]
    [InlineData("serverTime")]
    [InlineData("pages")]
    [InlineData("rows")]
    [InlineData("gaps")]
    [InlineData("first")]
    [InlineData("last")]
    [InlineData("file")]
    [InlineData("fingerprint")]
    public void AManifestThatLeavesOutWhatItMustSay_IsRefused_NamingWhatItLeftOut(string key)
    {
        var root = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(Sample().ToJson())!;

        root.Remove(key);

        var refused = Assert.Throws<FormatException>(() => LandingManifest.FromJson(JsonSerializer.Serialize(root)));

        Assert.Contains($"'{key}'", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("")]
    [InlineData("abc")]
    [InlineData("AB00000000000000000000000000000000000000000000000000000000000000")]
    [InlineData("zz00000000000000000000000000000000000000000000000000000000000000")]
    public void AFingerprintThatIsNotASha256InLowerCaseHex_IsRefused(string fingerprint)
    {
        var text = Sample().ToJson().Replace(Fingerprint, fingerprint, StringComparison.Ordinal);

        Assert.Throws<FormatException>(() => LandingManifest.FromJson(text));
    }

    [Theory]
    [InlineData("../BTCEUR.csv")]
    [InlineData("sub/BTCEUR.csv")]
    [InlineData("sub\\BTCEUR.csv")]
    [InlineData("")]
    public void AFileThatIsNotAFileNameInTheLandingsOwnFolder_IsRefused(string file)
    {
        var text = Sample().ToJson().Replace("BTCEUR-1d-20240101T000000Z-20240104T000000Z.csv", file.Replace("\\", "\\\\", StringComparison.Ordinal), StringComparison.Ordinal);

        Assert.Throws<FormatException>(() => LandingManifest.FromJson(text));
    }

    [Fact]
    public void NoText_IsNoManifest()
    {
        Assert.Throws<ArgumentNullException>(() => LandingManifest.FromJson(null!));
    }

    [Fact]
    public void APageOfThePagesThatIsNotAnObject_IsRefused_NamingThePages()
    {
        var root = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(Sample().ToJson())!;

        root["pages"] = Parsed("[7]");

        var refused = Assert.Throws<FormatException>(() => LandingManifest.FromJson(JsonSerializer.Serialize(root)));

        Assert.Contains("'pages'", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("")]
    [InlineData("abc")]
    [InlineData("CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC")]
    public void APagesDigestThatIsNotASha256InLowerCaseHex_IsRefused_NamingTheDigest(string digest)
    {
        var text = Sample().ToJson().Replace(new string('c', 64), digest, StringComparison.Ordinal);

        var refused = Assert.Throws<FormatException>(() => LandingManifest.FromJson(text));

        Assert.Contains("sha256", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("askedAt")]
    [InlineData("serverTime")]
    [InlineData("first")]
    [InlineData("last")]
    public void AMomentThatIsNoMoment_IsRefused_NamingWhatItWasMeantToSay(string key)
    {
        var root = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(Sample().ToJson())!;

        root[key] = Parsed("\"yesterday, about lunch\"");

        var refused = Assert.Throws<FormatException>(() => LandingManifest.FromJson(JsonSerializer.Serialize(root)));

        Assert.Contains($"'{key}'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("not a moment", refused.Message, StringComparison.Ordinal);
    }
}
