// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using System.Text;
using System.Text.Json;
using DeepSharp.Pipelines;
using Microsoft.Extensions.Time.Testing;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A window landed as a file, once: the venue is asked once, the candles are written whole beside a record of what was
/// asked and what came back, and from then on the file is the window and the venue is never asked again. What stands in the
/// folder decides what happens next, by a table: nothing — land; both — verify the bytes against the manifest and reuse them;
/// a record without its file, a file that is not the record's, a record of a newer version — refuse, touching nothing; a file
/// without its record — land again, and write the record only if the venue says the same. A landing is never replaced: the
/// move of the file is the commit, and the record is written after it.
/// </summary>
public sealed class BinanceLandingTests : IDisposable
{
    private static readonly DateTimeOffset Now = new(2026, 10, 6, 8, 0, 0, TimeSpan.Zero);

    private static readonly DateTime Jan1 = new(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);

    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-landing-").FullName;

    private readonly FakeTimeProvider _clock = new(Now);

    private readonly FakeBinanceVenue _venue;

    public BinanceLandingTests() => _venue = new FakeBinanceVenue(_clock);

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private BinanceCandles Asked(DateTime from, DateTime to, FakeBinanceVenue? venue = null) =>
        new BinanceCandles("BTCEUR", "1d", from, to).Through(venue ?? _venue).On(_clock).At(new Uri("https://venue.test"));

    private Task<string> LandAsync(BinanceCandles asked) =>
        _clock.RunAsync(asked.LandAsync(SourceFolder.Of(_folder), TestContext.Current.CancellationToken));

    private string At(string name) => Path.Join(_folder, name);

    private string[] Standing() => [.. Directory.GetFileSystemEntries(_folder).Select(entry => Path.GetFileName(entry)).Order(StringComparer.Ordinal)];

    private LandingManifest ManifestOf(BinanceCandles asked) => LandingManifest.FromJson(File.ReadAllText(At(asked.ManifestName)));

    [Fact]
    public async Task AWindowThatIsNotThereYet_IsLanded_AsAFileAndTheRecordBesideIt()
    {
        var asked = Asked(Jan1, Jan1.AddDays(5));
        var landed = await LandAsync(asked);

        Assert.Equal("BTCEUR-1d-20240101T000000Z-20240106T000000Z.csv", landed);
        Assert.Equal([asked.LandingName, asked.ManifestName], Standing());

        var bytes = File.ReadAllBytes(At(landed));
        var manifest = ManifestOf(asked);

        Assert.Equal(6, bytes.AsText().TrimEnd('\n').Split('\n').Length);
        Assert.Equal(bytes.Fingerprint(), manifest.Fingerprint);
        Assert.Equal(landed, manifest.File);
        Assert.Equal(5, manifest.Rows);
        Assert.Equal(0, manifest.Gaps);
        Assert.Equal(Jan1, manifest.First);
        Assert.Equal(Jan1.AddDays(4), manifest.Last);
        Assert.Equal("https://venue.test", manifest.Host);
    }

    [Fact]
    public async Task TheLandingIsTheCandlesTheVenueAnswered_WrittenAsTheFormatSays()
    {
        var asked = Asked(Jan1, Jan1.AddDays(3));

        await LandAsync(asked);

        var walked = await _clock.RunAsync(Asked(Jan1, Jan1.AddDays(3)).WalkAsync(TestContext.Current.CancellationToken));

        Assert.Equal(walked.Candles.AsCsv(), File.ReadAllBytes(At(asked.LandingName)));
    }

    [Fact]
    public async Task TheRecordSaysWhatWasAskedWhenAndWhatCameBack()
    {
        var asked = Asked(Jan1, Jan1.AddDays(5));

        await LandAsync(asked);

        var manifest = ManifestOf(asked);

        Assert.Equal("BTCEUR", manifest.Symbol);
        Assert.Equal("1d", manifest.Interval);
        Assert.Equal(Jan1, manifest.From);
        Assert.Equal(Jan1.AddDays(5), manifest.To);
        Assert.Equal(Now.UtcDateTime, manifest.AskedAt);
        Assert.Equal(60, manifest.SettleAfterCloseSeconds);
        Assert.Equal(1, manifest.PagesEstimated);
        Assert.Single(manifest.Pages);
        Assert.Equal(5, manifest.Pages[0].Rows);
        Assert.Matches("^[0-9a-f]{64}$", manifest.Pages[0].Sha256);
    }

    [Fact]
    public async Task ACandleTheVenueLacked_IsCountedInTheRecord_AndNotInTheFile()
    {
        var asked = Asked(Jan1, Jan1.AddDays(5));

        _venue.HoleAt(Jan1.AddDays(2));

        await LandAsync(asked);

        var manifest = ManifestOf(asked);

        Assert.Equal(4, manifest.Rows);
        Assert.Equal(1, manifest.Gaps);
        Assert.DoesNotContain("2024-01-03T00:00:00Z", File.ReadAllText(At(asked.LandingName)), StringComparison.Ordinal);
    }

    [Fact]
    public async Task AWindowLandedAlready_IsReused_AndTheVenueIsNotAskedAgain()
    {
        var asked = Asked(Jan1, Jan1.AddDays(5));

        await LandAsync(asked);

        var asks = _venue.Seen.Count;
        var csv = File.GetLastWriteTimeUtc(At(asked.LandingName));
        var again = await LandAsync(Asked(Jan1, Jan1.AddDays(5)));

        Assert.Equal(asked.LandingName, again);
        Assert.Equal(asks, _venue.Seen.Count);
        Assert.Equal(csv, File.GetLastWriteTimeUtc(At(asked.LandingName)));
        Assert.Equal([asked.LandingName, asked.ManifestName], Standing());
    }

    [Fact]
    public async Task AWindowLandedAlready_IsReusedWhateverTheVenueWouldNowAnswer()
    {
        var asked = Asked(Jan1, Jan1.AddDays(5));

        await LandAsync(asked);

        var before = File.ReadAllBytes(At(asked.LandingName));

        _venue.Seed = 99;

        await LandAsync(Asked(Jan1, Jan1.AddDays(5)));

        Assert.Equal(before, File.ReadAllBytes(At(asked.LandingName)));
    }

    [Fact]
    public async Task ALandingWhoseBytesWereEdited_IsRefused_AndLeftAsItIs()
    {
        var asked = Asked(Jan1, Jan1.AddDays(5));

        await LandAsync(asked);

        var edited = File.ReadAllBytes(At(asked.LandingName));

        edited[^3] ^= 1;
        File.WriteAllBytes(At(asked.LandingName), edited);

        var asks = _venue.Seen.Count;
        var refused = await Assert.ThrowsAsync<InvalidDataException>(() => LandAsync(Asked(Jan1, Jan1.AddDays(5))));

        Assert.Contains("other bytes", refused.Message, StringComparison.Ordinal);
        Assert.Contains(ManifestOf(asked).Fingerprint, refused.Message, StringComparison.Ordinal);
        Assert.Equal(edited, File.ReadAllBytes(At(asked.LandingName)));
        Assert.Equal(asks, _venue.Seen.Count);
    }

    [Fact]
    public async Task ARecordWithoutItsFile_IsRefused_AndNothingIsFetchedToReplaceIt()
    {
        var asked = Asked(Jan1, Jan1.AddDays(5));

        await LandAsync(asked);
        File.Delete(At(asked.LandingName));

        var asks = _venue.Seen.Count;
        var refused = await Assert.ThrowsAsync<InvalidDataException>(() => LandAsync(Asked(Jan1, Jan1.AddDays(5))));

        Assert.Contains("is not there", refused.Message, StringComparison.Ordinal);
        Assert.Equal([asked.ManifestName], Standing());
        Assert.Equal(asks, _venue.Seen.Count);
    }

    [Fact]
    public async Task AFileWithoutItsRecord_LandsAgain_AndTheRecordIsWrittenWhenTheVenueSaysTheSame()
    {
        var asked = Asked(Jan1, Jan1.AddDays(5));

        await LandAsync(asked);

        var before = File.ReadAllBytes(At(asked.LandingName));

        File.Delete(At(asked.ManifestName));

        var asks = _venue.Seen.Count;

        await LandAsync(Asked(Jan1, Jan1.AddDays(5)));

        Assert.True(_venue.Seen.Count > asks, "The venue was not asked again.");
        Assert.Equal(before, File.ReadAllBytes(At(asked.LandingName)));
        Assert.Equal(before.Fingerprint(), ManifestOf(asked).Fingerprint);
        Assert.Equal([asked.LandingName, asked.ManifestName], Standing());
    }

    [Fact]
    public async Task AFileWithoutItsRecord_IsRefused_WhenTheVenueNowAnswersOtherNumbers_AndTheFirstStays()
    {
        var asked = Asked(Jan1, Jan1.AddDays(5));

        await LandAsync(asked);

        var first = File.ReadAllBytes(At(asked.LandingName));

        File.Delete(At(asked.ManifestName));
        _venue.Seed = 99;

        var refused = await Assert.ThrowsAsync<InvalidDataException>(() => LandAsync(Asked(Jan1, Jan1.AddDays(5))));

        Assert.Contains("first stays", refused.Message, StringComparison.Ordinal);
        Assert.Equal(first, File.ReadAllBytes(At(asked.LandingName)));
        Assert.Equal([asked.LandingName], Standing());
    }

    [Fact]
    public async Task ARecordOfANewerVersion_IsRefused_NamingTheVersion()
    {
        var asked = Asked(Jan1, Jan1.AddDays(5));

        await LandAsync(asked);
        File.WriteAllText(At(asked.ManifestName), File.ReadAllText(At(asked.ManifestName)).Replace("\"version\": 1", "\"version\": 2", StringComparison.Ordinal));

        var refused = await Assert.ThrowsAsync<InvalidDataException>(() => LandAsync(Asked(Jan1, Jan1.AddDays(5))));

        Assert.Contains("version 2", refused.Message, StringComparison.Ordinal);
        Assert.Contains(asked.ManifestName, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ARecordThatCannotBeRead_IsRefused_AndLeftAsItIs()
    {
        var asked = Asked(Jan1, Jan1.AddDays(5));

        await LandAsync(asked);
        File.WriteAllText(At(asked.ManifestName), "not a manifest");

        var refused = await Assert.ThrowsAsync<InvalidDataException>(() => LandAsync(Asked(Jan1, Jan1.AddDays(5))));

        Assert.Contains("cannot be read", refused.Message, StringComparison.Ordinal);
        Assert.Equal("not a manifest", File.ReadAllText(At(asked.ManifestName)));
    }

    [Fact]
    public async Task ARecordThatNamesAnotherWindow_IsRefused()
    {
        var asked = Asked(Jan1, Jan1.AddDays(5));

        await LandAsync(asked);
        File.WriteAllText(At(asked.ManifestName), File.ReadAllText(At(asked.ManifestName)).Replace("\"symbol\": \"BTCEUR\"", "\"symbol\": \"ETHEUR\"", StringComparison.Ordinal));

        var refused = await Assert.ThrowsAsync<InvalidDataException>(() => LandAsync(Asked(Jan1, Jan1.AddDays(5))));

        Assert.Contains("another window", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TheFileIsTheCommit_ARecordThatCannotBeWritten_LeavesTheFileLanded_AndTheNextLandingRepairsIt()
    {
        var asked = Asked(Jan1, Jan1.AddDays(5));

        // A folder where the record goes refuses it on every system: the process is stopped between the file and its record.
        Directory.CreateDirectory(At(asked.ManifestName));

        var failure = await Record.ExceptionAsync(() => LandAsync(asked));

        Assert.True(failure is IOException or UnauthorizedAccessException, $"{failure}");
        Assert.True(File.Exists(At(asked.LandingName)));
        Assert.Equal([asked.LandingName, asked.ManifestName], Standing());

        Directory.Delete(At(asked.ManifestName));

        await LandAsync(Asked(Jan1, Jan1.AddDays(5)));

        Assert.Equal(File.ReadAllBytes(At(asked.LandingName)).Fingerprint(), ManifestOf(asked).Fingerprint);
    }

    [Fact]
    public async Task ALandingThatFails_LeavesNothingInTheFolder()
    {
        var asked = Asked(Jan1, Jan1.AddDays(5));

        _venue.Fail((HttpStatusCode)418, retryAfter: TimeSpan.FromMinutes(5), path: "/api/v3/klines");

        await Assert.ThrowsAsync<BinanceException>(() => LandAsync(asked));

        Assert.Empty(Standing());
    }

    [Fact]
    public async Task ALandingThatIsCancelled_LeavesNothingInTheFolder()
    {
        var asked = Asked(Jan1, Jan1.AddDays(5));
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);

        await stop.CancelAsync();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => _clock.RunAsync(asked.LandAsync(SourceFolder.Of(_folder), stop.Token)));

        Assert.Empty(Standing());
        Assert.Empty(_venue.Seen);
    }

    [Fact]
    public async Task AFolderThatIsNotThere_IsRefused_BeforeAnythingIsAskedOfTheVenue()
    {
        var asked = Asked(Jan1, Jan1.AddDays(5));
        var nowhere = SourceFolder.Of(At("nowhere"));

        await Assert.ThrowsAsync<DirectoryNotFoundException>(() => _clock.RunAsync(asked.LandAsync(nowhere, TestContext.Current.CancellationToken)));

        Assert.Empty(_venue.Seen);
    }

    [Fact]
    public async Task TwoLandingsOfOneWindowAtOnce_AreOneLanding_BothGettingItsName()
    {
        var one = LandAsync(Asked(Jan1, Jan1.AddDays(30)));
        var other = LandAsync(Asked(Jan1, Jan1.AddDays(30)));

        var names = await Task.WhenAll(one, other);

        Assert.Equal(names[0], names[1]);
        Assert.Equal(["BTCEUR-1d-20240101T000000Z-20240131T000000Z.csv", "BTCEUR-1d-20240101T000000Z-20240131T000000Z.manifest.json"], Standing());
        Assert.Equal(File.ReadAllBytes(At(names[0])).Fingerprint(), LandingManifest.FromJson(File.ReadAllText(At("BTCEUR-1d-20240101T000000Z-20240131T000000Z.manifest.json"))).Fingerprint);
    }

    [Fact]
    public async Task TwoLandingsOfOneWindowThatTheVenueAnswersDifferently_LetTheFirstStand_AndRefuseTheSecond()
    {
        var first = new FakeBinanceVenue(_clock) { Seed = 1 };
        var second = new FakeBinanceVenue(_clock) { Seed = 2 };

        async Task<Exception?> Settled(Task<string> landing)
        {
            try
            {
                await landing;

                return null;
            }
            catch (InvalidDataException refused)
            {
                return refused;
            }
        }

        var outcomes = await Task.WhenAll(
            Settled(LandAsync(Asked(Jan1, Jan1.AddDays(30), first))),
            Settled(LandAsync(Asked(Jan1, Jan1.AddDays(30), second))));

        Assert.Single(outcomes, outcome => outcome is null);
        Assert.Single(outcomes, outcome => outcome is InvalidDataException);

        var name = "BTCEUR-1d-20240101T000000Z-20240131T000000Z";
        var landed = File.ReadAllBytes(At($"{name}.csv"));

        Assert.Equal(landed.Fingerprint(), LandingManifest.FromJson(File.ReadAllText(At($"{name}.manifest.json"))).Fingerprint);
        Assert.Equal([$"{name}.csv", $"{name}.manifest.json"], Standing());
    }

    [Fact]
    public async Task TheRecordIsWrittenWithTheLineFeedsTheFormatSays_AndTheFileWithoutAMark()
    {
        var asked = Asked(Jan1, Jan1.AddDays(2));

        await LandAsync(asked);

        var csv = File.ReadAllBytes(At(asked.LandingName));
        var manifest = File.ReadAllBytes(At(asked.ManifestName));

        Assert.False(csv is [0xEF, 0xBB, 0xBF, ..]);
        Assert.False(manifest is [0xEF, 0xBB, 0xBF, ..]);
        Assert.DoesNotContain((byte)'\r', csv);
        Assert.DoesNotContain((byte)'\r', manifest);
        Assert.True(JsonDocument.Parse(Encoding.UTF8.GetString(manifest)).RootElement.TryGetProperty("fingerprint", out _));
    }

    [Fact]
    public async Task TheLandingReadsBack_WhereThePipelineWillFindIt_ByAPlainReadCsv()
    {
        var asked = Asked(Jan1, Jan1.AddDays(3));
        var landed = await LandAsync(asked);

        var rows = new ReadCsvStep(landed).Open(SourceFolder.Of(_folder));

        Assert.Equal(CandleCsv.Header.Split(','), rows.ColumnNames);
        Assert.Equal(["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"], rows.Rows.Select(row => row[0]));
    }

    [Fact]
    public async Task ARecordThatAppearsDuringTheWalk_AndCannotBeRead_IsNotTakenForThisLandings_TheFileStaysLanded_TheRecordStaysAsItIs()
    {
        var asked = Asked(Jan1, Jan1.AddDays(5));
        using var planting = new Planting(_venue, At(asked.ManifestName), "not a record at all\n");

        var failure = await Record.ExceptionAsync(() => LandAsync(asked.Through(planting)));

        Assert.IsAssignableFrom<IOException>(failure);
        Assert.True(File.Exists(At(asked.LandingName)));
        Assert.Equal("not a record at all\n", File.ReadAllText(At(asked.ManifestName)));
        Assert.Equal([asked.LandingName, asked.ManifestName], Standing());

        var refused = await Assert.ThrowsAsync<InvalidDataException>(() => LandAsync(Asked(Jan1, Jan1.AddDays(5))));

        Assert.Contains("cannot be read", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AFileThatOtherBytesTookTheNameOfDuringTheWalk_IsRefused_AndTheFirstStays()
    {
        var asked = Asked(Jan1, Jan1.AddDays(5));
        using var planting = new Planting(_venue, At(asked.LandingName), "somebody else's landing\n");

        var refused = await Assert.ThrowsAsync<InvalidDataException>(() => LandAsync(asked.Through(planting)));

        Assert.IsAssignableFrom<IOException>(refused.InnerException);
        Assert.Contains("other bytes", refused.Message, StringComparison.Ordinal);
        Assert.Equal("somebody else's landing\n", File.ReadAllText(At(asked.LandingName)));
        Assert.Equal([asked.LandingName], Standing());
    }

    // A venue that, as the walk asks it for its first page, has somebody else put a file where the landing is about to go.
    private sealed class Planting(HttpMessageHandler inner, string path, string text) : DelegatingHandler(inner)
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellation)
        {
            if (request.RequestUri!.AbsolutePath == "/api/v3/klines" && !File.Exists(path))
            {
                File.WriteAllText(path, text);
            }

            return base.SendAsync(request, cancellation);
        }
    }
}
