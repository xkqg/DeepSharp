// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using Microsoft.Extensions.Time.Testing;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The door a pipeline written in code goes in by: it lands the window if the window is not landed, and then declares an
/// ordinary read of the file. Nothing of Binance is in the pipeline it leaves: the file it saves names a landed file and
/// is the file of every other pipeline, so a project that serves a model brings nothing of HTTP, retries or pacing to read
/// it, and the pipeline reads the landing as it reads any file — by its bytes, every digest of which was fixed when it landed.
/// </summary>
public sealed class BinanceDoorTests : IDisposable
{
    private static readonly DateTimeOffset Now = new(2026, 10, 6, 8, 0, 0, TimeSpan.Zero);

    private static readonly DateTime Jan1 = new(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);

    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-door-").FullName;

    private readonly FakeTimeProvider _clock = new(Now);

    private readonly FakeBinanceVenue _venue;

    public BinanceDoorTests() => _venue = new FakeBinanceVenue(_clock);

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private BinanceCandles Asked(int days) =>
        new BinanceCandles("BTCEUR", "1d", Jan1, Jan1.AddDays(days)).Through(_venue).On(_clock).At(new Uri("https://venue.test"));

    private Task<PipelineBuilder> ThroughTheDoor(int days) =>
        _clock.RunAsync(Pdd.Create().ReadBinanceAsync(Asked(days), SourceFolder.Of(_folder), TestContext.Current.CancellationToken));

    [Fact]
    public async Task TheDoorLandsTheWindow_AndDeclaresAPlainReadCsvOfIt()
    {
        var builder = await ThroughTheDoor(5);
        var step = Assert.IsType<ReadCsvStep>(Assert.Single(builder.Declaration.Steps));

        Assert.Equal("BTCEUR-1d-20240101T000000Z-20240106T000000Z.csv", step.Path);
        Assert.True(File.Exists(Path.Join(_folder, step.Path)));
        Assert.Equal("read.csv", step.Verb);
    }

    [Fact]
    public async Task ASecondRunOfTheDoor_SendsNoRequest_AndDeclaresTheSameStep()
    {
        var first = await ThroughTheDoor(5);
        var asked = _venue.Seen.Count;
        var second = await ThroughTheDoor(5);

        Assert.True(asked > 0);
        Assert.Equal(asked, _venue.Seen.Count);
        Assert.Equal(first.Declaration, second.Declaration);
    }

    [Fact]
    public async Task ThePipelineTheDoorLeaves_IsSavedAsAnyPipelinesFileIs_AtTheVersionItWas_NamingNothingOfBinance()
    {
        var declaration = (await ThroughTheDoor(30))
            .Declare(schema => schema.Timestamp("timestamp").Number("close").Optional("trades", ColumnKind.Number))
            .OrderBy("timestamp")
            .SplitByTime("timestamp", train: 0.70, validation: 0.15, gap: 1)
            .Ahead("close", 1, AheadAs.Return)
            .Drop("timestamp")
            .Normalise(scale => scale.Columns("close").MaxAbs("trades"))
            .Declaration;

        var json = declaration.ToJson();

        Assert.Contains("\"version\": 8", json, StringComparison.Ordinal);
        Assert.Contains("\"step\": \"read.csv\"", json, StringComparison.Ordinal);
        Assert.Contains("\"path\": \"BTCEUR-1d-20240101T000000Z-20240131T000000Z.csv\"", json, StringComparison.Ordinal);
        Assert.DoesNotContain("binance", json, StringComparison.OrdinalIgnoreCase);

        // And it is read back by a catalog that knows nothing but what the library itself brings: a project that only serves a
        // model references none of this package.
        Assert.Equal(declaration, PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));
    }

    [Fact]
    public async Task ThePipelineReadsTheLanding_AsItReadsAnyFile_AndRunsToItsEnd()
    {
        var declaration = (await ThroughTheDoor(30))
            .Declare(schema => schema.Timestamp("timestamp").Number("close").Optional("trades", ColumnKind.Number))
            .OrderBy("timestamp")
            .SplitByTime("timestamp", train: 0.70, validation: 0.15, gap: 1)
            .Ahead("close", 1, AheadAs.Return)
            .Drop("timestamp")
            .Normalise(scale => scale.Columns("close").MaxAbs("trades"))
            .Declaration;

        var prepared = new Pipeline(declaration, rows: null, SourceFolder.Of(_folder)).Run();

        Assert.Equal(30, prepared.Table.RowCount);
        Assert.True(prepared.CountIn(Part.Train) > 0);
        Assert.True(prepared.CountIn(Part.Test) > 0);
    }

    [Fact]
    public async Task TheDoorRefusesNoPipeline_NoWindow_AndNoFolder()
    {
        await Assert.ThrowsAsync<ArgumentNullException>(() => BinanceSourceExtensions.ReadBinanceAsync(null!, Asked(3), TestContext.Current.CancellationToken));
        await Assert.ThrowsAsync<ArgumentNullException>(() => Pdd.Create().ReadBinanceAsync(null!, TestContext.Current.CancellationToken));
        await Assert.ThrowsAsync<ArgumentNullException>(() => Pdd.Create().ReadBinanceAsync(Asked(3), null!, TestContext.Current.CancellationToken));
    }
}

/// <summary>
/// The door a person uses: a window handed to a pipeline written in code, landed in the folder the pipeline's own relative
/// paths are read from, which for a pipeline written in code is the working directory.
/// </summary>
/// <remarks>
/// The working directory is the whole process's, so no other test runs while this one holds it.
/// </remarks>
[Collection(nameof(TheWorkingDirectory))]
public sealed class BinanceDoorInTheWorkingDirectoryTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-door-here-").FullName;

    private readonly string _before = Environment.CurrentDirectory;

    public void Dispose()
    {
        Environment.CurrentDirectory = _before;
        Directory.Delete(_folder, recursive: true);
    }

    [Fact]
    public async Task APipelineWrittenInCode_HasItsWindowLandedWhereItsRelativePathsAreRead()
    {
        var clock = new FakeTimeProvider(new DateTimeOffset(2026, 10, 6, 8, 0, 0, TimeSpan.Zero));
        var venue = new FakeBinanceVenue(clock);
        var asked = new BinanceCandles("BTCEUR", "1d", new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc), new DateTime(2024, 1, 4, 0, 0, 0, DateTimeKind.Utc))
            .Through(venue).On(clock).At(new Uri("https://venue.test"));

        Environment.CurrentDirectory = _folder;

        var builder = await clock.RunAsync(Pdd.Create().ReadBinanceAsync(asked, TestContext.Current.CancellationToken));
        var step = Assert.IsType<ReadCsvStep>(Assert.Single(builder.Declaration.Steps));

        Assert.True(File.Exists(Path.Join(_folder, step.Path)));
        Assert.Equal(["BTCEUR-1d-20240101T000000Z-20240104T000000Z.csv", "BTCEUR-1d-20240101T000000Z-20240104T000000Z.manifest.json"], Directory.GetFiles(_folder).Select(Path.GetFileName).Order(StringComparer.Ordinal));
    }
}

/// <summary>The tests that stand in the process's working directory, which no test beside them may change.</summary>
[CollectionDefinition(nameof(TheWorkingDirectory), DisableParallelization = true)]
public sealed class TheWorkingDirectory;
