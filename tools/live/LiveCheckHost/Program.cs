// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

// What an application that brought only the packages finds. Built twice: with the package that lands a window, where it
// lands thirty days of candles from a stand-in venue on this machine into a folder and writes the pipeline that reads the
// landing; and without it, where that pipeline is read back with the library's own catalog and run over the landing, and
// plainly nothing of the package and nothing of Polly is anywhere in the application.
using DeepSharp.Pipelines;
#if LANDS
using DeepSharp.Tests.Pipelines;
#endif

var what = args.Length > 0 ? args[0] : "reads";
var folder = args.Length > 1 ? args[1] : ".";

// A pipeline written in code reads a relative path from the working directory, and the door lands it there.
Environment.CurrentDirectory = folder;

#if LANDS
if (what == "lands")
{
    using var venue = new FakeBinanceVenue(TimeProvider.System);
    using var standIn = new StandIn(venue);

    var from = new DateTime(2024, 1, 1, 0, 0, 0, DateTimeKind.Utc);
    var candles = new BinanceCandles("BTCEUR", "1d", from, from.AddDays(30)).At(new Uri($"http://127.0.0.1:{standIn.Port}"));

    var pipeline = (await Pdd.Create().ReadBinanceAsync(candles))
        .Declare(schema => schema.Timestamp("timestamp").Number("close").Optional("trades", ColumnKind.Number))
        .OrderBy("timestamp")
        .SplitByTime("timestamp", train: 0.70, validation: 0.15, gap: 1)
        .Ahead("close", 1, AheadAs.Return)
        .Drop("timestamp")
        .Normalise(scale => scale.Columns("close").MaxAbs("trades"));

    File.WriteAllText("pipeline.json", pipeline.Declaration.ToJson());

    var asked = venue.Seen;
    var landed = Directory.GetFiles(".", "BTCEUR-1d-*").Select(Path.GetFileName).Order(StringComparer.Ordinal).ToArray();

    // The control for the second run: the package's own libraries are loaded here, so their absence there says something.
    var carriesPolly = AppDomain.CurrentDomain.GetAssemblies().Any(assembly => assembly.GetName().Name == "Polly.Core");

    Console.WriteLine($"    landed {landed.Length} files, {string.Join(" and ", landed)}, from {asked.Count} requests the stand-in answered");

    if (asked.Count < 3 || landed.Length != 2 || !carriesPolly)
    {
        Console.Error.WriteLine($"::error::the landing asked {asked.Count} times, left {landed.Length} files and {(carriesPolly ? "carried" : "did not carry")} Polly: it was meant to ask at least three times, leave two files and carry it");

        return 1;
    }

    Console.WriteLine("    and wrote the pipeline that reads it, which names a file and nothing of Binance");

    return 0;
}
#endif

if (what == "lands")
{
    Console.Error.WriteLine("::error::this build brought no landing, so it cannot land one");

    return 1;
}

var declaration = PipelineDeclaration.FromJson(File.ReadAllText("pipeline.json"), StepCatalog.BuiltIn());
var prepared = new Pipeline(declaration).Run();
var carries = AppDomain.CurrentDomain.GetAssemblies().Select(assembly => assembly.GetName().Name)
    .Concat(typeof(PipelineDeclaration).Assembly.GetReferencedAssemblies().Select(name => name.Name))
    .Any(name => name is not null && (name.StartsWith("Polly", StringComparison.Ordinal) || name.Contains("Binance", StringComparison.Ordinal)));

Console.WriteLine($"    read the pipeline with the library's own catalog and ran it over the landing: {prepared.Table.RowCount} rows, {prepared.CountIn(Part.Train)} to train on");

if (prepared.Table.RowCount != 30)
{
    Console.Error.WriteLine($"::error::the landing held {prepared.Table.RowCount} rows where thirty days were asked for");

    return 1;
}

if (carries)
{
    Console.Error.WriteLine("::error::this build was supposed to carry nothing of the landing's package or of Polly, and it does");

    return 1;
}

Console.WriteLine("    and this application carries nothing of Binance's package, and nothing of Polly, at all");

return 0;
