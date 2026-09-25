// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Text;
using DeepSharp.Pipelines;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// Seventy bands of a flock are taken in, or made the answer, with two ticks. The list ticks one column at a time, or a
/// range: the first tick of a range says where it starts and changes nothing, the second takes in every column between
/// in one change, each with the one kind picked for the range. Picking the mode or a range's kind changes nothing. A
/// range the output is made of takes in its bands with its kind, and a band taken in as text is made that kind, since a
/// range of answers is asked for as that kind.
/// </summary>
public sealed class RangeTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-ranges-").FullName;

    private static readonly string[] Bands = [.. Enumerable.Range(0, 70).Select(band => $"w{500 + (band * 50):0000}")];

    private const string Schema =
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "farm", "kind": "integer", "optional": false}, {"name": "chicks", "kind": "integer", "optional": false}]}""";

    private const string Split = """{"step": "split.atRandom", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 3}""";

    public RangeTests() => File.WriteAllText(Path.Join(_folder, "flock.csv"), Flock(rows: 4));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    // A flock file: two columns of its own, then seventy bands whose shares sum to one on every row.
    private static string Flock(int rows)
    {
        var text = new StringBuilder($"farm,chicks,{string.Join(',', Bands)}\n");
        var random = new Random(3);

        for (var row = 0; row < rows; row++)
        {
            var shares = Bands.Select(_ => random.NextDouble()).ToArray();
            var sum = shares.Sum();

            text.Append(CultureInfo.InvariantCulture, $"{row % 40},{1000 + (row % 500)}");

            foreach (var share in shares)
            {
                text.Append(',').Append((share / sum).ToString("0.000000", CultureInfo.InvariantCulture));
            }

            text.Append('\n');
        }

        return text.ToString();
    }

    private async Task<Notebook> NotebookAsync(params string[] blocks)
    {
        var notebook = await Notebook.OpenAsync(Path.Join(_folder, "flock.verso"));

        foreach (var block in blocks)
        {
            notebook.AddBlock(block);
        }

        return notebook;
    }

    private Task<Notebook> FlockAsync(params string[] more) =>
        NotebookAsync(["""{"step": "read.csv", "path": "flock.csv"}""", Schema, Split, .. more]);

    private static IReadOnlyList<IPipelineStep> Steps(Notebook notebook) =>
        [.. notebook.Scaffold.Cells.Where(cell => cell.Type == StepCellType.StepType).Select(cell => NotebookVerbs.Catalog().ReadStep(cell.Source))];

    private static DeclareStep Declared(Notebook notebook) => Steps(notebook).OfType<DeclareStep>().Single();

    private static ColumnKind? KindOf(Notebook notebook, string column) => Declared(notebook).Taking.FirstOrDefault(each => each.Name == column)?.Kind;

    private static CellModel SchemaBlock(Notebook notebook) => notebook.Scaffold.Cells[1];

    private static string List(Notebook notebook) =>
        SchemaBlock(notebook).Outputs.Single(output => output.Content.Contains("<tr data-column=", StringComparison.Ordinal)).Content;

    private static async Task ChooseAsync(Notebook notebook) => await notebook.GestureAsync(SchemaBlock(notebook), StepRenderer.Columns);

    // A pick as the router sends it: the select's value, to the list's block.
    private static async Task<bool> PickAsync(Notebook notebook, string gesture, string value, string? of = null) =>
        (await notebook.GestureAsync(SchemaBlock(notebook), List(notebook).SelectOf(gesture, of)!.Value.Action, value)).StateChanged;

    // A row's box, as the list is drawn now.
    private static async Task<bool> TickAsync(Notebook notebook, string column, bool output = false, bool ticked = true)
    {
        var row = List(notebook).Row(column);

        return (await notebook.GestureAsync(SchemaBlock(notebook), (output ? row.Output : row.Included).Action, ticked ? "true" : "false")).StateChanged;
    }

    private async Task<Notebook> InRangeModeAsync(params string[] more)
    {
        var notebook = await FlockAsync(more);

        await ChooseAsync(notebook);
        await PickAsync(notebook, StepRenderer.ListRange, "range");

        return notebook;
    }

    [Fact]
    public async Task TheModeAndARangesKind_ArePicks_ThatChangeNothing()
    {
        await using var notebook = await FlockAsync();

        await ChooseAsync(notebook);

        Assert.Equal(["one", "range"], List(notebook).SelectOf(StepRenderer.ListRange)!.Value.Options);
        Assert.Equal("one", List(notebook).SelectOf(StepRenderer.ListRange)!.Value.Value);
        Assert.Null(List(notebook).SelectOf(StepRenderer.ListRangeKind, "include"));

        Assert.False(await PickAsync(notebook, StepRenderer.ListRange, "range"));

        var kinds = List(notebook).SelectOf(StepRenderer.ListRangeKind, "include")!.Value;

        Assert.Equal("range", List(notebook).SelectOf(StepRenderer.ListRange)!.Value.Value);
        Assert.Equal(["text", "number", "integer", "boolean", "timestamp", "category"], kinds.Options);
        Assert.Equal("text", kinds.Value);
        Assert.False(await PickAsync(notebook, StepRenderer.ListRangeKind, "integer", "include"));
        Assert.Equal("integer", List(notebook).SelectOf(StepRenderer.ListRangeKind, "include")!.Value.Value);
        Assert.Equal(2, Declared(notebook).Taking.Count);
    }

    [Fact]
    public async Task AFirstTickInARange_SaysWhereItStarts_AndChangesNothing_NorDoesItsEcho()
    {
        await using var notebook = await InRangeModeAsync();
        var first = List(notebook).Row(Bands[0]).Included.Action;

        Assert.False(await TickAsync(notebook, Bands[0]));
        Assert.False((await notebook.GestureAsync(SchemaBlock(notebook), first, "true")).StateChanged);
        Assert.Contains("range from here", List(notebook).Row(Bands[0]).Mark, StringComparison.Ordinal);
        Assert.Equal(2, Declared(notebook).Taking.Count);
    }

    [Theory]
    [InlineData(0, 2)]
    [InlineData(2, 0)]
    public async Task ASecondTick_TakesInEveryColumnBetween_WithThePickedKind_InOneChange(int first, int second)
    {
        await using var notebook = await InRangeModeAsync();

        await PickAsync(notebook, StepRenderer.ListRangeKind, "integer", "include");
        await TickAsync(notebook, Bands[first]);

        Assert.True(await TickAsync(notebook, Bands[second]));
        Assert.Equal(["farm", "chicks", .. Bands[..3]], Declared(notebook).Taking.Select(column => column.Name));
        Assert.All(Bands[..3], band => Assert.Equal(ColumnKind.Integer, KindOf(notebook, band)));
        Assert.DoesNotContain("range from here", List(notebook), StringComparison.Ordinal);
        Assert.Equal("range", List(notebook).SelectOf(StepRenderer.ListRange)!.Value.Value);
    }

    [Fact]
    public async Task ARange_LeavesTheColumnsAlreadyInAsTheSchemaHasThem()
    {
        await using var notebook = await InRangeModeAsync();

        await notebook.GestureAsync(SchemaBlock(notebook), List(notebook).Row(Bands[1]).Kind.Action, "number");
        await PickAsync(notebook, StepRenderer.ListRangeKind, "integer", "include");
        await TickAsync(notebook, Bands[0]);
        await TickAsync(notebook, Bands[2]);

        Assert.Equal(ColumnKind.Integer, KindOf(notebook, Bands[0]));
        Assert.Equal(ColumnKind.Number, KindOf(notebook, Bands[1]));
        Assert.Equal(ColumnKind.Integer, KindOf(notebook, Bands[2]));
    }

    [Fact]
    public async Task UntickingInARange_LeavesOneColumnOut()
    {
        await using var notebook = await InRangeModeAsync();

        Assert.True(await TickAsync(notebook, "chicks", ticked: false));
        Assert.Equal(["farm"], Declared(notebook).Taking.Select(column => column.Name));
        Assert.DoesNotContain("range from here", List(notebook), StringComparison.Ordinal);
    }

    [Fact]
    public async Task ASecondTickFromAListDrawnBeforeTheBlocksChanged_DrawsTheListAgain_AndTakesNothingIn()
    {
        await using var notebook = await InRangeModeAsync();

        await TickAsync(notebook, Bands[0]);
        var second = List(notebook).Row(Bands[2]).Included.Action;

        await notebook.GestureAsync(SchemaBlock(notebook), List(notebook).Row("farm").Kind.Action, "number");

        Assert.False((await notebook.GestureAsync(SchemaBlock(notebook), second, "true")).StateChanged);
        Assert.Equal(["farm", "chicks"], Declared(notebook).Taking.Select(column => column.Name));
    }

    [Fact]
    public async Task ARangeStartedOnAColumnTheSourceLacks_IsTheColumnItEndsOnAlone()
    {
        // No list starts a range on a column the source lacks; a box carrying one anyway ends a range of one.
        await using var notebook = await InRangeModeAsync();
        var box = List(notebook).Row(Bands[2]).Included.Action.Replace(
            $"\"{StepRenderer.RangeKey}\":\"true\"", $"\"{StepRenderer.RangeKey}\":\"true\",\"{StepRenderer.IncludeFromKey}\":\"colour\"", StringComparison.Ordinal);

        Assert.True((await notebook.GestureAsync(SchemaBlock(notebook), box, "true")).StateChanged);
        Assert.Equal(["farm", "chicks", Bands[2]], Declared(notebook).Taking.Select(column => column.Name));
    }

    [Fact]
    public async Task AnOutputRangeEndedOnAColumnTheSourceLacks_IsThatColumnAlone_SoItIsRefused_AndItsStartIsKept()
    {
        await using var notebook = await NotebookAsync(
            """{"step": "read.csv", "path": "flock.csv"}""",
            Schema.Replace("{\"name\": \"chicks\"", "{\"name\": \"ghost\", \"kind\": \"number\", \"optional\": true}, {\"name\": \"chicks\"", StringComparison.Ordinal),
            Split);

        await ChooseAsync(notebook);
        await PickAsync(notebook, StepRenderer.ListRange, "range");
        await PickAsync(notebook, StepRenderer.ListType, "target.distribution");
        await TickAsync(notebook, Bands[0], output: true);

        Assert.False(await TickAsync(notebook, "ghost", output: true));
        Assert.Empty(Steps(notebook).OfType<INamesTheAnswer>());
        Assert.Contains(SchemaBlock(notebook).Outputs, output => output.IsError
            && WebUtility.HtmlDecode(output.Content).Contains("names at least two columns", StringComparison.Ordinal));
        Assert.Contains("range from here", List(notebook).Row(Bands[0]).Mark, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AnOutputRangesKind_IsOneTheOutputReads_AndNeverText()
    {
        await using var notebook = await InRangeModeAsync();

        await PickAsync(notebook, StepRenderer.ListType, "target.distribution");

        var kinds = List(notebook).SelectOf(StepRenderer.ListRangeKind, "output")!.Value;

        Assert.Equal(["number", "integer", "boolean"], kinds.Options);
        Assert.Equal("number", kinds.Value);
    }

    [Fact]
    public async Task TwoTicksOnAnOutputRange_MakeADistributionOverEveryBandBetween_TakingThemInAsItsKind_InOneChange()
    {
        await using var notebook = await InRangeModeAsync();

        await PickAsync(notebook, StepRenderer.ListType, "target.distribution");

        Assert.False(await TickAsync(notebook, Bands[0], output: true));
        Assert.Contains("range from here", List(notebook).Row(Bands[0]).Mark, StringComparison.Ordinal);
        Assert.True(await TickAsync(notebook, Bands[^1], output: true));
        Assert.Equal(Bands, ((DistributionStep)Steps(notebook).OfType<INamesTheAnswer>().Single()).Columns);
        Assert.All(Bands, band => Assert.Equal(ColumnKind.Number, KindOf(notebook, band)));
    }

    [Fact]
    public async Task TheKindPickedForAnOutputRange_IsTheKindItsBandsAreTakenInWith_AndPickingItKeepsWhereTheRangeStarts()
    {
        await using var notebook = await InRangeModeAsync();

        await PickAsync(notebook, StepRenderer.ListType, "target.distribution");
        await TickAsync(notebook, Bands[0], output: true);

        Assert.False(await PickAsync(notebook, StepRenderer.ListRangeKind, "integer", "output"));
        Assert.Equal("integer", List(notebook).SelectOf(StepRenderer.ListRangeKind, "output")!.Value.Value);
        Assert.Contains("range from here", List(notebook).Row(Bands[0]).Mark, StringComparison.Ordinal);
        Assert.True(await TickAsync(notebook, Bands[2], output: true));
        Assert.All(Bands[..3], band => Assert.Equal(ColumnKind.Integer, KindOf(notebook, band)));
    }

    [Fact]
    public async Task AnOutputRangeIntoADistributionStanding_PutsEachBandInItsPlace()
    {
        await using var notebook = await NotebookAsync(
            """{"step": "read.csv", "path": "flock.csv"}""",
            Schema.Replace(
                "{\"name\": \"chicks\", \"kind\": \"integer\", \"optional\": false}",
                "{\"name\": \"chicks\", \"kind\": \"integer\", \"optional\": false}, {\"name\": \"w0500\", \"kind\": \"number\", \"optional\": false}, "
                + "{\"name\": \"w3950\", \"kind\": \"number\", \"optional\": false}",
                StringComparison.Ordinal),
            Split,
            """{"step": "target.distribution", "columns": ["w0500", "w3950"]}""");

        await ChooseAsync(notebook);
        await PickAsync(notebook, StepRenderer.ListRange, "range");
        await TickAsync(notebook, Bands[1], output: true);

        Assert.True(await TickAsync(notebook, Bands[2], output: true));

        Assert.Equal([Bands[0], Bands[1], Bands[2], Bands[^1]], ((DistributionStep)Steps(notebook).OfType<INamesTheAnswer>().Single()).Columns);
    }

    [Fact]
    public async Task InARange_AnOutputOfOneColumn_IsTickedAsOne()
    {
        await using var notebook = await InRangeModeAsync();

        Assert.True(await TickAsync(notebook, "chicks", output: true));
        Assert.Equal(new TargetStep("chicks"), Steps(notebook).OfType<INamesTheAnswer>().Single());
    }

    [Fact]
    public async Task BandsTakenInAsText_AreMadeTheKindOfAnOutputRangeOverThem()
    {
        await using var notebook = await InRangeModeAsync();

        await TickAsync(notebook, Bands[0]);
        await TickAsync(notebook, Bands[^1]);

        Assert.All(Bands, band => Assert.Equal(ColumnKind.Text, KindOf(notebook, band)));

        await PickAsync(notebook, StepRenderer.ListType, "target.distribution");
        await TickAsync(notebook, Bands[0], output: true);

        Assert.True(await TickAsync(notebook, Bands[^1], output: true));
        Assert.Equal(Bands, ((DistributionStep)Steps(notebook).OfType<INamesTheAnswer>().Single()).Columns);
        Assert.All(Bands, band => Assert.Equal(ColumnKind.Number, KindOf(notebook, band)));
    }

    [Fact]
    public async Task AnOutputRangeWhileTheBlocksMakeNoPipeline_IsRefused_SayingWhichBlockStopsThem()
    {
        await using var notebook = await InRangeModeAsync();

        await PickAsync(notebook, StepRenderer.ListType, "target.distribution");
        await TickAsync(notebook, Bands[0], output: true);
        var second = List(notebook).Row(Bands[2]).Output.Action;

        notebook.AddBlock("""{"step": "normalise", "column": "colour", "scale": "standard", "outOfRange": "pass"}""");

        Assert.False((await notebook.GestureAsync(SchemaBlock(notebook), second, "true")).StateChanged);
        Assert.Empty(Steps(notebook).OfType<INamesTheAnswer>());
        Assert.Contains(SchemaBlock(notebook).Outputs, output => output.IsError
            && WebUtility.HtmlDecode(output.Content).Contains("the blocks do not make a pipeline yet: block 4", StringComparison.Ordinal));
    }

    [Fact]
    public async Task SeventyBandsTakenInAsIntegers_CanEachBeDividedByTheirRowsSum()
    {
        await using var notebook = await InRangeModeAsync();

        await PickAsync(notebook, StepRenderer.ListRangeKind, "integer", "include");
        await TickAsync(notebook, Bands[0]);
        await TickAsync(notebook, Bands[^1]);

        Assert.Empty(PipelineDeclaration.FaultsIn([.. Steps(notebook), new NormaliseRowStep(Bands)]));
    }

    [Fact]
    public async Task OnAFlockOfTwentyThousandRows_EachRangeIsOneChange_UnderASecond()
    {
        File.WriteAllText(Path.Join(_folder, "flock.csv"), Flock(rows: 20_000));

        await using var notebook = await InRangeModeAsync();

        await PickAsync(notebook, StepRenderer.ListRangeKind, "number", "include");
        await TickAsync(notebook, Bands[0]);
        var clock = Stopwatch.StartNew();

        Assert.True(await TickAsync(notebook, Bands[^1]));

        var included = clock.ElapsedMilliseconds;


        await PickAsync(notebook, StepRenderer.ListType, "target.distribution");
        await TickAsync(notebook, Bands[0], output: true);
        clock.Restart();

        Assert.True(await TickAsync(notebook, Bands[^1], output: true));

        var answered = clock.ElapsedMilliseconds;

        Assert.InRange(included, 0, 999);
        Assert.InRange(answered, 0, 999);
        TestContext.Current.SendDiagnosticMessage($"20 000 x 75: include range of 70 {included} ms, output range of 70 {answered} ms");
    }

    [Fact]
    public async Task SeventySingleIncludes_OnAFlockOfTwentyThousandRows_AreMeasured()
    {
        File.WriteAllText(Path.Join(_folder, "flock.csv"), Flock(rows: 20_000));

        await using var notebook = await FlockAsync();

        await ChooseAsync(notebook);
        var clock = Stopwatch.StartNew();

        foreach (var band in Bands)
        {
            Assert.True(await TickAsync(notebook, band));
        }

        Assert.Equal(72, Declared(notebook).Taking.Count);
        TestContext.Current.SendDiagnosticMessage($"20 000 x 75: 70 single includes {clock.ElapsedMilliseconds} ms");
    }
}
