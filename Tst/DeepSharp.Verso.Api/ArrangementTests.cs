// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Verso.Api;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using Verso.Extensions;
using Verso.Serializers;

namespace DeepSharp.Tests.Api;

/// <summary>
/// A notebook shown in a layout that draws an arrangement of its own — Verso's dashboard, its presentation — carries the
/// arrangement as the engine draws it, a slot in it for each cell the layout shows, with every version it changes in; the
/// notebook's own layout is its list of cells, which the version already says, and a layout that fails to draw holds up no
/// version. What a person does to the arrangement — moving a tile, resizing it, running it — goes to the layout's own part
/// as a change: it runs nothing it is not asked to and waits for no C# run elsewhere, and a tile's run of C# is a run,
/// told and stopped as any run is.
/// </summary>
[Collection(RunsLeftBehind.Name)]
public sealed class ArrangementTests : IDisposable
{
    private static readonly TimeSpan AtOnce = TimeSpan.FromSeconds(10);

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-api-arrangement-").FullName;

    public ArrangementTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose()
    {
        // A cell a stop left behind goes on once it is let go, and writes into this folder as it ends: a test that does not
        // wait for it has it write while the folder is taken away, which then refuses to go.
        var running = Directory.GetFiles(_folder, "*-began")
            .Where(began => !File.Exists(began[..^"began".Length] + "ended"))
            .Select(Path.GetFileName)
            .ToArray();

        Assert.True(running.Length == 0, $"Still running as the test ended: {string.Join(", ", running)}");
        Directory.Delete(_folder, recursive: true);
    }

    private static CellModel CSharp(string source) => new() { Type = "code", Language = "csharp", Source = source };

    private static CellModel Block(string source) => new() { Type = StepCellType.StepType, Language = StepKernel.Language, Source = source };

    private string At(string name) => Path.Join(_folder, name);

    // A cell that says it began, waits until it is let go, and says it ended a moment later, so a test that does not wait
    // for it to end is caught as it ends.
    private string Held(string name) =>
        $$"""System.IO.File.WriteAllText(@"{{At(name + "-began")}}", "on"); while (!System.IO.File.Exists(@"{{At(name + "-go")}}")) { await System.Threading.Tasks.Task.Delay(10); } await System.Threading.Tasks.Task.Delay(500); System.IO.File.WriteAllText(@"{{At(name + "-ended")}}", "on");""";

    // Lets a held cell go and, when it began, waits until it ended.
    private async Task LetGoAsync(string name)
    {
        await File.WriteAllTextAsync(At(name + "-go"), "go", TestContext.Current.CancellationToken);

        if (File.Exists(At(name + "-began")))
        {
            await UntilAsync(() => File.Exists(At(name + "-ended")), $"the cell '{name}' let go never ended");
        }
    }

    private static async Task UntilAsync(Func<bool> holds, string what)
    {
        for (var waited = 0; !holds(); waited += 20)
        {
            Assert.True(waited < 30_000, what);
            await Task.Delay(20, TestContext.Current.CancellationToken);
        }
    }

    // A notebook of the cells, saved shown in a layout of the engine's, or in none.
    private async Task<string> SaveAsync(string name, string? layout, CellModel[] cells)
    {
        var notebook = new NotebookModel { DefaultKernelId = "csharp" };

        if (layout is not null)
        {
            notebook.ActiveLayout = new LayoutReference($"verso.layout.{layout}", layout);
        }

        foreach (var cell in cells)
        {
            notebook.Cells.Add(cell);
        }

        await File.WriteAllTextAsync(At(name), await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        return At(name);
    }

    private async Task<NotebookHost> OpenAsync(OpenNotebooks notebooks, string name, string? layout, params CellModel[] cells) =>
        await notebooks.OpenAsync(await SaveAsync(name, layout, cells), TestContext.Current.CancellationToken);

    // A notebook of the cells, opened on an engine that also carries the given parts.
    private async Task<NotebookHost> OpenWithAsync(IExtension[] parts, params CellModel[] cells)
    {
        var path = await SaveAsync("parts.verso", null, cells);
        var engine = new ExtensionHost();

        foreach (var part in parts)
        {
            await engine.LoadExtensionAsync(part);
        }

        return await NotebookHost.OpenAsync(path, engine, TestContext.Current.CancellationToken);
    }

    private static string Slot(Guid cell) => $"data-cell-slot=\"{cell}\"";

    private static HostedLayoutInteraction Move(Guid tile, int row, int column, int width, int height) =>
        new("dashboard", "updateCellPosition", $$"""{"cellId":"{{tile}}","row":{{row}},"col":{{column}},"width":{{width}},"height":{{height}}}""", tile.ToString());

    private static HostedLayoutInteraction RunOf(Guid tile) => new("dashboard", "run", string.Empty, tile.ToString());

    [Fact]
    public async Task TheDashboard_IsArrangedAsTheEngineDrawsIt_ASlotForEachCellItShows()
    {
        var shown = CSharp("1 + 1");
        var hidden = CSharp("2 + 2");

        // Hidden in the dashboard, as its properties panel sets it for that layout.
        hidden.Metadata["verso:ui.layoutVisibility"] = new Dictionary<string, string> { ["dashboard"] = "Hidden" };

        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "tiles.verso", "dashboard", shown, hidden);
        var arranged = host.Current.Arrangement;

        Assert.Null(arranged.Fault);
        Assert.Contains("verso-dashboard-grid", arranged.Html, StringComparison.Ordinal);
        Assert.Contains(Slot(shown.Id), arranged.Html, StringComparison.Ordinal);
        Assert.DoesNotContain(Slot(hidden.Id), arranged.Html, StringComparison.Ordinal);
        Assert.Contains($"data-action=\"run\" data-target-id=\"{shown.Id}\"", arranged.Html, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ThePresentation_ArrangesOnlyCellsThatShowSomething_WithTheTextOfThoseShownWhole()
    {
        var shows = CSharp("1 + 1");
        var quiet = CSharp("var nothing = 0;");

        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "slides.verso", null, shows, quiet);

        await host.RunAsync(shows.Id);
        await host.SwitchLayoutAsync("presentation");

        var arranged = host.Current.Arrangement.Html;

        Assert.Contains("verso-presentation-view", arranged, StringComparison.Ordinal);
        Assert.Contains(Slot(shows.Id), arranged, StringComparison.Ordinal);
        Assert.Contains("<pre>1 + 1</pre>", arranged, StringComparison.Ordinal);
        Assert.DoesNotContain(Slot(quiet.Id), arranged, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TheNotebooksOwnLayout_CarriesNoArrangement_WhichItsCellsAlreadySay()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "own.verso", null, CSharp("1 + 1"));

        Assert.Equal(default, host.Current.Arrangement);

        await host.SwitchLayoutAsync("dashboard");
        Assert.NotNull(host.Current.Arrangement.Html);

        await host.SwitchLayoutAsync("notebook");
        Assert.Equal(default, host.Current.Arrangement);
    }

    [Fact]
    public async Task AnArrangementIsCarried_OnlyWithAVersionItChangesIn()
    {
        var tile = CSharp("1 + 1");

        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "tiles.verso", "dashboard", tile);
        using var view = host.Subscribe();

        // A run changes what the tile shows, and nothing of the arrangement.
        await host.RunAsync(tile.Id);

        var ran = new List<NotebookChange>();

        while (view.TryRead(out var change))
        {
            ran.Add(change);
        }

        Assert.NotEmpty(ran);
        Assert.All(ran, change => Assert.Null(change.Arrangement));

        await host.InteractAsync(Move(tile.Id, 2, 3, 4, 5));

        Assert.True(view.TryRead(out var moved));
        Assert.Contains("grid-column:4/span 4;grid-row:3/span 5", moved.Arrangement?.Html, StringComparison.Ordinal);
    }

    [Fact]
    public async Task MovingATile_RunsNothing_IsArrangedAnew_AndIsSavedWithTheNotebook()
    {
        var tile = CSharp(Held("tile"));

        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "tiles.verso", "dashboard", tile);

        // Saved with its tile where the dashboard first placed it, nothing is unsaved until the tile moves.
        await host.SaveAsync();
        Assert.False(host.Current.Unsaved);

        await host.InteractAsync(Move(tile.Id, 5, 6, 3, 2));

        Assert.Contains("grid-column:7/span 3;grid-row:6/span 2", host.Current.Arrangement.Html, StringComparison.Ordinal);
        Assert.False(File.Exists(At("tile-began")));
        Assert.Null(host.Running);
        Assert.Empty(host.Executing);
        Assert.True(host.Current.Unsaved);

        await host.SaveAsync();

        var saved = await new VersoSerializer().DeserializeAsync(await File.ReadAllTextAsync(At("tiles.verso"), TestContext.Current.CancellationToken));

        // Read back as the dashboard reads its tiles when the notebook opens.
        var tiles = (Dictionary<string, object>)((Dictionary<string, object>)saved.Layouts["verso.layout.dashboard:dashboard"])["cells"];
        var place = (Dictionary<string, object>)tiles[tile.Id.ToString()];

        Assert.Equal(5, Convert.ToInt32(place["row"], CultureInfo.InvariantCulture));
        Assert.Equal(6, Convert.ToInt32(place["col"], CultureInfo.InvariantCulture));
        Assert.Equal(3, Convert.ToInt32(place["width"], CultureInfo.InvariantCulture));
        Assert.Equal(2, Convert.ToInt32(place["height"], CultureInfo.InvariantCulture));
    }

    [Fact]
    public async Task DraggingATile_WaitsForNoCSharpRun()
    {
        // Notebook A's C# cell takes the process's C# turn until it is let go.
        await using var notebooks = new OpenNotebooks();
        var a = await OpenAsync(notebooks, "a.verso", null, CSharp(Held("a")));
        var holding = a.RunAsync(a.Cells[0].Id);

        try
        {
            await UntilAsync(() => File.Exists(At("a-began")), "notebook A's run never began");

            var tile = CSharp("1 + 1");
            var b = await OpenAsync(notebooks, "b.verso", "dashboard", tile);

            // The drag runs nothing, so it waits for no C# run: it is made while A's still runs.
            await b.InteractAsync(Move(tile.Id, 1, 1, 6, 4)).WaitAsync(AtOnce, TestContext.Current.CancellationToken);

            Assert.Contains("grid-column:2/span 6;grid-row:2/span 4", b.Current.Arrangement.Html, StringComparison.Ordinal);
            Assert.False(File.Exists(At("a-ended")));
        }
        finally
        {
            await LetGoAsync("a");
            await holding;
        }
    }

    [Fact]
    public async Task ATilesRun_IsToldAndStoppable()
    {
        var tile = CSharp(Held("b"));

        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "tiles.verso", "dashboard", tile);
        using var view = host.Subscribe();

        try
        {
            var acting = host.InteractAsync(RunOf(tile.Id));

            await UntilAsync(() => File.Exists(At("b-began")), "the tile's cell never began");

            // The tile's C# cell is a run: told to every view with its cell, and stopped as any run is.
            var run = host.Running!.Value;
            var told = new List<NotebookChange>();

            while (view.TryRead(out var change))
            {
                told.Add(change);
            }

            Assert.Equal(tile.Id, run.Cell);
            Assert.Contains(told, change => change.Running?.Number == run.Number);
            Assert.True(host.Stop(run.Number));
            Assert.Null(await acting.WaitAsync(AtOnce, TestContext.Current.CancellationToken));
            Assert.Null(host.Running);
        }
        finally
        {
            await LetGoAsync("b");
        }
    }

    [Fact]
    public async Task StoppingATileThenRunningIt_StopsTheSecondRun()
    {
        // Each run of the tile counts itself, says it began, and waits to be let go by its own count.
        var tile = CSharp($$"""
            var count = System.IO.File.Exists(@"{{At("count")}}") ? int.Parse(System.IO.File.ReadAllText(@"{{At("count")}}"), System.Globalization.CultureInfo.InvariantCulture) + 1 : 1;
            System.IO.File.WriteAllText(@"{{At("count")}}", count.ToString(System.Globalization.CultureInfo.InvariantCulture));
            System.IO.File.WriteAllText(@"{{At("began-")}}" + count, "on");
            while (!System.IO.File.Exists(@"{{At("go-")}}" + count)) { await System.Threading.Tasks.Task.Delay(10); }
            System.IO.File.WriteAllText(@"{{At("ended-")}}" + count, "on");
            """);

        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "tiles.verso", "dashboard", tile);

        try
        {
            var first = host.InteractAsync(RunOf(tile.Id));

            await UntilAsync(() => File.Exists(At("began-1")), "the tile's first run never began");
            Assert.True(host.Stop(host.Running!.Value.Number));
            await first.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

            var second = host.InteractAsync(RunOf(tile.Id));

            await UntilAsync(() => File.Exists(At("began-2")), "the tile's second run never began");

            var run = host.Running!.Value;

            // The first run, left behind, ends while the second runs: its end is its own, and the second's stop stops it.
            await File.WriteAllTextAsync(At("go-1"), "go", TestContext.Current.CancellationToken);
            await UntilAsync(() => File.Exists(At("ended-1")), "the first run never ended");

            Assert.Equal(run, host.Running);
            Assert.True(host.Stop(run.Number));
            await second.WaitAsync(AtOnce, TestContext.Current.CancellationToken);
            Assert.Null(host.Running);
            Assert.False(File.Exists(At("ended-2")));
        }
        finally
        {
            await File.WriteAllTextAsync(At("go-2"), "go", TestContext.Current.CancellationToken);

            if (File.Exists(At("began-2")))
            {
                await UntilAsync(() => File.Exists(At("ended-2")), "the tile's second run never ended");
            }
        }
    }

    [Fact]
    public async Task AnActOnALayoutNotShown_IsRefused_AndNothingChanges()
    {
        var tile = CSharp(Held("tile"));

        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "tiles.verso", "dashboard", tile);
        var before = host.Current;

        try
        {
            await Assert.ThrowsAsync<InvalidOperationException>(
                () => host.InteractAsync(new HostedLayoutInteraction("presentation", "run", string.Empty, tile.Id.ToString())).WaitAsync(AtOnce, TestContext.Current.CancellationToken));

            Assert.Equal(before, host.Current);
            Assert.False(File.Exists(At("tile-began")));
        }
        finally
        {
            await LetGoAsync("tile");
        }
    }

    [Fact]
    public async Task TheHostsOwnActs_AndActsNoPartTakes_ChangeNothing()
    {
        var tile = CSharp("1 + 1");

        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "tiles.verso", "dashboard", tile);
        var before = host.Current;

        // Every version carries the arrangement as it is drawn, so asking for it again has nothing left to do.
        Assert.Null(await host.InteractAsync(new HostedLayoutInteraction("dashboard", "verso/requestRender", string.Empty, null)));
        Assert.Null(await host.InteractAsync(new HostedLayoutInteraction("dashboard", "verso/whatever-comes-next", string.Empty, null)));

        // An act the dashboard's part does not know is its to ignore.
        Assert.Null(await host.InteractAsync(new HostedLayoutInteraction("dashboard", "no-such-act", string.Empty, null)));
        Assert.Equal(before, host.Current);

        // The presentation has no part to take an act at all.
        await host.SwitchLayoutAsync("presentation");

        var shown = host.Current;

        Assert.Null(await host.InteractAsync(new HostedLayoutInteraction("presentation", "run", string.Empty, tile.Id.ToString())));
        Assert.Equal(shown, host.Current);
    }

    [Fact]
    public async Task ALayoutThatFailsToDraw_HoldsUpNoVersion_AndSaysWhy()
    {
        var host = await OpenWithAsync([new ArrangingLayout("failing", _ => throw new InvalidOperationException("It could not draw."))], CSharp("1 + 1"));

        try
        {
            await host.SwitchLayoutAsync("failing");

            Assert.Equal(new HostedArrangement(null, "It could not draw."), host.Current.Arrangement);

            var before = host.Current.Version;

            await host.EditAsync(host.Cells[0].Id, "2 + 2");
            Assert.True(host.Current.Version > before);
            Assert.Equal("2 + 2", host.Cells[0].Source);
        }
        finally
        {
            await host.CloseAsync();
        }
    }

    [Fact]
    public async Task ALayoutThatDrawsNoHtml_OrNoArrangementOfItsOwn_OrOnlyInAFrameOfItsOwn_CarriesNone()
    {
        var host = await OpenWithAsync(
            [
                new ArrangingLayout("plain", _ => new RenderResult("text/plain", "a list")),
                new ArrangingLayout("listed", _ => new RenderResult("text/html", "<div></div>"), custom: false),
                new ArrangingLayout("framed", _ => new RenderResult("text/html", "<div></div>"), isolation: LayoutRendererIsolation.Isolated),
                new ArrangingLayout("drawn", cells => new RenderResult("text/html", string.Concat(cells.Select(cell => $"<div data-cell-slot=\"{cell.Id}\"></div>")))),
            ],
            CSharp("1 + 1"));

        try
        {
            foreach (var none in new[] { "plain", "listed", "framed" })
            {
                await host.SwitchLayoutAsync(none);
                Assert.Equal(default, host.Current.Arrangement);
            }

            await host.SwitchLayoutAsync("drawn");
            Assert.Equal(new HostedArrangement($"<div data-cell-slot=\"{host.Cells[0].Id}\"></div>", null), host.Current.Arrangement);
        }
        finally
        {
            await host.CloseAsync();
        }
    }

    [Fact]
    public async Task AnActGoesToTheLayoutsOwnPart_AsThePersonMadeIt_AndAFileItHandsOverGoesToWhoeverActed()
    {
        LayoutInteractionContext? seen = null;
        var host = await OpenWithAsync(
            [
                new ArrangingLayout(
                    "acting",
                    _ => new RenderResult("text/html", "<div></div>"),
                    async context =>
                    {
                        seen = context;
                        await context.Verso.RequestFileDownloadAsync("tiles.csv", "text/csv", [1, 2, 3]);
                    }),
            ],
            CSharp("1 + 1"));

        try
        {
            await host.SwitchLayoutAsync("acting");

            // An act in the host's own space never reaches the part.
            Assert.Null(await host.InteractAsync(new HostedLayoutInteraction("acting", "verso/requestRender", string.Empty, null)));
            Assert.Null(seen);

            var file = await host.InteractAsync(new HostedLayoutInteraction("acting", "export", "the payload", "the target"));

            Assert.Equal("tiles.csv", file?.Name);
            Assert.Equal([1, 2, 3], file?.Bytes);
            Assert.NotNull(seen);
            Assert.Equal("deepsharp.tests.layout.acting", seen.ExtensionId);
            Assert.Equal("acting", seen.LayoutId);
            Assert.Equal("export", seen.InteractionType);
            Assert.Equal("the payload", seen.Payload);
            Assert.Equal("the target", seen.TargetId);
        }
        finally
        {
            await host.CloseAsync();
        }
    }

    [Fact]
    public async Task AnActThatTakesABlockAway_TakesBackWhatTheRunHandedToCSharpCells()
    {
        var host = await OpenWithAsync(
            [
                new ArrangingLayout(
                    "acting",
                    _ => new RenderResult("text/html", "<div></div>"),
                    context => context.Verso.Notebook.RemoveCellAsync(Guid.Parse(context.TargetId!))),
            ],
            [.. Titanic.Select(Block)]);

        try
        {
            await host.RunToolbarAsync(RunPipelineAction.Id);
            Assert.True(host.Scaffold.Variables.TryGet<string>(StepKernel.HandOver, out _));

            await host.SwitchLayoutAsync("acting");
            await host.InteractAsync(new HostedLayoutInteraction("acting", "take-away", string.Empty, host.Cells[3].Id.ToString()));

            Assert.Equal(4, host.Cells.Count);
            Assert.False(host.Scaffold.Variables.TryGet<string>(StepKernel.HandOver, out _));
        }
        finally
        {
            await host.CloseAsync();
        }
    }
}
