// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Verso.Api;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using Verso.Extensions;
using Verso.Serializers;

namespace DeepSharp.Tests.Api;

/// <summary>
/// A notebook is opened and saved the way Verso's browser editor opens and saves one: what the engine fills in by itself —
/// the kernel a cell with no language runs in, the layout, the theme — is never written into the notebook; what the file
/// holds for its layouts and for the parts' settings is handed back to them when it opens, and taken from them again
/// before every save and every look at what is unsaved; and every save stamps the notebook's modified date.
/// </summary>
public sealed class LifecycleTests : IDisposable
{
    private const string Dashboard = "verso.layout.dashboard:dashboard";

    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-api-lifecycle-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private static CellModel Block(string source) => new() { Type = StepCellType.StepType, Language = StepKernel.Language, Source = source };

    private async Task<string> WriteAsync(string name, NotebookModel notebook)
    {
        var path = Path.Join(_folder, name);

        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        return path;
    }

    private static async Task<NotebookModel> ReadAsync(string path) =>
        await new VersoSerializer().DeserializeAsync(await File.ReadAllTextAsync(path, TestContext.Current.CancellationToken));

    // A layout's tiles, by cell, as the layout keeps them or as a file holds them.
    private static Dictionary<Guid, Tile> Tiles(object metadata)
    {
        var cells = metadata switch
        {
            Dictionary<string, object> kept => kept["cells"],
            JsonElement file => file.GetProperty("cells"),
            _ => throw new InvalidOperationException($"No tiles in {metadata}"),
        };
        var tiles = new Dictionary<Guid, Tile>();

        if (cells is JsonElement saved)
        {
            foreach (var cell in saved.EnumerateObject())
            {
                tiles[Guid.Parse(cell.Name)] = new Tile(
                    cell.Value.GetProperty("row").GetInt32(),
                    cell.Value.GetProperty("col").GetInt32(),
                    cell.Value.GetProperty("width").GetInt32(),
                    cell.Value.GetProperty("height").GetInt32());
            }

            return tiles;
        }

        foreach (var (id, tile) in (Dictionary<string, object>)cells)
        {
            var at = (Dictionary<string, object>)tile;

            tiles[Guid.Parse(id)] = new Tile(Convert.ToInt32(at["row"]), Convert.ToInt32(at["col"]), Convert.ToInt32(at["width"]), Convert.ToInt32(at["height"]));
        }

        return tiles;
    }

    [Fact]
    public async Task ANotebookNamingNeitherKernelNorLayout_IsShownInItsOwnLayout_AndSavedNamingNeither_WithItsDateStamped()
    {
        var notebook = new NotebookModel { Modified = new DateTimeOffset(2026, 1, 2, 0, 0, 0, TimeSpan.Zero) };

        notebook.Cells.Add(Block("""{"step": "read.csv", "path": "titanic.csv"}"""));

        var path = await WriteAsync("plain.verso", notebook);

        await using var notebooks = new OpenNotebooks();
        var host = await notebooks.OpenAsync(path, TestContext.Current.CancellationToken);

        Assert.Equal("notebook", host.Current.Layout.Id);

        var before = DateTimeOffset.UtcNow.AddSeconds(-1);

        await host.SaveAsync();

        var saved = await ReadAsync(path);

        Assert.Null(saved.DefaultKernelId);
        Assert.Null(saved.ActiveLayout);
        Assert.True(saved.Modified >= before, $"the notebook was saved with the date {saved.Modified}");
    }

    [Fact]
    public async Task ANotebookNamingALayoutTheEngineDoesNotHave_IsShownInTheNotebooksOwn_AndSavedStillNamingIt()
    {
        var notebook = new NotebookModel { ActiveLayout = new LayoutReference("elsewhere.layout", "no-such-layout") };

        notebook.Cells.Add(Block("""{"step": "read.csv", "path": "titanic.csv"}"""));

        var path = await WriteAsync("elsewhere.verso", notebook);

        await using var notebooks = new OpenNotebooks();
        var host = await notebooks.OpenAsync(path, TestContext.Current.CancellationToken);

        Assert.Equal("notebook", host.Current.Layout.Id);

        await host.SaveAsync();

        Assert.Equal("no-such-layout", (await ReadAsync(path)).ActiveLayout?.LayoutId);
    }

    [Fact]
    public async Task TheTilesOfADashboard_AreGivenBackToItWhenTheNotebookOpens_AndSavedAsTheyWere()
    {
        var read = Block("""{"step": "read.csv", "path": "titanic.csv"}""");
        var declare = Block("""{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}]}""");
        var notebook = new NotebookModel { ActiveLayout = new LayoutReference("verso.layout.dashboard", "dashboard") };

        notebook.Cells.Add(read);
        notebook.Cells.Add(declare);
        notebook.Layouts[Dashboard] = new Dictionary<string, object>
        {
            ["cells"] = new Dictionary<string, object>
            {
                [read.Id.ToString()] = new Dictionary<string, object> { ["row"] = 0, ["col"] = 0, ["width"] = 12, ["height"] = 3, ["visible"] = true },
                [declare.Id.ToString()] = new Dictionary<string, object> { ["row"] = 3, ["col"] = 0, ["width"] = 12, ["height"] = 5, ["visible"] = true },
            },
        };

        var path = await WriteAsync("dashboard.verso", notebook);

        await using var notebooks = new OpenNotebooks();
        var host = await notebooks.OpenAsync(path, TestContext.Current.CancellationToken);
        var tiles = Tiles(host.Extensions.GetLayouts().Single(layout => layout.LayoutId == "dashboard").GetLayoutMetadata());

        // As arranged in Verso, before anything draws the notebook.
        Assert.Equal(new Tile(0, 0, 12, 3), tiles[read.Id]);
        Assert.Equal(new Tile(3, 0, 12, 5), tiles[declare.Id]);

        await host.SaveAsync();

        var saved = Tiles((await ReadAsync(path)).Layouts[Dashboard]);

        Assert.Equal(new Tile(0, 0, 12, 3), saved[read.Id]);
        Assert.Equal(new Tile(3, 0, 12, 5), saved[declare.Id]);
    }

    [Fact]
    public async Task APartsSettings_AreGivenBackToItWhenTheNotebookOpens_AndOneChangedIsUnsavedUntilSaved()
    {
        var notebook = new NotebookModel();

        notebook.Cells.Add(Block("""{"step": "read.csv", "path": "titanic.csv"}"""));
        notebook.ExtensionSettings[SettingPart.Id] = new Dictionary<string, object?> { ["colour"] = "loud" };

        var path = await WriteAsync("settings.verso", notebook);
        var part = new SettingPart();
        var engine = new ExtensionHost();

        await engine.LoadExtensionAsync(part);

        var host = await NotebookHost.OpenAsync(path, engine, TestContext.Current.CancellationToken);
        var closed = false;

        try
        {
            Assert.Equal("loud", part.Colour);

            // Changed in the part and not saved: the notebook differs from its file, so no view is needed to keep it open.
            await part.OnSettingChangedAsync("colour", "quiet");

            Assert.True(await host.StaysOpenAsync(TimeSpan.Zero, () => { }));

            await host.SaveAsync();

            Assert.Equal("quiet", (await ReadAsync(path)).ExtensionSettings[SettingPart.Id]["colour"]?.ToString());

            closed = !await host.StaysOpenAsync(TimeSpan.Zero, () => { });

            Assert.True(closed);
        }
        finally
        {
            if (!closed)
            {
                await host.CloseAsync();
            }
        }
    }

    // Where a tile stands and how big it is.
    private readonly record struct Tile(int Row, int Col, int Width, int Height);
}
