// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using DeepSharp.Pipelines;
using DeepSharp.Verso.Api;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using Verso.Serializers;

namespace DeepSharp.Tests.Serve;

// Writing a notebook through the server, as a page does: the kinds a cell can be, a cell added, taken away, moved or
// turned into another kind, what a cell's kernel offers as its text is typed, and a new notebook in the folder served.
public sealed partial class NotebookEndpointTests
{
    [Fact]
    public async Task ThePage_IsToldTheKindsACellCanBe_AndWhatTheLayoutAllows()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);

        var snapshot = await socket.SnapshotAsync();

        Assert.Contains(new HostedKind(StepCellType.StepType, StepKernel.Language, "Pipeline step", Editable: true, Rendered: false), snapshot.Kinds);
        Assert.Contains(new HostedKind("code", "csharp", "C# (Roslyn)", Editable: true, Rendered: false), snapshot.Kinds);
        Assert.Contains(new HostedKind("markdown", null, "Markdown", Editable: true, Rendered: true), snapshot.Kinds);
        Assert.Contains(new HostedKind("parameters", null, "Parameters", Editable: false, Rendered: false), snapshot.Kinds);
        Assert.Equal(new HostedLayout("notebook", (LayoutAllows)255, HasPropertiesPanel: true), snapshot.Version.Layout);
    }

    [Fact]
    public async Task ACellAddedAfterAnother_AndAtTheEnd_StandsWhereItWasAskedFor_AndThePageIsTold()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);
        var split = (await socket.SnapshotAsync()).Version.Cells[2].Id;

        var inserted = (await socket.AskAsync("add", new { after = split, type = "code", language = "csharp" })).Result<HostedCell>();
        var added = (await socket.AskAsync("add", new { type = "markdown" })).Result<HostedCell>();

        var order = (await NotebookAsync(served)).Cells.Select(cell => cell.Id).ToArray();

        Assert.Equal(inserted.Id, order[3]);
        Assert.Equal(added.Id, order[^1]);
        Assert.Equal("code", inserted.Type);
        Assert.Equal(string.Empty, inserted.Source);
        Assert.Contains(inserted.Id, (await socket.ChangeAsync()).Order!);
    }

    [Fact]
    public async Task ACellIsTakenAway_Moved_AndTurnedIntoAnotherKind_ThroughTheSocket()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);
        var cells = (await socket.SnapshotAsync()).Version.Cells.Select(cell => cell.Id).ToArray();

        Assert.True((await socket.AskAsync("move", new { cell = cells[3], before = cells[2] })).IsNothing);
        Assert.True((await socket.AskAsync("move", new { cell = cells[1], after = cells[4] })).IsNothing);
        Assert.Equal([cells[0], cells[3], cells[2], cells[4], cells[1]], (await NotebookAsync(served)).Cells.Select(cell => cell.Id));

        Assert.True((await socket.AskAsync("remove", new { cell = cells[4] })).IsNothing);

        var turned = (await socket.AskAsync("kind", new { cell = cells[3], type = "markdown" })).Result<HostedCell>();

        Assert.Equal(cells[3], turned.Id);
        Assert.Equal("markdown", turned.Type);
        Assert.DoesNotContain(cells[4], (await NotebookAsync(served)).Cells.Select(cell => cell.Id));
    }

    [Fact]
    public async Task AMoveNamingNoNeighbour_OrTwo_IsABadRequest_AndMovesNothing()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);
        var cells = (await socket.SnapshotAsync()).Version.Cells.Select(cell => cell.Id).ToArray();

        Assert.Equal(400, (await socket.AskAsync("move", new { cell = cells[3] })).Status);
        Assert.Equal(400, (await socket.AskAsync("move", new { cell = cells[3], before = cells[2], after = cells[4] })).Status);
        Assert.Equal(cells, (await NotebookAsync(served)).Cells.Select(cell => cell.Id));
    }

    [Fact]
    public async Task ACellThatWent_IsRefusedNamingTheVersionItWentAt_AndAKindTheNotebookLacks_SaysWhy()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);
        var cells = (await socket.SnapshotAsync()).Version.Cells.Select(cell => cell.Id).ToArray();

        await socket.AskAsync("remove", new { cell = cells[4] });

        var gone = await socket.AskAsync("add", new { after = cells[4], type = "code", language = "csharp" });
        var unlisted = await socket.AskAsync("add", new { type = "code", language = StepKernel.Language });

        Assert.Equal(409, gone.Status);
        Assert.Equal((await NotebookAsync(served)).Version, gone.Frame.GetProperty("version").GetInt64());
        Assert.Equal(422, unlisted.Status);
    }

    [Fact]
    public async Task ANotebookSavedInTheDashboard_RefusesAddingACell_InTheEnginesOwnWords()
    {
        var notebook = new NotebookModel { ActiveLayout = new LayoutReference("verso.layout.dashboard", "dashboard") };

        notebook.Cells.Add(new CellModel { Type = StepCellType.StepType, Language = StepKernel.Language, Source = Titanic[0] });
        await File.WriteAllTextAsync(At("dashboard.verso"), await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served, "dashboard.verso");

        await socket.SnapshotAsync();

        var refused = await socket.AskAsync("add", new { type = "markdown" });

        Assert.Equal(422, refused.Status);
        Assert.Contains("CellInsert", refused.Why, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ABlocksKernel_OffersTheVerbs_AndSaysWhatAStepDoes_AndNothingUnderTheCursorIsNothing()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served);
        var fill = (await socket.SnapshotAsync()).Version.Cells[3].Id;
        const string typed = """{"step": "fill.""";

        var offered = (await socket.AskAsync("completions", new { cell = fill, code = typed, position = typed.Length })).Result<HostedCompletion[]>();
        var hover = (await socket.AskAsync("hover", new { cell = fill, code = Titanic[3], position = Titanic[3].IndexOf("fill.missing", StringComparison.Ordinal) + 3 })).Result<HostedHover>();
        var nothing = await socket.AskAsync("hover", new { cell = fill, code = Titanic[3], position = 0 });

        Assert.Contains(offered, completion => completion.Text == "fill.missing");
        Assert.Equal(StepCatalog.BuiltIn().Describe("fill.missing").Purpose, hover.Content);
        Assert.True(nothing.IsNothing);
    }

    [Fact]
    public async Task ANewNotebook_IsMadeInTheFolderServed_AndListedAtOnce()
    {
        await using var served = await StartAsync();

        var made = await served.Client.PostAsJsonAsync("/api/notebooks", new { name = "new.verso" }, TestContext.Current.CancellationToken);

        Assert.Equal(HttpStatusCode.Created, made.StatusCode);
        Assert.Equal("/?notebook=new.verso", made.Headers.Location?.OriginalString);
        Assert.Contains("new.verso", (await served.Client.GetFromJsonAsync<string[]>("/api/notebooks", TestContext.Current.CancellationToken))!);
        Assert.Equal(StepCellType.StepType, Assert.Single((await NotebookAsync(served, "new.verso")).Cells).Type);
    }

    [Theory]
    [InlineData("..\\up.verso")]
    [InlineData("../up.verso")]
    [InlineData("sub/new.verso")]
    [InlineData("sub\\new.verso")]
    [InlineData("new.ipynb")]
    [InlineData(".hidden.verso")]
    [InlineData("")]
    public async Task ANewNotebookNamedOtherThanABareVersoFileName_IsABadRequest_AndNothingIsWritten(string name)
    {
        // A folder served from inside this test's own, so a name that climbed out of it would land where the test looks.
        var inside = Directory.CreateDirectory(At("served")).FullName;
        var around = Directory.GetFileSystemEntries(_folder).Order().ToArray();
        await using var served = await StartAsync(inside);

        var refused = await served.Client.PostAsJsonAsync("/api/notebooks", new { name }, TestContext.Current.CancellationToken);

        Assert.Equal(HttpStatusCode.BadRequest, refused.StatusCode);
        Assert.Empty(Directory.GetFileSystemEntries(inside));
        Assert.Equal(around, Directory.GetFileSystemEntries(_folder).Order());
    }

    [Fact]
    public async Task ANewNotebookNamedAfterOneThatIsThere_IsAConflict_AndTheOneThereIsLeftAsItWas()
    {
        var there = await File.ReadAllTextAsync(At("titanic.verso"), TestContext.Current.CancellationToken);
        await using var served = await StartAsync();

        var refused = await served.Client.PostAsJsonAsync("/api/notebooks", new { name = "titanic.verso" }, TestContext.Current.CancellationToken);

        Assert.Equal(HttpStatusCode.Conflict, refused.StatusCode);
        Assert.Equal(there, await File.ReadAllTextAsync(At("titanic.verso"), TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task ThePage_IsToldWhetherTheServerServesAFolder_WhereANotebookCanBeMade()
    {
        await using (var folder = await StartAsync())
        {
            Assert.True((await folder.Client.GetFromJsonAsync<JsonElement>("/api/served", TestContext.Current.CancellationToken)).GetProperty("folder").GetBoolean());
        }

        await using var one = await StartAsync(At("titanic.verso"));

        Assert.False((await one.Client.GetFromJsonAsync<JsonElement>("/api/served", TestContext.Current.CancellationToken)).GetProperty("folder").GetBoolean());
    }

    [Fact]
    public async Task AServerServingOneNotebook_MakesNoNewOne()
    {
        await using var served = await StartAsync(At("titanic.verso"));

        var refused = await served.Client.PostAsJsonAsync("/api/notebooks", new { name = "new.verso" }, TestContext.Current.CancellationToken);

        Assert.Equal(HttpStatusCode.NotFound, refused.StatusCode);
        Assert.False(File.Exists(At("new.verso")));
    }

    [Fact]
    public async Task AServedMarkdownNotebookGivenABlock_IsRefusedItsSave_AndItsFileKeepsWhatItHeld()
    {
        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served, "notes.md");

        await socket.SnapshotAsync();

        var block = (await socket.AskAsync("add", new { type = StepCellType.StepType, language = StepKernel.Language })).Result<HostedCell>();

        await socket.AskAsync("edit", new { cell = block.Id, source = Titanic[0] });

        var refused = await socket.AskAsync("save");

        Assert.Equal(422, refused.Status);
        Assert.Contains(".verso", refused.Why, StringComparison.Ordinal);
        Assert.Equal("# Notes", await File.ReadAllTextAsync(At("notes.md"), TestContext.Current.CancellationToken));
    }
}
