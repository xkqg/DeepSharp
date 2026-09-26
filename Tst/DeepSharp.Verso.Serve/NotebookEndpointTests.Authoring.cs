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
    private async Task<NotebookVersion> NotebookAsync(Served served, string name = "titanic.verso") =>
        (await served.Client.GetFromJsonAsync<NotebookVersion>($"/api/notebooks/{name}", TestContext.Current.CancellationToken))!;

    [Fact]
    public async Task ThePage_IsToldTheKindsACellCanBe_AndWhatTheLayoutAllows()
    {
        await using var served = await StartAsync();

        var kinds = await served.Client.GetFromJsonAsync<HostedKind[]>("/api/notebooks/titanic.verso/kinds", TestContext.Current.CancellationToken);
        using var snapshot = JsonDocument.Parse(await served.Client.GetStringAsync("/api/notebooks/titanic.verso", TestContext.Current.CancellationToken));
        var layout = snapshot.RootElement.GetProperty("layout");

        Assert.Contains(new HostedKind(StepCellType.StepType, StepKernel.Language, "Pipeline step"), kinds!);
        Assert.Contains(new HostedKind("code", "csharp", "C# (Roslyn)"), kinds!);
        Assert.Equal("notebook", layout.GetProperty("id").GetString());
        Assert.Equal(255, layout.GetProperty("allows").GetInt32());
    }

    [Fact]
    public async Task ACellAddedAfterAnother_AndAtTheEnd_StandsWhereItWasAskedFor_AndTheViewIsTold()
    {
        await using var served = await StartAsync();
        await using var view = await ViewAsync(served, "titanic.verso");
        var snapshot = Read<NotebookVersion>(await NextAsync(view));
        var split = snapshot.Cells[2].Id;

        var inserted = await (await served.Client.PostAsJsonAsync(
            "/api/notebooks/titanic.verso/cells", new { after = split, type = "code", language = "csharp" }, TestContext.Current.CancellationToken))
            .Content.ReadFromJsonAsync<HostedCell>(TestContext.Current.CancellationToken);
        var added = await (await served.Client.PostAsJsonAsync(
            "/api/notebooks/titanic.verso/cells", new { type = "markdown" }, TestContext.Current.CancellationToken))
            .Content.ReadFromJsonAsync<HostedCell>(TestContext.Current.CancellationToken);

        var order = (await NotebookAsync(served)).Cells.Select(cell => cell.Id).ToArray();

        Assert.Equal(inserted.Id, order[3]);
        Assert.Equal(added.Id, order[^1]);
        Assert.Equal(("code", ""), (inserted.Type, inserted.Source));
        Assert.Contains(inserted.Id, Read<NotebookChange>(await NextAsync(view)).Order!);
    }

    [Fact]
    public async Task ACellIsTakenAway_Moved_AndTurnedIntoAnotherKind_ThroughTheServer()
    {
        await using var served = await StartAsync();
        var cells = (await NotebookAsync(served)).Cells.Select(cell => cell.Id).ToArray();

        Assert.Equal(HttpStatusCode.NoContent, (await served.Client.PostAsJsonAsync(
            $"/api/notebooks/titanic.verso/cells/{cells[3]}/move", new { before = cells[2] }, TestContext.Current.CancellationToken)).StatusCode);
        Assert.Equal(HttpStatusCode.NoContent, (await served.Client.PostAsJsonAsync(
            $"/api/notebooks/titanic.verso/cells/{cells[1]}/move", new { after = cells[4] }, TestContext.Current.CancellationToken)).StatusCode);
        Assert.Equal([cells[0], cells[3], cells[2], cells[4], cells[1]], (await NotebookAsync(served)).Cells.Select(cell => cell.Id));

        Assert.Equal(HttpStatusCode.NoContent, (await served.Client.PostAsync(
            $"/api/notebooks/titanic.verso/cells/{cells[4]}/remove", null, TestContext.Current.CancellationToken)).StatusCode);

        var turned = await (await served.Client.PostAsJsonAsync(
            $"/api/notebooks/titanic.verso/cells/{cells[3]}/kind", new { type = "markdown" }, TestContext.Current.CancellationToken))
            .Content.ReadFromJsonAsync<HostedCell>(TestContext.Current.CancellationToken);

        Assert.Equal((cells[3], "markdown"), (turned.Id, turned.Type));
        Assert.DoesNotContain(cells[4], (await NotebookAsync(served)).Cells.Select(cell => cell.Id));
    }

    [Fact]
    public async Task AMoveNamingNoNeighbour_OrTwo_IsABadRequest_AndMovesNothing()
    {
        await using var served = await StartAsync();
        var cells = (await NotebookAsync(served)).Cells.Select(cell => cell.Id).ToArray();

        Assert.Equal(HttpStatusCode.BadRequest, (await served.Client.PostAsJsonAsync(
            $"/api/notebooks/titanic.verso/cells/{cells[3]}/move", new { }, TestContext.Current.CancellationToken)).StatusCode);
        Assert.Equal(HttpStatusCode.BadRequest, (await served.Client.PostAsJsonAsync(
            $"/api/notebooks/titanic.verso/cells/{cells[3]}/move", new { before = cells[2], after = cells[4] }, TestContext.Current.CancellationToken)).StatusCode);
        Assert.Equal(cells, (await NotebookAsync(served)).Cells.Select(cell => cell.Id));
    }

    [Fact]
    public async Task ACellThatWent_AnswersTheVersionItWentAt_AndAKindTheNotebookLacks_SaysWhy()
    {
        await using var served = await StartAsync();
        var cells = (await NotebookAsync(served)).Cells.Select(cell => cell.Id).ToArray();

        await served.Client.PostAsync($"/api/notebooks/titanic.verso/cells/{cells[4]}/remove", null, TestContext.Current.CancellationToken);

        var gone = await served.Client.PostAsJsonAsync(
            "/api/notebooks/titanic.verso/cells", new { after = cells[4], type = "code", language = "csharp" }, TestContext.Current.CancellationToken);
        var unlisted = await served.Client.PostAsJsonAsync(
            "/api/notebooks/titanic.verso/cells", new { type = "code", language = StepKernel.Language }, TestContext.Current.CancellationToken);

        Assert.Equal(HttpStatusCode.Conflict, gone.StatusCode);
        Assert.Equal((await NotebookAsync(served)).Version, (await gone.Content.ReadFromJsonAsync<JsonElement>(TestContext.Current.CancellationToken)).GetProperty("version").GetInt64());
        Assert.Equal(HttpStatusCode.UnprocessableEntity, unlisted.StatusCode);
    }

    [Fact]
    public async Task ANotebookSavedInTheDashboard_RefusesAddingACell_InTheEnginesOwnWords()
    {
        var notebook = new NotebookModel { ActiveLayout = new LayoutReference("verso.layout.dashboard", "dashboard") };

        notebook.Cells.Add(new CellModel { Type = StepCellType.StepType, Language = StepKernel.Language, Source = Titanic[0] });
        await File.WriteAllTextAsync(At("dashboard.verso"), await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);
        await using var served = await StartAsync();

        var refused = await served.Client.PostAsJsonAsync("/api/notebooks/dashboard.verso/cells", new { type = "markdown" }, TestContext.Current.CancellationToken);

        Assert.Equal(HttpStatusCode.UnprocessableEntity, refused.StatusCode);
        Assert.Contains("CellInsert", (await refused.Content.ReadFromJsonAsync<JsonElement>(TestContext.Current.CancellationToken)).GetProperty("message").GetString(), StringComparison.Ordinal);
    }

    [Fact]
    public async Task ABlocksKernel_OffersTheVerbs_AndSaysWhatAStepDoes_AndNothingUnderTheCursorIsNoContent()
    {
        await using var served = await StartAsync();
        var fill = (await NotebookAsync(served)).Cells[3].Id;
        const string typed = """{"step": "fill.""";

        var offered = await served.Client.PostAsJsonAsync(
            $"/api/notebooks/titanic.verso/cells/{fill}/completions", new { code = typed, position = typed.Length }, TestContext.Current.CancellationToken);
        var hover = await served.Client.PostAsJsonAsync(
            $"/api/notebooks/titanic.verso/cells/{fill}/hover", new { code = Titanic[3], position = Titanic[3].IndexOf("fill.missing", StringComparison.Ordinal) + 3 }, TestContext.Current.CancellationToken);
        var nothing = await served.Client.PostAsJsonAsync(
            $"/api/notebooks/titanic.verso/cells/{fill}/hover", new { code = Titanic[3], position = 0 }, TestContext.Current.CancellationToken);

        Assert.Contains((await offered.Content.ReadFromJsonAsync<HostedCompletion[]>(TestContext.Current.CancellationToken))!, completion => completion.Text == "fill.missing");
        Assert.Equal(StepCatalog.BuiltIn().Describe("fill.missing").Purpose, (await hover.Content.ReadFromJsonAsync<HostedHover>(TestContext.Current.CancellationToken)).Content);
        Assert.Equal(HttpStatusCode.NoContent, nothing.StatusCode);
    }

    [Fact]
    public async Task ANewNotebook_IsMadeInTheFolderServed_AndListedAtOnce()
    {
        await using var served = await StartAsync();

        var made = await served.Client.PostAsJsonAsync("/api/notebooks", new { name = "new.verso" }, TestContext.Current.CancellationToken);

        Assert.Equal(HttpStatusCode.Created, made.StatusCode);
        Assert.Equal("/api/notebooks/new.verso", made.Headers.Location?.OriginalString);
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

        var block = await (await served.Client.PostAsJsonAsync(
            "/api/notebooks/notes.md/cells", new { type = StepCellType.StepType, language = StepKernel.Language }, TestContext.Current.CancellationToken))
            .Content.ReadFromJsonAsync<HostedCell>(TestContext.Current.CancellationToken);

        await served.Client.PostAsJsonAsync($"/api/notebooks/notes.md/cells/{block.Id}/source", new { source = Titanic[0] }, TestContext.Current.CancellationToken);

        var refused = await served.Client.PostAsync("/api/notebooks/notes.md/save", null, TestContext.Current.CancellationToken);

        Assert.Equal(HttpStatusCode.UnprocessableEntity, refused.StatusCode);
        Assert.Contains(".verso", (await refused.Content.ReadFromJsonAsync<JsonElement>(TestContext.Current.CancellationToken)).GetProperty("message").GetString(), StringComparison.Ordinal);
        Assert.Equal("# Notes", await File.ReadAllTextAsync(At("notes.md"), TestContext.Current.CancellationToken));
    }
}
