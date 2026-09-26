// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using DeepSharp.Verso.Api;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using Verso.Serializers;

namespace DeepSharp.Tests.Api;

/// <summary>
/// A notebook is written in its host as in Verso's own editor: a cell is added after another or at the end, of a kind the
/// engine has — code in a language it runs, Markdown, a pipeline block and the rest — and starts empty; a cell is taken
/// away, moved past the neighbour it passes, or turned into another kind in one step. Every such change goes through the
/// port the notebook's layout guards, so a layout that does not let cells be added, taken away, moved or typed into
/// refuses it, and every version says what the layout allows. Each change tells the notebook, so what was worked out
/// from the blocks as they were is taken back. And as a cell's text is typed, its kernel offers what may come next and
/// says what a word means, as Verso's editors ask it.
/// </summary>
public sealed class AuthoringTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-api-authoring-").FullName;

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
        """{"step": "normalise", "column": "fare", "scale": "standard", "outOfRange": "pass"}""",
    ];

    public AuthoringTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private static CellModel Block(string source) => new() { Type = StepCellType.StepType, Language = StepKernel.Language, Source = source };

    private static CellModel CSharp(string source) => new() { Type = "code", Language = "csharp", Source = source };

    private async Task<NotebookHost> OpenAsync(OpenNotebooks notebooks, string name, NotebookModel notebook)
    {
        var path = Path.Join(_folder, name);

        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        return await notebooks.OpenAsync(path, TestContext.Current.CancellationToken);
    }

    private Task<NotebookHost> OpenAsync(OpenNotebooks notebooks, string name, params CellModel[] cells)
    {
        var notebook = new NotebookModel();

        foreach (var cell in cells)
        {
            notebook.Cells.Add(cell);
        }

        return OpenAsync(notebooks, name, notebook);
    }

    private static HostedKind Kind(NotebookHost host, string type) => host.Kinds.Single(kind => kind.Type == type);

    private static string? HandedOver(NotebookHost host) =>
        host.Scaffold.Variables.TryGet<string>(StepKernel.HandOver, out var pipeline) ? pipeline : null;

    // A grid draws a box for every column it shows.
    private static bool ShowsAGrid(HostedCell cell) => cell.Outputs.Any(output => output.Content.Contains("deepsharp.include ", StringComparison.Ordinal));

    private static Guid[] Order(NotebookHost host) => [.. host.Cells.Select(cell => cell.Id)];

    [Fact]
    public async Task TheKindsACellCanBe_AreVersosEditorsList_WithCodeInEveryLanguageButTheBlocksOwn()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "empty.verso");

        Assert.Equal(
            [
                new HostedKind("code", "csharp", "C# (Roslyn)", Editable: true),
                new HostedKind("markdown", null, "Markdown", Editable: true),
                new HostedKind(StepCellType.StepType, StepKernel.Language, "Pipeline step", Editable: true),
                new HostedKind("html", "html", "HTML", Editable: true),
                new HostedKind("mermaid", "mermaid", "Mermaid", Editable: true),

                // The parameters form is drawn from the notebook's parameters, and nobody writes its text.
                new HostedKind("parameters", null, "Parameters", Editable: false),
            ],
            host.Kinds);
    }

    [Fact]
    public async Task EveryKindButCode_TakesTheLanguageTheEnginesOwnInsertGivesItsType()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "empty.verso");

        foreach (var kind in host.Kinds.Where(kind => kind.Type != "code"))
        {
            Assert.Equal(host.Scaffold.InsertCell(0, kind.Type).Language, kind.Language);
        }
    }

    [Fact]
    public async Task ACellInsertedAfterAnother_StandsRightAfterIt_Empty_OfTheKindAsked()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var before = Order(host);

        var inserted = await host.InsertAsync(before[1], Kind(host, "code"));

        Assert.Equal([before[0], before[1], inserted.Id, .. before[2..]], Order(host));
        Assert.Equal(("code", "csharp", ""), (inserted.Type, inserted.Language, inserted.Source));
        Assert.Equal(inserted, host.Cells[2]);
    }

    [Fact]
    public async Task ACellAdded_StandsLast_AndANotebookWithNoCellsTakesOne()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "empty.verso");

        var first = await host.AddAsync(Kind(host, "markdown"));
        var second = await host.AddAsync(Kind(host, StepCellType.StepType));

        Assert.Equal([first.Id, second.Id], Order(host));
        Assert.Equal((StepCellType.StepType, StepKernel.Language, ""), (second.Type, second.Language, second.Source));
    }

    [Fact]
    public async Task ACellTakenAway_IsGone_AndAVerbOnItAfterwardsIsGoneToo()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var normalise = host.Cells[4].Id;

        await host.RemoveAsync(normalise);

        Assert.DoesNotContain(normalise, Order(host));
        await Assert.ThrowsAsync<CellGoneException>(() => host.RemoveAsync(normalise));
        await Assert.ThrowsAsync<CellGoneException>(() => host.InsertAsync(normalise, Kind(host, "code")));
    }

    [Fact]
    public async Task MovingUp_PutsTheCellBeforeTheNeighbourAbove_AndMovingDown_AfterTheNeighbourBelow()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var (read, declare, split, fill, normalise) = (host.Cells[0].Id, host.Cells[1].Id, host.Cells[2].Id, host.Cells[3].Id, host.Cells[4].Id);

        await host.MoveBeforeAsync(fill, split);

        Assert.Equal([read, declare, fill, split, normalise], Order(host));

        await host.MoveAfterAsync(declare, normalise);

        Assert.Equal([read, fill, split, normalise, declare], Order(host));

        // A cell named as its own neighbour passes nothing.
        await host.MoveBeforeAsync(read, read);

        Assert.Equal([read, fill, split, normalise, declare], Order(host));
    }

    [Fact]
    public async Task AMoveNamingANeighbourThatWent_IsGone_AndMovesNothing()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var split = host.Cells[2].Id;
        var fill = host.Cells[3].Id;

        await host.RemoveAsync(split);
        var before = Order(host);

        await Assert.ThrowsAsync<CellGoneException>(() => host.MoveBeforeAsync(fill, split));

        Assert.Equal(before, Order(host));
    }

    [Fact]
    public async Task TurningACellIntoAnotherKind_IsOneStep_KeepingItsIdAndText_AndClearingWhatItShowed()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "printing.verso", CSharp("""Console.WriteLine("printed");"""));
        var printing = host.Cells[0].Id;

        await host.RunAsync(printing);
        var version = host.Current.Version;

        // The kind it is already changes nothing: what it printed stays, and there is no new version.
        var same = await host.ChangeKindAsync(printing, Kind(host, "code"));

        Assert.NotEmpty(same.Outputs);
        Assert.Equal(version, host.Current.Version);

        var turned = await host.ChangeKindAsync(printing, Kind(host, "markdown"));

        Assert.Equal((printing, "markdown", (string?)null, """Console.WriteLine("printed");"""), (turned.Id, turned.Type, turned.Language, turned.Source));
        Assert.Empty(turned.Outputs);
    }

    [Fact]
    public async Task AKindTheNotebookDoesNotList_IsRefused_CodeInTheBlocksOwnLanguageAmongThem()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var withheld = new HostedKind("code", StepKernel.Language, "Pipeline step", Editable: true);

        await Assert.ThrowsAsync<InvalidOperationException>(() => host.AddAsync(withheld));
        await Assert.ThrowsAsync<InvalidOperationException>(() => host.ChangeKindAsync(host.Cells[0].Id, withheld));
        Assert.Equal(5, host.Cells.Count);
    }

    [Fact]
    public async Task ANotebookSavedInTheDashboard_RefusesAddingTakingAwayMovingTypingAndChangingAKind_AndSaysSo()
    {
        var notebook = new NotebookModel { ActiveLayout = new LayoutReference("verso.layout.dashboard", "dashboard") };

        foreach (var block in Titanic)
        {
            notebook.Cells.Add(Block(block));
        }

        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "dashboard.verso", notebook);
        var (read, declare) = (host.Cells[0].Id, host.Cells[1].Id);

        Assert.Equal(new HostedLayout("dashboard", LayoutAllows.CellResize | LayoutAllows.CellExecute), host.Current.Layout);
        await Assert.ThrowsAsync<LayoutCapabilityException>(() => host.InsertAsync(read, Kind(host, "code")));
        await Assert.ThrowsAsync<LayoutCapabilityException>(() => host.AddAsync(Kind(host, "code")));
        await Assert.ThrowsAsync<LayoutCapabilityException>(() => host.RemoveAsync(read));
        await Assert.ThrowsAsync<LayoutCapabilityException>(() => host.MoveAfterAsync(read, declare));
        await Assert.ThrowsAsync<LayoutCapabilityException>(() => host.EditAsync(read, Titanic[0]));
        await Assert.ThrowsAsync<LayoutCapabilityException>(() => host.ChangeKindAsync(read, Kind(host, "code")));
        Assert.Equal(Titanic, host.Cells.Select(cell => cell.Source));
    }

    [Fact]
    public async Task ANotebookInItsOwnLayout_SaysEveryCellMayBeAddedTakenAwayMovedTypedIntoAndRun()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);

        Assert.Equal("notebook", host.Current.Layout.Id);
        Assert.True(host.Current.Layout.Allows.HasFlag(
            LayoutAllows.CellInsert | LayoutAllows.CellDelete | LayoutAllows.CellReorder | LayoutAllows.CellEdit | LayoutAllows.CellExecute));
    }

    [Fact]
    public async Task SwitchingTheLayout_MakesANewVersion_ThatSaysWhatTheNewLayoutAllows()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var before = host.Current.Version;

        // Verso's own button, which goes on to the next layout the engine has: the presentation, after the notebook. A press
        // takes its turn among the process's C# runs, and while another notebook's C# run holds that turn every view is told
        // the press waits — a version of its own — so the versions move on at least once.
        await host.RunToolbarAsync("verso.switchLayout");

        Assert.Equal(new HostedLayout("presentation", LayoutAllows.None), host.Current.Layout);
        Assert.True(host.Current.Version > before);
    }

    [Fact]
    public async Task ANotebookNamingALayoutTheEngineDoesNotHave_SaysNone_AndAllowsEveryCellAction()
    {
        var notebook = new NotebookModel { ActiveLayout = new LayoutReference("elsewhere.layout", "no-such-layout") };

        notebook.Cells.Add(Block(Titanic[0]));

        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "elsewhere.verso", notebook);

        Assert.Null(host.Current.Layout.Id);
        Assert.True(host.Current.Layout.Allows.HasFlag(
            LayoutAllows.CellInsert | LayoutAllows.CellDelete | LayoutAllows.CellReorder | LayoutAllows.CellEdit | LayoutAllows.CellExecute));
        var added = await host.AddAsync(Kind(host, "code"));

        Assert.Equal(added.Id, host.Cells[^1].Id);
    }

    [Fact]
    public async Task ABlockInsertedAboveAShownGrid_ClearsIt_AndTakesBackWhatTheRunHandedToCSharpCells()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);

        await host.RunToolbarAsync(RunPipelineAction.Id);

        Assert.True(ShowsAGrid(host.Cells[4]));
        Assert.Contains("\"fitted\"", HandedOver(host), StringComparison.Ordinal);

        await host.InsertAsync(host.Cells[2].Id, Kind(host, StepCellType.StepType));

        Assert.False(ShowsAGrid(host.Cells[5]));
        Assert.Null(HandedOver(host));
    }

    [Fact]
    public async Task ABlockTakenAway_TakesBackWhatTheRunHandedToCSharpCells()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);

        await host.RunToolbarAsync(RunPipelineAction.Id);
        await host.RemoveAsync(host.Cells[3].Id);

        Assert.Null(HandedOver(host));
        Assert.False(ShowsAGrid(host.Cells[3]));
    }

    [Fact]
    public async Task ABlockMoved_TakesBackWhatTheRunHandedToCSharpCells()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);

        await host.RunToolbarAsync(RunPipelineAction.Id);
        await host.MoveBeforeAsync(host.Cells[4].Id, host.Cells[3].Id);

        Assert.Null(HandedOver(host));
    }

    [Fact]
    public async Task ABlockTurnedIntoAnotherKind_TakesBackWhatTheRunHandedToCSharpCells()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);

        await host.RunToolbarAsync(RunPipelineAction.Id);
        await host.ChangeKindAsync(host.Cells[3].Id, Kind(host, "markdown"));

        Assert.Null(HandedOver(host));
    }

    [Fact]
    public async Task ACSharpCellInserted_LeavesThePipelineHandedOver_SinceTheBlocksStillMakeIt()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);

        await host.RunToolbarAsync(RunPipelineAction.Id);
        var handed = HandedOver(host);

        await host.InsertAsync(host.Cells[4].Id, Kind(host, "code"));

        Assert.Equal(handed, HandedOver(host));
        Assert.True(ShowsAGrid(host.Cells[4]));
    }

    [Fact]
    public async Task AMarkdownNotebookGivenABlock_IsRefusedItsSave_AndItsFileKeepsWhatItHeld()
    {
        var path = Path.Join(_folder, "notes.md");

        await File.WriteAllTextAsync(path, "# Notes\n", TestContext.Current.CancellationToken);
        await using var notebooks = new OpenNotebooks();
        var host = await notebooks.OpenAsync(path, TestContext.Current.CancellationToken);

        var block = await host.AddAsync(Kind(host, StepCellType.StepType));
        await host.EditAsync(block.Id, Titanic[0]);

        var refused = await Assert.ThrowsAsync<InvalidOperationException>(host.SaveAsync);

        Assert.Contains(".verso", refused.Message, StringComparison.Ordinal);
        Assert.Equal("# Notes\n", await File.ReadAllTextAsync(path, TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task ABlocksKernel_OffersTheVerbsAfterTheStepKey_AndSaysWhatAStepDoes()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var fill = host.Cells[3].Id;
        const string typed = """{"step": "fill.""";

        var offered = await host.CompletionsAsync(fill, typed, typed.Length);
        var hover = await host.HoverAsync(fill, Titanic[3], Titanic[3].IndexOf("fill.missing", StringComparison.Ordinal) + 3);

        Assert.Contains(offered, completion => completion.Text == "fill.missing");
        Assert.Equal(StepCatalog.BuiltIn().Describe("fill.missing").Purpose, hover?.Content);
        Assert.Null(hover?.Range);
    }

    [Fact]
    public async Task ACSharpCell_IsOfferedWhatTheCSharpKernelOffers_AndItsHoverSaysWhereTheWordStands()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "code.verso", CSharp(""));
        var code = host.Cells[0].Id;
        const string typed = "System.Console.";
        const string written = "System.Console.WriteLine(1);";

        var offered = await host.CompletionsAsync(code, typed, typed.Length);
        var hover = await host.HoverAsync(code, written, written.IndexOf("WriteLine", StringComparison.Ordinal) + 2);

        Assert.Contains(offered, completion => completion.Text == "WriteLine");
        Assert.Contains("WriteLine", hover?.Content, StringComparison.Ordinal);
        Assert.NotNull(hover?.Range);
    }

    [Fact]
    public async Task ACellWithNoKernelForItsText_IsOfferedNothing_AndAWordInItMeansNothing()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "empty.verso");
        var markdown = await host.AddAsync(Kind(host, "markdown"));
        var html = await host.AddAsync(Kind(host, "html"));

        Assert.Empty(await host.CompletionsAsync(markdown.Id, "# Notes", 3));
        Assert.Null(await host.HoverAsync(markdown.Id, "# Notes", 3));
        Assert.Empty(await host.CompletionsAsync(html.Id, "<p>", 2));
    }

    [Fact]
    public async Task ABlockWithNothingUnderTheCursor_SaysNothing_AndACellThatWentIsGone()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var normalise = host.Cells[4].Id;

        Assert.Null(await host.HoverAsync(normalise, Titanic[4], 0));

        await host.RemoveAsync(normalise);

        await Assert.ThrowsAsync<CellGoneException>(() => host.CompletionsAsync(normalise, Titanic[4], 0));
        await Assert.ThrowsAsync<CellGoneException>(() => host.HoverAsync(normalise, Titanic[4], 0));
    }
}
