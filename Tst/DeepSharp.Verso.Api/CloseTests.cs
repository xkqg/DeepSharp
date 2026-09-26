// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Api;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using Verso.Serializers;

namespace DeepSharp.Tests.Api;

/// <summary>
/// When an open notebook closes by itself: once no view has shown it for a while, nothing runs or waits, and nothing in
/// it differs from the file it was last saved to — what a block shows never counts, since it is never saved. A notebook
/// with changes not yet saved stays open until it is saved. An application can also close one notebook at once, and
/// what was asked of a notebook before it closed, and waited behind the close, is refused.
/// </summary>
public sealed partial class CloseTests : IDisposable
{
    private static readonly TimeSpan Grace = TimeSpan.FromMilliseconds(200);

    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-api-close-").FullName;

    private static readonly string[] Titanic =
    [
        """{"step": "read.csv", "path": "titanic.csv"}""",
        """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "pclass", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
        """{"step": "fill.missing", "column": "age", "with": "median"}""",
    ];

    public CloseTests() => File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "titanic.csv"));

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private string At(string name) => Path.Join(_folder, name);

    private static CellModel Block(string source) => new() { Type = StepCellType.StepType, Language = StepKernel.Language, Source = source };

    private static CellModel CSharp(string source) => new() { Type = "code", Language = "csharp", Source = source };

    private async Task<NotebookHost> OpenAsync(OpenNotebooks notebooks, string name, params CellModel[] cells)
    {
        var notebook = new NotebookModel();

        foreach (var cell in cells)
        {
            notebook.Cells.Add(cell);
        }

        await File.WriteAllTextAsync(At(name), await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        return await notebooks.OpenAsync(At(name), TestContext.Current.CancellationToken);
    }

    // Waits for the notebooks to let go of a notebook; one still held after ten seconds fails the test.
    private static async Task ForgottenAsync(OpenNotebooks notebooks, NotebookHost host)
    {
        for (var waited = TimeSpan.Zero; notebooks.Paths.Contains(host.FilePath); waited += Grace)
        {
            Assert.True(waited < TimeSpan.FromSeconds(10), $"{host.FilePath} is still open.");
            await Task.Delay(Grace, TestContext.Current.CancellationToken);
        }

        Assert.Throws<ObjectDisposedException>(host.Subscribe);
    }

    // Long enough for the notebooks to have looked at a notebook more than once.
    private static Task Beyond() => Task.Delay(Grace * 4, TestContext.Current.CancellationToken);

    [Fact]
    public async Task ANotebookNoViewShows_WithNothingUnsaved_IsClosedAfterTheGrace_AndOpenedAnewAfter()
    {
        await using var notebooks = new OpenNotebooks(Grace);
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);

        // What a block shows is never saved, so showing it leaves nothing unsaved.
        await host.RunAsync(host.Cells[0].Id);

        await ForgottenAsync(notebooks, host);

        Assert.NotSame(host, await notebooks.OpenAsync(At("titanic.verso"), TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task ANotebookWhoseRepeatedIdsWereRepaired_IsNotUnsavedForThat()
    {
        var twin = Guid.NewGuid();

        await using var notebooks = new OpenNotebooks(Grace);
        var host = await OpenAsync(notebooks, "twins.verso", new CellModel { Id = twin, Type = "markdown", Source = "# One" }, new CellModel { Id = twin, Type = "markdown", Source = "# Two" });

        await ForgottenAsync(notebooks, host);
    }

    [Fact]
    public async Task AView_KeepsTheNotebookOpen_AndWhenItEnds_TheNotebookCloses()
    {
        await using var notebooks = new OpenNotebooks(Grace);
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var view = host.Subscribe();

        await Beyond();

        Assert.Contains(host.FilePath, notebooks.Paths);

        view.Dispose();

        await ForgottenAsync(notebooks, host);
    }

    [Fact]
    public async Task ANotebookWithChangesNotYetSaved_StaysOpen_UntilItIsSaved()
    {
        await using var notebooks = new OpenNotebooks(Grace);
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);

        await host.EditAsync(host.Cells[2].Id, Titanic[2].Replace("median", "mean", StringComparison.Ordinal));
        await Beyond();

        Assert.Contains(host.FilePath, notebooks.Paths);

        await host.SaveAsync();

        await ForgottenAsync(notebooks, host);
    }

    [Fact]
    public async Task ARunUnderWay_OrSomethingAskedBehindTheLook_KeepsTheNotebookOpen()
    {
        var go = At("go");

        await using var notebooks = new OpenNotebooks(Grace);
        var host = await OpenAsync(notebooks, "wait.verso", CSharp($$"""while (!System.IO.File.Exists(@"{{go}}")) { await System.Threading.Tasks.Task.Delay(10); }"""));
        var running = host.RunAsync(host.Cells[0].Id);

        // The look the notebooks take waits behind the run; something asked meanwhile waits behind the look.
        await Beyond();

        var asked = host.ToolbarAsync();

        Assert.Contains(host.FilePath, notebooks.Paths);

        await File.WriteAllTextAsync(go, "go", TestContext.Current.CancellationToken);
        await running;

        Assert.NotEmpty(await asked);

        await ForgottenAsync(notebooks, host);
    }

    [Fact]
    public async Task ANotebookSavedUnderAnotherName_IsClosedAndForgottenUnderThatName()
    {
        await using var notebooks = new OpenNotebooks(Grace);
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var view = host.Subscribe();

        await notebooks.SaveAsAsync(host, At("renamed.verso"));
        view.Dispose();

        await ForgottenAsync(notebooks, host);

        Assert.Empty(notebooks.Paths);
    }

    [Fact]
    public async Task ClosingOneNotebook_ClosesItAtOnce_UnsavedOrNot_AndLeavesTheOthers()
    {
        await using var notebooks = new OpenNotebooks(Grace);
        var closed = await OpenAsync(notebooks, "a.verso", [.. Titanic.Select(Block)]);
        var kept = await OpenAsync(notebooks, "b.verso", [.. Titanic.Select(Block)]);
        var keeping = kept.Subscribe();

        await closed.EditAsync(closed.Cells[2].Id, Titanic[2].Replace("median", "mean", StringComparison.Ordinal));

        var view = closed.Subscribe();

        await notebooks.CloseAsync(closed);

        await foreach (var change in view.ReadAllAsync(TestContext.Current.CancellationToken))
        {
            Assert.Fail($"A closed notebook told a view of version {change.Version}.");
        }

        Assert.DoesNotContain(closed.FilePath, notebooks.Paths);
        Assert.Contains(kept.FilePath, notebooks.Paths);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => closed.ToolbarAsync());

        // Its watch ends with it, and the other notebook goes on.
        await Beyond();

        Assert.NotEmpty(await kept.ToolbarAsync());
        keeping.Dispose();

        await using var others = new OpenNotebooks();

        await Assert.ThrowsAsync<InvalidOperationException>(() => others.CloseAsync(kept));
    }

    [Fact]
    public async Task ClosingTheNotebooks_EndsTheWatch_AndClosesWhatIsStillViewed()
    {
        var notebooks = new OpenNotebooks(Grace);
        var host = await OpenAsync(notebooks, "titanic.verso", [.. Titanic.Select(Block)]);
        var view = host.Subscribe();

        await notebooks.DisposeAsync();

        await foreach (var change in view.ReadAllAsync(TestContext.Current.CancellationToken))
        {
            Assert.Fail($"A closed notebook told a view of version {change.Version}.");
        }

        Assert.Throws<ObjectDisposedException>(host.Subscribe);

        // A view that ends after its notebook closed ends as any other.
        view.Dispose();
    }

    [Fact]
    public void AGraceOfNothing_IsRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new OpenNotebooks(TimeSpan.Zero));
        Assert.Throws<ArgumentOutOfRangeException>(() => new OpenNotebooks(TimeSpan.FromSeconds(-1)));
    }
}
