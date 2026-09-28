// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Api;
using Verso.Abstractions;
using Verso.Extensions;
using Verso.Serializers;

namespace DeepSharp.Tests.Api;

/// <summary>
/// Every version says whether the notebook differs from the file it was last saved to, as Verso's own comparison of two
/// notebooks finds it — what a save would change in the file — so a view marks its Save while something is unsaved, and
/// a save is told as a version like any other change.
/// </summary>
public sealed class UnsavedTests : IDisposable
{
    private static readonly TimeSpan AtOnce = TimeSpan.FromSeconds(10);

    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-api-unsaved-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private string At(string name) => Path.Join(_folder, name);

    private static CellModel CSharp(string source) => new() { Type = "code", Language = "csharp", Source = source };

    private Task<string> WriteAsync(string name, params CellModel[] cells) => WriteAsync(name, new NotebookModel { DefaultKernelId = "csharp" }, cells);

    private async Task<string> WriteAsync(string name, NotebookModel notebook, params CellModel[] cells)
    {
        foreach (var cell in cells)
        {
            notebook.Cells.Add(cell);
        }

        await File.WriteAllTextAsync(At(name), await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        return At(name);
    }

    // Everything a view was told so far, one change taken together.
    private static NotebookChange ToldSoFar(NotebookSubscription view)
    {
        Assert.True(view.TryRead(out var told), "the view was told nothing");

        return told;
    }

    [Fact]
    public async Task AVersion_SaysWhetherTheNotebookDiffersFromItsFile_AndASaveIsToldAsAVersion()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await notebooks.OpenAsync(await WriteAsync("typed.verso", CSharp("1 + 1")), TestContext.Current.CancellationToken);

        Assert.False(host.Current.Unsaved);

        using var view = host.Subscribe();

        await host.EditAsync(host.Cells[0].Id, "2 + 2");

        var typed = ToldSoFar(view);

        Assert.True(typed.Unsaved);
        Assert.True(host.Current.Unsaved);

        await host.SaveAsync();

        // Nothing else changed: the save is told as a version of its own.
        var saved = ToldSoFar(view);

        Assert.False(saved.Unsaved);
        Assert.True(saved.Version > typed.Version);
        Assert.False(host.Current.Unsaved);
    }

    [Fact]
    public async Task ANotebookASaveWouldChange_IsUnsavedFromItsFirstVersion()
    {
        // A part's setting written out at its default, as a file written by hand may hold it: a save leaves it out, as
        // Verso's editors leave out a setting nobody changed.
        var notebook = new NotebookModel { DefaultKernelId = "csharp" };

        notebook.ExtensionSettings[SettingPart.Id] = new Dictionary<string, object?> { ["colour"] = SettingPart.Plain };

        var path = await WriteAsync("defaults.verso", notebook, CSharp("1 + 1"));
        var engine = new ExtensionHost();

        await engine.LoadExtensionAsync(new SettingPart());

        var host = await NotebookHost.OpenAsync(path, engine, TestContext.Current.CancellationToken);

        try
        {
            Assert.Equal(0, host.Current.Version);
            Assert.True(host.Current.Unsaved);

            await host.SaveAsync();

            Assert.False(host.Current.Unsaved);
        }
        finally
        {
            await host.CloseAsync();
        }
    }

    [Fact]
    public async Task ACSharpRunThatShowsSomething_IsToldUnsaved_BeforeItEnds()
    {
        var go = At("go");

        // Shown while the cell runs, as Verso's Display shows it — what a cell prints comes only once it ends.
        await using var notebooks = new OpenNotebooks();
        var host = await notebooks.OpenAsync(
            await WriteAsync("showing.verso", CSharp($$"""Verso.Abstractions.DisplayExtensions.Display("one"); while (!System.IO.File.Exists(@"{{go}}")) { await System.Threading.Tasks.Task.Delay(10); }""")),
            TestContext.Current.CancellationToken);
        using var view = host.Subscribe();
        var running = host.RunAsync(host.Cells[0].Id);

        try
        {
            using var patience = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);

            patience.CancelAfter(TimeSpan.FromSeconds(30));

            // What the cell shows is in the notebook while the run is under way, and a save would write it.
            await foreach (var change in view.ReadAllAsync(patience.Token))
            {
                if (change.Running is not null && change.Cells.Any(cell => cell.Outputs.Any(output => output.Content.Contains("one", StringComparison.Ordinal))))
                {
                    Assert.True(change.Unsaved);

                    break;
                }
            }
        }
        finally
        {
            await File.WriteAllTextAsync(go, "go", TestContext.Current.CancellationToken);
            await running.WaitAsync(AtOnce, TestContext.Current.CancellationToken);
        }

        Assert.True(host.Current.Unsaved);
    }
}
