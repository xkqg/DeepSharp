// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Api;
using Verso.Abstractions;
using Verso.Serializers;

namespace DeepSharp.Tests.Api;

/// <summary>
/// What the notebook says of itself, as Verso's Metadata panel shows it — its title, the kernel a cell that names none
/// runs in, when it was made and last saved, and the version of its format — comes with every version it changes in. Its
/// title is changed as Verso's editors change it, a change every view is told and unsaved until saved, and it names what
/// the notebook is exported as.
/// </summary>
public sealed class MetadataTests : IDisposable
{
    private static readonly DateTimeOffset Made = new(2020, 1, 2, 3, 4, 0, TimeSpan.Zero);

    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-api-metadata-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private async Task<string> SaveAsync(string name, NotebookModel notebook)
    {
        var path = Path.Join(_folder, name);

        notebook.Cells.Add(new CellModel { Type = "code", Language = "csharp", Source = "1 + 1" });
        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        return path;
    }

    [Fact]
    public async Task EveryVersion_SaysTheNotebooksTitle_DefaultKernel_Dates_AndFormat()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await notebooks.OpenAsync(
            await SaveAsync("said.verso", new NotebookModel { Title = "Passengers", DefaultKernelId = "csharp", Created = Made, Modified = Made.AddDays(1) }),
            TestContext.Current.CancellationToken);

        Assert.Equal(new HostedMetadata("Passengers", "csharp", Made, Made.AddDays(1), "1.1"), host.Current.Metadata);
    }

    [Fact]
    public async Task ATitle_IsAChangeEveryViewIsTold_UnsavedUntilSaved_AndItNamesTheExport()
    {
        var path = await SaveAsync("titled.verso", new NotebookModel { DefaultKernelId = "csharp" });

        await using var notebooks = new OpenNotebooks();
        var host = await notebooks.OpenAsync(path, TestContext.Current.CancellationToken);
        using var view = host.Subscribe();

        Assert.Null(host.Current.Metadata.Title);

        await host.RetitleAsync("Passengers");

        Assert.True(view.TryRead(out var titled));
        Assert.Equal("Passengers", titled.Metadata?.Title);
        Assert.True(host.Current.Unsaved);

        // As Verso's own export names it: by the title.
        Assert.Equal("Passengers.html", (await host.RunToolbarAsync("verso.action.export-html"))?.Name);

        // The same title again changes nothing.
        var before = host.Current.Version;

        await host.RetitleAsync("Passengers");
        Assert.Equal(before, host.Current.Version);

        await host.SaveAsync();

        var saved = await new VersoSerializer().DeserializeAsync(await File.ReadAllTextAsync(path, TestContext.Current.CancellationToken));

        Assert.Equal("Passengers", saved.Title);
        Assert.False(host.Current.Unsaved);
    }

    [Fact]
    public async Task ASave_TellsWhenTheNotebookWasLastSaved()
    {
        await using var notebooks = new OpenNotebooks();
        var host = await notebooks.OpenAsync(await SaveAsync("stamped.verso", new NotebookModel { DefaultKernelId = "csharp", Modified = Made }), TestContext.Current.CancellationToken);
        using var view = host.Subscribe();

        await host.SaveAsync();

        var told = default(NotebookChange);

        while (view.TryRead(out var change))
        {
            told = change;
        }

        Assert.True(told.Metadata?.Modified > Made);
        Assert.Equal(host.Current.Metadata.Modified, told.Metadata?.Modified);
    }
}
