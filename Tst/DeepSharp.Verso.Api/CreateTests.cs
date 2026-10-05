// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using DeepSharp.Verso.Api;
using DeepSharp.Verso.Notebooks;
using Verso.Abstractions;
using Verso.Serializers;

namespace DeepSharp.Tests.Api;

/// <summary>
/// A notebook can be made where there is none: a .verso file holding one block, the step that reads a CSV file with its path
/// still to say, written whole under a name of its own before it takes its place, so nothing ever meets half a notebook — and never over a file
/// that is there already, which is left as it was.
/// </summary>
public sealed class CreateTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-api-create-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private string At(string name) => Path.Join(_folder, name);

    // The step a new notebook starts with: the first step of a course, the file it reads still to say.
    private const string FirstStepOfACourse = "{\n  \"step\": \"read.csv\",\n  \"path\": null\n}";

    [Fact]
    public async Task ANewNotebook_HoldsOneBlockThatReadsACsvFile_AndIsOpen()
    {
        await using var notebooks = new OpenNotebooks();

        var host = await notebooks.CreateAsync(At("new.verso"), TestContext.Current.CancellationToken);

        var block = Assert.Single(host.Cells);
        Assert.Equal((StepCellType.StepType, StepKernel.Language), (block.Type, block.Language));
        Assert.Equal(FirstStepOfACourse, block.Source);
        Assert.Same(host, await notebooks.OpenAsync(At("new.verso"), TestContext.Current.CancellationToken));

        var saved = await new VersoSerializer().DeserializeAsync(await File.ReadAllTextAsync(At("new.verso"), TestContext.Current.CancellationToken));

        Assert.Equal([block.Source], saved.Cells.Select(cell => cell.Source));
        Assert.Equal("new", saved.Title);
        Assert.Equal([At("new.verso")], Directory.GetFiles(_folder));

        // It names no extension it needs: Verso's browser editor would wait on a consent no page can give while it opens it.
        Assert.Empty(saved.RequiredExtensions);
    }

    [Fact]
    public async Task ANotebookOfBlocksThatNamesNoExtension_SavedHere_NamesNoneStill()
    {
        var notebook = new NotebookModel();

        notebook.Cells.Add(new CellModel { Type = StepCellType.StepType, Language = StepKernel.Language, Source = StepCatalog.BuiltIn().Describe(ReadCsvStep.Name).Template });
        await File.WriteAllTextAsync(At("old.verso"), await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);
        await using var notebooks = new OpenNotebooks();

        var host = await notebooks.OpenAsync(At("old.verso"), TestContext.Current.CancellationToken);

        await host.SaveAsync();

        var saved = await new VersoSerializer().DeserializeAsync(await File.ReadAllTextAsync(At("old.verso"), TestContext.Current.CancellationToken));

        Assert.Empty(saved.RequiredExtensions);
    }

    [Fact]
    public async Task ANewNotebookWhereAFileIs_IsRefused_AndTheFileKeepsWhatItHeld()
    {
        await File.WriteAllTextAsync(At("there.verso"), "what was there", TestContext.Current.CancellationToken);
        await using var notebooks = new OpenNotebooks();

        await Assert.ThrowsAsync<IOException>(() => notebooks.CreateAsync(At("there.verso"), TestContext.Current.CancellationToken));

        Assert.Equal("what was there", await File.ReadAllTextAsync(At("there.verso"), TestContext.Current.CancellationToken));
        Assert.Equal([At("there.verso")], Directory.GetFiles(_folder));
        Assert.Empty(notebooks.Paths);
    }

    [Fact]
    public async Task ANewNotebookInAnotherFormat_IsRefused_AndNothingIsWritten()
    {
        await using var notebooks = new OpenNotebooks();

        await Assert.ThrowsAsync<ArgumentException>(() => notebooks.CreateAsync(At("new.ipynb"), TestContext.Current.CancellationToken));

        Assert.Empty(Directory.GetFiles(_folder));
    }

    [Fact]
    public async Task ANewNotebookWhereAnOpenNotebookIs_IsRefused_EvenWhenItsFileIsGone()
    {
        await using var notebooks = new OpenNotebooks();
        var open = await notebooks.CreateAsync(At("open.verso"), TestContext.Current.CancellationToken);

        File.Delete(At("open.verso"));

        await Assert.ThrowsAsync<InvalidOperationException>(() => notebooks.CreateAsync(At("open.verso"), TestContext.Current.CancellationToken));

        Assert.False(File.Exists(At("open.verso")));
        Assert.Same(open, await notebooks.OpenAsync(At("open.verso"), TestContext.Current.CancellationToken));
    }

    [Fact]
    public async Task NotebooksThatWereClosed_MakeNoNewOne()
    {
        var notebooks = new OpenNotebooks();

        await notebooks.DisposeAsync();

        await Assert.ThrowsAsync<ObjectDisposedException>(() => notebooks.CreateAsync(At("late.verso"), TestContext.Current.CancellationToken));

        Assert.Empty(Directory.GetFiles(_folder));
    }
}
