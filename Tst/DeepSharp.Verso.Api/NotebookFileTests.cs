// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using DeepSharp.Verso.Api;
using Verso.Abstractions;
using Verso.Extensions;
using Verso.Serializers;

namespace DeepSharp.Tests.Api;

/// <summary>
/// The notebook's file: what it last held, whether the notebook differs from it, and writing it whole before it takes the
/// file's place.
/// </summary>
/// <remarks>
/// The comparison that says what is unsaved reads a cell's outputs while a run may be writing them, and it reads them
/// whole rather than as they come, so it answers rather than failing. That is measured here rather than assumed: a version
/// of Verso that read them as they came would need the answer guarded, and this is where it would be said.
/// </remarks>
public sealed class NotebookFileTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-api-file-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    // The notebook a file holds, read as the notebook is read when it opens: what the comparison is against.
    private static async Task<NotebookModel> SavedAsync(ExtensionHost engine, string path) =>
        await engine.GetSerializers().First(each => each.CanImport(path)).ReadAsync(engine, await File.ReadAllTextAsync(path, TestContext.Current.CancellationToken), path);

    [Fact]
    public async Task TheComparison_AnswersWhileARunWritesWhatACellShows()
    {
        var path = Path.Join(_folder, "shown.verso");
        var notebook = new NotebookModel { DefaultKernelId = "csharp" };

        notebook.Cells.Add(new CellModel { Type = "code", Language = "csharp", Source = "1 + 1" });

        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        var engine = new ExtensionHost();
        var host = await NotebookHost.OpenAsync(path, engine, TestContext.Current.CancellationToken);

        try
        {
            var outputs = host.Scaffold.Notebook.Cells[0].Outputs;
            var shown = new CellOutput("text/plain", "x");

            // Saved with the cell showing it, so the notebook and its file hold the same thing from here on.
            outputs.Add(shown);
            await host.SaveAsync();

            var file = new NotebookFile(path, await SavedAsync(engine, path), engine, host.Scaffold);

            Assert.False(await file.UnsavedAsync(takeBack: false));

            // What a run does to the list while the comparison reads it: the same thing written again, so nothing a save
            // would write changes, and the list's own count of its writings moves under whoever is walking it.
            var writing = true;
            var writer = new Thread(() =>
            {
                while (Volatile.Read(ref writing))
                {
                    outputs[0] = shown;
                }
            });

            writer.Start();

            var answered = 0;
            var clock = Stopwatch.StartNew();

            try
            {
                while (clock.Elapsed < TimeSpan.FromSeconds(3))
                {
                    // The comparison reads the outputs whole rather than as they come, so it answers rather than failing.
                    // A version of Verso that read them as they came would fail here, and this is where that would be said.
                    Assert.False(await file.UnsavedAsync(takeBack: false));
                    answered++;
                }
            }
            finally
            {
                Volatile.Write(ref writing, false);
                writer.Join();
            }

            Assert.True(answered > 1000, $"{answered} answered");
        }
        finally
        {
            await host.CloseAsync();
        }
    }

    [Fact]
    public async Task ANotebookThatDiffersFromItsFile_IsUnsaved_AndIsNotOnceItIsSaved()
    {
        var path = Path.Join(_folder, "typed.verso");
        var notebook = new NotebookModel { DefaultKernelId = "csharp" };

        notebook.Cells.Add(new CellModel { Type = "code", Language = "csharp", Source = "1 + 1" });

        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        var engine = new ExtensionHost();
        var host = await NotebookHost.OpenAsync(path, engine, TestContext.Current.CancellationToken);

        try
        {
            var file = new NotebookFile(path, await SavedAsync(engine, path), engine, host.Scaffold);

            Assert.False(await file.UnsavedAsync(takeBack: true));

            host.Scaffold.Notebook.Cells[0].Source = "2 + 2";

            Assert.True(await file.UnsavedAsync(takeBack: false));

            await file.SaveAsync();

            Assert.False(await file.UnsavedAsync(takeBack: false));
            // Read back the way the notebook is read when it opens, since a file writes its text as JSON writes text.
            Assert.Equal("2 + 2", Assert.Single((await SavedAsync(engine, path)).Cells).Source);
        }
        finally
        {
            await host.CloseAsync();
        }
    }

    [Fact]
    public async Task AFileSavedUnderAnotherName_IsTheNotebooksFileFromThenOn()
    {
        var path = Path.Join(_folder, "first.verso");
        var moved = Path.Join(_folder, "second.verso");
        var notebook = new NotebookModel { DefaultKernelId = "csharp" };

        notebook.Cells.Add(new CellModel { Type = "code", Language = "csharp", Source = "1 + 1" });

        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        var engine = new ExtensionHost();
        var host = await NotebookHost.OpenAsync(path, engine, TestContext.Current.CancellationToken);

        try
        {
            var file = new NotebookFile(path, await SavedAsync(engine, path), engine, host.Scaffold);

            Assert.Equal(path, file.Path);

            await file.SaveAsAsync(moved);

            Assert.Equal(moved, file.Path);
            Assert.True(File.Exists(moved));
            Assert.False(await file.UnsavedAsync(takeBack: false));

            // Nothing is ever left behind beside the file it was written for.
            Assert.Empty(Directory.GetFiles(_folder, ".*.tmp"));
        }
        finally
        {
            await host.CloseAsync();
        }
    }
}
