// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Concurrent;
using DeepSharp.Verso.Api;
using Verso.Abstractions;
using Verso.Extensions;
using Verso.Serializers;

namespace DeepSharp.Tests.Api;

/// <summary>
/// A look at the notebook does nothing to it: a button asked whether it can be pressed, and a panel asked to draw its
/// section, are handed operations that refuse every verb, whoever wrote the part, so what is only looked at never runs a
/// cell or changes one.
/// </summary>
public sealed class LookTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-api-looks-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    // A notebook of one C# cell, opened on an engine that also carries the given part.
    private async Task<NotebookHost> OpenWithAsync(IExtension part)
    {
        var path = Path.Join(_folder, "look.verso");
        var notebook = new NotebookModel { DefaultKernelId = "csharp" };

        notebook.Cells.Add(new CellModel { Type = "code", Language = "csharp", Source = "System.Console.Write(1);" });
        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        var engine = new ExtensionHost();

        await engine.LoadExtensionAsync(part);

        return await NotebookHost.OpenAsync(path, engine, TestContext.Current.CancellationToken);
    }

    [Theory]
    [InlineData("button")]
    [InlineData("panel")]
    public async Task ALookThatRunsACell_IsRefused_AndNoCellBegins(string look)
    {
        var host = await OpenWithAsync(look == "button" ? new LookingButton() : new LookingPanel());
        var begun = new ConcurrentQueue<Guid>();

        host.Scaffold.OnCellExecuting += begun.Enqueue;

        try
        {
            var looking = look == "button" ? (Task)host.ToolbarAsync() : host.PropertiesAsync(host.Cells[0].Id);

            await Assert.ThrowsAsync<InvalidOperationException>(() => looking);
            Assert.Empty(begun);
            Assert.Empty(host.Cells[0].Outputs);
        }
        finally
        {
            await host.CloseAsync();
        }
    }

    [Fact]
    public async Task AReadPort_RefusesEveryVerb_AndAnswersTheLayoutAndTheTheme()
    {
        var host = await OpenWithAsync(new LookingPanel());
        var cell = host.Cells[0].Id;
        var look = new ReadPort(host.Scaffold);
        var begun = new ConcurrentQueue<Guid>();

        host.Scaffold.OnCellExecuting += begun.Enqueue;

        try
        {
            Assert.Equal(host.Scaffold.NotebookOps.ActiveLayoutId, look.ActiveLayoutId);
            Assert.Equal(host.Scaffold.NotebookOps.ActiveThemeId, look.ActiveThemeId);
            Assert.Equal(CancellationToken.None, look.Token);

            Func<Task>[] verbs =
            [
                () => look.ExecuteCellAsync(cell),
                () => look.ExecuteAllAsync(),
                () => look.ExecuteFromAsync(cell),
                () => look.ExecuteCodeAsync("1 + 1", "csharp"),
                () => look.ExecuteCodeCaptureOutputsAsync("1 + 1", "csharp"),
                () => look.ClearOutputAsync(cell),
                () => look.ClearAllOutputsAsync(),
                () => look.RestartKernelAsync("csharp"),
                () => look.InsertCellAsync(0, "code", "csharp"),
                () => look.RemoveCellAsync(cell),
                () => look.MoveCellAsync(cell, 0),
                () => { look.SetActiveLayout("dashboard"); return Task.CompletedTask; },
                () => { look.SetActiveTheme("dark"); return Task.CompletedTask; },
            ];

            foreach (var verb in verbs)
            {
                await Assert.ThrowsAsync<InvalidOperationException>(verb);
            }

            Assert.Empty(begun);
            Assert.Single(host.Cells);
        }
        finally
        {
            await host.CloseAsync();
        }
    }
}
