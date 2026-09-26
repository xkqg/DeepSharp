// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Net;
using Verso.Abstractions;
using Verso.Serializers;

namespace DeepSharp.Tests.Serve;

// No close waits for a run. A notebook whose run never ends closes when a person closes it, and the server stops as soon
// as it is told to, with that run's request still waiting for its answer, rather than waiting out the time it gives a
// request to finish.
public sealed partial class NotebookEndpointTests
{
    // A notebook of one C# cell that says it began, and never ends; the cell's id.
    private async Task<Guid> EndlessAsync(string name, string started)
    {
        var cell = new CellModel
        {
            Type = "code",
            Language = "csharp",
            Source = $$"""System.IO.File.WriteAllText(@"{{started}}", "on"); while (true) { await System.Threading.Tasks.Task.Delay(10); }""",
        };
        var notebook = new NotebookModel();

        notebook.Cells.Add(cell);
        await File.WriteAllTextAsync(At(name), await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        return cell.Id;
    }

    private static async Task BeganAsync(string started)
    {
        for (var waited = 0; !File.Exists(started); waited += 20)
        {
            Assert.True(waited < 30_000, "the endless cell never began");
            await Task.Delay(20, TestContext.Current.CancellationToken);
        }
    }

    [Fact]
    public async Task ClosingANotebookWhoseRunNeverEnds_AnswersAtOnce()
    {
        var started = At("started");
        var cell = await EndlessAsync("endless.verso", started);

        await using var served = await StartAsync();
        var running = served.Client.PostAsync($"/api/notebooks/endless.verso/cells/{cell}/run", null, TestContext.Current.CancellationToken);

        await BeganAsync(started);

        var closed = await served.Client.PostAsync("/api/notebooks/endless.verso/close", null, TestContext.Current.CancellationToken)
            .WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken);

        Assert.Equal(HttpStatusCode.NoContent, closed.StatusCode);

        // The run's own request is answered too, once the close stopped the run.
        Assert.Equal(HttpStatusCode.OK, (await running.WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken)).StatusCode);
    }

    [Fact]
    public async Task TheServerStops_WithARunThatNeverEndsInFlight_WithinFiveSeconds()
    {
        var started = At("started");
        var cell = await EndlessAsync("endless.verso", started);

        await using var served = await StartAsync();
        var running = served.Client.PostAsync($"/api/notebooks/endless.verso/cells/{cell}/run", null, TestContext.Current.CancellationToken);

        await BeganAsync(started);

        var stopping = Stopwatch.StartNew();

        await served.App.StopAsync(TestContext.Current.CancellationToken).WaitAsync(TimeSpan.FromSeconds(60), TestContext.Current.CancellationToken);

        Assert.True(stopping.Elapsed < TimeSpan.FromSeconds(5), $"the server took {stopping.Elapsed.TotalSeconds:0.0} s to stop");
        Assert.Equal(HttpStatusCode.OK, (await running.WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken)).StatusCode);
    }
}
