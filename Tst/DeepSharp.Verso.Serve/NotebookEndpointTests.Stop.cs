// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Net.WebSockets;
using System.Text.Json;
using Verso.Abstractions;
using Verso.Serializers;

namespace DeepSharp.Tests.Serve;

// No close waits for a run, and nothing a page asks keeps Stop from it. A notebook whose run never ends closes when a
// person closes it, the server stops as soon as it is told to, and a Stop is answered at once however many asks wait
// behind the run it stops.
[Collection(RunsLeftBehind.Name)]
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
        notebook.Cells.Add(new CellModel { Type = "code", Language = "csharp", Source = "1 + 1" });
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
    public async Task ClosingANotebookWhoseRunNeverEnds_IsAnsweredAtOnce_AndSoIsTheRun_ThenTheSocketEnds()
    {
        var started = At("started");
        var cell = await EndlessAsync("endless.verso", started);

        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served, "endless.verso");

        await socket.SnapshotAsync();

        var running = await socket.SendAsync("run", new { cell });

        await BeganAsync(started);

        var closed = await socket.AskAsync("close").WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken);

        Assert.True(closed.IsNothing);

        // The run's own ask is answered too, once the close stopped the run, before the socket ends.
        Assert.False((await running.WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken)).Refused);
        await socket.Ended.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
        Assert.Equal(WebSocketCloseStatus.NormalClosure, socket.CloseStatus);
    }

    [Fact]
    public async Task TheServerStops_WithARunThatNeverEndsInFlight_WithinFiveSeconds()
    {
        var started = At("started");
        var cell = await EndlessAsync("endless.verso", started);

        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served, "endless.verso");

        await socket.SnapshotAsync();
        await socket.SendAsync("run", new { cell });
        await BeganAsync(started);

        var stopping = Stopwatch.StartNew();

        await served.App.StopAsync(TestContext.Current.CancellationToken).WaitAsync(TimeSpan.FromSeconds(60), TestContext.Current.CancellationToken);

        Assert.True(stopping.Elapsed < TimeSpan.FromSeconds(5), $"the server took {stopping.Elapsed.TotalSeconds:0.0} s to stop");
        await socket.Ended.WaitAsync(TimeSpan.FromSeconds(10), TestContext.Current.CancellationToken);
    }

    [Fact]
    public async Task AStop_IsAnsweredAtOnce_WhileThirtyAsksWaitBehindTheRunItStops_WhichAreThenAnswered()
    {
        var started = At("started");
        var cell = await EndlessAsync("endless.verso", started);

        await using var served = await StartAsync();
        await using var socket = await SocketAsync(served, "endless.verso");
        var second = (await socket.SnapshotAsync()).Version.Cells[1].Id;
        var running = await socket.SendAsync("run", new { cell });

        await BeganAsync(started);

        // Thirty things asked of the notebook, each waiting its turn behind the run.
        var waiting = new List<Task<PageSocket.Answered>>();

        for (var ask = 0; ask < 30; ask++)
        {
            waiting.Add(await socket.SendAsync("edit", new { cell = second, source = $"{ask} + 1" }));
        }

        var run = (await NotebookAsync(served, "endless.verso")).Running!.Value.Number;
        var clock = Stopwatch.StartNew();
        var stopped = await socket.AskAsync("stop", new { run });

        Assert.True(stopped.Result<JsonElement>().GetProperty("stopped").GetBoolean());
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(1), $"the stop took {clock.Elapsed.TotalMilliseconds:0} ms");

        // The run ends, and every ask behind it is made in the order it came.
        await running.WaitAsync(TimeSpan.FromSeconds(30), TestContext.Current.CancellationToken);
        await Task.WhenAll(waiting).WaitAsync(TimeSpan.FromSeconds(30), TestContext.Current.CancellationToken);

        Assert.Equal("29 + 1", (await NotebookAsync(served, "endless.verso")).Cells[1].Source);
    }
}
