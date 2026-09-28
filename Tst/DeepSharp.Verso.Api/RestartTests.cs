// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Api;
using Verso.Abstractions;
using Verso.Extensions;
using Verso.Serializers;

namespace DeepSharp.Tests.Api;

/// <summary>
/// Every version says what became of the notebook's kernels, as Verso's editors show it beside the notebook's name: how
/// many times a kernel was started afresh — by a stop, or by Verso's Restart Kernel — each of which clears the notebook's
/// variables; whether one is being started afresh now; and why the last start failed, until a kernel starts afresh or a
/// cell begins. Kernels are real ones, loaded into the engine as any part is.
/// </summary>
public sealed class RestartTests : IDisposable
{
    private static readonly TimeSpan AtOnce = TimeSpan.FromSeconds(10);

    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-api-restarts-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private string At(string name) => Path.Join(_folder, name);

    private static CellModel Markdown(string source) => new() { Type = "markdown", Source = source };

    // A notebook whose kernel is the one given, opened on an engine that carries it: Verso's Restart Kernel restarts it.
    private async Task<NotebookHost> OpenOnAsync(ILanguageKernel kernel, params CellModel[] cells)
    {
        var path = At("restarts.verso");
        var notebook = new NotebookModel { DefaultKernelId = kernel.LanguageId };

        foreach (var cell in cells)
        {
            notebook.Cells.Add(cell);
        }

        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        var engine = new ExtensionHost();

        await engine.LoadExtensionAsync(kernel);

        return await NotebookHost.OpenAsync(path, engine, TestContext.Current.CancellationToken);
    }

    // What a view is told first that holds, within a while.
    private static async Task<NotebookChange> ToldAsync(NotebookSubscription view, Func<NotebookChange, bool> holds)
    {
        using var patience = CancellationTokenSource.CreateLinkedTokenSource(TestContext.Current.CancellationToken);

        patience.CancelAfter(TimeSpan.FromSeconds(30));

        await foreach (var change in view.ReadAllAsync(patience.Token))
        {
            if (holds(change))
            {
                return change;
            }
        }

        throw new InvalidOperationException("The view ended before it was told.");
    }

    [Fact]
    public async Task VersosRestartKernel_IsToldWhileItStartsAfresh_AndCountedOnceItHas()
    {
        var go = At("go");
        var host = await OpenOnAsync(new SlowStartKernel(go), Markdown("# A notebook"));

        try
        {
            Assert.Equal(new HostedKernels(0, Restarting: false, Fault: null), host.Current.Kernels);

            using var view = host.Subscribe();
            var pressing = host.RunToolbarAsync("verso.action.restart-kernel");

            // Told while the kernel starts afresh, which it does only once let go.
            Assert.Equal(new HostedKernels(0, Restarting: true, Fault: null), (await ToldAsync(view, change => change.Kernels.Restarting)).Kernels);

            await File.WriteAllTextAsync(go, "go", TestContext.Current.CancellationToken);
            await pressing.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

            Assert.Equal(new HostedKernels(1, Restarting: false, Fault: null), host.Current.Kernels);
        }
        finally
        {
            await host.CloseAsync();
        }
    }

    [Fact]
    public async Task AStop_IsCountedAsAKernelStartedAfresh()
    {
        var started = At("started");

        await using var notebooks = new OpenNotebooks();
        var path = At("endless.verso");
        var notebook = new NotebookModel { DefaultKernelId = "csharp" };

        notebook.Cells.Add(new CellModel { Type = "code", Language = "csharp", Source = $$"""System.IO.File.WriteAllText(@"{{started}}", "on"); while (true) { await System.Threading.Tasks.Task.Delay(10); }""" });
        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        var host = await notebooks.OpenAsync(path, TestContext.Current.CancellationToken);
        var running = host.RunAsync(host.Cells[0].Id);

        while (!File.Exists(started))
        {
            await Task.Delay(20, TestContext.Current.CancellationToken);
        }

        Assert.True(host.Stop(host.Running!.Value.Number));
        await running.WaitAsync(AtOnce, TestContext.Current.CancellationToken);

        Assert.Equal(new HostedKernels(1, Restarting: false, Fault: null), host.Current.Kernels);
    }

    [Fact]
    public async Task AKernelThatFailsToStartAfresh_IsToldWithWhy_UntilACellBegins()
    {
        var host = await OpenOnAsync(new FailingOnceKernel(), Markdown("# A notebook"));

        try
        {
            var refused = await Assert.ThrowsAsync<InvalidOperationException>(() => host.RunToolbarAsync("verso.action.restart-kernel"));

            Assert.Equal(FailingOnceKernel.Why, refused.Message);
            Assert.Equal(new HostedKernels(0, Restarting: false, FailingOnceKernel.Why), host.Current.Kernels);

            // As Verso's editor stops showing the failure once a cell runs.
            await host.RunAsync(host.Cells[0].Id);

            Assert.Equal(new HostedKernels(0, Restarting: false, Fault: null), host.Current.Kernels);
        }
        finally
        {
            await host.CloseAsync();
        }
    }

    [Fact]
    public async Task AKernelThatFailedToStartAfresh_IsToldAsStartedOnceItIs()
    {
        var host = await OpenOnAsync(new FailingOnceKernel(), Markdown("# A notebook"));

        try
        {
            await Assert.ThrowsAsync<InvalidOperationException>(() => host.RunToolbarAsync("verso.action.restart-kernel"));
            await host.RunToolbarAsync("verso.action.restart-kernel");

            Assert.Equal(new HostedKernels(1, Restarting: false, Fault: null), host.Current.Kernels);
        }
        finally
        {
            await host.CloseAsync();
        }
    }
}
