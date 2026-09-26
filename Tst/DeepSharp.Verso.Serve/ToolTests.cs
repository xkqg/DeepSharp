// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Diagnostics;
using System.Runtime.InteropServices;
using DeepSharp.Verso.Serve;
using Verso.Abstractions;
using Verso.Serializers;

namespace DeepSharp.Tests.Serve;

/// <summary>
/// The tool as a person runs it, in a process of its own: stopped from its terminal, it ends within moments whatever a
/// notebook runs — a run that never ends, or one that keeps a thread of its own going.
/// </summary>
public sealed class ToolTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-serve-tool-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    [Fact]
    public async Task AskedForItsUsage_TheToolSaysIt_AndARefusedCommandLineEndsItWithTwo()
    {
        var helped = await ToolAsync("--help");

        Assert.Equal(0, helped.Code);
        Assert.Contains(ServeOptions.Usage, helped.Said, StringComparison.Ordinal);

        var refused = await ToolAsync("--nope");

        Assert.Equal(2, refused.Code);
        Assert.Contains("--nope", refused.Errors, StringComparison.Ordinal);
    }

    [Fact]
    public Task StoppedFromItsTerminal_TheToolEnds_WhileARunNeverEnds() =>
        EndsAsync("while (true) { await System.Threading.Tasks.Task.Delay(10); }");

    [Fact]
    public Task StoppedFromItsTerminal_TheToolEnds_WhileARunKeepsAThreadOfItsOwnGoing() =>
        EndsAsync("new System.Threading.Thread(() => { while (true) { System.Threading.Thread.Sleep(10); } }).Start(); while (true) { await System.Threading.Tasks.Task.Delay(10); }");

    private async Task EndsAsync(string code)
    {
#if !NET10_0_OR_GREATER
        // Windows hands a terminal's stop only to a process started in a group of its own, which .NET 8 cannot start.
        Assert.SkipWhen(OperatingSystem.IsWindows(), "Only .NET 10 starts a process in a group of its own on Windows.");
#endif

        var started = Path.Join(_folder, "started");
        var cell = new CellModel { Type = "code", Language = "csharp", Source = $$"""System.IO.File.WriteAllText(@"{{started}}", "on"); {{code}}""" };
        var notebook = new NotebookModel();

        notebook.Cells.Add(cell);
        await File.WriteAllTextAsync(Path.Join(_folder, "endless.verso"), await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        var start = new ProcessStartInfo("dotnet") { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true };

        start.ArgumentList.Add(Path.Join(AppContext.BaseDirectory, "DeepSharp.Verso.Serve.dll"));
        start.ArgumentList.Add(_folder);
        start.ArgumentList.Add("--no-browser");
#if NET10_0_OR_GREATER
        if (OperatingSystem.IsWindows())
        {
            start.CreateNewProcessGroup = true;
        }
#endif

        using var tool = Process.Start(start)!;

        try
        {
            var said = await tool.StandardOutput.ReadLineAsync(TestContext.Current.CancellationToken).AsTask()
                .WaitAsync(TimeSpan.FromSeconds(60), TestContext.Current.CancellationToken);

            Assert.NotNull(said);

            var address = new Uri(said[(said.IndexOf(" at ", StringComparison.Ordinal) + 4)..]);
            using var browser = new HttpClient { BaseAddress = new Uri(address.GetLeftPart(UriPartial.Authority)) };
            var running = browser.PostAsync($"/api/notebooks/endless.verso/cells/{cell.Id}/run{address.Query}", null, TestContext.Current.CancellationToken);

            for (var waited = 0; !File.Exists(started); waited += 20)
            {
                Assert.True(waited < 60_000, "the endless cell never began");
                Assert.False(running.IsCompleted, "the run was answered before it began");
                await Task.Delay(20, TestContext.Current.CancellationToken);
            }

            Interrupt(tool);

            await tool.WaitForExitAsync(TestContext.Current.CancellationToken).WaitAsync(TimeSpan.FromSeconds(5), TestContext.Current.CancellationToken);

            Assert.Equal(0, tool.ExitCode);
        }
        finally
        {
            if (!tool.HasExited)
            {
                tool.Kill(entireProcessTree: true);
            }
        }
    }

    // The tool run to its end with a command line: what it said, what went wrong, and the code it ended with.
    private static async Task<Ended> ToolAsync(params string[] args)
    {
        var start = new ProcessStartInfo("dotnet") { UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true };

        start.ArgumentList.Add(Path.Join(AppContext.BaseDirectory, "DeepSharp.Verso.Serve.dll"));

        foreach (var arg in args)
        {
            start.ArgumentList.Add(arg);
        }

        using var tool = Process.Start(start)!;
        var said = tool.StandardOutput.ReadToEndAsync(TestContext.Current.CancellationToken);
        var errors = tool.StandardError.ReadToEndAsync(TestContext.Current.CancellationToken);

        await tool.WaitForExitAsync(TestContext.Current.CancellationToken).WaitAsync(TimeSpan.FromSeconds(60), TestContext.Current.CancellationToken);

        return new Ended(tool.ExitCode, await said, await errors);
    }

    private readonly record struct Ended(int Code, string Said, string Errors);

    // What a terminal sends on Ctrl+C: SIGINT on Linux and macOS, and on Windows its break, which .NET stops a host on
    // the same way.
    private static void Interrupt(Process tool)
    {
        if (OperatingSystem.IsWindows())
        {
            Assert.True(GenerateConsoleCtrlEvent(CtrlBreak, (uint)tool.Id), $"the break was not sent: {Marshal.GetLastPInvokeError()}");
        }
        else
        {
            Assert.Equal(0, Kill(tool.Id, Sigint));
        }
    }

    private const uint CtrlBreak = 1;

    private const int Sigint = 2;

    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GenerateConsoleCtrlEvent(uint ctrlEvent, uint processGroupId);

    [DllImport("libc", EntryPoint = "kill", SetLastError = true)]
    private static extern int Kill(int process, int signal);
}
