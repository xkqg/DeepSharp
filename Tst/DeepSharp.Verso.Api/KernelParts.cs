// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Tests.Api;

/// <summary>
/// A cell type that brings a kernel of its own and registers it nowhere else, as a part may: the engine finds that kernel
/// by its language for any cell that names it.
/// </summary>
/// <param name="began">The file its kernel writes as a run begins.</param>
/// <param name="go">The file its kernel waits for before the run ends.</param>
internal sealed class KernelledType(string began, string go) : ICellType
{
    public string ExtensionId => "deepsharp.tests.kernelled-type";

    public string Name => "A cell type with a kernel of its own";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public string CellTypeId => "deepsharp.tests.kernelled";

    public string DisplayName => "Kernelled";

    public string? Icon => null;

    public ICellRenderer Renderer { get; } = new DrawingPart();

    public ILanguageKernel? Kernel { get; } = new HeldKernel(began, go);

    public bool IsEditable => true;

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    public string GetDefaultContent() => string.Empty;
}

/// <summary>A kernel whose runs go on until they are let go.</summary>
/// <param name="began">The file it writes as a run begins.</param>
/// <param name="go">The file it waits for before the run ends.</param>
internal sealed class HeldKernel(string began, string go) : ILanguageKernel
{
    public const string Language = "held";

    public string ExtensionId => "deepsharp.tests.held-kernel";

    public string Name => "A kernel that runs until it is let go";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public string LanguageId => Language;

    public string DisplayName => "Held";

    public IReadOnlyList<string> FileExtensions => [];

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    public Task InitializeAsync() => Task.CompletedTask;

    public async Task<IReadOnlyList<CellOutput>> ExecuteAsync(string code, IExecutionContext context)
    {
        await File.WriteAllTextAsync(began, "on", TestContext.Current.CancellationToken);

        while (!File.Exists(go))
        {
            await Task.Delay(10, TestContext.Current.CancellationToken);
        }

        return [];
    }

    public Task<IReadOnlyList<Completion>> GetCompletionsAsync(string code, int cursorPosition) => Task.FromResult<IReadOnlyList<Completion>>([]);

    public Task<IReadOnlyList<Diagnostic>> GetDiagnosticsAsync(string code) => Task.FromResult<IReadOnlyList<Diagnostic>>([]);

    public Task<HoverInfo?> GetHoverInfoAsync(string code, int cursorPosition) => Task.FromResult<HoverInfo?>(null);

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}

/// <summary>A kernel that starts at once the first time and, once stopped, starts again only when it is let go.</summary>
/// <param name="go">The file it waits for before it starts again.</param>
internal sealed class SlowStartKernel(string go) : ILanguageKernel
{
    public const string Language = "slow-start";

    private int _stopped;

    public string ExtensionId => "deepsharp.tests.slow-start-kernel";

    public string Name => "A kernel that starts again only when it is let go";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public string LanguageId => Language;

    public string DisplayName => "Slow start";

    public IReadOnlyList<string> FileExtensions => [];

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    public async Task InitializeAsync()
    {
        while (Volatile.Read(ref _stopped) == 1 && !File.Exists(go))
        {
            await Task.Delay(10, TestContext.Current.CancellationToken);
        }
    }

    public Task<IReadOnlyList<CellOutput>> ExecuteAsync(string code, IExecutionContext context) => Task.FromResult<IReadOnlyList<CellOutput>>([]);

    public Task<IReadOnlyList<Completion>> GetCompletionsAsync(string code, int cursorPosition) => Task.FromResult<IReadOnlyList<Completion>>([]);

    public Task<IReadOnlyList<Diagnostic>> GetDiagnosticsAsync(string code) => Task.FromResult<IReadOnlyList<Diagnostic>>([]);

    public Task<HoverInfo?> GetHoverInfoAsync(string code, int cursorPosition) => Task.FromResult<HoverInfo?>(null);

    public ValueTask DisposeAsync()
    {
        Volatile.Write(ref _stopped, 1);

        return ValueTask.CompletedTask;
    }
}

/// <summary>
/// A kernel whose first start afresh fails, as one whose runtime went away fails, and whose starts after that succeed.
/// </summary>
internal sealed class FailingOnceKernel : ILanguageKernel
{
    public const string Language = "failing-once";

    /// <summary>Why its first start afresh fails.</summary>
    public const string Why = "It cannot start again yet.";

    // How many times it was stopped.
    private int _stopped;

    public string ExtensionId => "deepsharp.tests.failing-once-kernel";

    public string Name => "A kernel whose first start afresh fails";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public string LanguageId => Language;

    public string DisplayName => "Failing once";

    public IReadOnlyList<string> FileExtensions => [];

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    public Task InitializeAsync() => Volatile.Read(ref _stopped) == 1 ? Task.FromException(new InvalidOperationException(Why)) : Task.CompletedTask;

    public Task<IReadOnlyList<CellOutput>> ExecuteAsync(string code, IExecutionContext context) => Task.FromResult<IReadOnlyList<CellOutput>>([]);

    public Task<IReadOnlyList<Completion>> GetCompletionsAsync(string code, int cursorPosition) => Task.FromResult<IReadOnlyList<Completion>>([]);

    public Task<IReadOnlyList<Diagnostic>> GetDiagnosticsAsync(string code) => Task.FromResult<IReadOnlyList<Diagnostic>>([]);

    public Task<HoverInfo?> GetHoverInfoAsync(string code, int cursorPosition) => Task.FromResult<HoverInfo?>(null);

    public ValueTask DisposeAsync()
    {
        Interlocked.Increment(ref _stopped);

        return ValueTask.CompletedTask;
    }
}

/// <summary>A kernel that writes down the token a run hands it, and does nothing else.</summary>
internal sealed class RecordingKernel : ILanguageKernel
{
    public const string Language = "recording";

    /// <summary>The token the last run handed it.</summary>
    public CancellationToken Handed { get; private set; }

    public string ExtensionId => "deepsharp.tests.recording-kernel";

    public string Name => "A kernel that writes down the token a run hands it";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public string LanguageId => Language;

    public string DisplayName => "Recording";

    public IReadOnlyList<string> FileExtensions => [];

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    public Task InitializeAsync() => Task.CompletedTask;

    public Task<IReadOnlyList<CellOutput>> ExecuteAsync(string code, IExecutionContext context)
    {
        Handed = context.CancellationToken;

        return Task.FromResult<IReadOnlyList<CellOutput>>([]);
    }

    public Task<IReadOnlyList<Completion>> GetCompletionsAsync(string code, int cursorPosition) => Task.FromResult<IReadOnlyList<Completion>>([]);

    public Task<IReadOnlyList<Diagnostic>> GetDiagnosticsAsync(string code) => Task.FromResult<IReadOnlyList<Diagnostic>>([]);

    public Task<HoverInfo?> GetHoverInfoAsync(string code, int cursorPosition) => Task.FromResult<HoverInfo?>(null);

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}

/// <summary>
/// A kernel that, as Run All's reset puts it away, has the run stopped and then holds the reset until it is let go — as a
/// person pressing Stop while a slow kernel is put away does.
/// </summary>
/// <param name="stop">Stops the run.</param>
/// <param name="go">Ends when the reset may go on.</param>
internal sealed class StoppingKernel(Action stop, Task go) : ILanguageKernel
{
    public const string Language = "stopping";

    public string ExtensionId => "deepsharp.tests.stopping-kernel";

    public string Name => "A kernel whose putting away stops the run and waits";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public string LanguageId => Language;

    public string DisplayName => "Stopping";

    public IReadOnlyList<string> FileExtensions => [];

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    public Task InitializeAsync() => Task.CompletedTask;

    public Task<IReadOnlyList<CellOutput>> ExecuteAsync(string code, IExecutionContext context) => Task.FromResult<IReadOnlyList<CellOutput>>([]);

    public Task<IReadOnlyList<Completion>> GetCompletionsAsync(string code, int cursorPosition) => Task.FromResult<IReadOnlyList<Completion>>([]);

    public Task<IReadOnlyList<Diagnostic>> GetDiagnosticsAsync(string code) => Task.FromResult<IReadOnlyList<Diagnostic>>([]);

    public Task<HoverInfo?> GetHoverInfoAsync(string code, int cursorPosition) => Task.FromResult<HoverInfo?>(null);

    public async ValueTask DisposeAsync()
    {
        stop();
        await go;
    }
}

/// <summary>
/// A kernel that offers one completion, once it is let go: a read of it can be held while the notebook does something
/// else, such as starting the kernel afresh.
/// </summary>
/// <param name="asked">Ended when it was asked.</param>
/// <param name="go">Ends when it may answer.</param>
internal sealed class GatedKernel(TaskCompletionSource asked, Task go) : ILanguageKernel
{
    public const string Language = "gated";

    public string ExtensionId => "deepsharp.tests.gated-kernel";

    public string Name => "A kernel that answers a read once it is let go";

    public string Version => "1.0.0";

    public string? Author => null;

    public string? Description => null;

    public string LanguageId => Language;

    public string DisplayName => "Gated";

    public IReadOnlyList<string> FileExtensions => [];

    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    public Task OnUnloadedAsync() => Task.CompletedTask;

    public Task InitializeAsync() => Task.CompletedTask;

    public Task<IReadOnlyList<CellOutput>> ExecuteAsync(string code, IExecutionContext context) => Task.FromResult<IReadOnlyList<CellOutput>>([]);

    public async Task<IReadOnlyList<Completion>> GetCompletionsAsync(string code, int cursorPosition)
    {
        asked.TrySetResult();
        await go;

        return [new Completion("offered", "offered", "Keyword")];
    }

    public Task<IReadOnlyList<Diagnostic>> GetDiagnosticsAsync(string code) => Task.FromResult<IReadOnlyList<Diagnostic>>([]);

    public Task<HoverInfo?> GetHoverInfoAsync(string code, int cursorPosition) => Task.FromResult<HoverInfo?>(null);

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}
