// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Tests.Serve.Parts;

/// <summary>
/// A kernel that takes a while to start afresh, as a kernel with a large runtime does: long enough for a page to show that
/// it is being started afresh.
/// </summary>
[VersoExtension]
public sealed class SlowRestartKernel : ILanguageKernel
{
    /// <summary>The language it runs, which a notebook names as its kernel.</summary>
    public const string Language = "slow-restart";

    /// <summary>How long it takes to start afresh.</summary>
    public static readonly TimeSpan Taking = TimeSpan.FromSeconds(2);

    // Whether it was stopped once: it starts at once the first time.
    private int _stopped;

    /// <inheritdoc />
    public string ExtensionId => "deepsharp.tests.slow-restart-kernel";

    /// <inheritdoc />
    public string Name => "A kernel that takes a while to start afresh";

    /// <inheritdoc />
    public string Version => "1.0.0";

    /// <inheritdoc />
    public string? Author => null;

    /// <inheritdoc />
    public string? Description => null;

    /// <inheritdoc />
    public string LanguageId => Language;

    /// <inheritdoc />
    public string DisplayName => "Slow restart";

    /// <inheritdoc />
    public IReadOnlyList<string> FileExtensions => [];

    /// <inheritdoc />
    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    /// <inheritdoc />
    public Task OnUnloadedAsync() => Task.CompletedTask;

    /// <inheritdoc />
    public Task InitializeAsync() => Volatile.Read(ref _stopped) == 1 ? Task.Delay(Taking) : Task.CompletedTask;

    /// <inheritdoc />
    public Task<IReadOnlyList<CellOutput>> ExecuteAsync(string code, IExecutionContext context) => Task.FromResult<IReadOnlyList<CellOutput>>([]);

    /// <inheritdoc />
    public Task<IReadOnlyList<Completion>> GetCompletionsAsync(string code, int cursorPosition) => Task.FromResult<IReadOnlyList<Completion>>([]);

    /// <inheritdoc />
    public Task<IReadOnlyList<Diagnostic>> GetDiagnosticsAsync(string code) => Task.FromResult<IReadOnlyList<Diagnostic>>([]);

    /// <inheritdoc />
    public Task<HoverInfo?> GetHoverInfoAsync(string code, int cursorPosition) => Task.FromResult<HoverInfo?>(null);

    /// <inheritdoc />
    public ValueTask DisposeAsync()
    {
        Volatile.Write(ref _stopped, 1);

        return ValueTask.CompletedTask;
    }
}

/// <summary>A kernel that cannot start afresh, as one whose runtime went away cannot.</summary>
[VersoExtension]
public sealed class FailingRestartKernel : ILanguageKernel
{
    /// <summary>The language it runs, which a notebook names as its kernel.</summary>
    public const string Language = "failing-restart";

    /// <summary>Why it does not start afresh.</summary>
    public const string Why = "Its runtime went away.";

    // Whether it was stopped once: it starts the first time.
    private int _stopped;

    /// <inheritdoc />
    public string ExtensionId => "deepsharp.tests.failing-restart-kernel";

    /// <inheritdoc />
    public string Name => "A kernel that cannot start afresh";

    /// <inheritdoc />
    public string Version => "1.0.0";

    /// <inheritdoc />
    public string? Author => null;

    /// <inheritdoc />
    public string? Description => null;

    /// <inheritdoc />
    public string LanguageId => Language;

    /// <inheritdoc />
    public string DisplayName => "Failing restart";

    /// <inheritdoc />
    public IReadOnlyList<string> FileExtensions => [];

    /// <inheritdoc />
    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    /// <inheritdoc />
    public Task OnUnloadedAsync() => Task.CompletedTask;

    /// <inheritdoc />
    public Task InitializeAsync() => Volatile.Read(ref _stopped) == 1 ? Task.FromException(new InvalidOperationException(Why)) : Task.CompletedTask;

    /// <inheritdoc />
    public Task<IReadOnlyList<CellOutput>> ExecuteAsync(string code, IExecutionContext context) => Task.FromResult<IReadOnlyList<CellOutput>>([]);

    /// <inheritdoc />
    public Task<IReadOnlyList<Completion>> GetCompletionsAsync(string code, int cursorPosition) => Task.FromResult<IReadOnlyList<Completion>>([]);

    /// <inheritdoc />
    public Task<IReadOnlyList<Diagnostic>> GetDiagnosticsAsync(string code) => Task.FromResult<IReadOnlyList<Diagnostic>>([]);

    /// <inheritdoc />
    public Task<HoverInfo?> GetHoverInfoAsync(string code, int cursorPosition) => Task.FromResult<HoverInfo?>(null);

    /// <inheritdoc />
    public ValueTask DisposeAsync()
    {
        Volatile.Write(ref _stopped, 1);

        return ValueTask.CompletedTask;
    }
}
