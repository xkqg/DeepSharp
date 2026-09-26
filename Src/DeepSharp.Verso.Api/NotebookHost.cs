// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Notebooks;
using Verso;
using Verso.Abstractions;
using Verso.Extensions;

namespace DeepSharp.Verso.Api;

/// <summary>One open notebook: Verso's engine with DeepSharp's parts, holding the notebook one file saves.</summary>
/// <remarks>
/// <para>
/// It is opened the way Verso's own editors open a notebook — through the serializer for its format, past the guards
/// that run after reading, with the cells that are only ever shown rendered drawn — and DeepSharp's parts are
/// registered before Verso looks for any beside the application, so an application published as a single file, which
/// has no assemblies beside it, still has them. An extension a notebook asks for is refused: the engine runs what the
/// application carries, and fetches nothing. A host is opened and closed by the <see cref="OpenNotebooks"/> that
/// holds it, never by whoever it was handed to.
/// </para>
/// <para>
/// Everything done to the notebook takes its turn, one at a time and in the order it came, because the engine serves
/// one caller at a time and a change made while another is under way would act on blocks that are going away. A
/// request about a cell a change before it rewrote or took away is refused with <see cref="CellGoneException"/>.
/// </para>
/// </remarks>
public sealed class NotebookHost
{
    private const string CSharp = "csharp";

    // The one thing in this package as wide as the process, because what it guards is: a C# kernel takes over the
    // process's console while it runs (Console.SetOut in Verso's C# kernel), so two C# runs anywhere in the process —
    // in two notebooks, or under two holders of notebooks — print into each other (measured). Each run takes its turn.
    private static readonly Lane CSharpRuns = new();

    private readonly Lane _turns = new();
    private Run? _running;
    private int _closed;

    private NotebookHost(string filePath, ExtensionHost extensions, Scaffold scaffold)
    {
        FilePath = filePath;
        Extensions = extensions;
        Scaffold = scaffold;
    }

    /// <summary>The file the notebook is saved in, as a full path.</summary>
    public string FilePath { get; }

    /// <summary>The notebook's cells as they stand, in order.</summary>
    public IReadOnlyList<HostedCell> Cells => [.. Scaffold.Cells.Select(cell => cell.Hosted())];

    /// <summary>The engine's extensions: Verso's own and DeepSharp's.</summary>
    internal ExtensionHost Extensions { get; }

    /// <summary>The notebook as Verso's engine holds and runs it.</summary>
    internal Scaffold Scaffold { get; }

    /// <summary>Sets a cell's text, as typing it does; nothing runs.</summary>
    /// <param name="cell">The cell.</param>
    /// <param name="source">Its new text.</param>
    /// <returns>The cell as it stands after.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell or took it away.</exception>
    public Task<HostedCell> EditAsync(Guid cell, string source)
    {
        ArgumentNullException.ThrowIfNull(source);

        return TurnAsync(() =>
        {
            Standing(cell);
            Scaffold.UpdateCellSource(cell, source);

            return Task.FromResult(Standing(cell).Hosted());
        });
    }

    /// <summary>Runs a cell, as pressing its run button does.</summary>
    /// <param name="cell">The cell.</param>
    /// <returns>The cell as it stands after the run, or after <see cref="Stop"/> ended it.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell or took it away.</exception>
    /// <remarks>
    /// A C# run takes its turn among every C# run in the process, because a C# kernel takes over the process's console
    /// while it runs, and two at once would print into each other.
    /// </remarks>
    public Task<HostedCell> RunAsync(Guid cell) => TurnAsync(async () =>
    {
        var running = Standing(cell);

        await (string.Equals(running.Language, CSharp, StringComparison.OrdinalIgnoreCase)
            ? CSharpRuns.TakeTurnAsync(() => RunUntilStoppedAsync(() => Scaffold.ExecuteCellAsync(running.Id), running.Language))
            : RunUntilStoppedAsync(() => Scaffold.ExecuteCellAsync(running.Id), running.Language));

        return running.Hosted();
    });

    /// <summary>
    /// Stops the run under way — a cell's, or a toolbar button's — if there is one: the only way Verso's engine stops a
    /// run that does not end is a fresh kernel, so the run's kernel is restarted, and what that kernel held — the
    /// notebook's variables, the pipeline handed to C# cells among them — is gone. A run that never ends goes on in the
    /// background until the application does.
    /// </summary>
    public void Stop() => Volatile.Read(ref _running)?.Stop();

    /// <summary>Hands a click on a control a cell drew to the part the control names.</summary>
    /// <param name="gesture">The click.</param>
    /// <returns>Whether it changed the cells, and what the part answered, which the cell then shows.</returns>
    /// <exception cref="CellGoneException">A change before this one rewrote the cell or took it away.</exception>
    /// <exception cref="InvalidOperationException">No part of that name answers a click.</exception>
    public Task<GestureResult> GestureAsync(HostedGesture gesture) => TurnAsync(async () =>
    {
        Standing(gesture.Cell);

        var part = Extensions.GetInteractionHandler(gesture.ExtensionId)
            ?? throw new InvalidOperationException($"No part named '{gesture.ExtensionId}' answers a click.");
        var context = new CellInteractionContext
        {
            Region = CellRegion.Output,
            InteractionType = gesture.Action,
            Payload = gesture.Payload,
            CellId = gesture.Cell,
            ExtensionId = gesture.ExtensionId,
            CancellationToken = CancellationToken.None,
            Variables = Scaffold.Variables,
            Notebook = Scaffold.NotebookOps,
            NotebookModel = Scaffold.Notebook,
        };
        var answer = await part.OnCellInteractionAsync(context);

        if (answer is not null)
        {
            // As Verso's browser editor shows one: the answer is what the cell shows now, while the cell still stands.
            foreach (var shown in Scaffold.Cells.Where(each => each.Id == gesture.Cell))
            {
                shown.Outputs.Clear();
                shown.Outputs.Add(new CellOutput("text/html", answer));
            }
        }

        return new GestureResult(context.StateChanged, answer);
    });

    /// <summary>Every toolbar button the engine has — Verso's own and DeepSharp's — each saying whether it can be pressed now.</summary>
    /// <returns>The buttons, by place and then in their order.</returns>
    public Task<IReadOnlyList<HostedToolbarAction>> ToolbarAsync() => TurnAsync(async () =>
    {
        var context = new ToolbarContext(Scaffold, []);
        var buttons = new List<HostedToolbarAction>();

        foreach (var action in Extensions.GetToolbarActions())
        {
            buttons.Add(new HostedToolbarAction(
                action.ActionId,
                action.DisplayName,
                action.Icon,
                action.IconOnly,
                action.IsPrimary,
                action.ConfirmationPrompt,
                Enum.Parse<ToolbarPlace>(action.Placement.ToString()),
                action.Order,
                await action.IsEnabledAsync(context)));
        }

        return (IReadOnlyList<HostedToolbarAction>)[.. buttons.OrderBy(button => button.Place).ThenBy(button => button.Order)];
    });

    /// <summary>Presses a toolbar button.</summary>
    /// <param name="id">The button.</param>
    /// <param name="cells">The cells it is pressed for, for a button on a cell's toolbar.</param>
    /// <returns>The file it handed over, for whoever pressed it; nothing when it handed none.</returns>
    /// <exception cref="InvalidOperationException">The engine has no button of that name.</exception>
    /// <remarks>
    /// A button may run cells, C# among them, so a press takes its turn among the process's C# runs as a C# run does. A file is
    /// never written beside the notebook: where it is saved is for whoever pressed the button to say.
    /// </remarks>
    public Task<HostedFile?> RunToolbarAsync(string id, params Guid[] cells) => TurnAsync(() => CSharpRuns.TakeTurnAsync(async () =>
    {
        var action = Extensions.GetToolbarActions().FirstOrDefault(each => each.ActionId == id)
            ?? throw new InvalidOperationException($"The notebook has no toolbar button '{id}'.");
        var context = new ToolbarContext(Scaffold, cells);

        // A button that runs the cells can meet one that never ends, and only the notebook's own kernel runs away.
        await RunUntilStoppedAsync(() => action.ExecuteAsync(context), Scaffold.DefaultKernelId);

        return context.Handed;
    }));

    /// <summary>Registers DeepSharp's parts with an engine, before it looks for any beside the application.</summary>
    /// <param name="extensions">The engine.</param>
    /// <returns>When they are registered.</returns>
    /// <remarks>
    /// First, because Verso's own look beside the application passes over a part already registered, while one
    /// registered after it would be refused as a second.
    /// </remarks>
    internal static async Task RegisterAsync(ExtensionHost extensions)
    {
        foreach (var part in Parts())
        {
            await extensions.LoadExtensionAsync(part);
        }
    }

    /// <summary>Every part DeepSharp's notebook has, made anew for one engine.</summary>
    /// <returns>The parts.</returns>
    internal static IExtension[] Parts() =>
    [
        new StepCellType(),
        new StepKernel(),
        new StepRenderer(),
        new StepForm(),
        new RunPipelineAction(),
        new ExportPipelineAction(),
        new TakeOverAction(),
        new JupyterGuard(),
    ];

    /// <summary>Opens the notebook a file holds on an engine, and closes the engine when it cannot be opened.</summary>
    /// <param name="filePath">The notebook's file, as a full path.</param>
    /// <param name="extensions">The engine to open it on; closed when the notebook cannot be opened.</param>
    /// <param name="cancellationToken">Stops the open.</param>
    /// <returns>The host.</returns>
    /// <exception cref="NotSupportedException">No format Verso knows reads the file.</exception>
    internal static async Task<NotebookHost> OpenAsync(string filePath, ExtensionHost extensions, CancellationToken cancellationToken)
    {
        Scaffold? scaffold = null;

        try
        {
            var content = await File.ReadAllTextAsync(filePath, cancellationToken);

            extensions.ConsentHandler = static (_, _) => Task.FromResult(false);
            await RegisterAsync(extensions);
            await extensions.LoadBuiltInExtensionsAsync();

            var serializer = extensions.GetSerializers().FirstOrDefault(each => each.CanImport(filePath))
                ?? throw new NotSupportedException($"No format Verso knows reads '{Path.GetFileName(filePath)}'.");
            var notebook = await serializer.DeserializeAsync(content);

            foreach (var guard in extensions.GetPostProcessors().Where(each => each.CanProcess(filePath, serializer.FormatId)).OrderBy(each => each.Priority))
            {
                notebook = await guard.PostDeserializeAsync(notebook, filePath);
            }

            // As Verso's own editors leave a notebook that names neither.
            notebook.DefaultKernelId ??= CSharp;
            notebook.ActiveLayout ??= LayoutDefaults.Reference;

            scaffold = new Scaffold(notebook, extensions, filePath);
            scaffold.InitializeSubsystems();
            await scaffold.RenderTransientCellsAsync(cancellationToken);

            return new NotebookHost(filePath, extensions, scaffold);
        }
        catch
        {
            if (scaffold is not null)
            {
                await scaffold.DisposeAsync();
            }

            await extensions.DisposeAsync();

            throw;
        }
    }

    /// <summary>Closes the notebook once what is under way is done, refusing anything asked after; then its engine.</summary>
    /// <returns>When both are closed.</returns>
    internal async ValueTask CloseAsync()
    {
        Interlocked.Exchange(ref _closed, 1);

        await _turns.TakeTurnAsync(async () =>
        {
            await Scaffold.DisposeAsync();
            await Extensions.DisposeAsync();

            return true;
        });
    }

    // Everything done to the notebook: refused once it closes, and otherwise one at a time, in the order it came.
    private Task<T> TurnAsync<T>(Func<Task<T>> change)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _closed) != 0, this);

        return _turns.TakeTurnAsync(change);
    }

    private CellModel Standing(Guid cell) => Scaffold.GetCell(cell) ?? throw new CellGoneException(cell);

    // Runs a cell until it ends, or until Stop. A run is not asked to stop, since one that does not end does not listen
    // either (measured: a C# loop that awaits goes on with its token cancelled); a fresh kernel ends the turn, and the
    // run is left behind. Only the run under way can be stopped, so the slot holds one, and a Stop after it ended
    // meets a run nobody waits for any more.
    private async Task<bool> RunUntilStoppedAsync(Func<Task> start, string? kernel)
    {
        var run = new Run();

        Volatile.Write(ref _running, run);

        try
        {
            var ran = start();

            if (await Task.WhenAny(ran, run.Stopped) == ran)
            {
                await ran;
            }
            else
            {
                await Scaffold.RestartKernelAsync(kernel);
            }

            return true;
        }
        finally
        {
            Volatile.Write(ref _running, null);
        }
    }

    // A run under way, and the way to stop it.
    private sealed class Run
    {
        private readonly TaskCompletionSource _stopped = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public Task Stopped => _stopped.Task;

        public void Stop() => _stopped.TrySetResult();
    }
}
