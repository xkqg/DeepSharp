// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;
using Verso.Abstractions;

namespace DeepSharp.Notebooks.Verso;

/// <summary>What is known of the bytes a notebook's source holds now.</summary>
/// <param name="Known">Whether they were read at all: a gesture is handed no file, so it knows nothing of them.</param>
/// <param name="Fingerprint">A SHA-256 of the bytes, when they were read; nothing when they could not be.</param>
internal readonly record struct SourceBytes(bool Known, string? Fingerprint)
{
    /// <summary>Nothing read: what a gesture knows, which is handed no file.</summary>
    public static SourceBytes Unknown { get; } = new(false, null);

    /// <summary>Read, and gone: the file could not be opened, so nothing learned from it holds.</summary>
    public static SourceBytes Unreadable { get; } = new(true, null);

    /// <summary>Read, with this fingerprint.</summary>
    /// <param name="fingerprint">A SHA-256 of the bytes.</param>
    /// <returns>What is known.</returns>
    public static SourceBytes Of(string fingerprint) => new(true, fingerprint);

    /// <summary>Whether a fit learned from bytes with this fingerprint may still stand.</summary>
    /// <param name="learnedFrom">The fingerprint of the bytes it was learned from.</param>
    /// <returns><see langword="true"/> unless the bytes were read and are other ones.</returns>
    public bool Allow(string learnedFrom) => !Known || Fingerprint == learnedFrom;
}

/// <summary>
/// What only holds between the calls on one notebook: which block shows what, what a gesture asked a block for, what
/// the notebook hands to C# cells, and that one gesture on it runs at a time.
/// </summary>
/// <remarks>
/// Kept by the block type Verso loaded, the one object every part of a notebook reaches; nothing here is saved, and
/// nothing is shared with another notebook. Everything is taken under a lock, since no host promises one call at a
/// time.
/// <para>
/// One view is kept: the last one worked out, under the key of the steps it is worked out from and the fingerprint of
/// the bytes it was read from. A view is those two and nothing else, so while both stay it is the same view, and
/// another page of it, or the same view shown again, runs no step. One, because a view holds every row at its block;
/// the rows as read are kept apart, by the source's own cache.
/// </para>
/// <para>
/// The pipeline handed to C# cells goes through here and nowhere else, always into the variables the caller was
/// handed: the declaration the blocks make, or — only while the blocks declare the same steps and the source holds
/// the same bytes — what this notebook's own last run learned from them. Whenever that may no longer hold, it is
/// taken back, and a C# cell sees none rather than one the blocks no longer make.
/// </para>
/// </remarks>
internal sealed class NotebookSession
{
    private readonly Lock _lock = new();
    private readonly Dictionary<Guid, ViewRequest> _requests = [];
    private readonly Dictionary<Guid, ShownView> _shown = [];
    private readonly Dictionary<Guid, Refusal> _refusals = [];
    private NotebookPipeline? _assembled;
    private KeptView? _view;
    private RunStamp? _run;
    private int _viewsRun;
    private int _runsFitted;

    // The last gesture's turn: the next one waits for it, so they take the lane in the order they came.
    private Task _lane = Task.CompletedTask;

    /// <summary>The rows the notebook's source opened last, kept for the next view.</summary>
    public SourceCache Sources { get; } = new();

    /// <summary>The pipeline the blocks made at the last gesture, when there was one.</summary>
    public NotebookPipeline? Assembled
    {
        get
        {
            lock (_lock)
            {
                return _assembled;
            }
        }
    }

    /// <summary>How many views were worked out from the source up: the number keeping the last one exists to keep down.</summary>
    public int ViewsRun
    {
        get
        {
            lock (_lock)
            {
                return _viewsRun;
            }
        }
    }

    /// <summary>How many times a run of the whole pipeline was fitted and handed over.</summary>
    public int RunsFitted
    {
        get
        {
            lock (_lock)
            {
                return _runsFitted;
            }
        }
    }

    /// <summary>The blocks that show data, each with the key of the steps its rows were worked out from.</summary>
    public IReadOnlyDictionary<Guid, string> Shown
    {
        get
        {
            lock (_lock)
            {
                return _shown.ToDictionary(each => each.Key, each => each.Value.Key);
            }
        }
    }

    /// <summary>Runs one gesture on the notebook once every gesture made before it has finished.</summary>
    /// <typeparam name="T">What the gesture answers.</typeparam>
    /// <param name="gesture">The gesture: it may ask a block for something and run the block.</param>
    /// <returns>Its answer.</returns>
    /// <remarks>
    /// A gesture leaves a request and then runs the block that takes it; two gestures that interleave would each take
    /// the other's. One host sends one request at a time, another runs them side by side, and this holds for both.
    /// </remarks>
    public async Task<T> OneAtATimeAsync<T>(Func<Task<T>> gesture)
    {
        var mine = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        Task before;

        lock (_lock)
        {
            before = _lane;
            _lane = mine.Task;
        }

        await before;

        try
        {
            return await gesture();
        }
        finally
        {
            mine.SetResult();
        }
    }

    /// <summary>Keeps the pipeline a gesture assembled, for the parts that ask what is around a block.</summary>
    /// <param name="assembled">The pipeline.</param>
    public void Publish(NotebookPipeline assembled)
    {
        lock (_lock)
        {
            _assembled = assembled;
        }
    }

    /// <summary>Hands C# cells the pipeline the blocks make, knowing nothing of the source's bytes.</summary>
    /// <param name="variables">The notebook's variables, as the caller was handed them.</param>
    /// <param name="assembled">The pipeline the blocks make now.</param>
    public void HandOver(IVariableStore variables, NotebookPipeline assembled) =>
        HandOver(variables, assembled, SourceBytes.Unknown);

    /// <summary>Hands C# cells the pipeline the blocks make, knowing what the source's bytes are now.</summary>
    /// <param name="variables">The notebook's variables, as the caller was handed them.</param>
    /// <param name="assembled">The pipeline the blocks make now.</param>
    /// <param name="bytes">What is known of the source's bytes.</param>
    public void HandOver(IVariableStore variables, NotebookPipeline assembled, SourceBytes bytes)
    {
        ArgumentNullException.ThrowIfNull(variables);

        if (EnvelopeFor(variables, assembled, bytes) is { } envelope)
        {
            variables.Set(StepKernel.HandOver, envelope);
        }
        else
        {
            variables.Remove(StepKernel.HandOver);
        }
    }

    /// <summary>
    /// What a notebook hands on as its pipeline: what its own last run learned, while the blocks declare those steps,
    /// the bytes are the ones it learned from and it is still what C# cells were handed; otherwise the declaration.
    /// </summary>
    /// <param name="variables">The notebook's variables.</param>
    /// <param name="assembled">The pipeline the blocks make now.</param>
    /// <param name="bytes">What is known of the source's bytes.</param>
    /// <returns>The pipeline as text, or nothing while the blocks make none.</returns>
    public string? EnvelopeFor(IVariableStore variables, NotebookPipeline assembled, SourceBytes bytes)
    {
        ArgumentNullException.ThrowIfNull(variables);

        if (!assembled.Whole)
        {
            return null;
        }

        RunStamp? run;

        lock (_lock)
        {
            run = _run;
        }

        return run is { } stamped
               && stamped.Key == KeyOf(assembled.Readable)
               && bytes.Allow(stamped.Fingerprint)
               && variables.TryGet<string>(StepKernel.HandOver, out var handed)
               && handed == stamped.Envelope
            ? stamped.Envelope
            : assembled.Readable.ToJson();
    }

    /// <summary>
    /// Hands C# cells what a run of the whole pipeline learned, beside its declaration — only when the run is of the
    /// steps the blocks declare now, since blocks can change while a run is on its way.
    /// </summary>
    /// <param name="variables">The notebook's variables.</param>
    /// <param name="prepared">The run.</param>
    /// <param name="fingerprint">The fingerprint of the bytes it read.</param>
    /// <returns><see langword="true"/> when it was handed over.</returns>
    public bool HandOverFit(IVariableStore variables, PreparedData prepared, string fingerprint)
    {
        ArgumentNullException.ThrowIfNull(variables);
        ArgumentNullException.ThrowIfNull(prepared);

        var envelope = prepared.ToJson();

        lock (_lock)
        {
            if (_assembled?.Readable.Equals(prepared.Declaration) != true)
            {
                return false;
            }

            _run = new RunStamp(KeyOf(prepared.Declaration), fingerprint, envelope);
            _runsFitted++;
        }

        variables.Set(StepKernel.HandOver, envelope);

        return true;
    }

    /// <summary>
    /// Hands over again what the last run learned, when it was a run of these steps over these bytes: a run asked
    /// for again then runs nothing.
    /// </summary>
    /// <param name="variables">The notebook's variables.</param>
    /// <param name="declaration">The steps asked to run.</param>
    /// <param name="fingerprint">The fingerprint of the bytes they would read.</param>
    /// <returns><see langword="true"/> when the last run was that run, and is handed over again.</returns>
    public bool HandOverFitAgain(IVariableStore variables, PipelineDeclaration declaration, string fingerprint)
    {
        ArgumentNullException.ThrowIfNull(variables);
        ArgumentNullException.ThrowIfNull(declaration);

        RunStamp? run;

        lock (_lock)
        {
            run = _run;
        }

        if (run is not { } stamped || stamped.Fingerprint != fingerprint || stamped.Key != KeyOf(declaration))
        {
            return false;
        }

        variables.Set(StepKernel.HandOver, stamped.Envelope);

        return true;
    }

    /// <summary>
    /// Takes back what was handed to C# cells: the blocks changed where no gesture saw it, so what was handed may no
    /// longer be what they make.
    /// </summary>
    /// <param name="variables">The notebook's variables, as the caller was handed them.</param>
    public static void Withdraw(IVariableStore variables)
    {
        ArgumentNullException.ThrowIfNull(variables);

        variables.Remove(StepKernel.HandOver);
    }

    /// <summary>Hands C# cells the folder a relative path in the pipeline is read from, or takes it back.</summary>
    /// <param name="variables">The notebook's variables.</param>
    /// <param name="folder">The notebook's folder, or nothing for a notebook never saved, which reads from the working directory.</param>
    public static void HandOverFolder(IVariableStore variables, string? folder)
    {
        ArgumentNullException.ThrowIfNull(variables);

        if (folder is null)
        {
            variables.Remove(StepKernel.Folder);
        }
        else
        {
            variables.Set(StepKernel.Folder, folder);
        }
    }

    /// <summary>Remembers why a change made in a block's form was not made, while its text stays as it was.</summary>
    /// <param name="cell">The block's cell.</param>
    /// <param name="source">The block's text when the change was refused.</param>
    /// <param name="why">Why.</param>
    public void Refused(Guid cell, string source, string why)
    {
        lock (_lock)
        {
            _refusals[cell] = new Refusal(source, why);
        }
    }

    /// <summary>Forgets a refusal: a change to the block's form was made, or asked for what already was.</summary>
    /// <param name="cell">The block's cell.</param>
    public void Accepted(Guid cell)
    {
        lock (_lock)
        {
            _refusals.Remove(cell);
        }
    }

    /// <summary>Why the last change to a block's form was not made, while the block's text is what it was then.</summary>
    /// <param name="cell">The block's cell.</param>
    /// <param name="source">The block's text now.</param>
    /// <returns>Why, or nothing when no change was refused or the block has changed since.</returns>
    public string? RefusalFor(Guid cell, string source)
    {
        lock (_lock)
        {
            return _refusals.TryGetValue(cell, out var refusal) && refusal.Source == source ? refusal.Why : null;
        }
    }

    /// <summary>Asks a block to show something the next time it runs.</summary>
    /// <param name="cell">The block's cell.</param>
    /// <param name="request">What to show.</param>
    public void Request(Guid cell, ViewRequest request)
    {
        lock (_lock)
        {
            _requests[cell] = request;
        }
    }

    /// <summary>Takes what a block was asked to show, once.</summary>
    /// <param name="cell">The block's cell.</param>
    /// <returns>The request, or nothing when the block runs because somebody ran it.</returns>
    public ViewRequest? Take(Guid cell)
    {
        lock (_lock)
        {
            return _requests.Remove(cell, out var request) ? request : null;
        }
    }

    /// <summary>The view kept last, when it is the one these steps work out over these bytes.</summary>
    /// <param name="key">The key of the steps the view is worked out from.</param>
    /// <param name="fingerprint">The fingerprint of the source's bytes.</param>
    /// <returns>The view, or nothing when it has to be worked out.</returns>
    public PipelineView? ViewFor(string key, string fingerprint)
    {
        lock (_lock)
        {
            return _view is { } kept && kept.Key == key && kept.Fingerprint == fingerprint ? kept.View : null;
        }
    }

    /// <summary>Counts a view worked out just now, and keeps it in place of the one kept before.</summary>
    /// <param name="key">The key of the steps it was worked out from.</param>
    /// <param name="fingerprint">The fingerprint of the bytes it was read from.</param>
    /// <param name="view">The view.</param>
    /// <returns>The same view.</returns>
    public PipelineView Keep(string key, string fingerprint, PipelineView view)
    {
        lock (_lock)
        {
            _viewsRun++;
            _view = new KeptView(key, fingerprint, view);

            return view;
        }
    }

    /// <summary>Remembers that a block shows data: what its rows were worked out from, and what its grid offers.</summary>
    /// <param name="cell">The block's cell.</param>
    /// <param name="key">The key of the steps the rows at the block were worked out from.</param>
    /// <param name="header">The columns the grid shows, and what its header offers for each.</param>
    public void Showing(Guid cell, string key, GridHeader header)
    {
        lock (_lock)
        {
            _shown[cell] = new ShownView(key, header);
        }
    }

    /// <summary>Forgets that a block shows data: it shows its card alone, or nothing.</summary>
    /// <param name="cell">The block's cell.</param>
    public void Hidden(Guid cell)
    {
        lock (_lock)
        {
            _shown.Remove(cell);
        }
    }

    /// <summary>
    /// Forgets every view the blocks as they are now no longer show: its rows are worked out from other steps, or its
    /// grid offers what no longer holds — and says which, for the caller to clear.
    /// </summary>
    /// <param name="now">The pipeline the blocks make now.</param>
    /// <param name="except">The block a gesture was made on, which that gesture shows again itself.</param>
    /// <returns>The blocks whose views were forgotten.</returns>
    public IReadOnlyList<Guid> ForgetStale(NotebookPipeline now, Guid? except)
    {
        ArgumentNullException.ThrowIfNull(now);

        lock (_lock)
        {
            Guid[] stale =
            [
                .. _shown
                    .Where(each => each.Key != except
                                   && (now.ViewKeyOf(each.Key) != each.Value.Key || !each.Value.Header.StillHolds(now.Readable)))
                    .Select(each => each.Key),
            ];

            foreach (var cell in stale)
            {
                _shown.Remove(cell);
            }

            return stale;
        }
    }

    // The key of a whole declaration: that of its last step, which covers every step above it.
    private static string KeyOf(PipelineDeclaration declaration) =>
        declaration.Steps.Count == 0 ? string.Empty : declaration.KeyAt(declaration.Steps.Count - 1);

    /// <summary>Why a change was not made, and the text it was refused on.</summary>
    /// <param name="Source">The block's text.</param>
    /// <param name="Why">Why.</param>
    private sealed record Refusal(string Source, string Why);

    /// <summary>A view, and what it was worked out from.</summary>
    /// <param name="Key">The key of the steps.</param>
    /// <param name="Fingerprint">The fingerprint of the source's bytes.</param>
    /// <param name="View">The rows at the block, and where each stands.</param>
    private sealed record KeptView(string Key, string Fingerprint, PipelineView View);

    /// <summary>What a block shows: the key its rows were worked out under, and what its grid offered.</summary>
    /// <param name="Key">The view's key.</param>
    /// <param name="Header">The grid's header.</param>
    private sealed record ShownView(string Key, GridHeader Header);

    /// <summary>What the notebook's own last run learned, and from which steps and bytes.</summary>
    /// <param name="Key">The key of the steps it ran.</param>
    /// <param name="Fingerprint">The fingerprint of the bytes it read.</param>
    /// <param name="Envelope">The pipeline it handed over: the declaration with what it learned.</param>
    private sealed record RunStamp(string Key, string Fingerprint, string Envelope);
}
