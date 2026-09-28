// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Threading.Channels;

namespace DeepSharp.Verso.Api;

/// <summary>
/// A view of an open notebook: the notebook as it stood when the view began, and then each change after it. A view that
/// reads slower than the notebook changes is never waited for; what it has not read yet is kept as one change, the
/// latest look at each cell, so it never holds more than the notebook itself.
/// </summary>
public sealed class NotebookSubscription : IDisposable
{
    private readonly NotebookHost _host;

    // The one change waiting for the reader. Only whoever holds the notebook's turn writes here, taking the change that
    // waits and writing it back together with the new one, so the notebook never waits on a reader.
    private readonly Channel<NotebookChange> _waiting = Channel.CreateBounded<NotebookChange>(1);

    internal NotebookSubscription(NotebookHost host) => _host = host;

    /// <summary>The notebook as it stood when the view began.</summary>
    public NotebookVersion Snapshot { get; internal set; }

    /// <summary>Each change after <see cref="Snapshot"/>, in order, until the view ends or the notebook closes.</summary>
    /// <param name="cancellationToken">Stops the reading; the view goes on.</param>
    /// <returns>The changes.</returns>
    /// <remarks>
    /// The first may repeat part of the snapshot, when the notebook changed just as the view began; applying it changes
    /// nothing, since a change holds each cell as it stands.
    /// </remarks>
    public IAsyncEnumerable<NotebookChange> ReadAllAsync(CancellationToken cancellationToken = default) => _waiting.Reader.ReadAllAsync(cancellationToken);

    /// <summary>Takes the change waiting for the reader, when one does; never waits.</summary>
    /// <param name="change">The change: every change since the last one taken, taken as one.</param>
    /// <returns>Whether one waited.</returns>
    /// <remarks>
    /// For a reader that tells the changes in turn with messages of its own: a change the notebook made before one of
    /// those messages is waiting by the time the message is, so taking it first keeps the two in the order they came.
    /// </remarks>
    public bool TryRead(out NotebookChange change) => _waiting.Reader.TryRead(out change);

    /// <summary>Waits until a change waits for the reader, or the view has ended.</summary>
    /// <param name="cancellationToken">Stops the waiting; the view goes on.</param>
    /// <returns>Whether a change waits; nothing will once the view has ended and its last change was taken.</returns>
    public ValueTask<bool> WaitToReadAsync(CancellationToken cancellationToken = default) => _waiting.Reader.WaitToReadAsync(cancellationToken);

    /// <summary>Ends the view: it is told nothing more.</summary>
    public void Dispose()
    {
        _host.Leave(this);
        End();
    }

    /// <summary>Hands the view a change, together with the one it has not read yet.</summary>
    /// <param name="change">The change.</param>
    internal void Offer(NotebookChange change) => _waiting.Writer.TryWrite(_waiting.Reader.TryRead(out var waiting) ? waiting.Then(change) : change);

    /// <summary>Tells the view nothing more; what it has not read yet it still reads.</summary>
    internal void End() => _waiting.Writer.TryComplete();
}
