// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using Verso.Abstractions;

namespace DeepSharp.Tests.Serve.Parts;

/// <summary>
/// A part that answers a click on a control naming it with what the control carried, in brackets, or says it was handed
/// nothing at all: a click is seen reaching a part with the payload Verso hands every part — words, empty when the control
/// carries none.
/// </summary>
[VersoExtension]
public sealed class AnsweringPart : IExtension, ICellInteractionHandler
{
    /// <summary>The part's name, which a control names to reach it.</summary>
    public const string Id = "deepsharp.tests.answering-part";

    /// <inheritdoc />
    public string ExtensionId => Id;

    /// <inheritdoc />
    public string Name => "A part that answers a click with what it carried";

    /// <inheritdoc />
    public string Version => "1.0.0";

    /// <inheritdoc />
    public string? Author => null;

    /// <inheritdoc />
    public string? Description => null;

    /// <inheritdoc />
    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    /// <inheritdoc />
    public Task OnUnloadedAsync() => Task.CompletedTask;

    /// <inheritdoc />
    public Task<string?> OnCellInteractionAsync(CellInteractionContext context)
    {
        ArgumentNullException.ThrowIfNull(context);

        return Task.FromResult<string?>(context.Payload is null ? "handed nothing" : $"[{context.Payload}]");
    }
}
