// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using Verso.Abstractions;

namespace DeepSharp.Tests.Serve.Parts;

/// <summary>
/// A part whose fields hold what a notebook's file can hold — a switch written as a word, several choices written as one
/// line of words, one of them not among the choices — and which writes down, for each field changed, the form of the value
/// it was handed: a page's panel is seen reading its fields and handing their values on as Verso's browser editor does.
/// </summary>
[VersoExtension]
public sealed class ReadingPart : ICellPropertyProvider
{
    /// <summary>The part's id.</summary>
    public const string Part = "deepsharp.tests.reading-part";

    /// <summary>What a cell carries that the part draws its fields for.</summary>
    public const string Marked = "test:reading";

    /// <summary>Where the part writes down, by the field's name after it, what it was handed.</summary>
    public const string Got = "test:got:";

    /// <inheritdoc />
    public string ExtensionId => Part;

    /// <inheritdoc />
    public string Name => "Reads its fields";

    /// <inheritdoc />
    public string Version => "1.0.0";

    /// <inheritdoc />
    public string? Author => null;

    /// <inheritdoc />
    public string? Description => null;

    /// <inheritdoc />
    public int Order => 100;

    /// <inheritdoc />
    public Task OnLoadedAsync(IExtensionHostContext context) => Task.CompletedTask;

    /// <inheritdoc />
    public Task OnUnloadedAsync() => Task.CompletedTask;

    /// <inheritdoc />
    public bool AppliesTo(CellModel cell, ICellRenderContext context) => cell.Metadata.ContainsKey(Marked);

    /// <inheritdoc />
    public Task<PropertySection> GetPropertiesSectionAsync(CellModel cell, ICellRenderContext context) => Task.FromResult(new PropertySection(
        "Reading",
        null,
        [
            new PropertyField("flag", "Flag", PropertyFieldType.Toggle, "True"),
            new PropertyField("choices", "Choices", PropertyFieldType.MultiSelect, "A, zz", Options: [new("a", "A"), new("b", "B"), new("c", "C")]),
            new PropertyField("count", "Count", PropertyFieldType.Number, 2),
        ]));

    /// <inheritdoc />
    public Task OnPropertyChangedAsync(CellModel cell, string propertyName, object? value, ICellRenderContext context)
    {
        cell.Metadata[Got + propertyName] = value is IEnumerable<string> words
            ? $"{value.GetType().Name}:{string.Join('|', words)}"
            : string.Create(CultureInfo.InvariantCulture, $"{value?.GetType().Name}:{value}");

        return Task.CompletedTask;
    }
}
