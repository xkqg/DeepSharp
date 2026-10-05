// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Frozen;
using System.Text.Json;
using DeepSharp.Verso.Api;

namespace DeepSharp.Verso.Serve;

/// <summary>
/// What a page asks of a notebook over its socket, by name, and the notebook's verb each ask is: typing, running, a click
/// on a control a cell drew, a cell added, taken away, moved or turned into another kind, what a kernel offers and what a
/// word means, a toolbar button pressed, a cell's panel read or one of its fields changed, an act on what a layout drew,
/// the layout or theme switched, the notebook retitled, a save and a close. Each ask is answered to the page that asked it
/// alone: with what the verb gives back — the cell as it stands, the new cell, what is offered, a file a button or a
/// layout handed over — or with nothing when it gives nothing back.
/// </summary>
/// <remarks>A Stop is not among them: the socket answers it at once, past everything asked before it.</remarks>
internal static class PageAsks
{
    // Every ask a page may make, and what it does to the notebook.
    private static readonly FrozenDictionary<string, Func<Asking, Task<object?>>> Verbs = new Dictionary<string, Func<Asking, Task<object?>>>
    {
        // Typing is sent as it is typed, so an edit is answered with nothing: the text comes with the version, to every page.
        ["edit"] = async asking =>
        {
            await asking.Host.EditAsync(asking.Cell, asking.Asked.Source ?? throw Missing(asking, "text"));

            return null;
        },
        ["run"] = async asking => await asking.Host.RunAsync(asking.Cell),
        ["gesture"] = async asking => await asking.Host.GestureAsync(new HostedGesture(
            asking.Cell,
            asking.Asked.ExtensionId ?? throw Missing(asking, "part"),
            asking.Asked.Action ?? throw Missing(asking, "action"),
            asking.Asked.Payload ?? string.Empty)),

        // A cell added after the one the page names, or at the end when it names none; it starts empty, but for a pipeline
        // block, which starts as the step of the course that belongs there.
        ["add"] = async asking => asking.Asked.After is { } after
            ? await asking.Host.InsertAsync(after, asking.Kind)
            : await asking.Host.AddAsync(asking.Kind),
        ["remove"] = async asking =>
        {
            await asking.Host.RemoveAsync(asking.Cell);

            return null;
        },

        // A move names the one neighbour it passes: the cell above for a move up, the cell below for a move down.
        ["move"] = async asking =>
        {
            switch (asking.Asked)
            {
                case { Before: { } before, After: null }:
                    await asking.Host.MoveBeforeAsync(asking.Cell, before);
                    break;

                case { Before: null, After: { } after }:
                    await asking.Host.MoveAfterAsync(asking.Cell, after);
                    break;

                default:
                    throw new ArgumentException("A move names the one neighbour it passes: the cell it goes before, or the cell it goes after.");
            }

            return null;
        },
        ["kind"] = async asking => await asking.Host.ChangeKindAsync(asking.Cell, asking.Kind),
        ["completions"] = async asking => await asking.Host.CompletionsAsync(asking.Cell, asking.Code, asking.Position),
        ["hover"] = async asking => await asking.Host.HoverAsync(asking.Cell, asking.Code, asking.Position),

        // A file a button hands over goes to the page that pressed it, under its own name, and never beside the notebook.
        ["press"] = async asking => await asking.Host.RunToolbarAsync(asking.Asked.Button ?? throw Missing(asking, "button"), [.. asking.Asked.Cells ?? []]),
        ["properties"] = async asking => await asking.Host.PropertiesAsync(asking.Cell),

        // The value is handed to the part in the form Verso's browser editor hands one on, which each part reads.
        ["property"] = async asking =>
        {
            await asking.Host.SetPropertyAsync(
                asking.Cell, asking.Asked.Part ?? throw Missing(asking, "part"), asking.Asked.Field ?? throw Missing(asking, "field"), Handed(asking.Asked.Value));

            return null;
        },
        // What a person does to the arrangement a layout drew — a tile moved, resized or run — goes to the layout's own part;
        // a file it hands over goes to the page that acted, as a pressed button's does.
        ["interact"] = async asking => await asking.Host.InteractAsync(new HostedLayoutInteraction(
            asking.Asked.Layout ?? throw Missing(asking, "layout"),
            asking.Asked.Action ?? throw Missing(asking, "action"),
            asking.Asked.Payload ?? string.Empty,
            asking.Asked.Target)),

        // As Verso's View panel switches them: by id, the layout the notebook is shown in and the theme it chose.
        ["layout"] = async asking =>
        {
            await asking.Host.SwitchLayoutAsync(asking.Asked.Layout ?? throw Missing(asking, "layout"));

            return null;
        },
        ["theme"] = async asking =>
        {
            await asking.Host.SwitchThemeAsync(asking.Asked.Theme ?? throw Missing(asking, "theme"));

            return null;
        },

        // As Verso's Metadata panel retitles it: one with no title takes the title away.
        ["title"] = async asking =>
        {
            await asking.Host.RetitleAsync(asking.Asked.Title);

            return null;
        },
        ["save"] = async asking =>
        {
            await asking.Host.SaveAsync();

            return null;
        },

        // Closes it now, whatever it holds unsaved — what a person asks for who discards the changes; every page's socket
        // on it ends, once every ask made of it is answered.
        ["close"] = async asking =>
        {
            await asking.Notebooks.CloseAsync(asking.Host);

            return null;
        },
    }.ToFrozenDictionary(StringComparer.Ordinal);

    /// <summary>Makes an ask of a notebook: what it does to the notebook begins now, its answer comes when it is done.</summary>
    /// <param name="asked">The ask, as the page sent it.</param>
    /// <param name="host">The notebook.</param>
    /// <param name="notebooks">The notebooks the server holds open, which close one.</param>
    /// <returns>What the page is answered: what the verb gave back, or nothing.</returns>
    /// <exception cref="ArgumentException">There is no such ask, or it lacks what it names.</exception>
    public static Task<object?> Make(Asked asked, NotebookHost host, OpenNotebooks notebooks) =>
        Verbs.TryGetValue(asked.Ask ?? string.Empty, out var verb)
            ? verb(new Asking(asked, host, notebooks))
            : throw new ArgumentException($"A page asks nothing called '{asked.Ask}' of a notebook.");

    private static ArgumentException Missing(Asking asking, string what) => new($"'{asking.Asked.Ask}' names no {what}.");

    // A field's value as Verso's browser editor hands one on to a part, read from the JSON a page sent it in: a number as a
    // double, a word as a string, a switch as a bool, several choices as a list of words; anything else as the JSON it came
    // as. Nothing comes as nothing — the JSON reader hands a null over as none — and is handed on as nothing.
    private static object? Handed(object? value) => value is not JsonElement json ? value : json.ValueKind switch
    {
        JsonValueKind.Number => json.GetDouble(),
        JsonValueKind.String => json.GetString(),
        JsonValueKind.True => true,
        JsonValueKind.False => false,
        JsonValueKind.Array => json.EnumerateArray().Select(each => each.ValueKind == JsonValueKind.String ? each.GetString()! : each.GetRawText()).ToList(),
        _ => json,
    };

    /// <summary>An ask as a page sends it: its id, which it is, and whatever it names.</summary>
    /// <param name="Id">What the page knows its answer by.</param>
    /// <param name="Ask">Which ask it is.</param>
    /// <param name="Cell">The cell it is about.</param>
    /// <param name="Source">A cell's new text.</param>
    /// <param name="Run">The run a stop means, by its number.</param>
    /// <param name="ExtensionId">The part a click goes to.</param>
    /// <param name="Action">What the click is.</param>
    /// <param name="Payload">What the click carries.</param>
    /// <param name="After">The cell a new one follows, or a moved one goes after.</param>
    /// <param name="Before">The cell a moved one goes before.</param>
    /// <param name="Type">A kind's type.</param>
    /// <param name="Language">A kind's language.</param>
    /// <param name="Code">A cell's text as the person has it.</param>
    /// <param name="Position">Where the cursor stands in it.</param>
    /// <param name="Button">The toolbar button pressed.</param>
    /// <param name="Cells">The cells a button is pressed for.</param>
    /// <param name="Part">The part a panel field's section came from.</param>
    /// <param name="Field">The panel field.</param>
    /// <param name="Value">What the field now holds, as the page sent it in JSON; the part is handed it as Verso's browser editor hands a value on.</param>
    /// <param name="Layout">The layout the notebook is to be shown in, or the one whose arrangement was acted on.</param>
    /// <param name="Theme">The theme the notebook is to be drawn in.</param>
    /// <param name="Target">What an act on a layout's arrangement is aimed at.</param>
    /// <param name="Title">The notebook's new title, or none to take it away.</param>
    internal readonly record struct Asked(
        long Id,
        string? Ask,
        Guid? Cell,
        string? Source,
        long? Run,
        string? ExtensionId,
        string? Action,
        string? Payload,
        Guid? After,
        Guid? Before,
        string? Type,
        string? Language,
        string? Code,
        int? Position,
        string? Button,
        IReadOnlyList<Guid>? Cells,
        string? Part,
        string? Field,
        object? Value,
        string? Layout,
        string? Theme,
        string? Target,
        string? Title);

    // An ask on its way to a notebook: what the page sent, and the notebook, read the way each verb needs it.
    private readonly record struct Asking(Asked Asked, NotebookHost Host, OpenNotebooks Notebooks)
    {
        public Guid Cell => Asked.Cell ?? throw Missing(this, "cell");

        public string Code => Asked.Code ?? throw Missing(this, "text");

        public int Position => Asked.Position ?? throw Missing(this, "position");

        // A kind as a page names it, by its type and language: the host finds it among the notebook's kinds by those alone.
        public HostedKind Kind => new(Asked.Type ?? throw Missing(this, "kind"), Asked.Language, Asked.Type, Editable: true, Rendered: false);
    }
}
