// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Notebooks.Verso;

/// <summary>What kind of place a cursor stands at in the text of a step.</summary>
internal enum Spot
{
    /// <summary>Nowhere an editor has anything to offer.</summary>
    Nowhere,

    /// <summary>Inside the quotes of a key at the top of the step.</summary>
    InKey,

    /// <summary>Where a key at the top of the step goes: after the opening brace or a comma.</summary>
    BeforeKey,

    /// <summary>Inside the quotes of a value at the top of the step.</summary>
    InValue,

    /// <summary>Where a value at the top of the step goes: after a colon.</summary>
    BeforeValue,

    /// <summary>Inside the quotes of an item of a list that is a value at the top of the step.</summary>
    InItem,
}

/// <summary>Where a cursor stands in the text of a step, as far as an editor needs to know.</summary>
/// <param name="Spot">The kind of place.</param>
/// <param name="Key">The key whose value, or whose list, the cursor is in; nothing where there is none.</param>
/// <param name="Typed">What has been typed of the word the cursor is in.</param>
internal readonly record struct CursorPlace(Spot Spot, string? Key, string Typed);

/// <summary>A piece of text between quotes, and where in the step it stands.</summary>
/// <param name="Start">Where its opening quote is.</param>
/// <param name="End">Where its closing quote is.</param>
/// <param name="Text">What it says, as written.</param>
/// <param name="IsKey">Whether it is a key rather than a value.</param>
/// <param name="Depth">How many objects and lists it stands inside.</param>
/// <param name="Key">For a value, the key it belongs to: its own, or that of the list it is an item of.</param>
internal readonly record struct Quoted(int Start, int End, string Text, bool IsKey, int Depth, string? Key);

/// <summary>
/// The text of one step, read the way an editor needs it while it is still being typed.
/// </summary>
/// <remarks>
/// Half a step is not JSON, so it cannot be parsed; it is walked character by character instead, keeping only what
/// an editor asks: which verb the step names, which keys it already holds, and what the cursor stands in. Nothing
/// here decides whether the step is right — the catalog does that, once the text is a step.
/// </remarks>
internal sealed class StepText
{
    private readonly string _text;

    private StepText(string text) => _text = text;

    /// <summary>The text of a step.</summary>
    /// <param name="text">The text, complete or not.</param>
    /// <returns>The step's text.</returns>
    public static StepText Of(string text) => new(text);

    /// <summary>The verb the step is written under, once the text names it in full.</summary>
    public string? Verb =>
        Walk(_text.Length).Quoted
            .Where(quoted => quoted is { Depth: 1, IsKey: false } && quoted.Key == StepCatalog.StepKey)
            .Select(quoted => quoted.Text)
            .FirstOrDefault();

    /// <summary>The keys the step already holds, at its top.</summary>
    public IReadOnlySet<string> Keys =>
        Walk(_text.Length).Quoted.Where(quoted => quoted is { Depth: 1, IsKey: true }).Select(quoted => quoted.Text)
            .ToHashSet(StringComparer.Ordinal);

    /// <summary>What the cursor stands in.</summary>
    /// <param name="cursor">The cursor's place, counting characters from nought.</param>
    /// <returns>The place; nowhere when the cursor is outside the text or where nothing can be offered.</returns>
    public CursorPlace PlaceOf(int cursor)
    {
        if (cursor < 0 || cursor > _text.Length)
        {
            return new CursorPlace(Spot.Nowhere, null, string.Empty);
        }

        var walked = Walk(cursor);
        var depth = walked.Open.Count;
        var top = depth > 0 ? walked.Open[^1] : null;

        if (walked.InQuotes)
        {
            var typed = _text[(walked.QuoteStart + 1)..cursor];

            return (depth, top) switch
            {
                (1, { IsObject: true }) when walked.QuoteIsKey => new CursorPlace(Spot.InKey, null, typed),
                (1, { IsObject: true }) => new CursorPlace(Spot.InValue, top.Key, typed),
                (2, { IsObject: false }) => new CursorPlace(Spot.InItem, top.Owner, typed),
                _ => new CursorPlace(Spot.Nowhere, null, string.Empty),
            };
        }

        // Outside the quotes only where nothing has been typed yet since the brace, the comma or the colon.
        var before = _text[..cursor].TrimEnd();
        var last = before.Length > 0 ? before[^1] : '\0';

        return (depth, top, last) switch
        {
            (1, { IsObject: true, ExpectsKey: true }, '{' or ',') => new CursorPlace(Spot.BeforeKey, null, string.Empty),
            (1, { IsObject: true, ExpectsKey: false }, ':') => new CursorPlace(Spot.BeforeValue, top.Key, string.Empty),
            _ => new CursorPlace(Spot.Nowhere, null, string.Empty),
        };
    }

    /// <summary>The items a list already holds under a key, leaving out the one the cursor stands in.</summary>
    /// <param name="key">The list's key.</param>
    /// <param name="cursor">The cursor's place, counting characters from nought.</param>
    /// <returns>The items, as written.</returns>
    public IReadOnlyList<string> ItemsOf(string key, int cursor) =>
        [.. Walk(_text.Length).Quoted
            .Where(quoted => quoted is { Depth: 2, IsKey: false } && quoted.Key == key && !(cursor > quoted.Start && cursor <= quoted.End))
            .Select(quoted => quoted.Text)];

    /// <summary>The text between quotes that the cursor stands in, when it stands in one.</summary>
    /// <param name="cursor">The cursor's place, counting characters from nought.</param>
    /// <returns>The quoted text, or nothing.</returns>
    public Quoted? QuotedAt(int cursor) =>
        Walk(_text.Length).Quoted.Where(quoted => cursor > quoted.Start && cursor <= quoted.End)
            .Select(quoted => (Quoted?)quoted)
            .FirstOrDefault();

    private Walked Walk(int until)
    {
        var quoted = new List<Quoted>();
        var open = new List<Container>();
        var inQuotes = false;
        var escaped = false;
        var start = 0;
        var isKey = false;

        for (var at = 0; at < until; at++)
        {
            var character = _text[at];
            var top = open.Count > 0 ? open[^1] : null;

            if (inQuotes)
            {
                if (escaped)
                {
                    escaped = false;
                }
                else if (character == '\\')
                {
                    escaped = true;
                }
                else if (character == '"')
                {
                    inQuotes = false;

                    var text = _text[(start + 1)..at];
                    quoted.Add(new Quoted(start, at, text, isKey, open.Count, isKey ? null : top?.Owning));

                    if (isKey)
                    {
                        top!.Pending = text;
                    }
                }

                continue;
            }

            switch (character)
            {
                case '"':
                    inQuotes = true;
                    start = at;
                    isKey = top is { IsObject: true, ExpectsKey: true };
                    break;

                case '{' or '[':
                    open.Add(new Container(character == '{', top?.Owning));
                    break;

                case '}' or ']' when top is not null:
                    open.RemoveAt(open.Count - 1);
                    break;

                case ':' when top is { IsObject: true }:
                    top.Key = top.Pending;
                    top.ExpectsKey = false;
                    break;

                case ',' when top is { IsObject: true }:
                    top.Key = null;
                    top.ExpectsKey = true;
                    break;
            }
        }

        return new Walked(quoted, open, inQuotes, start, isKey);
    }

    /// <summary>What walking the text up to a place found.</summary>
    private readonly record struct Walked(
        IReadOnlyList<Quoted> Quoted, IReadOnlyList<Container> Open, bool InQuotes, int QuoteStart, bool QuoteIsKey);

    /// <summary>An object or a list that is open where the walk stands.</summary>
    private sealed class Container(bool isObject, string? owner)
    {
        /// <summary>Whether it is an object rather than a list.</summary>
        public bool IsObject { get; } = isObject;

        /// <summary>The key the container is the value of.</summary>
        public string? Owner { get; } = owner;

        /// <summary>For an object, whether a key comes next rather than a value.</summary>
        public bool ExpectsKey { get; set; } = isObject;

        /// <summary>For an object, the key whose value comes next.</summary>
        public string? Key { get; set; }

        /// <summary>For an object, the key read last, waiting for its colon.</summary>
        public string? Pending { get; set; }

        /// <summary>The key a value in this container belongs to: an object's current key, or the list's own.</summary>
        public string? Owning => IsObject ? Key : Owner;
    }
}
