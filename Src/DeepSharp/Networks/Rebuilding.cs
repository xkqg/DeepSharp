// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Networks;

/// <summary>
/// What a kind is handed as it is rebuilt from a network file: its settings read in the file's own words, the layers it
/// holds rebuilt through the catalog, and draws to start from.
/// </summary>
/// <remarks>
/// A setting the kind does not read is refused once it is rebuilt, and one it reads that is missing or of the wrong kind is
/// refused where it stands; every fault in the file is named at once, each at its line and column.
/// </remarks>
public sealed class Rebuilding
{
    private readonly NetworkCatalog _catalog;
    private readonly NetworkText _text;
    private readonly Stack<Reading> _readings = new();

    internal Rebuilding(NetworkCatalog catalog, NetworkText text)
    {
        _catalog = catalog;
        _text = text;
    }

    /// <summary>Draws a layer being rebuilt can start from: whatever it draws is replaced by the numbers the file holds.</summary>
    public Draws Draws { get; } = new RandomStream(0).Draw("rebuilding", 0, 0);

    /// <summary>A whole number a kind was written with.</summary>
    /// <param name="settings">The object the kind was written as.</param>
    /// <param name="key">The setting.</param>
    /// <returns>Its value.</returns>
    /// <exception cref="FormatException">The setting is missing, or not a whole number.</exception>
    public int Whole(JsonElement settings, string key) =>
        Setting(settings, key, "a whole number").AsWhole() ?? throw Refusal(key, "a whole number");

    /// <summary>A number a kind was written with.</summary>
    /// <param name="settings">The object the kind was written as.</param>
    /// <param name="key">The setting.</param>
    /// <returns>Its value.</returns>
    /// <exception cref="FormatException">The setting is missing, or not a finite number.</exception>
    public double Number(JsonElement settings, string key) =>
        Setting(settings, key, "a number").AsNumber() ?? throw Refusal(key, "a number");

    /// <summary>A list of whole numbers a kind was written with.</summary>
    /// <param name="settings">The object the kind was written as.</param>
    /// <param name="key">The setting.</param>
    /// <returns>Its values.</returns>
    /// <exception cref="FormatException">The setting is missing, or not a list of whole numbers.</exception>
    public IReadOnlyList<int> Wholes(JsonElement settings, string key) =>
        [.. Listed(settings, key, "a list of whole numbers").Select(element => element.AsWhole() ?? throw Refusal(key, "a list of whole numbers"))];

    /// <summary>A whole number a kind was written with, or the one it means by not having been written with it.</summary>
    /// <param name="settings">The object the kind was written as.</param>
    /// <param name="key">The setting.</param>
    /// <param name="whenAbsent">What the kind means where the setting is not written.</param>
    /// <returns>Its value, or <paramref name="whenAbsent"/>.</returns>
    /// <exception cref="FormatException">The setting is written as something that is not a whole number.</exception>
    /// <remarks>
    /// How a setting is added to a kind without asking anything of a file written before it: the file that says nothing of it
    /// means what it always meant, and a kind writes it only when it says something else.
    /// </remarks>
    public int Whole(JsonElement settings, string key, int whenAbsent) =>
        settings.Member(key) is null ? whenAbsent : Whole(settings, key);

    /// <summary>A list of whole numbers a kind was written with, or what it means by not having been written with them.</summary>
    /// <param name="settings">The object the kind was written as.</param>
    /// <param name="key">The setting.</param>
    /// <param name="whenAbsent">What the kind means where the setting is not written.</param>
    /// <returns>Its values, or <paramref name="whenAbsent"/>.</returns>
    /// <exception cref="FormatException">The setting is written as something that is not a list of whole numbers.</exception>
    public IReadOnlyList<int> Wholes(JsonElement settings, string key, IReadOnlyList<int> whenAbsent) =>
        settings.Member(key) is null ? whenAbsent : Wholes(settings, key);

    /// <summary>A number a kind was written with, or the one it means by not having been written with it.</summary>
    /// <param name="settings">The object the kind was written as.</param>
    /// <param name="key">The setting.</param>
    /// <param name="whenAbsent">What the kind means where the setting is not written.</param>
    /// <returns>Its value, or <paramref name="whenAbsent"/>.</returns>
    /// <exception cref="FormatException">The setting is written as something that is not a finite number.</exception>
    public double Number(JsonElement settings, string key, double whenAbsent) =>
        settings.Member(key) is null ? whenAbsent : Number(settings, key);

    /// <summary>A refusal of a setting in the kind's own words, placed at that setting in the file.</summary>
    /// <param name="key">The setting the kind refuses.</param>
    /// <param name="message">What is wrong with it, as the kind says it.</param>
    /// <returns>The refusal, to be thrown.</returns>
    /// <remarks>
    /// For what a kind knows about a setting that the readers here cannot: how many numbers a window's stride holds for the
    /// axes it walks, say. Every fault is still named at the line and column of the setting itself, with the rest.
    /// </remarks>
    public FormatException Refuse(string key, string message) => Placed(key, message);

    /// <summary>Whether a setting is written as a list rather than as the one number or word that stands for every axis.</summary>
    /// <param name="settings">The object the kind was written as.</param>
    /// <param name="key">The setting.</param>
    /// <returns>Whether it is a list.</returns>
    /// <remarks>
    /// A setting that says one thing for every axis and another for each axis apart is one key of two kinds, as a window's
    /// padding is a number or a word. Reading which it is asks nothing of the setting, so a kind that does not read it on
    /// is still named for what it left unread.
    /// </remarks>
    public bool Lists(JsonElement settings, string key) => settings.Member(key) is { ValueKind: JsonValueKind.Array };

    /// <summary>A list of numbers a kind was written with.</summary>
    /// <param name="settings">The object the kind was written as.</param>
    /// <param name="key">The setting.</param>
    /// <returns>Its values.</returns>
    /// <exception cref="FormatException">The setting is missing, or not a list of finite numbers.</exception>
    public IReadOnlyList<double> Numbers(JsonElement settings, string key) =>
        [.. Listed(settings, key, "a list of numbers").Select(element => element.AsNumber() ?? throw Refusal(key, "a list of numbers"))];

    /// <summary>A list of words a kind was written with, or what it means by not having been written with them.</summary>
    /// <param name="settings">The object the kind was written as.</param>
    /// <param name="key">The setting.</param>
    /// <param name="whenAbsent">What the kind means where the setting is not written.</param>
    /// <returns>Its words, or <paramref name="whenAbsent"/>.</returns>
    /// <exception cref="FormatException">The setting is written as something that is not a list of text.</exception>
    public IReadOnlyList<string> Words(JsonElement settings, string key, IReadOnlyList<string> whenAbsent) =>
        settings.Member(key) is null
            ? whenAbsent
            : [.. Listed(settings, key, "a list of words").Select(element => element.ValueKind == JsonValueKind.String ? element.GetString()! : throw Refusal(key, "a list of words"))];

    /// <summary>A list of lists of words a kind was written with, or what it means by not having been written with them.</summary>
    /// <param name="settings">The object the kind was written as.</param>
    /// <param name="key">The setting.</param>
    /// <param name="whenAbsent">What the kind means where the setting is not written.</param>
    /// <returns>Its lists, or <paramref name="whenAbsent"/>.</returns>
    /// <exception cref="FormatException">The setting is written as something that is not a list of lists of text.</exception>
    public IReadOnlyList<IReadOnlyList<string>> WordLists(JsonElement settings, string key, IReadOnlyList<IReadOnlyList<string>> whenAbsent) =>
        settings.Member(key) is null
            ? whenAbsent
            : [.. Listed(settings, key, "a list of lists of words").Select(list => list.ValueKind == JsonValueKind.Array
                ? list.EnumerateArray().Select(element => element.ValueKind == JsonValueKind.String ? element.GetString()! : throw Refusal(key, "a list of lists of words")).ToArray()
                : throw Refusal(key, "a list of lists of words"))];

    /// <summary>The layers a kind holds, written as a list under a setting, each rebuilt through the catalog.</summary>
    /// <param name="settings">The object the kind was written as.</param>
    /// <param name="key">The setting.</param>
    /// <returns>The layers, in their order.</returns>
    /// <exception cref="FormatException">The setting is missing or not a list; a layer that cannot be rebuilt is named at its own place.</exception>
    public IReadOnlyList<Layer> Layers(JsonElement settings, string key)
    {
        var elements = Listed(settings, key, "a list of layers");
        var reading = _readings.Peek();
        var layers = elements.Select((element, place) => Rebuild(element, [.. reading.Path, key, place.ToString(CultureInfo.InvariantCulture)], NetworkCatalog.Role.Layer)).ToArray();

        return layers.Any(layer => layer is null) ? throw new Unreadable() : [.. layers.Cast<Layer>()];
    }

    /// <summary>A learning-rate schedule a kind holds, written as an object under a setting, rebuilt through the catalog.</summary>
    /// <param name="settings">The object the kind was written as.</param>
    /// <param name="key">The setting.</param>
    /// <returns>The schedule.</returns>
    /// <exception cref="FormatException">
    /// The setting is missing; what stands there — no object naming its kind, a kind of another role, a schedule its settings
    /// refuse — is named at its own place.
    /// </exception>
    /// <remarks>A schedule that hands over to another holds it this way, as a stack holds its layers.</remarks>
    public LearningRateSchedule Schedule(JsonElement settings, string key)
    {
        var element = Setting(settings, key, "a learning-rate schedule");
        var reading = _readings.Peek();

        return Rebuild(element, [.. reading.Path, key], NetworkCatalog.Role.Schedule) as LearningRateSchedule ?? throw new Unreadable();
    }

    /// <summary>
    /// Which of the words a kind takes in place of a number a setting is written as — a window's padding as <c>same</c> or
    /// <c>causal</c>, say. A word that is none of them is refused where it stands, in the words that say what the setting is;
    /// written as anything but text, it is not read here, and is left to be read as its number.
    /// </summary>
    /// <returns>The word the setting says; nothing when it is written as something else.</returns>
    internal string? SaysOneOf(JsonElement settings, string key, string what, params ReadOnlySpan<string> words)
    {
        if (settings.Member(key) is not { ValueKind: JsonValueKind.String } written)
        {
            return null;
        }

        _readings.Peek().Read.Add(key);

        foreach (var word in words)
        {
            if (written.GetString() == word)
            {
                return word;
            }
        }

        throw Refusal(key, what);
    }

    /// <summary>Whether a kind was written with a setting that it writes only when it is true: false, where it is left out.</summary>
    /// <param name="settings">The object the kind was written as.</param>
    /// <param name="key">The setting.</param>
    /// <returns>Whether it says true.</returns>
    /// <exception cref="FormatException">The setting is written as anything but true or false.</exception>
    public bool Holds(JsonElement settings, string key)
    {
        if (settings.Member(key) is not { } written)
        {
            return false;
        }

        _readings.Peek().Read.Add(key);

        return written.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => throw Refusal(key, "true or false"),
        };
    }

    /// <summary>Whether a kind was written with a setting that it writes only when it is false: true, where it is left out.</summary>
    /// <param name="settings">The object the kind was written as.</param>
    /// <param name="key">The setting.</param>
    /// <returns>Whether it says true.</returns>
    /// <exception cref="FormatException">The setting is written as anything but true or false.</exception>
    /// <remarks>
    /// The other way a setting is added to a kind without asking anything of a file written before it: a layer writes that it
    /// has no bias, or no scale, and says nothing where it has one, so the file that says nothing means what it always meant.
    /// </remarks>
    public bool Yes(JsonElement settings, string key)
    {
        if (settings.Member(key) is not { } written)
        {
            return true;
        }

        _readings.Peek().Read.Add(key);

        return written.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            _ => throw Refusal(key, "true or false"),
        };
    }

    /// <summary>Rebuilds the kind written at a place of the file, noting every fault at its place; nothing when it cannot be.</summary>
    internal object? Rebuild(JsonElement element, IReadOnlyList<string> path, NetworkCatalog.Role role)
    {
        if (element.Member(NetworkDocument.KindKey) is not { ValueKind: JsonValueKind.String } kind)
        {
            _text.Fault(path, $"Where {role.Named()} stands, an object names its kind under '{NetworkDocument.KindKey}'.");

            return null;
        }

        var package = element.Member(NetworkDocument.PackageKey) is { ValueKind: JsonValueKind.String } from ? from.GetString() : null;
        var reading = new Reading([.. path]);
        _readings.Push(reading);

        try
        {
            var rebuilt = _catalog.Rebuilder(kind.GetString()!, package, role)(element, this);

            foreach (var unread in element.EnumerateObject().Select(property => property.Name)
                         .Where(name => name is not (NetworkDocument.KindKey or NetworkDocument.PackageKey) && !reading.Read.Contains(name)))
            {
                _text.Fault([.. path, unread], $"'{unread.Quoted()}' is not a setting of '{kind.GetString()!.Quoted()}'.");
            }

            return rebuilt;
        }
        catch (Unreadable)
        {
            return null;
        }
        catch (Exception fault) when (fault is FormatException or ArgumentException)
        {
            _text.Fault(reading.Faulted ?? path, fault);

            return null;
        }
        finally
        {
            _readings.Pop();
        }
    }

    // The value of a setting the kind being rebuilt reads, which is then no setting it leaves unread.
    private JsonElement Setting(JsonElement settings, string key, string what)
    {
        _readings.Peek().Read.Add(key);

        return settings.Member(key) ?? throw Placed(key, $"'{key}' is missing here: it is {what}.");
    }

    private JsonElement[] Listed(JsonElement settings, string key, string what) =>
        Setting(settings, key, what) is { ValueKind: JsonValueKind.Array } list ? [.. list.EnumerateArray()] : throw Refusal(key, what);

    // A setting of the wrong kind, placed at the setting itself.
    private FormatException Refusal(string key, string what) => Placed(key, $"'{key}' is {what} here.");

    // A refusal of a setting, placed at the setting — or, when it is missing, at the kind that should have written it.
    private FormatException Placed(string key, string message)
    {
        var reading = _readings.Peek();
        reading.Faulted = [.. reading.Path, key];

        return new FormatException(message);
    }

    // A layer held by the one being rebuilt could not be rebuilt, and has been named at its own place already.
    private sealed class Unreadable : Exception
    {
    }

    // The kind being rebuilt now: where it stands, the settings it read, and the setting a fault is about.
    private sealed class Reading(string[] path)
    {
        public string[] Path { get; } = path;

        public HashSet<string> Read { get; } = new(StringComparer.Ordinal);

        public string[]? Faulted { get; set; }
    }
}
