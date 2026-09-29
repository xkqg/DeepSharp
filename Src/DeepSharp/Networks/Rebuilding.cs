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

    /// <summary>A list of numbers a kind was written with.</summary>
    /// <param name="settings">The object the kind was written as.</param>
    /// <param name="key">The setting.</param>
    /// <returns>Its values.</returns>
    /// <exception cref="FormatException">The setting is missing, or not a list of finite numbers.</exception>
    public IReadOnlyList<double> Numbers(JsonElement settings, string key) =>
        [.. Listed(settings, key, "a list of numbers").Select(element => element.AsNumber() ?? throw Refusal(key, "a list of numbers"))];

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
            var rebuilt = _catalog.Rebuild(kind.GetString()!, package, role, element, this);

            foreach (var unread in element.EnumerateObject().Select(property => property.Name)
                         .Where(name => name is not (NetworkDocument.KindKey or NetworkDocument.PackageKey) && !reading.Read.Contains(name)))
            {
                _text.Fault([.. path, unread], $"'{unread}' is not a setting of '{kind.GetString()}'.");
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
