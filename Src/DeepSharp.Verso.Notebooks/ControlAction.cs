// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using System.Text.Json.Nodes;

namespace DeepSharp.Verso.Notebooks;

/// <summary>What a control on a grid or a card sends: the gesture's name, and what the control is about.</summary>
/// <param name="Gesture">The gesture's name.</param>
/// <param name="Carried">What the control is about, as the JSON it carries.</param>
/// <remarks>
/// A control carries both in its <c>data-action</c> — the name, a space and the JSON — and no <c>data-payload</c>, so
/// Verso's router sends the control's own state with it: for a checkbox, whether it is ticked, on a click, on a change
/// and on every key. A key that changes nothing sends the state the control is in, and asking for what already is
/// changes nothing, so the same state sent twice does it once.
/// </remarks>
internal readonly record struct ControlAction(string Gesture, JsonObject Carried)
{
    /// <summary>The text a control carries in its <c>data-action</c>.</summary>
    /// <param name="gesture">The gesture's name.</param>
    /// <param name="carried">What the control is about.</param>
    /// <returns>The name, a space and the JSON.</returns>
    public static string Of(string gesture, JsonObject carried) => $"{gesture} {carried.ToJsonString()}";

    /// <summary>Reads back the text a control carried.</summary>
    /// <param name="interaction">The text, as the router handed it on.</param>
    /// <returns>The action, or nothing when the text is not a gesture's name followed by a JSON object.</returns>
    public static ControlAction? Read(string interaction)
    {
        var space = interaction.IndexOf(' ', StringComparison.Ordinal);

        if (space <= 0)
        {
            return null;
        }

        try
        {
            return JsonNode.Parse(interaction[(space + 1)..]) is JsonObject carried ? new ControlAction(interaction[..space], carried) : null;
        }
        catch (JsonException)
        {
            return null;
        }
    }

    /// <summary>The text the control carries under a key.</summary>
    /// <param name="key">The key.</param>
    /// <returns>The text, or nothing when the control carries none there.</returns>
    public string? Text(string key) => Carried[key] is JsonValue value && value.TryGetValue<string>(out var text) ? text : null;
}
