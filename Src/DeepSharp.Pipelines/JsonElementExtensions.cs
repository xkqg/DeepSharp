// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;

namespace DeepSharp.Pipelines;

/// <summary>
/// Reading a declared value out of a file, where absent and wrongly typed are both faults of the file.
/// </summary>
/// <remarks>
/// The BCL throws a different exception for each of those, and neither says which step or which parameter
/// was at fault. A person editing a pipeline by hand has to be told where to look, so both become one
/// <see cref="FormatException"/> that names the parameter.
/// <para>
/// The inside of the parameter kinds, and nothing more: a package that adds a verb builds its parameters
/// from the kinds and never reads a key by hand, so the file-loading boundary means one thing whichever
/// package wrote the step.
/// </para>
/// </remarks>
internal static class JsonElementExtensions
{
    /// <summary>The text value of a named property, or a fault naming it.</summary>
    /// <param name="element">The object to read from.</param>
    /// <param name="name">The property's name.</param>
    /// <returns>The text.</returns>
    /// <exception cref="FormatException">The property is absent or is not text.</exception>
    public static string RequiredString(this JsonElement element, string name)
    {
        if (!element.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.String)
        {
            throw new FormatException($"The step is missing a text value for '{name}'.");
        }

        return value.GetString()!;
    }

    /// <summary>The true or false of a named property, or a fault naming it.</summary>
    /// <param name="element">The object to read from.</param>
    /// <param name="name">The property's name.</param>
    /// <returns>The value.</returns>
    /// <exception cref="FormatException">The property is absent or is not true or false.</exception>
    public static bool RequiredBoolean(this JsonElement element, string name)
    {
        if (!element.TryGetProperty(name, out var value)
            || value.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
        {
            throw new FormatException($"The step is missing a true or false for '{name}'.");
        }

        return value.GetBoolean();
    }

    /// <summary>The number of a named property, or a fault naming it.</summary>
    /// <param name="element">The object to read from.</param>
    /// <param name="name">The property's name.</param>
    /// <returns>The number.</returns>
    /// <exception cref="FormatException">The property is absent, is not a number, or is too large to hold.</exception>
    public static double RequiredNumber(this JsonElement element, string name)
    {
        if (!element.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.Number)
        {
            throw new FormatException($"The step is missing a number for '{name}'.");
        }

        return Held(value, name);
    }

    /// <summary>The whole number of a named property, or a fault naming it.</summary>
    /// <param name="element">The object to read from.</param>
    /// <param name="name">The property's name.</param>
    /// <returns>The number.</returns>
    /// <exception cref="FormatException">The property is absent, has a fraction, or is too large to hold.</exception>
    /// <remarks>
    /// Read as a whole number rather than as a number and then cut: a seed of 3.5 used to become 3 and one
    /// of ten billion the largest value that fits, so the file said one thing and the run did another. A
    /// whole number written with a fraction of nought, <c>3.0</c>, is that whole number — nothing is lost,
    /// and it is what a JSON Schema calls an integer too.
    /// </remarks>
    public static int RequiredWholeNumber(this JsonElement element, string name)
    {
        if (!element.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.Number)
        {
            throw new FormatException($"The step is missing a whole number for '{name}'.");
        }

        return value.TryGetWholeNumber(out var whole)
            ? whole
            : throw new FormatException(string.Create(
                CultureInfo.InvariantCulture,
                $"The step's '{name}' is {value.GetRawText()}, which is not a whole number between {int.MinValue} and {int.MaxValue}."));
    }

    /// <summary>A value as a whole number, when it is one: the one rule for a whole number anywhere in a file.</summary>
    /// <param name="value">The value.</param>
    /// <param name="whole">The whole number, when the value is one.</param>
    /// <returns><see langword="true"/> when the value is a number with no fraction that fits in an <see cref="int"/>.</returns>
    public static bool TryGetWholeNumber(this JsonElement value, out int whole)
    {
        if (value.ValueKind == JsonValueKind.Number)
        {
            if (value.TryGetInt32(out whole))
            {
                return true;
            }

            if (value.TryGetDouble(out var number) && number == Math.Floor(number) && number is >= int.MinValue and <= int.MaxValue)
            {
                whole = (int)number;

                return true;
            }
        }

        whole = 0;

        return false;
    }

    /// <summary>The number of a named property, or a given value where the file does not mention it.</summary>
    /// <param name="element">The object to read from.</param>
    /// <param name="name">The property's name.</param>
    /// <param name="whenAbsent">What the value is when the file says nothing about it.</param>
    /// <returns>The number, or <paramref name="whenAbsent"/>.</returns>
    /// <exception cref="FormatException">The property is there but is not a number it can hold.</exception>
    /// <remarks>
    /// For a value whose absence is an ordinary thing to say rather than an omission — a share of the data
    /// that is simply not held back. A property that is present but is not a number is still a fault,
    /// because that is a person having meant something the file cannot express.
    /// </remarks>
    public static double OptionalNumber(this JsonElement element, string name, double whenAbsent = 0)
    {
        if (!element.TryGetProperty(name, out var value))
        {
            return whenAbsent;
        }

        if (value.ValueKind != JsonValueKind.Number)
        {
            throw new FormatException($"The step has something other than a number for '{name}'.");
        }

        return Held(value, name);
    }

    /// <summary>A number as a double, or a fault when it is larger than a double holds.</summary>
    private static double Held(JsonElement value, string name) =>
        value.TryGetDouble(out var number) && double.IsFinite(number)
            ? number
            : throw new FormatException(
                $"The step's '{name}' is {value.GetRawText()}, which is a number too large to hold.");
}
