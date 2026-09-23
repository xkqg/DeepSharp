// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

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
/// Public because a package that adds a verb reads its parameters the same way, and a reader that threw
/// three other exception types instead would make the file-loading boundary mean something different
/// depending on which package wrote the step.
/// </para>
/// </remarks>
public static class JsonElementExtensions
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

    /// <summary>One of a named set of words, or a fault listing the set.</summary>
    /// <typeparam name="TEnum">The set of words.</typeparam>
    /// <param name="element">The object to read from.</param>
    /// <param name="name">The property's name.</param>
    /// <returns>The value the word stands for.</returns>
    /// <exception cref="FormatException">The property is absent, or is a word nobody defined.</exception>
    public static TEnum RequiredEnum<TEnum>(this JsonElement element, string name)
        where TEnum : struct, Enum
    {
        var written = element.RequiredString(name);

        if (!Enum.TryParse<TEnum>(written, ignoreCase: true, out var value) || !Enum.IsDefined(value))
        {
            throw new FormatException(
                $"'{written}' is not one of the things '{name}' can be: "
                + string.Join(", ", Enum.GetNames<TEnum>().Select(each => each.ToLowerInvariant())) + ".");
        }

        return value;
    }

    /// <summary>The number of a named property, or a fault naming it.</summary>
    /// <param name="element">The object to read from.</param>
    /// <param name="name">The property's name.</param>
    /// <returns>The number.</returns>
    /// <exception cref="FormatException">The property is absent or is not a number.</exception>
    public static double RequiredNumber(this JsonElement element, string name)
    {
        if (!element.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.Number)
        {
            throw new FormatException($"The step is missing a number for '{name}'.");
        }

        return value.GetDouble();
    }

    /// <summary>The number of a named property, or a given value where the file does not mention it.</summary>
    /// <param name="element">The object to read from.</param>
    /// <param name="name">The property's name.</param>
    /// <param name="whenAbsent">What the value is when the file says nothing about it.</param>
    /// <returns>The number, or <paramref name="whenAbsent"/>.</returns>
    /// <exception cref="FormatException">The property is there but is not a number.</exception>
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

        return value.GetDouble();
    }
}
