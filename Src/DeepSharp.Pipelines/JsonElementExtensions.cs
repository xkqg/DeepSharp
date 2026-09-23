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
/// </remarks>
internal static class JsonElementExtensions
{
    internal static string RequiredString(this JsonElement element, string name)
    {
        if (!element.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.String)
        {
            throw new FormatException($"The step is missing a text value for '{name}'.");
        }

        return value.GetString()!;
    }

    internal static bool RequiredBoolean(this JsonElement element, string name)
    {
        if (!element.TryGetProperty(name, out var value)
            || value.ValueKind is not (JsonValueKind.True or JsonValueKind.False))
        {
            throw new FormatException($"The step is missing a true or false for '{name}'.");
        }

        return value.GetBoolean();
    }

    internal static TEnum RequiredEnum<TEnum>(this JsonElement element, string name)
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

    internal static double RequiredNumber(this JsonElement element, string name)
    {
        if (!element.TryGetProperty(name, out var value) || value.ValueKind != JsonValueKind.Number)
        {
            throw new FormatException($"The step is missing a number for '{name}'.");
        }

        return value.GetDouble();
    }
}
