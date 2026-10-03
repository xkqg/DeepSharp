// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Verso.Notebooks;

/// <summary>
/// The name a format is given when an extension is asked whether it takes part in reading or writing a notebook.
/// </summary>
/// <remarks>
/// Verso names its own format twice: its writer's name, <c>verso</c>, which DeepSharp's own host and Verso's browser
/// editor hand an extension, and the name its post-processors' contract gives it, <c>verso-native</c>, which Verso's VS
/// Code host hands an extension when it saves a notebook as .verso. Every part that asks treats the two as one.
/// </remarks>
internal static class FormatIdExtensions
{
    extension(string formatId)
    {
        /// <summary>Whether the format is Verso's own, under either of its names.</summary>
        /// <returns>Whether it names Verso's own format.</returns>
        public bool IsVersosOwn() => formatId is "verso" or "verso-native";
    }
}
