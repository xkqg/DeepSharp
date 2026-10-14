// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Networks;

/// <summary>
/// Something a network file names and rebuilds: a layer, a network written as code, a loss, an optimizer, a learning-rate
/// schedule.
/// </summary>
/// <remarks>
/// A file holds what a thing is rebuilt from — its kind and its settings — and never a number it learned, which the slots
/// carry by their paths. A kind is known to a reader only when it is registered with the catalog that reads, so a file
/// naming one nobody registered is refused rather than read as something else.
/// </remarks>
public interface ISaved
{
    /// <summary>The name its kind is registered under.</summary>
    string Kind { get; }

    /// <summary>Writes the settings it is rebuilt from, as properties of the object it is written as.</summary>
    /// <param name="writer">The writer, inside that object, after its kind.</param>
    void WriteSettings(Utf8JsonWriter writer);

    /// <summary>
    /// The oldest version of the part it is written into that can say what this one holds: the second, unless it says
    /// something a library of that version would not understand.
    /// </summary>
    /// <remarks>
    /// A network's part is written as the oldest version that says everything in it, which is the largest of these over the
    /// layers it holds; a checkpoint's training part, the largest of these over its optimizer and its schedule, and what it
    /// records of the run. A kind says the version per instance rather than per kind, so a convolution whose window is not
    /// dilated is still written as the second version and read by every library that could read it before — and one that is
    /// dilated names the version it needs, so an older library says which version wrote the part rather than which key it
    /// does not know. An optimizer or a schedule written outside this library says it the same way.
    /// </remarks>
    int FileVersion => NetworkDocument.SecondVersion;
}

/// <summary>A kind a network file names and rebuilds, registered under its name.</summary>
/// <typeparam name="TSelf">The kind itself.</typeparam>
public interface ISaved<TSelf> : ISaved
    where TSelf : ISaved<TSelf>
{
    /// <summary>The name the kind is registered under: one word, as a file writes it.</summary>
    static abstract string Name { get; }

    /// <inheritdoc />
    string ISaved.Kind => TSelf.Name;

    /// <summary>Rebuilds one from the settings it wrote.</summary>
    /// <param name="settings">The object it was written as.</param>
    /// <param name="rebuilding">Reads the settings in the file's own words, and rebuilds the layers it holds.</param>
    /// <returns>The rebuilt one, its numbers still to be read into its slots.</returns>
    static abstract TSelf Rebuild(JsonElement settings, Rebuilding rebuilding);
}
