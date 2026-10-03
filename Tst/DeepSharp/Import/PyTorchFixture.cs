// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;

namespace DeepSharp.Tests.Import;

/// <summary>
/// What PyTorch saved and answered, kept beside these tests with the scripts that made it (Fixtures/pytorch.py,
/// Fixtures/torch-save.py) and what they printed (Fixtures/pytorch.txt, Fixtures/torch-save.txt): the safetensors files and
/// the files torch.save wrote of the same state, pytorch.json — the rows each network was handed, what it answered, and
/// what safetensors' own reader made of files written wrongly on purpose — and torch-save.json.
/// </summary>
internal static class PyTorchFixture
{
    private static readonly Lazy<JsonElement> Read = new(() => JsonDocument.Parse(File.ReadAllText(Path("pytorch.json"))).RootElement);

    private static readonly Lazy<JsonElement> ReadSaved = new(() => JsonDocument.Parse(File.ReadAllText(Path("torch-save.json"))).RootElement);

    /// <summary>pytorch.json.</summary>
    public static JsonElement Json => Read.Value;

    /// <summary>
    /// torch-save.json, which torch-save.py wrote beside the files torch.save wrote: the numbers of the deep network, and
    /// files written wrongly or with hostile intent on purpose, each with what PyTorch's own weights-only reader made of it.
    /// </summary>
    public static JsonElement Saved => ReadSaved.Value;

    /// <summary>A file of the fixture, by its name.</summary>
    public static string Path(string file) => System.IO.Path.Join(Repository.Root, "Tst", "DeepSharp", "Import", "Fixtures", file);

    /// <summary>A file of the fixture, opened to be read.</summary>
    public static Stream Open(string file) => File.OpenRead(Path(file));

    extension(JsonElement list)
    {
        /// <summary>The numbers of a list in pytorch.json, each as the 32-bit float it was written from.</summary>
        public float[] Floats() => [.. list.EnumerateArray().Select(value => (float)value.GetDouble())];

        /// <summary>The numbers of a list in pytorch.json, as written.</summary>
        public double[] Doubles() => [.. list.EnumerateArray().Select(value => value.GetDouble())];
    }
}
