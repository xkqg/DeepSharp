// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.IO.Compression;
using System.Text;
using System.Text.Json;
using DeepSharp.Networks;
using PureHDF;

namespace DeepSharp.Import.Keras;

/// <summary>
/// What a file Keras saved a model to holds, whichever of its two kinds it is: the model's description, the loss it was
/// compiled with, and each layer's numbers — with where the file keeps the first two, to name a fault there.
/// </summary>
/// <param name="Where">Where the file keeps the description: <c>config.json</c>, or the <c>model_config</c> attribute.</param>
/// <param name="LossWhere">Where it keeps the loss.</param>
/// <param name="Model">The description: the model's class and its settings, its layers among them.</param>
/// <param name="Loss">The loss as the file writes it; nothing when the model was saved without one.</param>
internal sealed record KerasSaved(string Where, string LossWhere, JsonElement Model, JsonElement? Loss)
{
    // How a zip archive starts, as a .keras file does, and how an HDF5 file does.
    private static readonly byte[] Archive = [0x50, 0x4B, 0x03, 0x04];
    private static readonly byte[] Hdf5 = [0x89, 0x48, 0x44, 0x46, 0x0D, 0x0A, 0x1A, 0x0A];

    /// <summary>The numbers of each layer the file holds numbers for, by the name the file gives the layer.</summary>
    public IReadOnlyList<LayerNumbers> Numbers { get; init; } = [];

    /// <summary>Reads a file Keras saved a model to, of either kind, as it starts.</summary>
    /// <exception cref="FormatException">It is neither, or lacks what Keras writes into one.</exception>
    public static KerasSaved Open(Stream file)
    {
        // Copied whole: an entry of an archive cannot be sought, and the HDF5 reader seeks.
        var bytes = new MemoryStream();
        file.CopyTo(bytes);
        bytes.Position = 0;

        var start = bytes.GetBuffer().AsSpan(0, (int)bytes.Length);

        return start.StartsWith(Archive) ? KerasArchive.Read(bytes)
            : start.StartsWith(Hdf5) ? LegacyHdf5.Read(bytes)
            : throw new FormatException("A Keras file is the archive Keras 3 saves a model to, .keras, or the HDF5 file it saved one to before, .h5, and this is neither.");
    }

    /// <summary>JSON a file holds, read whole and kept apart from the text it was read from.</summary>
    /// <exception cref="FormatException">It is not JSON.</exception>
    public static JsonElement Json(string text, string where)
    {
        try
        {
            using var document = JsonDocument.Parse(text);

            return document.RootElement.Clone();
        }
        catch (JsonException refused)
        {
            throw new FormatException($"{where} is not JSON: {refused.Message.Quoted()}", refused);
        }
    }
}

/// <summary>The numbers a file holds for one layer, in the order Keras keeps a layer's numbers.</summary>
/// <param name="Layer">The layer's name, as the file gives it.</param>
/// <param name="Numbers">Its numbers.</param>
internal readonly record struct LayerNumbers(string Layer, IReadOnlyList<HeldNumber> Numbers);

/// <summary>One of a layer's numbers: its name in the file, where the file holds it, and its lengths.</summary>
/// <param name="Name">Its name among the layer's numbers.</param>
/// <param name="Source">Where the file holds it, in the file's own words.</param>
/// <param name="Lengths">The length of each of its axes, outermost first.</param>
internal readonly record struct HeldNumber(string Name, string Source, ulong[] Lengths)
{
    /// <summary>Its values, row-major; nothing when the file writes them as other than single-precision numbers.</summary>
    public float[]? Values { get; init; }

    /// <summary>What it is written as, when that is not single-precision numbers: <c>FloatingPoint of 8 bytes each</c>.</summary>
    public string? WrittenAs { get; init; }

    /// <summary>A dataset of an HDF5 file, read.</summary>
    public static HeldNumber Of(IH5Dataset dataset, string name, string source) =>
        dataset.Type is { Class: H5DataTypeClass.FloatingPoint, Size: 4 }
            ? new(name, source, dataset.Space.Dimensions) { Values = dataset.Read<float[]>() }
            : new(name, source, dataset.Space.Dimensions) { WrittenAs = $"{dataset.Type.Class} of {dataset.Type.Size} bytes each" };
}

/// <summary>
/// The archive Keras 3 saves a model to, <c>.keras</c>: its description in <c>config.json</c> and its numbers in
/// <c>model.weights.h5</c>, each layer's under a group named after its class — <c>layers/dense_1/vars</c> — whose
/// <c>name</c> says which layer they are.
/// </summary>
internal static class KerasArchive
{
    /// <summary>Reads the archive.</summary>
    /// <exception cref="FormatException">It lacks its description or its numbers, or its description is not JSON.</exception>
    public static KerasSaved Read(MemoryStream bytes)
    {
        using var zip = new ZipArchive(bytes, ZipArchiveMode.Read);
        var config = zip.GetEntry("config.json") ?? throw new FormatException("This archive holds no config.json, where Keras 3 describes a model.");
        var weights = zip.GetEntry("model.weights.h5")
            ?? throw new FormatException("This archive holds no model.weights.h5, where Keras 3 keeps a model's numbers: a model whose numbers were kept otherwise is not read here.");

        string text;

        using (var reader = new StreamReader(config.Open(), Encoding.UTF8))
        {
            text = reader.ReadToEnd();
        }

        var model = KerasSaved.Json(text, "config.json");
        JsonElement? loss = model.TryGetProperty("compile_config", out var compile) && compile.ValueKind == JsonValueKind.Object && compile.TryGetProperty("loss", out var named)
            ? named
            : null;

        var held = new MemoryStream();

        using (var entry = weights.Open())
        {
            entry.CopyTo(held);
        }

        held.Position = 0;

        using var file = H5File.Open(held);

        return new KerasSaved("config.json", "config.json, compile_config", model, loss)
        {
            Numbers = file.LinkExists("layers") ? [.. file.Group("layers").Children().OfType<IH5Group>().Where(group => group.LinkExists("vars")).Select(Numbers)] : [],
        };
    }

    // A layer's numbers in the order Keras numbers them, under the name the group says; a group that says none is named by
    // its path, which no layer is.
    private static LayerNumbers Numbers(IH5Group group)
    {
        var vars = group.Group("vars");
        var layer = vars.AttributeExists("name") ? vars.Attribute("name").Read<string>() : $"layers/{group.Name}";

        return new LayerNumbers(
            layer,
            [.. vars.Children().OfType<IH5Dataset>()
                .OrderBy(dataset => dataset.Name.Length).ThenBy(dataset => dataset.Name, StringComparer.Ordinal)
                .Select(dataset => HeldNumber.Of(dataset, dataset.Name, $"model.weights.h5, layers/{group.Name}/vars/{dataset.Name}"))]);
    }
}

/// <summary>
/// The HDF5 file Keras saved a model to before its own archive, <c>.h5</c>: its description and its loss in attributes of
/// the file, <c>model_config</c> and <c>training_config</c>, and each layer's numbers in a group named after the layer,
/// under <c>model_weights</c>, in the order its <c>weight_names</c> lists them.
/// </summary>
internal static class LegacyHdf5
{
    /// <summary>Reads the file.</summary>
    /// <exception cref="FormatException">It holds no description, or its description or its loss is not JSON.</exception>
    public static KerasSaved Read(MemoryStream bytes)
    {
        using var file = H5File.Open(bytes);

        if (!file.AttributeExists("model_config"))
        {
            throw new FormatException(
                "This HDF5 file holds no model_config: a model Keras saved whole describes itself there, and a file of numbers alone names no layers to hold them.");
        }

        var model = KerasSaved.Json(file.Attribute("model_config").Read<string>(), "model_config");
        JsonElement? loss = file.AttributeExists("training_config")
            && KerasSaved.Json(file.Attribute("training_config").Read<string>(), "training_config").TryGetProperty("loss", out var named)
                ? named
                : null;

        return new KerasSaved("model_config", "training_config", model, loss)
        {
            Numbers = file.LinkExists("model_weights") ? [.. file.Group("model_weights").Children().OfType<IH5Group>().Select(Numbers)] : [],
        };
    }

    // A layer's numbers in the order its weight names list them, each where the name says; a name the group holds nothing
    // at is left out, and so is its number.
    private static LayerNumbers Numbers(IH5Group group) =>
        new(group.Name, [.. WeightNames(group).Where(group.LinkExists).Select(name => HeldNumber.Of(group.Dataset(name), name, $"model_weights/{group.Name}/{name}"))]);

    // Keras writes a layer that holds no numbers an empty list of names — of numbers, since the list has no text in it.
    private static string[] WeightNames(IH5Group group) =>
        group.AttributeExists("weight_names") && group.Attribute("weight_names") is { Space.Dimensions: not [0] } names ? names.Read<string[]>() : [];
}
