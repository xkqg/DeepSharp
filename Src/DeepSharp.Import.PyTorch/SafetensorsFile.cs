// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using Onnxify.Safetensors;

namespace DeepSharp.Import.PyTorch;

/// <summary>
/// A safetensors file PyTorch saved a network's state in, read into the same network written here: every tensor into the
/// slot its name names, turned into the layout that slot keeps, as <see cref="PyTorchFile"/> says.
/// </summary>
/// <remarks>
/// The file is read whole into memory by a port of safetensors' own reader, so a file that reader refuses is refused here
/// with its words, as a <see cref="FormatException"/>; so is one whose note says its tensors are laid out as another
/// framework lays them out — anything but <c>"format": "pt"</c>, which Hugging Face's transformers writes as it saves a
/// model. A file with no such note is read as PyTorch's, this being PyTorch's reader: safetensors' own
/// <c>safetensors.torch.save_file</c> and <c>save_model</c> write none unless handed one, as measured with safetensors 0.8.0.
/// </remarks>
public sealed class SafetensorsFile : PyTorchFile
{
    // The note a safetensors file says whose layout its tensors are in under, and PyTorch's word for its own.
    private const string Format = "format";
    private const string PyTorchs = "pt";

    /// <summary>A reader of a safetensors file into the network its numbers belong to.</summary>
    /// <param name="network">The network the numbers belong to, its layers numbered as PyTorch numbered its modules.</param>
    /// <param name="loss">What it answers through, handed back with it.</param>
    /// <exception cref="ArgumentNullException">The network or the loss is missing.</exception>
    public SafetensorsFile(Network network, Loss loss)
        : base(network, loss)
    {
    }

    /// <inheritdoc />
    private protected override IReadOnlyList<StoredTensor> Tensors(Stream file) =>
        [.. Archive(file).Tensors().Select(pair => new StoredTensor(pair.Key, () => pair.Value))];

    // The file as safetensors' own reader reads it, refused in its words, and refused when it says it is another framework's.
    private static SafeTensors Archive(Stream file)
    {
        using var bytes = Whole(file);

        SafeTensors archive;

        try
        {
            archive = SafeTensors.Deserialize(bytes.GetBuffer().AsMemory(0, (int)bytes.Length));
        }
        catch (SafeTensorException refused)
        {
            throw new FormatException($"The file is no safetensors file: {refused.Message.Quoted()}", refused);
        }

        if (archive.Metadata.MetadataEntries?.GetValueOrDefault(Format) is { } format && format != PyTorchs)
        {
            throw new FormatException(
                $"The file says its tensors are laid out as '{format.Quoted()}' lays them out, and this reader reads PyTorch's layout, which a file marks as \"{Format}\": \"{PyTorchs}\".");
        }

        return archive;
    }
}
