// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;

namespace DeepSharp.Import.PyTorch;

/// <summary>
/// A file <c>torch.save(model.state_dict(), file)</c> wrote — a <c>.pt</c> or <c>.pth</c> — read into the same network
/// written here: every tensor into the slot its name names, turned into the layout that slot keeps, as
/// <see cref="PyTorchFile"/> says.
/// </summary>
/// <remarks>
/// <para>
/// The file is the archive torch.save has written since PyTorch 1.6: the storages the numbers are in, and <c>data.pkl</c>,
/// the pickle the state dictionary is written in. A pickle is a program, so it is read as PyTorch's own weights-only
/// reader, <c>torch.load(weights_only=True)</c>, reads one: only the instructions that reader carries out are carried out,
/// and a name the pickle writes stands only for what a state dictionary is made of — <c>collections.OrderedDict</c>, a
/// tensor rebuilt by <c>torch._utils._rebuild_tensor_v2</c> or <c>_rebuild_parameter</c>, and the storages of a kind of
/// number, <c>torch.FloatStorage</c> and its like. Any other name — <c>os.system</c>, <c>builtins.eval</c>, a type of
/// .NET's, or one PyTorch allows beyond a state dictionary's own, such as <c>torch.Size</c> — is refused as a
/// <see cref="FormatException"/> naming it and the byte where the pickle names it, before another byte is read: nothing a
/// file names is ever looked up, built or run.
/// </para>
/// <para>
/// A tensor may view its storage from any offset and with any strides, and several may share one storage; each is read as
/// the numbers it views. A file written on a big-endian machine has every number's bytes turned round. Refused as a
/// <see cref="FormatException"/>: a file that is no such archive — the format before PyTorch 1.6, a TorchScript archive, an
/// archive of a version PyTorch does not read or of a byte order it does not write, one missing a storage its pickle loads
/// or holding other bytes than that storage's numbers take — and a pickle holding anything but a dict of tensors by name,
/// so a checkpoint keeping the state under a key of its own is read once the state is saved alone. A tensor saved as a
/// negated or conjugated view is refused too, and so is one that repeats its storage's numbers — a stride of nought — which
/// would take more numbers than the file holds.
/// </para>
/// <para>
/// What a file can make the reader do is bounded by the file and the network: a dict is keyed only by values compared as
/// they are, never by a structure the file could make as deep as it likes, and hashed as no file can aim at; the numbers of
/// a tensor are gathered out of its storage only when a slot of the network takes them — a hundred views of one storage for
/// no slot copy none of it; and the records the archive holds together hold no more bytes than the file itself, as
/// torch.save's own files never do, each read to the length it says and checked against the check the archive keeps of it.
/// </para>
/// <para>
/// The archive is read by PyTorch's rules: a file is one only when it begins as a zip archive does; a record is found by its
/// name whatever the case of its ASCII letters, and a name held twice is refused; text is decoded as PyTorch decodes it,
/// half of a surrogate pair written on its own included. Every name the file holds is shown in a refusal as
/// <see cref="DeepSharp.Networks.FileTextExtensions"/> shows a file's text.
/// </para>
/// </remarks>
public sealed class TorchSaveFile : PyTorchFile
{
    /// <summary>A reader of a file torch.save wrote into the network its numbers belong to.</summary>
    /// <param name="network">The network the numbers belong to, its layers numbered as PyTorch numbered its modules.</param>
    /// <param name="loss">What it answers through, handed back with it.</param>
    /// <exception cref="ArgumentNullException">The network or the loss is missing.</exception>
    public TorchSaveFile(Network network, Loss loss)
        : base(network, loss)
    {
    }

    /// <inheritdoc />
    private protected override IReadOnlyList<StoredTensor> Tensors(Stream file)
    {
        using var archive = new TorchArchive(Whole(file));

        return archive.StateDictionary();
    }
}
