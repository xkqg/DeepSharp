// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Collections.Frozen;
using DeepSharp.Networks;
using Onnxify.Safetensors;

namespace DeepSharp.Import.PyTorch;

/// <summary>
/// A name a pickle may name, standing for what a state dictionary torch.save writes is made of: the only thing the reader
/// ever puts in the pickle's hands for a name. Nothing is looked up by a name a file holds; a name that is none of these is
/// refused where the file names it.
/// </summary>
/// <param name="name">Its name, as a pickle writes it: the module, a dot, the name.</param>
internal abstract class PickleGlobal(string name)
{
    /// <summary>
    /// Every name a pickle may name, and what it stands for: the part of PyTorch's weights-only allowlist a state dictionary
    /// is made of — the OrderedDict it is, the two functions its tensors are rebuilt by, and the storages that hold a kind of
    /// number a slot can be read from.
    /// </summary>
    public static readonly FrozenDictionary<string, PickleGlobal> Allowed = new PickleGlobal[]
    {
        new OrderedDictClass(),
        new TensorRebuild(),
        new ParameterRebuild(),
        new StorageKind("torch.FloatStorage", DataType.F32, 4, 4),
        new StorageKind("torch.DoubleStorage", DataType.F64, 8, 8),
        new StorageKind("torch.HalfStorage", DataType.F16, 2, 2),
        new StorageKind("torch.BFloat16Storage", DataType.Bf16, 2, 2),
        new StorageKind("torch.LongStorage", DataType.I64, 8, 8),
        new StorageKind("torch.IntStorage", DataType.I32, 4, 4),
        new StorageKind("torch.ShortStorage", DataType.I16, 2, 2),
        new StorageKind("torch.CharStorage", DataType.I8, 1, 1),
        new StorageKind("torch.ByteStorage", DataType.U8, 1, 1),
        new StorageKind("torch.BoolStorage", DataType.Bool, 1, 1),
        new StorageKind("torch.ComplexFloatStorage", DataType.C64, 8, 4),
    }.ToFrozenDictionary(global => global.Name, StringComparer.Ordinal);

    /// <summary>Its name, as a pickle writes it.</summary>
    public string Name => name;

    /// <summary>What calling it makes, as REDUCE calls it.</summary>
    /// <param name="arguments">What it is called with.</param>
    /// <param name="at">The byte of the instruction that calls it.</param>
    /// <exception cref="FormatException">It is nothing to call, or not with these.</exception>
    public virtual object? Called(object?[] arguments, int at) => throw NothingToCall(Name, at);

    /// <summary>What building it as an object makes, as NEWOBJ builds it.</summary>
    /// <param name="arguments">What it is built of.</param>
    /// <param name="at">The byte of the instruction that builds it.</param>
    /// <exception cref="FormatException">It is nothing to build, or not of these.</exception>
    public virtual object? Built(object?[] arguments, int at) => throw NotBuilt(Name, at);

    /// <summary>Its name.</summary>
    public override string ToString() => Name;

    /// <summary>The refusal of a call to what is nothing to call.</summary>
    public static FormatException NothingToCall(string what, int at) => new($"data.pkl calls {what} at byte {at}, which is nothing to call.");

    /// <summary>The refusal of building what is no OrderedDict as an object.</summary>
    public static FormatException NotBuilt(string what, int at) =>
        new($"data.pkl builds {what} as an object at byte {at}, and the only object built is an OrderedDict.");
}

/// <summary><c>collections.OrderedDict</c>: what a state dictionary is, made empty and filled item by item.</summary>
internal sealed class OrderedDictClass() : PickleGlobal("collections.OrderedDict")
{
    /// <inheritdoc />
    public override object? Called(object?[] arguments, int at) =>
        arguments.Length == 0
            ? new PickledDict(ordered: true)
            : throw new FormatException($"data.pkl makes an OrderedDict at byte {at} of {arguments.Length} arguments, where torch.save makes one of none and fills it.");

    /// <inheritdoc />
    public override object? Built(object?[] arguments, int at) => Called(arguments, at);
}

/// <summary>
/// <c>torch._utils._rebuild_tensor_v2</c>: a tensor as torch.save writes one — a storage, the offset into it, the length
/// and the stride of each axis, whether it takes gradients, its hooks, and marks of a view when there are any.
/// </summary>
internal sealed class TensorRebuild() : PickleGlobal("torch._utils._rebuild_tensor_v2")
{
    /// <inheritdoc />
    /// <remarks>
    /// The tensor reaches no number its storage does not hold and holds no more numbers than its storage does; whether it
    /// takes gradients and its hooks change none of its numbers and are not read; a view marked negated or conjugated, which
    /// PyTorch turns as it reads it, is refused.
    /// </remarks>
    public override object? Called(object?[] arguments, int at)
    {
        if (arguments is not [PickledStorage storage, long offset and >= 0, object?[] lengths, object?[] strides, bool, _, .. var marks]
            || marks is not ([] or [null or PickledDict])
            || Axes(lengths) is not { } axes
            || Axes(strides) is not { } steps
            || axes.Length != steps.Length)
        {
            throw Unfit(at);
        }

        foreach (var (mark, set) in marks is [PickledDict view] ? view.Entries : [])
        {
            if (set is not false)
            {
                throw new FormatException(
                    $"data.pkl rebuilds a tensor at byte {at} as a view of its numbers marked {(mark is string named ? $"'{named.Quoted()}'" : WeightsOnlyPickle.Described(mark))}, which this reader does not read: save the tensor the view resolves to.");
            }
        }

        long count = 1, last = offset;

        try
        {
            for (var axis = 0; axis < axes.Length; axis++)
            {
                count = checked(count * axes[axis]);
                last = checked(last + ((axes[axis] - 1) * steps[axis]));
            }

            _ = checked((int)(count * storage.Kind.Size));
        }
        catch (OverflowException)
        {
            throw Unfit(at);
        }

        if (count > 0 && last >= storage.Count)
        {
            throw new FormatException($"data.pkl rebuilds a tensor at byte {at} reaching number {last + 1} of storage '{storage.Key.Quoted()}', which holds {storage.Count}.");
        }

        // A view that repeats numbers — a stride of nought — would be gathered into more numbers than the file holds.
        if (count > storage.Count)
        {
            throw new FormatException(
                $"data.pkl rebuilds a tensor at byte {at} of {count} numbers out of storage '{storage.Key.Quoted()}', which holds {storage.Count}: a view that repeats numbers, which this reader does not read.");
        }

        return new PickledTensor(storage, offset, axes, steps);
    }

    // The lengths or strides of a tensor's axes: whole numbers, none below nought; nothing when they are not.
    private static long[]? Axes(object?[] values) => values.All(value => value is long and >= 0) ? [.. values.Cast<long>()] : null;

    private static FormatException Unfit(int at) =>
        new($"data.pkl rebuilds a tensor at byte {at} from arguments that do not fit torch._utils._rebuild_tensor_v2's: a storage, the offset into it, the length and the stride of each axis — none below nought — whether it takes gradients, and its hooks.");
}

/// <summary><c>torch._utils._rebuild_parameter</c>: a parameter, which to a state dictionary is the tensor it holds.</summary>
internal sealed class ParameterRebuild() : PickleGlobal("torch._utils._rebuild_parameter")
{
    /// <inheritdoc />
    public override object? Called(object?[] arguments, int at) =>
        arguments is [PickledTensor tensor, bool, _]
            ? tensor
            : throw new FormatException($"data.pkl rebuilds a parameter at byte {at} from arguments that do not fit torch._utils._rebuild_parameter's: a tensor, whether it takes gradients, and its hooks.");
}

/// <summary>A storage class of PyTorch's, named in a storage's persistent id: the kind of number the storage holds.</summary>
/// <param name="name">Its name: <c>torch.FloatStorage</c>.</param>
/// <param name="kind">The kind of number, as a slot's reader names it.</param>
/// <param name="size">How many bytes one number takes.</param>
/// <param name="swapped">How many bytes are turned round together when the file was written on a machine of the other byte order.</param>
internal sealed class StorageKind(string name, DataType kind, int size, int swapped) : PickleGlobal(name)
{
    /// <summary>The kind of number.</summary>
    public DataType Kind => kind;

    /// <summary>How many bytes one number takes.</summary>
    public int Size => size;

    /// <summary>How many bytes are turned round together for the other byte order: a number's, or each half of a complex one's.</summary>
    public int Swapped => swapped;
}
