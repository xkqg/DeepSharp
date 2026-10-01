// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers.Binary;
using System.Collections.Frozen;
using System.Globalization;
using System.Numerics;
using System.Text;
using DeepSharp.Networks;

namespace DeepSharp.Import.PyTorch;

/// <summary>
/// The pickle torch.save writes a state dictionary into, carried out as PyTorch's own weights-only reader carries it out —
/// <c>torch/_weights_only_unpickler.py</c> — and no further: the same instructions, and for a name nothing but what a state
/// dictionary is made of.
/// </summary>
/// <remarks>
/// A pickle is a program: its instructions name functions and call them. So every name is refused where the pickle names
/// it unless it is one of <see cref="PickleGlobal.Allowed"/>, before anything after it is read; what an allowed name stands
/// for is fixed here, never looked up; and an instruction PyTorch's reader does not carry out is refused by its name. What
/// the pickle builds is values alone — numbers, text, tuples, lists, dicts, storages and tensors.
/// </remarks>
/// <param name="pickle">The pickle's bytes: an archive's <c>data.pkl</c>.</param>
internal sealed class WeightsOnlyPickle(byte[] pickle)
{
    // The instructions PyTorch's weights-only reader carries out, by the byte a pickle writes each as.
    private const byte Mark = (byte)'(';
    private const byte Stop = (byte)'.';
    private const byte BinFloat = (byte)'G';
    private const byte BinInt = (byte)'J';
    private const byte BinInt1 = (byte)'K';
    private const byte BinInt2 = (byte)'M';
    private const byte None = (byte)'N';
    private const byte BinPersId = (byte)'Q';
    private const byte Reduce = (byte)'R';
    private const byte ShortBinString = (byte)'U';
    private const byte BinUnicode = (byte)'X';
    private const byte EmptyList = (byte)']';
    private const byte Append = (byte)'a';
    private const byte Build = (byte)'b';
    private const byte Global = (byte)'c';
    private const byte Appends = (byte)'e';
    private const byte BinGet = (byte)'h';
    private const byte Inst = (byte)'i';
    private const byte LongBinGet = (byte)'j';
    private const byte BinPut = (byte)'q';
    private const byte LongBinPut = (byte)'r';
    private const byte SetItem = (byte)'s';
    private const byte Tuple = (byte)'t';
    private const byte SetItems = (byte)'u';
    private const byte EmptyDict = (byte)'}';
    private const byte EmptyTuple = (byte)')';
    private const byte Proto = 0x80;
    private const byte NewObj = 0x81;
    private const byte Tuple1 = 0x85;
    private const byte Tuple2 = 0x86;
    private const byte Tuple3 = 0x87;
    private const byte NewTrue = 0x88;
    private const byte NewFalse = 0x89;
    private const byte Long1 = 0x8A;
    private const byte EmptySet = 0x8F;
    private const byte StackGlobal = 0x93;

    // Every other instruction of pickle's, by the name pickletools gives it: none is carried out.
    private static readonly FrozenDictionary<byte, string> NotCarriedOut = new Dictionary<byte, string>
    {
        [(byte)'0'] = "POP", [(byte)'1'] = "POP_MARK", [(byte)'2'] = "DUP", [(byte)'F'] = "FLOAT", [(byte)'I'] = "INT",
        [(byte)'L'] = "LONG", [(byte)'P'] = "PERSID", [(byte)'S'] = "STRING", [(byte)'T'] = "BINSTRING", [(byte)'V'] = "UNICODE",
        [(byte)'d'] = "DICT", [(byte)'g'] = "GET", [Inst] = "INST", [(byte)'l'] = "LIST", [(byte)'o'] = "OBJ", [(byte)'p'] = "PUT",
        [(byte)'B'] = "BINBYTES", [(byte)'C'] = "SHORT_BINBYTES", [0x82] = "EXT1", [0x83] = "EXT2", [0x84] = "EXT4", [0x8B] = "LONG4",
        [0x8C] = "SHORT_BINUNICODE", [0x8D] = "BINUNICODE8", [0x8E] = "BINBYTES8", [0x90] = "ADDITEMS", [0x91] = "FROZENSET",
        [0x92] = "NEWOBJ_EX", [StackGlobal] = "STACK_GLOBAL", [0x94] = "MEMOIZE", [0x95] = "FRAME", [0x96] = "BYTEARRAY8",
        [0x97] = "NEXT_BUFFER", [0x98] = "READONLY_BUFFER",
    }.ToFrozenDictionary();

    private static readonly UTF8Encoding Strict = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    private readonly Dictionary<long, object?> _memo = [];
    private readonly Stack<List<object?>> _marks = new();
    private readonly Dictionary<string, PickledStorage> _storages = new(StringComparer.Ordinal);
    private List<object?> _stack = [];
    private int _at;
    private int _instruction;

    /// <summary>Carries the pickle out, to its STOP.</summary>
    /// <returns>What it built last: for torch.save, the state dictionary.</returns>
    /// <exception cref="FormatException">
    /// It names what is not among what a state dictionary is made of, holds an instruction PyTorch's weights-only reader does
    /// not carry out, or does with a value what cannot be done with it; each refused at the byte where it stands.
    /// </exception>
    public object? Load()
    {
        while (true)
        {
            if (_at == pickle.Length)
            {
                throw new FormatException($"data.pkl ends at byte {_at} without the STOP that closes a pickle.");
            }

            _instruction = _at;

            var instruction = pickle[_at++];

            switch (instruction)
            {
                case Stop:
                    return Pop();
                case Proto:
                    Bytes(1);
                    break;
                case Global:
                    Push(Allowed(Line(), Line()));
                    break;
                case NewObj:
                    Push(Built(Pop(), Pop()));
                    break;
                case Reduce:
                    Push(Called(Pop(), Pop()));
                    break;
                case Build:
                    SetState(Pop());
                    break;
                case Append:
                    Appended([Pop()]);
                    break;
                case Appends:
                    Appended(PopMark());
                    break;
                case SetItem:
                {
                    var value = Pop();
                    var key = Pop();

                    Keyed([key, value]);
                    break;
                }
                case SetItems:
                    Keyed(PopMark());
                    break;
                case Mark:
                    _marks.Push(_stack);
                    _stack = [];
                    break;
                case Tuple:
                    Push(PopMark().ToArray());
                    break;
                case Tuple1:
                case Tuple2:
                case Tuple3:
                    Push(Last(instruction - Tuple1 + 1));
                    break;
                case None:
                    Push(null);
                    break;
                case NewTrue:
                    Push(true);
                    break;
                case NewFalse:
                    Push(false);
                    break;
                case EmptyTuple:
                    Push(Array.Empty<object?>());
                    break;
                case EmptyList:
                    Push(new List<object?>());
                    break;
                case EmptyDict:
                    Push(new PickledDict(ordered: false));
                    break;
                case EmptySet:
                    Push(new HashSet<object?>());
                    break;
                case BinInt:
                    Push((long)BinaryPrimitives.ReadInt32LittleEndian(Bytes(4)));
                    break;
                case BinInt1:
                    Push((long)Bytes(1)[0]);
                    break;
                case BinInt2:
                    Push((long)BinaryPrimitives.ReadUInt16LittleEndian(Bytes(2)));
                    break;
                case Long1:
                    Push(Whole(Bytes(Bytes(1)[0])));
                    break;
                case BinFloat:
                    Push(BinaryPrimitives.ReadDoubleBigEndian(Bytes(8)));
                    break;
                case BinUnicode:
                    Push(Unicode(Bytes(BinaryPrimitives.ReadUInt32LittleEndian(Bytes(4)))));
                    break;
                case ShortBinString:
                    Push(Text(Bytes(Bytes(1)[0])));
                    break;
                case BinPersId:
                    Push(Loaded(Pop()));
                    break;
                case BinGet:
                    Push(Memo(Bytes(1)[0]));
                    break;
                case LongBinGet:
                    Push(Memo(BinaryPrimitives.ReadUInt32LittleEndian(Bytes(4))));
                    break;
                case BinPut:
                    _memo[Bytes(1)[0]] = Peek();
                    break;
                case LongBinPut:
                    _memo[BinaryPrimitives.ReadUInt32LittleEndian(Bytes(4))] = Peek();
                    break;
                default:
                    throw NotCarried(instruction);
            }
        }
    }

    /// <summary>What a value is, in a refusal's words: <c>the number 1</c>, <c>the text 'a'</c>, <c>a list</c>, <c>storage '0'</c>.</summary>
    public static string Described(object? value) => value switch
    {
        null => "None",
        bool truth => truth.ToString(),
        IFormattable number => $"the number {number.ToString(null, CultureInfo.InvariantCulture)}",
        string text => $"the text '{text.Quoted()}'",
        object?[] => "a tuple",
        List<object?> => "a list",
        HashSet<object?> => "a set",
        _ => value.ToString()!,
    };

    // What an allowed name stands for; any other is refused here, before another byte is read.
    private PickleGlobal Allowed(string module, string name) =>
        PickleGlobal.Allowed.TryGetValue($"{module}.{name}", out var global)
            ? global
            : throw new FormatException(
                $"data.pkl names {module.Quoted()}.{name.Quoted()} at byte {_instruction}, which is not among what a state dictionary is made of — collections.OrderedDict, torch._utils._rebuild_tensor_v2 and _rebuild_parameter, and the storages torch.save keeps numbers in — so nothing it names is looked up, built or run.");

    private object? Called(object? arguments, object? callee) =>
        callee is PickleGlobal global
            ? global.Called(Arguments(arguments, global), _instruction)
            : throw PickleGlobal.NothingToCall(Described(callee), _instruction);

    private object? Built(object? arguments, object? callee) =>
        callee is PickleGlobal global
            ? global.Built(Arguments(arguments, global), _instruction)
            : throw PickleGlobal.NotBuilt(Described(callee), _instruction);

    private object?[] Arguments(object? arguments, PickleGlobal callee) =>
        arguments as object?[]
        ?? throw new FormatException($"data.pkl calls {callee.Name} at byte {_instruction} with {Described(arguments)}, where the arguments are a tuple.");

    // BUILD: the state of an OrderedDict — PyTorch keeps what it notes of a state dictionary there — and of nothing else.
    private void SetState(object? state)
    {
        if (Peek() is not PickledDict { Ordered: true })
        {
            throw new FormatException($"data.pkl sets the state of {Described(Peek())} at byte {_instruction}, and only an OrderedDict's state is set.");
        }

        if (state is not PickledDict)
        {
            throw new FormatException($"data.pkl sets an OrderedDict's state at byte {_instruction} to {Described(state)}, where a state is a dict.");
        }
    }

    private void Appended(List<object?> items) =>
        (Peek() as List<object?> ?? throw new FormatException($"data.pkl appends to {Described(Peek())} at byte {_instruction}, and only a list is appended to.")).AddRange(items);

    private void Keyed(List<object?> items)
    {
        if (Peek() is not PickledDict dict)
        {
            throw new FormatException($"data.pkl sets an item of {Described(Peek())} at byte {_instruction}, and only a dict or an OrderedDict has items.");
        }

        if (items.Count % 2 != 0)
        {
            throw new FormatException($"data.pkl sets items at byte {_instruction} from {items.Count} values, where keys and values come in pairs.");
        }

        for (var at = 0; at < items.Count; at += 2)
        {
            if (!Keyable(items[at]))
            {
                throw new FormatException(
                    $"data.pkl keys a dict by {Described(items[at])} at byte {_instruction}, and a key here is None, a truth, a number, text or a tuple of those.");
            }

            dict.Set(items[at], items[at + 1]);
        }
    }

    // What a dict is keyed by: a value compared by what it is, a tuple no deeper than one level of such values, so that
    // comparing two keys never walks a structure a file could make as deep as it likes.
    private static bool Keyable(object? key) => key is object?[] tuple ? tuple.All(Atom) : Atom(key);

    private static bool Atom(object? value) => value is null or bool or IFormattable or string;

    // BINPERSID: a storage, and only a storage — ('storage', its kind, its key, where it was, how many numbers) — each key one.
    private PickledStorage Loaded(object? id)
    {
        if (id is not object?[] parts || parts is not ["storage", StorageKind kind, string key, string, long count and >= 0])
        {
            throw new FormatException(
                $"data.pkl loads {Described(id)} as a persistent id at byte {_instruction}, and the only one it loads is a storage: ('storage', its kind, its key, where it was, how many numbers it holds).");
        }

        if (!_storages.TryGetValue(key, out var storage))
        {
            return _storages[key] = new PickledStorage(key, kind, count, _instruction);
        }

        return storage.Kind == kind && storage.Count == count
            ? storage
            : throw new FormatException(
                $"data.pkl names storage '{key.Quoted()}' at byte {_instruction} as {count} numbers of {kind.Name}, and earlier as {storage.Count} numbers of {storage.Kind.Name}.");
    }

    private object? Memo(long key) =>
        _memo.TryGetValue(key, out var value) ? value : throw new FormatException($"data.pkl gets memo {key} at byte {_instruction}, which was never put.");

    // An instruction not carried out, refused by its name — and by the name it would look up, where it names one.
    private FormatException NotCarried(byte instruction)
    {
        var named = instruction switch
        {
            Inst => $"{Line().Quoted()}.{Line().Quoted()}",
            StackGlobal when _stack is [.., string module, string name] => $"{module.Quoted()}.{name.Quoted()}",
            _ => null,
        };

        return named is not null
            ? new FormatException(
                $"data.pkl names {named} through {NotCarriedOut[instruction]} at byte {_instruction}, an instruction PyTorch's weights-only reader does not carry out either, so nothing it names is looked up, built or run.")
            : NotCarriedOut.TryGetValue(instruction, out var instructionName)
                ? new FormatException($"data.pkl holds the instruction {instructionName} at byte {_instruction}, which PyTorch's weights-only reader does not carry out either.")
                : new FormatException($"data.pkl holds 0x{instruction:X2} at byte {_instruction}, which is no instruction of pickle's.");
    }

    private void Push(object? value) => _stack.Add(value);

    // The last values on the stack, taken off it, as a tuple in the order they were put on.
    private object?[] Last(int count)
    {
        var values = new object?[count];

        for (var at = count - 1; at >= 0; at--)
        {
            values[at] = Pop();
        }

        return values;
    }

    private object? Pop()
    {
        var top = Peek();

        _stack.RemoveAt(_stack.Count - 1);

        return top;
    }

    private object? Peek() =>
        _stack.Count > 0 ? _stack[^1] : throw new FormatException($"data.pkl takes a value at byte {_instruction} where none is left.");

    // Every value since the last MARK, the stack below it back in its place.
    private List<object?> PopMark()
    {
        if (!_marks.TryPop(out var below))
        {
            throw new FormatException($"data.pkl closes a MARK at byte {_instruction} that was never opened.");
        }

        var items = _stack;
        _stack = below;

        return items;
    }

    private ReadOnlySpan<byte> Bytes(long count)
    {
        if (count > pickle.Length - _at)
        {
            throw new FormatException($"data.pkl ends inside the instruction at byte {_instruction}.");
        }

        _at += (int)count;

        return pickle.AsSpan(_at - (int)count, (int)count);
    }

    // A line of a name, its end not kept.
    private string Line()
    {
        var end = Array.IndexOf(pickle, (byte)'\n', _at);

        if (end < 0)
        {
            throw new FormatException($"data.pkl ends inside the instruction at byte {_instruction}.");
        }

        var line = Text(pickle.AsSpan(_at, end - _at));
        _at = end + 1;

        return line;
    }

    private string Text(ReadOnlySpan<byte> bytes)
    {
        try
        {
            return Strict.GetString(bytes);
        }
        catch (DecoderFallbackException)
        {
            throw new FormatException($"data.pkl holds text at byte {_instruction} that is no UTF-8.");
        }
    }

    // BINUNICODE as PyTorch's reader decodes it, UTF-8 with surrogatepass: half of a surrogate pair written in UTF-8's three
    // bytes, ED and A0 to BF and a continuation, is that half standing alone; everything else is strict UTF-8. Two halves
    // written apart, one after the other, are refused: Python keeps them as two characters, and a string here would be one.
    private string Unicode(ReadOnlySpan<byte> bytes)
    {
        var text = new StringBuilder();
        var afterHigh = false;

        for (var at = Half(bytes); at >= 0; at = Half(bytes))
        {
            if (at + 2 >= bytes.Length || bytes[at + 2] is < 0x80 or > 0xBF)
            {
                throw new FormatException($"data.pkl holds text at byte {_instruction} that is no UTF-8.");
            }

            var half = (char)(0xD000 | ((bytes[at + 1] & 0x3F) << 6) | (bytes[at + 2] & 0x3F));

            if (at == 0 && afterHigh && char.IsLowSurrogate(half))
            {
                throw new FormatException(
                    $"data.pkl holds text at byte {_instruction} that writes the two halves of a surrogate pair apart, which Python keeps as two characters and this reader would read as one: which is meant is not guessed.");
            }

            text.Append(Text(bytes[..at])).Append(half);
            afterHigh = char.IsHighSurrogate(half);
            bytes = bytes[(at + 3)..];
        }

        return text.Append(Text(bytes)).ToString();
    }

    // Where the first half of a surrogate pair written in UTF-8's three bytes begins: ED, and A0 to BF after it.
    private static int Half(ReadOnlySpan<byte> bytes)
    {
        for (var from = 0; bytes[from..].IndexOf((byte)0xED) is var found and >= 0; from += found + 1)
        {
            if (from + found + 1 < bytes.Length && bytes[from + found + 1] is >= 0xA0 and <= 0xBF)
            {
                return from + found;
            }
        }

        return -1;
    }

    // LONG1: a whole number in as many bytes as it takes, little-endian, in two's complement — a long, as every other whole
    // number here is, unless it is too large for one.
    private static object Whole(ReadOnlySpan<byte> bytes)
    {
        var whole = new BigInteger(bytes, isUnsigned: false, isBigEndian: false);

        return whole >= long.MinValue && whole <= long.MaxValue ? (object)(long)whole : whole;
    }
}
