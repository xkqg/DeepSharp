// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers.Binary;
using System.IO.Compression;
using System.Text;
using DeepSharp.Import.PyTorch;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;
using static DeepSharp.Tests.Import.HandWrittenPickle;
using static DeepSharp.Tests.Import.PyTorchNetworks;

namespace DeepSharp.Tests.Import;

/// <summary>
/// A network's state as torch.save writes it, read into the same network written here: the very numbers the safetensors
/// reader puts into each slot, to the bit — and the pickle it is written in read as the program it is, by an interpreter that
/// carries out only what PyTorch's own weights-only reader carries out and builds nothing a pickle names but what a state
/// dictionary is made of, refusing every other name where it stands, before anything is looked up, built or run.
/// </summary>
public class TorchSaveFileTests
{
    // What a payload writes if anything ever runs it: the harmful pickles torch-save.py wrote name this file.
    private const string Ran = "deepsharp-i6-ran.txt";

    // Where this reader refuses, on purpose, a file PyTorch's weights-only reader reads, and the words it refuses it with.
    private static readonly Dictionary<string, string> RefusedWherePyTorchReads = new()
    {
        ["torch.Size, which PyTorch allows"] = "data.pkl names torch.Size at byte 2, ",
        ["bytes, which PyTorch allows through _codecs.encode"] = "data.pkl names _codecs.encode at byte 2, ",
        ["collections.Counter, which PyTorch allows"] = "data.pkl names collections.Counter at byte 2, ",
        ["a state saved in the format before PyTorch 1.6"] = "The file is written in the format PyTorch wrote before 1.6",
        ["a checkpoint holding the state under a key"] = "The file's entry 'model' is an OrderedDict, and a state dictionary holds a tensor under each name",
        ["a bias saved as a negated view"] = "as a view of its numbers marked 'neg', which this reader does not read",
    };

    [Theory]
    [InlineData("titanic.pt", "titanic.safetensors")]
    [InlineData("titanic-f16.pt", "titanic-f16.safetensors")]
    [InlineData("titanic-bf16.pt", "titanic-bf16.safetensors")]
    [InlineData("titanic-f64.pt", "titanic-f64.safetensors")]
    [InlineData("titanic-parameters.pt", "titanic.safetensors")]
    [InlineData("titanic-views.pt", "titanic.safetensors")]
    [InlineData("titanic-values.pt", "titanic.safetensors")]
    [InlineData("titanic-big-endian.pt", "titanic.safetensors")]
    public void TheTitanicState_AsTorchSaveWritesIt_GoesIntoTheSlotsTheSafetensorsReaderPutsItIn_BitForBit(string saved, string safetensors)
    {
        var read = TitanicInKerasWords();
        var expected = TitanicInKerasWords();

        var imported = new TorchSaveFile(read, new BinaryCrossEntropy()).Read(PyTorchFixture.Open(saved));
        new SafetensorsFile(expected, new BinaryCrossEntropy()).Read(PyTorchFixture.Open(safetensors));

        Assert.Same(read, imported.Network);
        Assert.IsType<BinaryCrossEntropy>(imported.Loss);
        Assert.Equal(Values(expected).Select(Bits), Values(read).Select(Bits));
    }

    [Fact]
    public void TheReadmesPassenger_AnswersAsPyTorchsTitanicNetworkDoes_FromTheFileTorchSaveWrote()
    {
        var served = PyTorchFixture.Json.GetProperty("titanic").GetProperty("served");
        var features = Rows(Titanic.Value.Served(Passengers, Needs.OneScale).Features);
        var saved = new TorchSaveFile(TitanicInKerasWords(), new BinaryCrossEntropy()).Read(PyTorchFixture.Open("titanic.pt"));
        var safetensors = new SafetensorsFile(TitanicInKerasWords(), new BinaryCrossEntropy()).Read(PyTorchFixture.Open("titanic.safetensors"));

        var chances = saved.Network.Predict(features, saved.Loss, Backend).Values.ToArray();

        Assert.Equal(Bits(safetensors.Network.Predict(features, safetensors.Loss, Backend).Values.ToArray()), Bits(chances));
        Assert.All(Roundings(chances, served.GetProperty("chances").Doubles()), apart => Assert.InRange(apart, 0, 8));
    }

    [Fact]
    public void ABatchNormalisationsNumbers_GoInAsTheSafetensorsReaderPutsThem_ItsCountOfBatchesLeftOut()
    {
        var read = new Sequential().Dense(8).BatchNorm().Relu().Dense(1).Lower(new Shape(14), new RandomStream(7));
        var expected = new Sequential().Dense(8).BatchNorm().Relu().Dense(1).Lower(new Shape(14), new RandomStream(7));

        new TorchSaveFile(read, new BinaryCrossEntropy()).Read(PyTorchFixture.Open("batchnorm.pt"));
        new SafetensorsFile(expected, new BinaryCrossEntropy()).Read(PyTorchFixture.Open("batchnorm.safetensors"));

        Assert.Equal(Values(expected).Select(Bits), Values(read).Select(Bits));
    }

    [Fact]
    public void ConvolutionsFlattenedIntoALinearLayer_GoInAsTheSafetensorsReaderPutsThem_AndAnswerAsPyTorchDoes()
    {
        var saved = new TorchSaveFile(ConvolutionInKerasWords(), new MeanSquaredError()) { Example = new Shape(8, 6, 2) }
            .Read(PyTorchFixture.Open("convolution.pt"));
        var safetensors = new SafetensorsFile(ConvolutionInKerasWords(), new MeanSquaredError()) { Example = new Shape(8, 6, 2) }
            .Read(PyTorchFixture.Open("convolution.safetensors"));

        var outputs = saved.Network.Predict(Images(), saved.Loss, Backend).Values.ToArray();
        var expected = PyTorchFixture.Json.GetProperty("convolution").GetProperty("outputs").Floats();

        Assert.Equal(Values(safetensors.Network).Select(Bits), Values(saved.Network).Select(Bits));
        Assert.All(outputs.Zip(expected, (ours, theirs) => Math.Abs(ours - theirs)), apart => Assert.InRange(apart, 0, 1e-5));
    }

    [Fact]
    public void WithNothingSaidOfWhatAFlattenIsHanded_TheNumbersThatTurnOnIt_AreRefusedAsTheSafetensorsReaderRefusesThem()
    {
        var network = ConvolutionInKerasWords();

        var refused = Assert.Throws<SlotLoadException>(() => new TorchSaveFile(network, new MeanSquaredError()).Read(PyTorchFixture.Open("convolution.pt")));
        var alike = Assert.Throws<SlotLoadException>(() => new SafetensorsFile(ConvolutionInKerasWords(), new MeanSquaredError()).Read(PyTorchFixture.Open("convolution.safetensors")));

        Assert.Equal(alike.Faults.OrderBy(fault => fault.Slot, StringComparer.Ordinal), refused.Faults.OrderBy(fault => fault.Slot, StringComparer.Ordinal));
    }

    [Fact]
    public void FortyLayersInARow_WhosePickleNumbersItsMemoPastTwoHundredAndFiftyFive_GoInAsPyTorchHoldsThem()
    {
        var network = new LayerStack([.. Enumerable.Range(0, 40).Select(_ => (Layer)new Dense(Tensor.Zeros(new Shape(1, 1)), Tensor.Zeros(new Shape(1))))]);
        var deep = PyTorchFixture.Saved.GetProperty("deep");

        new TorchSaveFile(network, new MeanSquaredError()).Read(PyTorchFixture.Open("deep.pt"));

        Assert.Equal(80, network.Slots().Count());
        Assert.All(network.Slots(), named => Assert.Equal(Bits(deep.GetProperty(named.Path).Floats()), Bits(named.Slot.Value.Values.ToArray())));
    }

    [Fact]
    public void EveryFilePyTorchsWeightsOnlyReaderRefuses_IsRefusedHere_AndEveryOneItReads_IsRead_ButTheFewThisReaderRefusesByName()
    {
        var cases = PyTorchFixture.Saved.GetProperty("cases").EnumerateArray().ToArray();
        var titanic = TitanicInKerasWords();
        new SafetensorsFile(titanic, new BinaryCrossEntropy()).Read(PyTorchFixture.Open("titanic.safetensors"));

        Assert.Equal(49, cases.Length);
        Assert.All(RefusedWherePyTorchReads.Keys, name => Assert.Contains(cases, each => each.GetProperty("case").GetString() == name));

        foreach (var each in cases)
        {
            var name = each.GetProperty("case").GetString()!;
            var torch = each.GetProperty("torch").GetString()!;
            var network = TitanicInKerasWords();
            var before = network.Slots().Select(named => named.Slot.Value).ToArray();
            using var file = new MemoryStream(Convert.FromBase64String(each.GetProperty("file").GetString()!));

            var refused = Record.Exception(() => new TorchSaveFile(network, new BinaryCrossEntropy()).Read(file));

            if (torch.StartsWith("refuses", StringComparison.Ordinal))
            {
                // Refused as a file, never as numbers that do not fit, and nothing put in.
                Assert.True(refused?.GetType() == typeof(FormatException), $"{name}: {refused?.GetType().Name} {refused?.Message}");
                Assert.True(before.SequenceEqual(network.Slots().Select(named => named.Slot.Value), ReferenceEqualityComparer.Instance), name);
            }
            else if (RefusedWherePyTorchReads.TryGetValue(name, out var words))
            {
                Assert.True(refused?.GetType() == typeof(FormatException), $"{name}: {refused?.GetType().Name}");
                Assert.Contains(words, refused!.Message, StringComparison.Ordinal);
                Assert.True(before.SequenceEqual(network.Slots().Select(named => named.Slot.Value), ReferenceEqualityComparer.Instance), name);
            }
            else if (name == "a bias saved as whole numbers")
            {
                // Read as PyTorch reads it, and refused where the numbers meet the network, as a safetensors file of them is.
                var fault = Assert.Single(Assert.IsType<SlotLoadException>(refused).Faults);
                Assert.Equal(new SlotLoadFault("0.bias", "0.bias", "'0.bias' holds I32 numbers, and a slot holds 32-bit floats: F16 and BF16 are widened to them and F64 is narrowed, and no other kind of number is read."), fault);
            }
            else
            {
                Assert.True(refused is null, $"{name}: {refused?.Message}");
                Assert.Equal(Values(titanic).Select(Bits), Values(network).Select(Bits));
            }
        }

        Assert.False(File.Exists(Path.Join(Environment.CurrentDirectory, Ran)));
    }

    [Theory]
    [InlineData("os.system, as pickle writes a call to it", "nt.system at byte 2,")]
    [InlineData("os.system, named by hand", "os.system at byte 2,")]
    [InlineData("os.system, named by an old instruction", "os.system through INST at byte 10,")]
    [InlineData("os.system, among the state's tensors", "nt.system at byte 378,")]
    [InlineData("builtins.eval, as pickle writes a call to it", "__builtin__.eval at byte 2,")]
    [InlineData("builtins.eval, named by hand", "builtins.eval at byte 2,")]
    [InlineData("subprocess.Popen, as pickle writes a call to it", "commands.Popen at byte 2,")]
    [InlineData("subprocess.Popen, named by hand", "subprocess.Popen at byte 2,")]
    [InlineData("subprocess.Popen, named on the stack by hand", "subprocess.Popen through STACK_GLOBAL at byte 27,")]
    public void AFunctionOutsideWhatAStateDictionaryIsMadeOf_IsRefusedByNameWhereThePickleNamesIt_AndNothingItNamesRuns(string harm, string named)
    {
        var file = Case(harm);
        var network = TitanicInKerasWords();
        var before = network.Slots().Select(each => each.Slot.Value).ToArray();

        var refused = Assert.Throws<FormatException>(() => new TorchSaveFile(network, new BinaryCrossEntropy()).Read(file));

        Assert.StartsWith($"data.pkl names {named}", refused.Message, StringComparison.Ordinal);
        Assert.EndsWith("so nothing it names is looked up, built or run.", refused.Message, StringComparison.Ordinal);
        Assert.False(File.Exists(Path.Join(Environment.CurrentDirectory, Ran)));
        Assert.False(File.Exists(PyTorchFixture.Path(Ran)));
        Assert.Equal(before, network.Slots().Select(each => each.Slot.Value), ReferenceEqualityComparer.Instance);
    }

    [Fact]
    public void AForeignName_IsRefusedAtTheByteThatNamesIt_AndNothingAfterItIsRead()
    {
        // os.system named, then bytes no pickle holds: had anything after the name been read, those would be refused instead.
        byte[] pickle = [.. new HandWrittenPickle().Global("os", "system").Unfinished, 0xFF, 0xFF];

        var refused = Assert.Throws<FormatException>(() => new TorchSaveFile(TitanicInKerasWords(), new BinaryCrossEntropy()).Read(Archive(pickle)));

        Assert.Equal(
            "data.pkl names os.system at byte 2, which is not among what a state dictionary is made of — collections.OrderedDict, "
            + "torch._utils._rebuild_tensor_v2 and _rebuild_parameter, and the storages torch.save keeps numbers in — so nothing it "
            + "names is looked up, built or run.",
            refused.Message);
    }

    [Fact]
    public void ATypeOfThisRuntime_NamedByAPickle_IsNeverLookedUpNorBuilt()
    {
        var process = Assert.Throws<FormatException>(() => new TorchSaveFile(TitanicInKerasWords(), new BinaryCrossEntropy()).Read(Case("a .NET type, named by hand")));
        var tripwire = Assert.Throws<FormatException>(() => new TorchSaveFile(TitanicInKerasWords(), new BinaryCrossEntropy()).Read(Case("a type of the tests, named by hand")));

        Assert.StartsWith("data.pkl names System.Diagnostics.Process at byte 2, ", process.Message, StringComparison.Ordinal);
        Assert.StartsWith("data.pkl names DeepSharp.Tests.Import.Tripwire at byte 2, ", tripwire.Message, StringComparison.Ordinal);
        Assert.Equal(0, Tripwire.Built);
    }

    [Theory]
    [InlineData("torch", "Size")]
    [InlineData("torch", "device")]
    [InlineData("torch", "Tensor")]
    [InlineData("torch", "float32")]
    [InlineData("torch", "UntypedStorage")]
    [InlineData("torch", "ComplexDoubleStorage")]
    [InlineData("torch", "QInt8Storage")]
    [InlineData("torch.nn.parameter", "Parameter")]
    [InlineData("torch._utils", "_rebuild_tensor")]
    [InlineData("torch._utils", "_rebuild_tensor_v3")]
    [InlineData("torch._utils", "_rebuild_qtensor")]
    [InlineData("torch._utils", "_rebuild_parameter_with_state")]
    [InlineData("torch._tensor", "_rebuild_from_type_v2")]
    [InlineData("torch.serialization", "_get_layout")]
    [InlineData("_codecs", "encode")]
    [InlineData("builtins", "set")]
    [InlineData("builtins", "bytearray")]
    [InlineData("builtins", "complex")]
    [InlineData("collections", "Counter")]
    public void WhatPyTorchsWeightsOnlyReaderAllowsBeyondAStateDictionarysOwn_IsRefusedByNameToo(string module, string name)
    {
        var pickle = new HandWrittenPickle().Global(module, name).Stop();

        var refused = Assert.Throws<FormatException>(() => new TorchSaveFile(TitanicInKerasWords(), new BinaryCrossEntropy()).Read(Archive(pickle)));

        Assert.StartsWith($"data.pkl names {module}.{name} at byte 2, which is not among", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(new byte[] { 0x95, 0, 0, 0, 0, 0, 0, 0, 0 }, "FRAME")]
    [InlineData(new byte[] { 0x8C, 1, (byte)'a' }, "SHORT_BINUNICODE")]
    [InlineData(new byte[] { (byte)'B', 1, 0, 0, 0, (byte)'a' }, "BINBYTES")]
    [InlineData(new byte[] { 0x94 }, "MEMOIZE")]
    [InlineData(new byte[] { (byte)'P', (byte)'a', (byte)'\n' }, "PERSID")]
    [InlineData(new byte[] { 0x82, 1 }, "EXT1")]
    [InlineData(new byte[] { (byte)'I', (byte)'1', (byte)'\n' }, "INT")]
    [InlineData(new byte[] { 0x90 }, "ADDITEMS")]
    [InlineData(new byte[] { (byte)'0' }, "POP")]
    [InlineData(new byte[] { (byte)'2' }, "DUP")]
    public void AnInstructionPyTorchsWeightsOnlyReaderDoesNotCarryOut_IsRefusedByItsName(byte[] instruction, string name)
    {
        var pickle = new HandWrittenPickle().Op(instruction[0], instruction[1..]).Stop();

        var refused = Assert.Throws<FormatException>(() => new TorchSaveFile(TitanicInKerasWords(), new BinaryCrossEntropy()).Read(Archive(pickle)));

        Assert.Equal($"data.pkl holds the instruction {name} at byte 2, which PyTorch's weights-only reader does not carry out either.", refused.Message);
    }

    public static TheoryData<string, byte[], string> Refusals()
    {
        var data = new TheoryData<string, byte[], string>();

        void Add(string what, HandWrittenPickle pickle, Func<int, string> refusal, bool stop = true)
        {
            var at = pickle.Last;
            data.Add(what, stop ? pickle.Stop() : pickle.Unfinished, refusal(at));
        }

        Add("a byte that is no instruction", new HandWrittenPickle().Op(0xFF), at => $"data.pkl holds 0xFF at byte {at}, which is no instruction of pickle's.");
        Add("a byte below every instruction", new HandWrittenPickle().Op(0x00), at => $"data.pkl holds 0x00 at byte {at}, which is no instruction of pickle's.");
        Add("a byte between instructions", new HandWrittenPickle().Op(0x7F), at => $"data.pkl holds 0x7F at byte {at}, which is no instruction of pickle's.");
        Add("STACK_GLOBAL on what is no name", new HandWrittenPickle().Number(1).Number(2).Op(0x93),
            at => $"data.pkl holds the instruction STACK_GLOBAL at byte {at}, which PyTorch's weights-only reader does not carry out either.");
        Add("an OrderedDict built by OBJ", new HandWrittenPickle().Mark().Global("collections", "OrderedDict").Op('o'),
            at => $"data.pkl holds the instruction OBJ at byte {at}, which PyTorch's weights-only reader does not carry out either.");
        data.Add("PROTO without its protocol", [0x80], "data.pkl ends inside the instruction at byte 0.");
        data.Add("a name without its line's end", [0x80, 0x02, (byte)'c', (byte)'o', (byte)'s'], "data.pkl ends inside the instruction at byte 2.");
        data.Add("text longer than the pickle", [0x80, 0x02, (byte)'X', 9, 0, 0, 0, (byte)'a'], "data.pkl ends inside the instruction at byte 2.");
        data.Add("no STOP", [0x80, 0x02, (byte)'N'], "data.pkl ends at byte 3 without the STOP that closes a pickle.");
        data.Add("text that is no UTF-8", [0x80, 0x02, (byte)'X', 1, 0, 0, 0, 0xFF, (byte)'.'], "data.pkl holds text at byte 2 that is no UTF-8.");
        data.Add("a short string that is no UTF-8", [0x80, 0x02, (byte)'U', 1, 0xFF, (byte)'.'], "data.pkl holds text at byte 2 that is no UTF-8.");
        data.Add("the two halves of a surrogate pair written apart", [0x80, 0x02, (byte)'X', 6, 0, 0, 0, 0xED, 0xA0, 0x80, 0xED, 0xB0, 0x80, (byte)'.'],
            "data.pkl holds text at byte 2 that writes the two halves of a surrogate pair apart, which Python keeps as two characters and this reader would read as one: which is meant is not guessed.");
        data.Add("half of a surrogate pair cut short", [0x80, 0x02, (byte)'X', 2, 0, 0, 0, 0xED, 0xA0, (byte)'.'], "data.pkl holds text at byte 2 that is no UTF-8.");
        data.Add("half of a surrogate pair ending in a letter", [0x80, 0x02, (byte)'X', 3, 0, 0, 0, 0xED, 0xA0, 0x41, (byte)'.'], "data.pkl holds text at byte 2 that is no UTF-8.");
        data.Add("a short string holding half of a surrogate pair", [0x80, 0x02, (byte)'U', 3, 0xED, 0xB2, 0x80, (byte)'.'], "data.pkl holds text at byte 2 that is no UTF-8.");
        data.Add("a name that is no UTF-8", [0x80, 0x02, (byte)'c', 0xFF, (byte)'\n', (byte)'a', (byte)'\n', (byte)'.'], "data.pkl holds text at byte 2 that is no UTF-8.");
        Add("STOP with nothing made", new HandWrittenPickle().Op('.'), at => $"data.pkl takes a value at byte {at} where none is left.", stop: false);
        Add("a tuple with no mark", new HandWrittenPickle().Number(1).Tuple(), at => $"data.pkl closes a MARK at byte {at} that was never opened.");
        Add("a memo never put", new HandWrittenPickle().Op('h', 5), at => $"data.pkl gets memo 5 at byte {at}, which was never put.");
        Add("a long memo never put", new HandWrittenPickle().Op('j', 0, 1, 0, 0), at => $"data.pkl gets memo 256 at byte {at}, which was never put.");
        Add("BUILD on a list", new HandWrittenPickle().Op(']').Op('}').Op('b'), at => $"data.pkl sets the state of a list at byte {at}, and only an OrderedDict's state is set.");
        Add("an OrderedDict's state that is no dict", new HandWrittenPickle().Global("collections", "OrderedDict").Op(')').Op('R').Number(1).Op('b'),
            at => $"data.pkl sets an OrderedDict's state at byte {at} to the number 1, where a state is a dict.");
        Add("APPEND to a dict", new HandWrittenPickle().Op('}').Number(1).Op('a'), at => $"data.pkl appends to a dict at byte {at}, and only a list is appended to.");
        Add("APPENDS to a dict", new HandWrittenPickle().Op('}').Mark().Number(1).Op('e'), at => $"data.pkl appends to a dict at byte {at}, and only a list is appended to.");
        Add("SETITEM on a list", new HandWrittenPickle().Op(']').Number(1).Number(2).Op('s'), at => $"data.pkl sets an item of a list at byte {at}, and only a dict or an OrderedDict has items.");
        Add("SETITEMS on a tuple", new HandWrittenPickle().Op(')').Mark().Number(1).Number(2).Op('u'), at => $"data.pkl sets an item of a tuple at byte {at}, and only a dict or an OrderedDict has items.");
        Add("SETITEMS with a key and no value", new HandWrittenPickle().Op('}').Mark().Number(1).Number(2).Number(3).Op('u'), at => $"data.pkl sets items at byte {at} from 3 values, where keys and values come in pairs.");
        Add("a dict keyed by a list", new HandWrittenPickle().Op('}').Op(']').Number(1).Op('s'),
            at => $"data.pkl keys a dict by a list at byte {at}, and a key here is None, a truth, a number, text or a tuple of those.");
        Add("a dict keyed by a tuple of tuples", new HandWrittenPickle().Op('}').Number(1).Op(0x85).Op(0x85).Number(1).Op('s'),
            at => $"data.pkl keys a dict by a tuple at byte {at}, and a key here is None, a truth, a number, text or a tuple of those.");
        Add("REDUCE on text", new HandWrittenPickle().Text("a").Op(')').Op('R'), at => $"data.pkl calls the text 'a' at byte {at}, which is nothing to call.");
        Add("REDUCE on a storage's kind", new HandWrittenPickle().Global("torch", "FloatStorage").Op(')').Op('R'), at => $"data.pkl calls torch.FloatStorage at byte {at}, which is nothing to call.");
        Add("REDUCE with no tuple", new HandWrittenPickle().Global("collections", "OrderedDict").Number(1).Op('R'),
            at => $"data.pkl calls collections.OrderedDict at byte {at} with the number 1, where the arguments are a tuple.");
        Add("an OrderedDict of arguments", new HandWrittenPickle().Global("collections", "OrderedDict").Numbers(1, 2).Op('R'),
            at => $"data.pkl makes an OrderedDict at byte {at} of 2 arguments, where torch.save makes one of none and fills it.");
        Add("NEWOBJ of a function", new HandWrittenPickle().Global("torch._utils", "_rebuild_tensor_v2").Op(')').Op(0x81),
            at => $"data.pkl builds torch._utils._rebuild_tensor_v2 as an object at byte {at}, and the only object built is an OrderedDict.");
        Add("NEWOBJ of text", new HandWrittenPickle().Text("a").Op(')').Op(0x81),
            at => $"data.pkl builds the text 'a' as an object at byte {at}, and the only object built is an OrderedDict.");
        Add("a persistent id that is a number", new HandWrittenPickle().Number(1).Op('Q'), at => $"data.pkl loads the number 1 as a persistent id at byte {at}, {Persistent}");
        Add("a persistent id that is no storage's", new HandWrittenPickle().Mark().Text("module").Number(1).Tuple().Op('Q'), at => $"data.pkl loads a tuple as a persistent id at byte {at}, {Persistent}");
        Add("a storage of a count below nought", new HandWrittenPickle().Mark().Text("storage").Global("torch", "FloatStorage").Text("0").Text("cpu").Number(-1).Tuple().Op('Q'),
            at => $"data.pkl loads a tuple as a persistent id at byte {at}, {Persistent}");
        Add("one storage named as two", new HandWrittenPickle().Storage("0", 2).Storage("0", 3),
            at => $"data.pkl names storage '0' at byte {at} as 3 numbers of torch.FloatStorage, and earlier as 2 numbers of torch.FloatStorage.");
        Add("one storage named as two kinds", new HandWrittenPickle().Storage("0", 2).Storage("0", 2, "DoubleStorage"),
            at => $"data.pkl names storage '0' at byte {at} as 2 numbers of torch.DoubleStorage, and earlier as 2 numbers of torch.FloatStorage.");
        Add("a tensor of five arguments", new HandWrittenPickle().Global("torch._utils", "_rebuild_tensor_v2").Mark().Storage("0", 2).Number(0).Numbers(2).Numbers(1).Op(0x89).Tuple().Op('R'),
            at => $"data.pkl rebuilds a tensor at byte {at} {TensorArguments}");
        Add("a tensor of no storage", new HandWrittenPickle().Global("torch._utils", "_rebuild_tensor_v2").Mark().Number(0).Number(0).Numbers(2).Numbers(1).Op(0x89).Op('N').Tuple().Op('R'),
            at => $"data.pkl rebuilds a tensor at byte {at} {TensorArguments}");
        Add("a tensor at an offset below nought", new HandWrittenPickle().Global("torch._utils", "_rebuild_tensor_v2").Mark().Storage("0", 2).Number(-1).Numbers(2).Numbers(1).Op(0x89).Op('N').Tuple().Op('R'),
            at => $"data.pkl rebuilds a tensor at byte {at} {TensorArguments}");
        Add("a tensor of more lengths than strides", new HandWrittenPickle().Global("torch._utils", "_rebuild_tensor_v2").Mark().Storage("0", 2).Number(0).Numbers(1, 2).Numbers(1).Op(0x89).Op('N').Tuple().Op('R'),
            at => $"data.pkl rebuilds a tensor at byte {at} {TensorArguments}");
        Add("a tensor of a stride below nought", new HandWrittenPickle().Global("torch._utils", "_rebuild_tensor_v2").Mark().Storage("0", 2).Number(0).Numbers(2).Numbers(-1).Op(0x89).Op('N').Tuple().Op('R'),
            at => $"data.pkl rebuilds a tensor at byte {at} {TensorArguments}");
        Add("a tensor of a length that is text", new HandWrittenPickle().Global("torch._utils", "_rebuild_tensor_v2").Mark().Storage("0", 2).Number(0).Mark().Text("2").Tuple().Numbers(1).Op(0x89).Op('N').Tuple().Op('R'),
            at => $"data.pkl rebuilds a tensor at byte {at} {TensorArguments}");
        Add("a tensor told to take gradients by a number", new HandWrittenPickle().Global("torch._utils", "_rebuild_tensor_v2").Mark().Storage("0", 2).Number(0).Numbers(2).Numbers(1).Number(0).Op('N').Tuple().Op('R'),
            at => $"data.pkl rebuilds a tensor at byte {at} {TensorArguments}");
        Add("a tensor whose marks are no dict", new HandWrittenPickle().Global("torch._utils", "_rebuild_tensor_v2").Mark().Storage("0", 2).Number(0).Numbers(2).Numbers(1).Op(0x89).Op('N').Number(1).Tuple().Op('R'),
            at => $"data.pkl rebuilds a tensor at byte {at} {TensorArguments}");
        Add("a tensor marked conjugated", new HandWrittenPickle().Global("torch._utils", "_rebuild_tensor_v2").Mark().Storage("0", 2).Number(0).Numbers(2).Numbers(1).Op(0x89).Op('N').Op('}').Text("conj").Op(0x88).Op('s').Tuple().Op('R'),
            at => $"data.pkl rebuilds a tensor at byte {at} as a view of its numbers marked 'conj', which this reader does not read: save the tensor the view resolves to.");
        Add("a tensor marked by what is no name", new HandWrittenPickle().Global("torch._utils", "_rebuild_tensor_v2").Mark().Storage("0", 2).Number(0).Numbers(2).Numbers(1).Op(0x89).Op('N').Op('}').Number(1).Op(0x88).Op('s').Tuple().Op('R'),
            at => $"data.pkl rebuilds a tensor at byte {at} as a view of its numbers marked the number 1, which this reader does not read: save the tensor the view resolves to.");
        Add("a memo put of nothing", new HandWrittenPickle().Op('q', 0), at => $"data.pkl takes a value at byte {at} where none is left.");
        Add("a tensor repeating its storage's numbers", new HandWrittenPickle().Global("torch._utils", "_rebuild_tensor_v2").Mark().Storage("0", 2).Number(0).Numbers(3).Numbers(0).Op(0x89).Op('N').Tuple().Op('R'),
            at => $"data.pkl rebuilds a tensor at byte {at} of 3 numbers out of storage '0', which holds 2: a view that repeats numbers, which this reader does not read.");
        Add("a tensor reaching past its storage", new HandWrittenPickle().Global("torch._utils", "_rebuild_tensor_v2").Mark().Storage("0", 2).Number(1).Numbers(2).Numbers(1).Op(0x89).Op('N').Tuple().Op('R'),
            at => $"data.pkl rebuilds a tensor at byte {at} reaching number 3 of storage '0', which holds 2.");
        Add("a tensor of more numbers than can be counted", new HandWrittenPickle().Global("torch._utils", "_rebuild_tensor_v2").Mark().Storage("0", 2).Number(0)
                .Mark().Op('\u008a', 8, 0, 0, 0, 0, 0, 0, 0, 0x40).Op('\u008a', 8, 0, 0, 0, 0, 0, 0, 0, 0x40).Tuple().Numbers(1, 1).Op(0x89).Op('N').Tuple().Op('R'),
            at => $"data.pkl rebuilds a tensor at byte {at} {TensorArguments}");
        Add("a parameter of no tensor", new HandWrittenPickle().Global("torch._utils", "_rebuild_parameter").Mark().Number(1).Op(0x88).Op('N').Tuple().Op('R'),
            at => $"data.pkl rebuilds a parameter at byte {at} from arguments that do not fit torch._utils._rebuild_parameter's: a tensor, whether it takes gradients, and its hooks.");

        return data;
    }

    [Theory]
    [MemberData(nameof(Refusals))]
    public void APickleThatIsNoStateDictionaryPyTorchsWeightsOnlyReaderReads_IsRefusedWhereItStrays(string what, byte[] pickle, string refusal)
    {
        var network = TitanicInKerasWords();
        var before = network.Slots().Select(each => each.Slot.Value).ToArray();

        var refused = Assert.Throws<FormatException>(() => new TorchSaveFile(network, new BinaryCrossEntropy()).Read(Archive(pickle, new Dictionary<string, byte[]> { ["0"] = Floats(1, 2) })));

        Assert.True(refusal == refused.Message, $"{what}:{Environment.NewLine}{refused.Message}");
        Assert.Equal(before, network.Slots().Select(each => each.Slot.Value), ReferenceEqualityComparer.Instance);
    }

    [Fact]
    public void EveryKindOfValueAPickleOfNumbersHolds_IsReadBesideTheState_WhereNoSlotIs()
    {
        // The state in an OrderedDict, and in the state PyTorch keeps beside it every value PyTorch's weights-only reader reads.
        var pickle = new HandWrittenPickle()
            .Global("collections", "OrderedDict").Op(')').Op('R')
            .Mark().Text("0.weight").Tensor("0", 2, [1, 2], [2, 1]).Text("0.bias").Tensor("1", 1, [1], [1]).Op('u')
            .Op('}').Text("_metadata")
            .Mark()
            .Op(0x8F)
            .Op('U', 2, (byte)'h', (byte)'i')
            .Op(0x8A, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0x80)
            .Op(0x8A, 0)
            .Op('J', 0xFF, 0xFF, 0xFF, 0xFF)
            .Op('M', 0x2C, 0x01)
            .Op('G', 0x3F, 0xF8, 0, 0, 0, 0, 0, 0)
            .Op('N').Op(0x88).Op(0x89)
            .Number(1).Op(0x85).Number(1).Number(2).Op(0x86).Number(1).Number(2).Number(3).Op(0x87)
            .Op(']').Number(4).Op('a').Mark().Number(5).Number(6).Op('e')
            .Op('r', 0, 1, 0, 0).Op('j', 0, 1, 0, 0).Op('q', 7).Op('h', 7)
            .Op(')').Op('}').Mark().Text("a").Number(1).Text("a").Number(2).Op('u')
            .Tuple()
            .Op('s').Op('b');
        var network = OneLinearLayer();

        new TorchSaveFile(network, new MeanSquaredError()).Read(Archive(pickle.Stop(), OneLinearLayersStorages));

        Assert.Equal([1.5f, -2f], network.Slots().First().Slot.Value.Values.ToArray());
        Assert.Equal([0.5f], network.Slots().Last().Slot.Value.Values.ToArray());
    }

    [Fact]
    public void KeysOfEveryKind_AreKeptApart_AndEqualOnesAreOneKey_AsPythonKeepsThem()
    {
        // Beside the state, a dict keyed by None, the truths, whole numbers of both widths, floats — nought and minus
        // nought, and two not-a-numbers, being equal as the runtime compares them — text and a tuple.
        static byte[] Float(double value) => [.. Enumerable.Reverse(BitConverter.GetBytes(value))];

        var pickle = new HandWrittenPickle()
            .Global("collections", "OrderedDict").Op(')').Op('R')
            .Mark().Text("0.weight").Tensor("0", 2, [1, 2], [2, 1]).Text("0.bias").Tensor("1", 1, [1], [1]).Text("0.bias").Tensor("1", 1, [1], [1]).Op('u')
            .Op('}').Text("_metadata").Op('}').Mark()
            .Op('N').Number(1).Op(0x88).Number(2).Op(0x89).Number(3).Number(7).Number(4)
            .Op(0x8A, 9, 0, 0, 0, 0, 0, 0, 0, 0, 1).Number(5).Op(0x8A, 9, 0, 0, 0, 0, 0, 0, 0, 0, 1).Number(6)
            .Op('G', Float(0.0)).Number(7).Op('G', Float(-0.0)).Number(8).Op('G', Float(double.NaN)).Number(9).Op('G', [0x7F, 0xF8, 0, 0, 0, 0, 0, 1]).Number(10)
            .Op('G', Float(1.5)).Number(11).Text("a").Number(12).Numbers(1, 2).Number(13)
            .Op('u').Op('s').Op('b');
        var network = OneLinearLayer();

        new TorchSaveFile(network, new MeanSquaredError()).Read(Archive(pickle.Stop(), OneLinearLayersStorages));

        // The state's name set twice is one slot, read once, as Python's dict keeps one key.
        Assert.Equal([0.5f], network.Slots().Last().Slot.Value.Values.ToArray());
    }

    [Fact]
    public void AKeyTorchSaveWritesWithALoneSurrogate_IsReadAsPyTorchReadsIt()
    {
        // As Python's surrogatepass reads UTF-8: half of a surrogate pair written in three bytes is that half, standing alone.
        var pickle = new HandWrittenPickle().Op('}').Mark()
            .Text("0.weight").Tensor("0", 2, [1, 2], [2, 1])
            .Text("0.bias").Tensor("1", 1, [1], [1])
            .Op('X', [8, 0, 0, 0, .. "extra"u8, 0xED, 0xB2, 0x80]).Tensor("1", 1, [1], [1])
            .Op('X', [5, 0, 0, 0, (byte)'h', 0xED, 0xA0, 0x80, (byte)'i']).Tensor("1", 1, [1], [1])
            .Op('X', [3, 0, 0, 0, 0xED, 0x9F, 0xBF]).Tensor("1", 1, [1], [1])
            .Op('u');

        var refused = Assert.Throws<SlotLoadException>(() => new TorchSaveFile(OneLinearLayer(), new MeanSquaredError()).Read(Archive(pickle.Stop(), OneLinearLayersStorages)));

        string[] keys = ["extra" + (char)0xDC80, "h" + (char)0xD800 + "i", ((char)0xD7FF).ToString()];
        Assert.Equal(keys, refused.Faults.Select(fault => fault.Source));
        Assert.Equal($"extra{(char)92}udc80: 'extra{(char)92}udc80' is no slot of this network.", refused.Faults[0].ToString());
        Assert.DoesNotContain(refused.Message, char.IsSurrogate);
    }

    [Fact]
    public void AStateDictionary_GoesIn_EachTensorTurnedIntoItsSlotsLayout()
    {
        var network = OneLinearLayer();

        new TorchSaveFile(network, new MeanSquaredError()).Read(Archive(OneLinearLayersState().Stop(), OneLinearLayersStorages));

        Assert.Equal([1.5f, -2f], network.Slots().First().Slot.Value.Values.ToArray());
        Assert.Equal([0.5f], network.Slots().Last().Slot.Value.Values.ToArray());
    }

    [Fact]
    public void AStateDictionaryWhoseWholeNumbersAreWrittenLong_IsRead_AsTorchSaveWritesEveryOneFromTwoToTheThirtyFirstUp()
    {
        // Every whole number of the state — counts, offsets, lengths, strides — written as LONG1, as each is past 32 bits.
        static HandWrittenPickle Long(HandWrittenPickle pickle, int value) => pickle.Op(0x8A, 1, (byte)value);

        static HandWrittenPickle Longs(HandWrittenPickle pickle, int[] values)
        {
            pickle.Mark();

            foreach (var value in values)
            {
                Long(pickle, value);
            }

            return pickle.Tuple();
        }

        static HandWrittenPickle Rebuilt(HandWrittenPickle pickle, string key, int count, int[] lengths, int[] strides)
        {
            pickle.Global("torch._utils", "_rebuild_tensor_v2").Mark().Mark().Text("storage").Global("torch", "FloatStorage").Text(key).Text("cpu");
            Long(pickle, count).Tuple().Op('Q');
            Long(pickle, 0);
            Longs(Longs(pickle, lengths), strides);

            return pickle.Op(0x89).Op('N').Tuple().Op('R');
        }

        var pickle = new HandWrittenPickle().Op('}').Mark().Text("0.weight");
        Rebuilt(pickle, "0", 2, [1, 2], [2, 1]).Text("0.bias");
        Rebuilt(pickle, "1", 1, [1], [1]).Op('u');
        var network = OneLinearLayer();

        new TorchSaveFile(network, new MeanSquaredError()).Read(Archive(pickle.Stop(), OneLinearLayersStorages));

        Assert.Equal([1.5f, -2f], network.Slots().First().Slot.Value.Values.ToArray());
        Assert.Equal([0.5f], network.Slots().Last().Slot.Value.Values.ToArray());
    }

    [Fact]
    public void TheNumbersOfATensorNoSlotTakes_AreNeverGatheredOutOfItsStorage()
    {
        // Beside the state, a hundred tensors for no slot, each viewing the whole of a storage of a quarter of a million
        // numbers: gathered, they would take a hundred megabytes; refused by name, as no slot of the network, they take none.
        var pickle = new HandWrittenPickle().Op('}').Mark()
            .Text("0.weight").Tensor("0", 2, [1, 2], [2, 1])
            .Text("0.bias").Tensor("1", 1, [1], [1]);

        for (var extra = 0; extra < 100; extra++)
        {
            pickle.Text($"extra{extra}").Tensor("9", 1 << 18, [1 << 18], [1]);
        }

        var file = Archive(pickle.Op('u').Stop(), new Dictionary<string, byte[]>(OneLinearLayersStorages) { ["9"] = new byte[4 << 18] });
        var network = OneLinearLayer();

        var before = GC.GetAllocatedBytesForCurrentThread();
        var refused = Assert.Throws<SlotLoadException>(() => new TorchSaveFile(network, new MeanSquaredError()).Read(file));
        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        Assert.Equal(100, refused.Faults.Count);
        Assert.All(refused.Faults, fault => Assert.Equal($"'{fault.Source}' is no slot of this network.", fault.Message));
        Assert.InRange(allocated, 0, 16 << 20);
    }

    [Fact]
    public void AStateDictionaryWhoseVersionTorchSaveKeepsUnderData_IsRead()
    {
        var network = OneLinearLayer();

        new TorchSaveFile(network, new MeanSquaredError()).Read(Zip(
        [
            new("archive/data.pkl", OneLinearLayersState().Stop()),
            new("archive/.data/version", "3\n"u8.ToArray()),
            new("archive/data/0", Floats(1.5f, -2f)),
            new("archive/data/1", Floats(0.5f)),
        ]));

        Assert.Equal([0.5f], network.Slots().Last().Slot.Value.Values.ToArray());
    }

    [Fact]
    public void AStateDictionaryKeptDeflated_IsRead()
    {
        var network = OneLinearLayer();

        new TorchSaveFile(network, new MeanSquaredError()).Read(Zip(
            [
                new("archive/data.pkl", OneLinearLayersState().Stop()),
                new("archive/version", "3\n"u8.ToArray()),
                new("archive/data/0", Floats(1.5f, -2f)),
                new("archive/data/1", Floats(0.5f)),
            ],
            CompressionLevel.Optimal));

        Assert.Equal([1.5f, -2f], network.Slots().First().Slot.Value.Values.ToArray());
    }

    public static TheoryData<string, MemoryStream, string> ArchiveRefusals()
    {
        var state = OneLinearLayersState().Stop();
        IEnumerable<KeyValuePair<string, byte[]>> Files(string version = "3\n", string? order = "little") =>
        [
            new("archive/data.pkl", state),
            new("archive/version", Encoding.UTF8.GetBytes(version)),
            .. order is null ? [] : new KeyValuePair<string, byte[]>[] { new("archive/byteorder", Encoding.UTF8.GetBytes(order)) },
            new("archive/data/0", Floats(1.5f, -2f)),
            new("archive/data/1", Floats(0.5f)),
        ];

        // data.pkl deflated, its first block given the one type deflate reserves, which no reader can read.
        var corrupt = Zip([new("archive/data.pkl", new byte[64]), new("archive/version", "3\n"u8.ToArray())], CompressionLevel.Optimal).ToArray();
        corrupt[30 + BinaryPrimitives.ReadUInt16LittleEndian(corrupt.AsSpan(26)) + BinaryPrimitives.ReadUInt16LittleEndian(corrupt.AsSpan(28))] = 0x07;

        return new TheoryData<string, MemoryStream, string>
        {
            { "no archive at all", new MemoryStream("not an archive"u8.ToArray()), "The file is no archive torch.save writes: " },
            { "the format before 1.6", new MemoryStream([0x80, 0x02, 0x8A, 0x0A, 0x6C, 0xFC, 0x9C, 0x46, 0xF9, 0x20, 0x6A, 0xA8, 0x50, 0x19, 0x2E]),
                "The file is written in the format PyTorch wrote before 1.6, one pickle after another, which this reader does not read: load it with PyTorch and save it again with torch.save, which has written an archive since." },
            { "an archive holding nothing", Zip([]), "The file is no archive torch.save writes: " },
            { "a record's header and an end that says none follow", new MemoryStream(HeaderThenEmptyEnd()),
                "The file is an archive holding nothing, where torch.save writes data.pkl and the storages it names." },
            { "junk before an archive", new MemoryStream([.. "NOT A ZIP, NOT A PICKLE"u8, .. Zip(Files()).ToArray()]), "The file is no archive torch.save writes: " },
            { "an archive after the format before 1.6", new MemoryStream([0x80, 0x02, 0x8A, 0x0A, 0x6C, 0xFC, 0x9C, 0x46, 0xF9, 0x20, 0x6A, 0xA8, 0x50, 0x19, 0x2E, .. Zip(Files()).ToArray()]),
                "The file is written in the format PyTorch wrote before 1.6" },
            { "a record named twice", Zip([.. Files(), new("archive/data/0", Floats(9f, 9f))]), "The file's archive names 'archive/data/0' twice" },
            { "a record named twice in other case", Zip([.. Files(), new("archive/DATA/0", Floats(9f, 9f))]), "The file's archive names 'archive/data/0' twice" },
            { "a central directory that is broken", new MemoryStream(BrokenCentralDirectory(Zip(Files()).ToArray())), "The file is no archive torch.save writes: " },
            { "a central directory whose name runs past the archive", new MemoryStream(RunningPastTheEnd(Zip(Files()).ToArray(), 0, 28, 0xFFFF)), "The file is no archive torch.save writes: " },
            { "a central directory whose extra field runs past the archive", new MemoryStream(RunningPastTheEnd(Zip(Files()).ToArray(), 2, 30, 256)), "The file is no archive torch.save writes: " },
            { "an archive in no folder", Zip([new("data.pkl", state)]), "The file is an archive whose file 'data.pkl' is in no folder, where torch.save writes every file of an archive into one." },
            { "a TorchScript archive", Zip([.. Files(), new("archive/constants.pkl", [0x80, 0x02, (byte)')', (byte)'.'])]),
                "The file is a TorchScript archive, a program PyTorch runs, which PyTorch's weights-only reader refuses too: save the network's state with torch.save(model.state_dict(), file)." },
            { "no version", Zip(Files().Where(file => !file.Key.EndsWith("version", StringComparison.Ordinal))), "The file is an archive without a version, which torch.save writes into every archive." },
            { "a version that is no number", Zip(Files("three")), "The file is an archive whose version, 'three', is no number." },
            { "version 0", Zip(Files("0\n")), "The file is an archive of version 0, and PyTorch reads versions 1 to 10." },
            { "version 11", Zip(Files("11\n")), "The file is an archive of version 11, and PyTorch reads versions 1 to 10." },
            { "a version past counting", Zip(Files("99999999999999999999\n")), "The file is an archive of version 99999999999999999999, and PyTorch reads versions 1 to 10." },
            { "a byte order nobody has", Zip(Files(order: "middle")), "The file says its numbers are in 'middle' byte order, and PyTorch writes them in little or big." },
            { "no data.pkl", Zip(Files().Where(file => !file.Key.EndsWith("data.pkl", StringComparison.Ordinal))), "The file is an archive without data.pkl, the pickle torch.save writes the state into." },
            { "no storage", Zip(Files().Where(file => !file.Key.EndsWith("/data/1", StringComparison.Ordinal))), "The file holds no storage '1', which data.pkl loads at byte 241." },
            { "a storage of other bytes", Zip(Files().Select(file => file.Key.EndsWith("/data/0", StringComparison.Ordinal) ? new KeyValuePair<string, byte[]>(file.Key, Floats(1.5f)) : file)),
                "The file's storage '0' holds 4 bytes, where data.pkl says 2 numbers of 4 bytes." },
            { "a record that cannot be read", new MemoryStream(corrupt), "The file's 'archive/data.pkl' cannot be read: " },
            { "a state that is no dictionary", Zip([new("archive/data.pkl", new HandWrittenPickle().Op(']').Stop()), new("archive/version", "3\n"u8.ToArray())]),
                "The file holds a list, where a state dictionary maps each slot's name to a tensor." },
            { "a name that is no text", Zip([new("archive/data.pkl", new HandWrittenPickle().Op('}').Number(1).Number(2).Op('s').Stop()), new("archive/version", "3\n"u8.ToArray())]),
                "The file's state dictionary holds the number 1 as a name, where each name is text." },
            { "an entry that is True", Zip([new("archive/data.pkl", new HandWrittenPickle().Op('}').Text("0.weight").Op(0x88).Op('s').Stop()), new("archive/version", "3\n"u8.ToArray())]),
                "The file's entry '0.weight' is True, and a state dictionary holds a tensor under each name" },
            { "an entry that is a set", Zip([new("archive/data.pkl", new HandWrittenPickle().Op('}').Text("0.weight").Op(0x8F).Op('s').Stop()), new("archive/version", "3\n"u8.ToArray())]),
                "The file's entry '0.weight' is a set, and a state dictionary holds a tensor under each name" },
            { "an entry that is a storage", Zip([new("archive/data.pkl", new HandWrittenPickle().Op('}').Text("0.weight").Storage("0", 2).Op('s').Stop()), new("archive/version", "3\n"u8.ToArray())]),
                "The file's entry '0.weight' is storage '0', and a state dictionary holds a tensor under each name" },
            { "an entry that is a kind of storage", Zip([new("archive/data.pkl", new HandWrittenPickle().Op('}').Text("0.weight").Global("torch", "FloatStorage").Op('s').Stop()), new("archive/version", "3\n"u8.ToArray())]),
                "The file's entry '0.weight' is torch.FloatStorage, and a state dictionary holds a tensor under each name" },
            { "a tensor as a key", Zip([new("archive/data.pkl", new HandWrittenPickle().Op('}').Tensor("0", 2, [2], [1]).Text("x").Op('s').Stop()), new("archive/version", "3\n"u8.ToArray()), new("archive/data/0", Floats(1, 2))]),
                "data.pkl keys a dict by a tensor at byte " },
            { "an entry that is no tensor", Zip([new("archive/data.pkl", new HandWrittenPickle().Op('}').Text("0.weight").Op('N').Op('s').Stop()), new("archive/version", "3\n"u8.ToArray())]),
                "The file's entry '0.weight' is None, and a state dictionary holds a tensor under each name: a checkpoint keeping the state under a key of its own is read once the state is saved alone, torch.save(model.state_dict(), file)." },
        };
    }

    [Theory]
    [MemberData(nameof(ArchiveRefusals))]
    public void AFileThatIsNoArchiveTorchSaveWrites_IsRefused_AndNothingGoesIn(string what, MemoryStream file, string refusal)
    {
        var network = OneLinearLayer();
        var before = network.Slots().Select(each => each.Slot.Value).ToArray();

        var refused = Assert.Throws<FormatException>(() => new TorchSaveFile(network, new MeanSquaredError()).Read(file));

        Assert.True(refused.Message.StartsWith(refusal, StringComparison.Ordinal), $"{what}:{Environment.NewLine}{refused.Message}");
        Assert.Equal(before, network.Slots().Select(each => each.Slot.Value), ReferenceEqualityComparer.Instance);
    }

    public static TheoryData<string, byte[], string> FileTextInRefusals() => new()
    {
        { "a line break", new HandWrittenPickle().Text("x\r\n[INFO] model.pt verified").Op(')').Op('R').Stop(), @"data.pkl calls the text 'x\r\n[INFO] model.pt verified' at byte " },
        { "a name of a million letters", new HandWrittenPickle().Global(new string('m', 1_000_000), "x").Stop(), "… (1000000 characters).x at byte 2, " },
    };

    [Theory]
    [MemberData(nameof(FileTextInRefusals))]
    public void FileTextInARefusal_IsShownEscapedAndCut_NeverAsTheFileHoldsIt(string what, byte[] pickle, string shown)
    {
        var refused = Assert.Throws<FormatException>(() => new TorchSaveFile(OneLinearLayer(), new MeanSquaredError()).Read(Archive(pickle)));

        Assert.True(refused.Message.Contains(shown, StringComparison.Ordinal), $"{what}:{Environment.NewLine}{refused.Message[..Math.Min(refused.Message.Length, 2000)]}");
        Assert.DoesNotContain(refused.Message, char.IsControl);
        Assert.InRange(refused.Message.Length, 0, 999);
    }

    [Fact]
    public void ANameNoSlotTakes_IsShownEscapedInTheSlotRefusal()
    {
        var pickle = new HandWrittenPickle().Op('}').Mark()
            .Text("0.weight").Tensor("0", 2, [1, 2], [2, 1])
            .Text("0.bias").Tensor("1", 1, [1], [1])
            .Text("x\n[INFO] ok").Tensor("1", 1, [1], [1])
            .Op('u');

        var refused = Assert.Throws<SlotLoadException>(() => new TorchSaveFile(OneLinearLayer(), new MeanSquaredError()).Read(Archive(pickle.Stop(), OneLinearLayersStorages)));

        var fault = Assert.Single(refused.Faults);
        Assert.Equal("x\n[INFO] ok", fault.Source);
        Assert.Equal(@"x\n[INFO] ok: 'x\n[INFO] ok' is no slot of this network.", fault.ToString());
        Assert.DoesNotContain(refused.Message, char.IsControl);
    }

    [Fact]
    public void ARecordNamedInOtherCase_IsRead_AsPyTorchReadsIt()
    {
        // PyTorch finds a record by its name, whatever the case of its letters; torch.save never writes two that differ in case only.
        var network = OneLinearLayer();

        new TorchSaveFile(network, new MeanSquaredError()).Read(Zip(
        [
            new("archive/data.pkl", OneLinearLayersState().Stop()),
            new("archive/version", "3\n"u8.ToArray()),
            new("archive/DATA/0", Floats(1.5f, -2f)),
            new("archive/data/1", Floats(0.5f)),
        ]));

        Assert.Equal([1.5f, -2f], network.Slots().First().Slot.Value.Values.ToArray());
        Assert.Equal([0.5f], network.Slots().Last().Slot.Value.Values.ToArray());
    }

    [Fact]
    public void AStorageRecordInflatingPastWhatDataPklSays_IsRefusedBeforeItIsInflated()
    {
        var file = Zip(
            [
                new("archive/data.pkl", OneLinearLayersState().Stop()),
                new("archive/version", Version),
                new("archive/data/0", new byte[32 << 20]),
                new("archive/data/1", Floats(0.5f)),
            ],
            CompressionLevel.SmallestSize);

        var (refused, allocated) = Refused(file);

        Assert.Equal("The file's storage '0' holds 33554432 bytes, where data.pkl says 2 numbers of 4 bytes.", refused.Message);
        Assert.InRange(allocated, 0, 4 << 20);
    }

    [Fact]
    public void APickleInflatingPastTheFile_IsRefusedBeforeItIsInflated()
    {
        var pickle = new byte[32 << 20];
        Array.Fill(pickle, (byte)'N');
        var file = Zip([new("archive/data.pkl", pickle), new("archive/version", Version)], CompressionLevel.SmallestSize);

        var (refused, allocated) = Refused(file);

        Assert.StartsWith("The file's 'archive/data.pkl' holds 33554432 bytes, ", refused.Message, StringComparison.Ordinal);
        Assert.InRange(allocated, 0, 4 << 20);
    }

    [Fact]
    public void RecordsTogetherHoldingMoreBytesThanTheFile_AreRefused_AtTheFirstPastIt()
    {
        // Two hundred storages of twenty thousand bytes each, deflated to almost nothing: together far more than the file.
        var pickle = new HandWrittenPickle().Op('}').Mark();

        for (var storage = 0; storage < 200; storage++)
        {
            pickle.Text($"t{storage}").Tensor($"{storage}", 5000, [5000], [1]);
        }

        var state = pickle.Op('u').Stop();
        var file = Zip(
            [
                new("archive/data.pkl", state),
                new("archive/version", Version),
                .. Enumerable.Range(0, 200).Select(storage => new KeyValuePair<string, byte[]>($"archive/data/{storage}", new byte[20_000])),
            ],
            CompressionLevel.SmallestSize);

        // The version's two bytes and data.pkl are read first, then the storages in the order data.pkl loads them.
        var past = (file.Length - Version.Length - state.Length) / 20_000;
        var (refused, allocated) = Refused(file);

        Assert.StartsWith($"The file's 'archive/data/{past}' holds 20000 bytes, ", refused.Message, StringComparison.Ordinal);
        Assert.InRange(allocated, 0, 4 << 20);
    }

    [Fact]
    public void ARecordHoldingMoreBytesThanItsHeadersSay_IsRefusedAtTheFirstByteBeyond()
    {
        // Stored as it is, sixty-four thousand bytes, its headers saying eight.
        var file = Zip(
                [
                    new("archive/data.pkl", OneLinearLayersState().Stop()),
                    new("archive/version", Version),
                    new("archive/data/0", new byte[64 << 10]),
                    new("archive/data/1", Floats(0.5f)),
                ])
            .ToArray();
        Said(file, "archive/data/0", 8);
        using var archive = new MemoryStream(file);

        var (refused, allocated) = Refused(archive);

        Assert.Equal("The file's 'archive/data/0' holds more bytes than the 8 it says.", refused.Message);
        Assert.InRange(allocated, 0, 4 << 20);
    }

    [Fact]
    public void ARecordInflatingToMoreThanItsHeadersSay_IsRefusedByItsCheck_HavingInflatedNoMoreThanTheySay()
    {
        // Deflated, thirty-two mebibytes, its headers saying eight: eight are inflated, and their check is not the record's.
        var file = Zip(
                [
                    new("archive/data.pkl", OneLinearLayersState().Stop()),
                    new("archive/version", Version),
                    new("archive/data/0", new byte[32 << 20]),
                    new("archive/data/1", Floats(0.5f)),
                ],
                CompressionLevel.SmallestSize)
            .ToArray();
        Said(file, "archive/data/0", 8);
        using var archive = new MemoryStream(file);

        var (refused, allocated) = Refused(archive);

        Assert.StartsWith("The file's 'archive/data/0' holds other bytes than its check says: ", refused.Message, StringComparison.Ordinal);
        Assert.InRange(allocated, 0, 4 << 20);
    }

    [Fact]
    public void ARecordHoldingFewerBytesThanItsHeadersSay_IsRefused()
    {
        var file = Zip(
                [
                    new("archive/data.pkl", OneLinearLayersState().Stop()),
                    new("archive/version", Version),
                    new("archive/data/0", Floats(1.5f, -2f)),
                    new("archive/data/1", new byte[2]),
                ],
                CompressionLevel.SmallestSize)
            .ToArray();
        Said(file, "archive/data/1", 4);
        using var archive = new MemoryStream(file);

        var (refused, _) = Refused(archive);

        Assert.StartsWith("The file's 'archive/data/1' holds fewer bytes than the 4 it says", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFileLongerThanOneArrayHolds_IsRefusedByName_BeforeAByteOfItIsRead()
    {
        var torch = Assert.Throws<FormatException>(() => new TorchSaveFile(OneLinearLayer(), new MeanSquaredError()).Read(new Endless()));
        var safetensors = Assert.Throws<FormatException>(() => new SafetensorsFile(OneLinearLayer(), new MeanSquaredError()).Read(new Endless()));

        Assert.Equal($"The file holds more than {Array.MaxLength} bytes, the most a file read here whole into memory holds.", torch.Message);
        Assert.Equal(torch.Message, safetensors.Message);
    }

    [Fact]
    public void AFileThatCannotBeWoundBack_IsReadAsItComes()
    {
        var network = OneLinearLayer();

        new TorchSaveFile(network, new MeanSquaredError()).Read(new OnlyForward(Archive(OneLinearLayersState().Stop(), OneLinearLayersStorages)));

        Assert.Equal([1.5f, -2f], network.Slots().First().Slot.Value.Values.ToArray());
    }

    [Theory]
    [InlineData("tuples")]
    [InlineData("whole numbers")]
    [InlineData("floats")]
    public void DictKeysAFileMakesShareAHash_CostNoMoreThanDistinctKeys(string keys)
    {
        // Eighty thousand keys a file chose so that the hash the runtime gives each kind of value is the same for every one:
        // a tuple ending in eight Nones; a whole number, and the bits of a float, whose two halves are equal.
        var pickle = new HandWrittenPickle().Op('}').Mark();

        for (var key = 0L; key < 80_000; key++)
        {
            var halves = new byte[8];
            System.Buffers.Binary.BinaryPrimitives.WriteInt64LittleEndian(halves, key * ((1L << 32) + 1));

            _ = keys switch
            {
                "tuples" => pickle.Mark().Number(checked((int)key) + 256).Op('N').Op('N').Op('N').Op('N').Op('N').Op('N').Op('N').Op('N').Tuple(),
                "whole numbers" => pickle.Op(0x8A, [8, .. halves]),
                _ => pickle.Op('G', [.. Enumerable.Reverse(halves)]),
            };
            pickle.Op('N');
        }

        var file = Archive(pickle.Op('u').Stop());
        var clock = System.Diagnostics.Stopwatch.StartNew();

        Assert.Throws<FormatException>(() => new TorchSaveFile(OneLinearLayer(), new MeanSquaredError()).Read(file));

        Assert.InRange(clock.Elapsed.TotalSeconds, 0, 5);
    }

    [Fact]
    public void NothingIsReadWithoutANetwork_ALoss_AndAFile()
    {
        Assert.Throws<ArgumentNullException>(() => new TorchSaveFile(null!, new MeanSquaredError()));
        Assert.Throws<ArgumentNullException>(() => new TorchSaveFile(OneLinearLayer(), null!));
        Assert.Throws<ArgumentNullException>(() => new TorchSaveFile(OneLinearLayer(), new MeanSquaredError()).Read(null!));
        Assert.Throws<ArgumentNullException>(() => new TorchSaveFile(OneLinearLayer(), new MeanSquaredError()) { Flattened = null! });
    }

    // One linear layer from two inputs to one output, holding nothing yet.
    private static LayerStack OneLinearLayer() => new(new Dense(Tensor.Zeros(new Shape(2, 1)), Tensor.Zeros(new Shape(1))));

    // Its state as torch.save writes it: the weight one row of two in PyTorch's layout, the bias one number.
    private static HandWrittenPickle OneLinearLayersState() => new HandWrittenPickle().Op('}').Mark()
        .Text("0.weight").Tensor("0", 2, [1, 2], [2, 1])
        .Text("0.bias").Tensor("1", 1, [1], [1])
        .Op('u');

    private static readonly Dictionary<string, byte[]> OneLinearLayersStorages = new() { ["0"] = Floats(1.5f, -2f), ["1"] = Floats(0.5f) };

    // A record's local header, for a file of no bytes, and after it the end of a zip archive saying its directory holds none.
    private static byte[] HeaderThenEmptyEnd() =>
    [
        0x50, 0x4B, 0x03, 0x04, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, (byte)'x',
        0x50, 0x4B, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 31, 0x00, 0x00, 0x00, 0x00, 0x00,
    ];

    // A zip archive whose first central directory entry says it was made by a version no zip reader knows the layout of.
    private static byte[] BrokenCentralDirectory(byte[] archive)
    {
        archive[archive.AsSpan().IndexOf(new byte[] { 0x50, 0x4B, 0x01, 0x02 }) + 2] = 0x09;

        return archive;
    }

    private static readonly byte[] Version = "3\n"u8.ToArray();

    // What reading a file into one linear layer is refused with, and how many bytes the reading took on this thread.
    private static Refusal Refused(Stream file)
    {
        var network = OneLinearLayer();
        var before = GC.GetAllocatedBytesForCurrentThread();
        var refused = Assert.Throws<FormatException>(() => new TorchSaveFile(network, new MeanSquaredError()).Read(file));

        return new Refusal(refused, GC.GetAllocatedBytesForCurrentThread() - before);
    }

    // The size a record's local header and its central directory entry say it holds, rewritten.
    private static void Said(byte[] archive, string record, int length)
    {
        var name = System.Text.Encoding.UTF8.GetBytes(record);
        var local = archive.AsSpan().IndexOf(name);
        var central = local + name.Length + archive.AsSpan(local + name.Length).IndexOf(name);

        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(archive.AsSpan(local - 30 + 22), length);
        System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(archive.AsSpan(central - 46 + 24), length);
    }

    // What a reading was refused with, and the bytes it took on the thread that read.
    private readonly record struct Refusal(FormatException Refused, long Allocated);

    // A file that says it holds one byte more than an array can, and refuses to be read.
    private sealed class Endless : Stream
    {
        public override bool CanRead => true;

        public override bool CanSeek => true;

        public override bool CanWrite => false;

        public override long Length => Array.MaxLength + 1L;

        public override long Position { get; set; }

        public override int Read(byte[] buffer, int offset, int count) => throw new InvalidOperationException("A file this long is never read.");

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

        public override void Flush() => throw new NotSupportedException();

        public override void SetLength(long value) => throw new NotSupportedException();

        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    // A file read as it comes, as from a network, which cannot be wound back.
    private sealed class OnlyForward(Stream inner) : Stream
    {
        public override bool CanRead => true;

        public override bool CanSeek => false;

        public override bool CanWrite => false;

        public override long Length => throw new NotSupportedException();

        public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }

        public override int Read(byte[] buffer, int offset, int count) => inner.Read(buffer, offset, Math.Min(count, 7));

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

        public override void Flush() => throw new NotSupportedException();

        public override void SetLength(long value) => throw new NotSupportedException();

        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    // A zip archive one of whose central directory entries — the first is 0 — says a length of its own, its name's at 28 or
    // its extra field's at 30, longer than all that follows it.
    private static byte[] RunningPastTheEnd(byte[] archive, int entry, int length, int said)
    {
        var at = archive.AsSpan().IndexOf(new byte[] { 0x50, 0x4B, 0x01, 0x02 });

        for (var skipped = 0; skipped < entry; skipped++)
        {
            at += 1 + archive.AsSpan(at + 1).IndexOf(new byte[] { 0x50, 0x4B, 0x01, 0x02 });
        }

        System.Buffers.Binary.BinaryPrimitives.WriteUInt16LittleEndian(archive.AsSpan(at + length), (ushort)said);

        return archive;
    }

    private const string Persistent = "and the only one it loads is a storage: ('storage', its kind, its key, where it was, how many numbers it holds).";

    private const string TensorArguments =
        "from arguments that do not fit torch._utils._rebuild_tensor_v2's: a storage, the offset into it, the length and the stride of each axis — none below nought — whether it takes gradients, and its hooks.";

    // A file of torch-save.json's, by the name of its case.
    private static MemoryStream Case(string name) =>
        new(Convert.FromBase64String(PyTorchFixture.Saved.GetProperty("cases").EnumerateArray()
            .Single(each => each.GetProperty("case").GetString() == name).GetProperty("file").GetString()!));
}

/// <summary>A type of the tests a pickle names: were a reader ever to look a name up and build what it names, this would count it.</summary>
public sealed class Tripwire
{
    /// <summary>How many were ever built.</summary>
    public static int Built;

    /// <summary>Counts itself.</summary>
    public Tripwire() => Interlocked.Increment(ref Built);
}
