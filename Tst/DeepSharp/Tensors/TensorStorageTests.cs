// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DeepSharp.Tensors;
using DeepSharp.Tests.Backends.Parts;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// A tensor's values live where the engine that made it keeps them. An engine of somebody else's, in an assembly the library
/// opens nothing of its own to, keeps what it makes on a storage of its own; a tensor on such a storage reads its values
/// back here the first time they are read, once, and keeps that copy — a tensor never changes, so the copy never goes out
/// of date.
/// </summary>
public class TensorStorageTests
{
    [Fact]
    public void AnEngineOutsideTheLibrary_BuildsATensorOnItsOwnStorage()
    {
        // The premise: the library opens its internals to its own tests, and to nothing the engine is compiled into.
        var opened = typeof(Tensor).Assembly.GetCustomAttributes<InternalsVisibleToAttribute>().Select(granted => granted.AssemblyName);
        Assert.DoesNotContain(typeof(CopyCountingBackend).Assembly.GetName().Name, opened);

        var engine = new CopyCountingBackend();
        var sum = engine.Add(Tensor.From(new Shape(2, 2), [1f, 2f, 3f, 4f]), Tensor.From(new Shape(2, 2), [10f, 20f, 30f, 40f]));

        Assert.Same(engine.Made[^1], sum.Storage);
        Assert.Equal(new Shape(2, 2), sum.Shape);
        Assert.Equal<float[]>([11f, 22f, 33f, 44f], sum.Values.ToArray());
        Assert.Equal(2, engine.TakenIn);
    }

    [Fact]
    public void ATensorOnAnEnginesStorage_IsCopiedOutOnce_HoweverOftenItsValuesAreRead()
    {
        var storage = new CopyCountingStorage([1f, 2f, 3f]);
        var tensor = Tensor.On(new Shape(3), storage);

        Assert.Equal(0, storage.Copies);
        Assert.Equal(1f, tensor.Values[0]);
        Assert.Equal<float[]>([1f, 2f, 3f], tensor.Values.ToArray());
        Assert.Equal(3f, tensor.Values[2]);
        Assert.Equal(1, storage.Copies);
    }

    [Fact]
    public void ATensorOnAnEnginesStorage_IsCopiedOutOnce_WhenManyThreadsReadItAtTheSameMoment()
    {
        // Whichever reader comes first, and however they interleave, one copy is made and every reader reads that one.
        var storage = new CopyCountingStorage([.. Enumerable.Range(0, 1000).Select(at => (float)at)]);
        var tensor = Tensor.On(new Shape(10, 100), storage);
        const int Readers = 16;
        using var together = new Barrier(Readers);
        var seen = new float[Readers][];
        var threads = Enumerable.Range(0, Readers).Select(reader => new Thread(() =>
        {
            together.SignalAndWait();
            seen[reader] = tensor.Values.ToArray();
        })).ToArray();

        foreach (var thread in threads)
        {
            thread.Start();
        }

        foreach (var thread in threads)
        {
            thread.Join();
        }

        Assert.Equal(1, storage.Copies);
        Assert.All(seen, values => Assert.Equal<float[]>([.. Enumerable.Range(0, 1000).Select(at => (float)at)], values));
    }

    [Fact]
    public void AStorageHoldingAnotherNumberOfValuesThanTheShape_IsRefused_AsIsNoStorage()
    {
        var wrong = Assert.Throws<ArgumentException>(() => Tensor.On(new Shape(2, 3), new CopyCountingStorage([1f, 2f, 3f, 4f, 5f])));

        Assert.Equal("storage", wrong.ParamName);
        Assert.StartsWith("A 2x3 tensor holds 6 values and the storage given holds 5.", wrong.Message, StringComparison.Ordinal);
        Assert.Equal("storage", Assert.Throws<ArgumentNullException>(() => Tensor.On(new Shape(1), null!)).ParamName);
    }

    [Fact]
    public void ATensorMadeHere_HandsOutOneStorageOverItsValues()
    {
        var tensor = Tensor.From(new Shape(2, 2), [1f, 2f, 3f, 4f]);
        var storage = tensor.Storage;
        var copied = new float[storage.Count];

        storage.CopyTo(copied);

        Assert.Same(storage, tensor.Storage);
        Assert.Equal(4, storage.Count);
        Assert.Equal<float[]>([1f, 2f, 3f, 4f], copied);
    }

    [Fact]
    public void ATensorStandingOnTheStorageOfOneMadeHere_ReadsItsValuesWhereTheyAre()
    {
        var made = Tensor.From(new Shape(2, 3), [1f, 2f, 3f, 4f, 5f, 6f]);

        var laidOut = Tensor.On(new Shape(3, 2), made.Storage);

        Assert.Same(made.Storage, laidOut.Storage);
        Assert.Equal(new Shape(3, 2), laidOut.Shape);
        Assert.True(Unsafe.AreSame(ref MemoryMarshal.GetReference(made.Values), ref MemoryMarshal.GetReference(laidOut.Values)));
    }

    [Theory]
    [InlineData(LanguageVersion.CSharp12)]
    [InlineData(LanguageVersion.CSharp14)]
    public void ATensorOnAStorage_HasANameOfItsOwn_SoEveryCallThatMadeATensorFromValuesStillCompiles(LanguageVersion language)
    {
        // A second form of From taking the storage would make a bare null or default fit both, and the compiler would refuse
        // such a call as ambiguous: .NET's rules for a library forbid an overload that stops an existing call compiling, so a
        // tensor standing on a storage is made by a name of its own, and every way 0.4 wrote a tensor from values still reads.
        Assert.Empty(ErrorsOf("Tensor.From(new Shape(0), null)", language));
        Assert.Empty(ErrorsOf("Tensor.From(new Shape(0), default)", language));
        Assert.Empty(ErrorsOf("Tensor.From(new Shape(0), [])", language));
        Assert.Empty(ErrorsOf("Tensor.Zeros(new Shape(0))", language));
        Assert.Empty(ErrorsOf("Tensor.From(new Shape(2), new float[] { 1f, 2f })", language));
        Assert.Empty(ErrorsOf("Tensor.From(new Shape(2), new float[] { 1f, 2f }.AsSpan())", language));
        Assert.Empty(ErrorsOf("Tensor.From(new Shape(2), [1f, 2f])", language));
        Assert.Empty(ErrorsOf("Tensor.On(new Shape(0), Tensor.Zeros(new Shape(0)).Storage)", language));
    }

    // The errors a line of code gets from the compiler when it makes a tensor, against the library as this suite runs it.
    private static string[] ErrorsOf(string made, LanguageVersion language)
    {
        var source = $"using System; using DeepSharp.Tensors; public static class Written {{ public static Tensor Made() => {made}; }}";
        var compilation = CSharpCompilation.Create(
            "Written",
            [CSharpSyntaxTree.ParseText(source, new CSharpParseOptions(language))],
            ((string)AppContext.GetData("TRUSTED_PLATFORM_ASSEMBLIES")!).Split(Path.PathSeparator).Select(path => MetadataReference.CreateFromFile(path)),
            new CSharpCompilationOptions(OutputKind.DynamicallyLinkedLibrary));

        return [.. compilation.GetDiagnostics().Where(each => each.Severity == DiagnosticSeverity.Error).Select(each => each.Id)];
    }
}
