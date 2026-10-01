// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using DeepSharp.Backends.TorchSharp;
using DeepSharp.Tensors;
using DeepSharp.Tests.Backends.TorchGpu;
using TorchSharp;

namespace DeepSharp.Tests.Backends.Torch;

/// <summary>
/// The engine on libtorch: what every engine is held to runs on it from the contract; this is what is its own — where what
/// it makes lives, how a tensor from anywhere else reaches it, and what it says when the application brings no libtorch it
/// can run on.
/// </summary>
public class TorchBackendTests
{
    [Fact]
    public void TheEngine_SaysWhichOneItIs_OnTheProcessorAsOnACard()
    {
        Assert.Equal("torch", TorchBackend.OnCpu().Name);
    }

    [Fact]
    public void TheEngine_NamesTheLibtorchItRunsOn_AsTorchSharpStatesIt_AndTheDeviceItWasMadeFor()
    {
        INamesItsVersionAndDevice engine = TorchBackend.OnCpu();

        Assert.Equal(torch.__version__, engine.Version);
        Assert.StartsWith($"{Libtorch.Version}.", engine.Version, StringComparison.Ordinal);
        Assert.Equal("cpu", engine.Device);
    }

    [Fact]
    public void WhatItMakes_StandsOnLibtorchsMemory_OnTheDeviceItWasMadeFor_AndIsReadBackHere()
    {
        var engine = TorchBackend.OnCpu();

        var sum = engine.Add(Tensor.From(new Shape(2, 2), [1f, 2f, 3f, 4f]), Tensor.From(new Shape(2, 2), [10f, 20f, 30f, 40f]));

        var storage = Assert.IsType<TorchStorage>(sum.Storage);
        Assert.True(storage.IsOn("cpu"));
        Assert.False(storage.IsOn("cuda:0"));
        Assert.Equal(4, storage.Count);
        Assert.Equal<float[]>([11f, 22f, 33f, 44f], sum.Values.ToArray());
    }

    [Fact]
    public void ATensorFromThisMachinesMemory_IsTakenInOnce_HoweverOftenItIsHanded()
    {
        var engine = TorchBackend.OnCpu();
        var rows = Tensor.From(new Shape(2, 3), [1f, 2f, 3f, 4f, 5f, 6f]);

        // A reshape stands on the storage the engine holds the tensor on: handed twice, it is the one taken in the first time.
        var first = engine.Reshape(rows, new Shape(3, 2));
        var second = engine.Reshape(rows, new Shape(6));

        Assert.Same(first.Storage, second.Storage);
        Assert.IsType<TorchStorage>(first.Storage);
        Assert.Equal<float[]>([1f, 2f, 3f, 4f, 5f, 6f], second.Values.ToArray());
    }

    [Fact]
    public void AReshape_StandsOnTheStorageItWasHanded_AndIsReadUnderItsOwnShape()
    {
        var engine = TorchBackend.OnCpu();
        var made = engine.Add(Tensor.From(new Shape(2, 3), [1f, 2f, 3f, 4f, 5f, 6f]), Tensor.Zeros(new Shape(2, 3)));

        var laidOut = engine.Reshape(made, new Shape(3, 2));
        var turned = engine.Transpose(laidOut);

        Assert.Same(made.Storage, laidOut.Storage);
        Assert.Equal(new Shape(2, 3), turned.Shape);
        Assert.Equal<float[]>([1f, 3f, 5f, 2f, 4f, 6f], turned.Values.ToArray());
    }

    [Fact]
    public void ATensorTheLightEngineMade_IsTakenIn_AndWhatComesOfItIsTheEngines()
    {
        var engine = TorchBackend.OnCpu();
        var light = new CpuBackend().Fill(new Shape(2), 1.5f);

        var product = engine.Multiply(light, light);

        Assert.IsType<TorchStorage>(product.Storage);
        Assert.Equal<float[]>([2.25f, 2.25f], product.Values.ToArray());
        Assert.Equal<float[]>([4.5f, 4.5f], new CpuBackend().Add(product, Tensor.From(new Shape(2), [2.25f, 2.25f])).Values.ToArray());
    }

    [Fact]
    public void TwoEnginesOnTheProcessor_EachTakeTheOthersTensorsAsTheirOwn()
    {
        var one = TorchBackend.OnCpu();
        var other = TorchBackend.OnCpu();
        var made = one.Fill(new Shape(3), 2f);

        var reshaped = other.Reshape(made, new Shape(1, 3));

        Assert.Same(made.Storage, reshaped.Storage);
    }

    [Fact]
    public void WithoutLibtorch_TheEngineIsRefused_NamingThePackagesThatBringIt()
    {
        // What TorchSharp throws the first time it is touched in an application that brings no libtorch: its type initializer
        // fails, looking for the processor's libtorch for the platform, and fails the same way every time after.
        var cause = new TypeInitializationException(typeof(torch).FullName, new NotSupportedException("This application or script uses TorchSharp but doesn't contain a reference to libtorch-cpu-win-x64, Version=2.10.0.0."));

        var refused = Assert.Throws<InvalidOperationException>(() => Libtorch.Loaded<int>(() => throw cause));

        Assert.Same(cause, refused.InnerException);
        Assert.StartsWith("libtorch, the native library DeepSharp.Backends.TorchSharp runs its arithmetic on, was not found", refused.Message, StringComparison.Ordinal);

        foreach (var package in new[]
                 {
                     "libtorch-cpu-win-x64", "libtorch-cpu-linux-x64", "libtorch-cpu-osx-arm64 2.10.0", "TorchSharp-cpu 0.107.0",
                     "TorchSharp-cuda-windows", "TorchSharp-cuda-linux 0.107.0", System.Runtime.InteropServices.RuntimeInformation.RuntimeIdentifier,
                 })
        {
            Assert.Contains(package, refused.Message, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void AnotherTypesFailure_IsNotTakenForAMissingLibtorch()
    {
        var cause = new TypeInitializationException("Somebody.Else", new InvalidOperationException("not libtorch"));

        Assert.Same(cause, Assert.Throws<TypeInitializationException>(() => Libtorch.Loaded<int>(() => throw cause)));
    }

    [Fact]
    public void TheVersionsTheEngineNames_AreTheOnesItRunsOn()
    {
        var torchSharp = typeof(torch).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()!.InformationalVersion.Split('+')[0];

        Assert.Equal(Libtorch.TorchSharpVersion, torchSharp);
        Assert.StartsWith($"{Libtorch.Version}.", torch.__version__, StringComparison.Ordinal);
    }

    [Fact]
    public void ACardNumberedBelowNothing_IsRefused()
    {
        Assert.Equal("index", Assert.Throws<ArgumentOutOfRangeException>(() => TorchBackend.OnGpu(-1)).ParamName);
    }

    [Fact]
    public void WhereLibtorchFindsNoCard_AnEngineOnOne_IsRefused_NamingThePackagesThatBringACardsLibtorch()
    {
        Assert.SkipWhen(Card.Present, "libtorch runs on a card here.");

        var refused = Assert.Throws<InvalidOperationException>(() => TorchBackend.OnGpu(0));

        Assert.StartsWith("libtorch finds no graphics card it can run on, so there is no card 0:", refused.Message, StringComparison.Ordinal);
        Assert.Contains("TorchSharp-cuda-windows or TorchSharp-cuda-linux 0.107.0", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(1, "libtorch finds 1 graphics card here, numbered from nought, so there is no card 1.")]
    [InlineData(2, "libtorch finds 2 graphics cards here, numbered from nought, so there is no card 1.")]
    public void ACardBeyondTheOnesThere_IsRefusedAsTheNumberAskedFor_NamingHowManyThereAre(int cards, string words)
    {
        var refused = Assert.IsType<ArgumentOutOfRangeException>(Libtorch.NoCard(1, cards));

        Assert.Equal("index", refused.ParamName);
        Assert.Equal(1, refused.ActualValue);
        Assert.StartsWith(words, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void WhereThereIsNoCardAtAll_TheRefusalIsOfWhatTheApplicationBrings_NotOfTheNumberAskedFor()
    {
        var refused = Assert.IsType<InvalidOperationException>(Libtorch.NoCard(0, 0));

        Assert.StartsWith("libtorch finds no graphics card it can run on, so there is no card 0:", refused.Message, StringComparison.Ordinal);
    }
}
