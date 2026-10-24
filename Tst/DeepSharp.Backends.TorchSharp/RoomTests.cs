// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Runtime.CompilerServices;
using DeepSharp.Backends.TorchSharp;

namespace DeepSharp.Tests.Backends.Torch;

/// <summary>
/// What libtorch holds is let go of by the .NET collector, which runs when it is told of enough managed memory to run for and not when the
/// card is full: an allocation that fails while the card holds garbage the collector has not reached is asked for again, once, after the
/// collector has been through, and any other failure is the caller's.
/// </summary>
public class RoomTests
{
    private const string NoRoom = "CUDA out of memory. Tried to allocate 2.00 GiB. GPU 1 has a total capacity of 15.92 GiB";

    private static bool _letGo;

    [Fact]
    public void AnAllocationThatFailsForWantOfRoom_IsAskedAgain_OnceTheGarbageIsLetGoOf()
    {
        var calls = 0;

        Garbage();

        var made = TorchBackend.WithRoom(() =>
        {
            calls++;

            return calls == 1 ? throw new InvalidOperationException(NoRoom) : _letGo;
        });

        Assert.True(made);
        Assert.Equal(2, calls);
    }

    [Fact]
    public void AnAllocationThatFailsForWantOfRoom_AfterTheCollector_FailsAsItDid()
    {
        var calls = 0;

        var failed = Assert.Throws<InvalidOperationException>(() => TorchBackend.WithRoom<int>(() =>
        {
            calls++;

            throw new InvalidOperationException(NoRoom);
        }));

        Assert.Equal(NoRoom, failed.Message);
        Assert.Equal(2, calls);
    }

    [Fact]
    public void AFailureThatIsNotForWantOfRoom_IsNotAskedAgain()
    {
        var calls = 0;

        var failed = Assert.Throws<InvalidOperationException>(() => TorchBackend.WithRoom<int>(() =>
        {
            calls++;

            throw new InvalidOperationException("A tensor of another shape.");
        }));

        Assert.Equal("A tensor of another shape.", failed.Message);
        Assert.Equal(1, calls);
    }

    [Fact]
    public void AnAllocationThatSucceeds_IsNotAskedTwice_And_TheCollectorIsLeftAlone()
    {
        var calls = 0;

        Garbage();

        Assert.Equal(7, TorchBackend.WithRoom(() =>
        {
            calls++;

            return 7;
        }));
        Assert.Equal(1, calls);
        Assert.False(_letGo);
    }

    // An object nothing refers to once this returns, whose finalizer says it was let go of.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void Garbage()
    {
        _letGo = false;
        _ = new Finalised();
    }

    private sealed class Finalised
    {
        ~Finalised() => _letGo = true;
    }
}
