// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Backends.TorchSharp;
using TorchSharp;

namespace DeepSharp.Tests.Backends.TorchGpu;

/// <summary>
/// The graphics card a test runs on: the first one libtorch can run on, or the test is skipped — where the suite is built
/// with the processor's libtorch, as the coverage check and the release build it, or where the machine has no NVIDIA card.
/// </summary>
internal static class Card
{
    /// <summary>Why a test that needs a card did not run.</summary>
    public const string Absent =
        "No graphics card libtorch can run on: the suite runs the card's tests when it is built with -p:Libtorch=cuda on a machine with an NVIDIA card.";

    /// <summary>Whether libtorch can run on a card here.</summary>
    public static bool Present => torch.cuda.is_available();

    /// <summary>The engine on the first card, or the test that asks for it is skipped.</summary>
    public static TorchBackend First()
    {
        Assert.SkipUnless(Present, Absent);

        return TorchBackend.OnGpu(0);
    }
}
