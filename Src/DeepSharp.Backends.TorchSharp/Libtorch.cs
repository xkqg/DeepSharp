// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Runtime.InteropServices;
using TorchSharp;

namespace DeepSharp.Backends.TorchSharp;

/// <summary>
/// libtorch, the native library the engine runs its arithmetic on, as an application brings it: the versions the engine is
/// built against, and what is said when the application brings none, or none that runs on a graphics card.
/// </summary>
/// <remarks>
/// TorchSharp looks for libtorch the first time anything of it is touched, and when there is none it fails there, in a type
/// initializer, and every time after. The engine touches it once, where it is made, and says what to bring.
/// </remarks>
internal static class Libtorch
{
    /// <summary>The TorchSharp the engine is built against.</summary>
    internal const string TorchSharpVersion = "0.107.0";

    /// <summary>The libtorch that TorchSharp runs on.</summary>
    internal const string Version = "2.10.0";

    /// <summary>What an application is told to bring when it brings no libtorch.</summary>
    internal static string Missing =>
        string.Create(
            CultureInfo.InvariantCulture,
            $"libtorch, the native library DeepSharp.Backends.TorchSharp runs its arithmetic on, was not found: an application brings the libtorch it runs on. "
            + $"Reference libtorch-cpu-win-x64, libtorch-cpu-linux-x64 or libtorch-cpu-osx-arm64 {Version}, the processor's for the platform it runs on — this one is {RuntimeInformation.RuntimeIdentifier}; "
            + $"TorchSharp-cpu {TorchSharpVersion}, which brings the processor's for all three; "
            + $"or TorchSharp-cuda-windows or TorchSharp-cuda-linux {TorchSharpVersion}, for an NVIDIA graphics card.");

    /// <summary>What touching libtorch gives — or, when the application brings none, the refusal that names what to bring.</summary>
    /// <typeparam name="T">What is read.</typeparam>
    /// <param name="touch">The first thing read of libtorch.</param>
    /// <returns>What it read.</returns>
    /// <exception cref="InvalidOperationException">There is no libtorch to run on; the packages that bring one are named.</exception>
    internal static T Loaded<T>(Func<T> touch)
    {
        try
        {
            return touch();
        }
        catch (TypeInitializationException missing) when (missing.TypeName == typeof(torch).FullName)
        {
            throw new InvalidOperationException(Missing, missing);
        }
    }

    /// <summary>
    /// The refusal of a graphics card libtorch cannot run on: where it finds none, what the application brings or the machine
    /// lacks, naming the packages that bring a card's libtorch; where it finds some, the number asked for, naming how many.
    /// </summary>
    /// <param name="index">The card asked for.</param>
    /// <param name="cards">How many cards libtorch can run on here.</param>
    /// <returns>The refusal, to be thrown.</returns>
    internal static Exception NoCard(int index, int cards) =>
        cards == 0
            ? new InvalidOperationException(string.Create(
                CultureInfo.InvariantCulture,
                $"libtorch finds no graphics card it can run on, so there is no card {index}: the libtorch this application brings is the processor's, or this machine has no NVIDIA card with its driver. "
                + $"An application runs on a card by bringing TorchSharp-cuda-windows or TorchSharp-cuda-linux {TorchSharpVersion} in place of the processor's libtorch, on a machine that has one."))
            : new ArgumentOutOfRangeException(nameof(index), index, string.Create(
                CultureInfo.InvariantCulture,
                $"libtorch finds {cards} graphics {(cards == 1 ? "card" : "cards")} here, numbered from nought, so there is no card {index}."));
}
