// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

namespace DeepSharp.Networks;

/// <summary>One step of a stack, as it is asked for: a layer, the name it goes by, and the steps it takes what it works on from.</summary>
/// <param name="Layer">The layer the step runs; held by no other network.</param>
/// <param name="Name">What the step is called, so a later step can take what it made; its place in the stack, counted from nought, unless said.</param>
/// <param name="Takes">
/// The steps whose work this one takes, by name — <see cref="Input"/> for what the network was handed; the step before it, unless said.
/// A layer that joins takes two or more, and any other takes one.
/// </param>
/// <remarks>
/// A name is some text with no dot and no comma in it, which is not a whole number — those are the places unnamed steps go by —
/// and is not <see cref="Input"/>. It is the path of everything the step's layer holds, so a step called <c>second</c> holds
/// <c>second.weight</c>, as the module a network written as code calls second does.
/// </remarks>
public readonly record struct Step(Layer Layer, string? Name = null, IReadOnlyList<string>? Takes = null)
{
    /// <summary>What a step calls the rows the network was handed.</summary>
    public const string Input = "input";
}

/// <summary>A step of a stack with everything it says worked out: the name it goes by, its layer, and the names of the steps it takes.</summary>
/// <param name="Name">What the step is called: the name it was given, or its place in the stack, counted from nought.</param>
/// <param name="Layer">The layer the step runs.</param>
/// <param name="Takes">The names of the steps whose work it takes, <see cref="Step.Input"/> for the rows the network was handed.</param>
public readonly record struct PlacedStep(string Name, Layer Layer, IReadOnlyList<string> Takes);
