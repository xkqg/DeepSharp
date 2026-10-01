// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Tensors;

namespace DeepSharp.Networks;

/// <summary>Keeps every value above nothing and makes the rest nothing: the rectifier.</summary>
public sealed class Relu : Layer, ISaved<Relu>
{
    /// <inheritdoc />
    public static string Name => "relu";

    /// <inheritdoc />
    public static Relu Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    /// <remarks>Nothing: it has no settings.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass) => pass.Backend.Relu(input);
}

/// <summary>Bends every value between minus one and one along the hyperbolic tangent.</summary>
public sealed class Tanh : Layer, ISaved<Tanh>
{
    /// <inheritdoc />
    public static string Name => "tanh";

    /// <inheritdoc />
    public static Tanh Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    /// <remarks>Nothing: it has no settings.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass) => pass.Backend.Tanh(input);
}

/// <summary>Bends every value between nothing and one along the logistic curve.</summary>
/// <remarks>
/// A network trained with <see cref="BinaryCrossEntropy"/> ends without one: the loss takes the logits and applies the
/// curve itself, which is both steadier and what PyTorch does, and a prediction goes through it on its way out. A stack
/// that ends in one is refused where it is compiled with such a loss, and a description in Keras's words that ends in one
/// leaves it out as it is compiled.
/// </remarks>
public sealed class Sigmoid : Layer, ISaved<Sigmoid>
{
    /// <inheritdoc />
    public static string Name => "sigmoid";

    /// <inheritdoc />
    public static Sigmoid Rebuild(JsonElement settings, Rebuilding rebuilding) => new();

    /// <inheritdoc />
    /// <remarks>Nothing: it has no settings.</remarks>
    public void WriteSettings(Utf8JsonWriter writer)
    {
    }

    /// <inheritdoc />
    protected override Tensor Compute(Tensor input, Pass pass) => pass.Backend.Sigmoid(input);
}
