// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Backends.Contract;

/// <summary>
/// How closely an engine's arithmetic comes to the light engine's, operation by operation. A total is held to three
/// roundings of the size of the terms it adds up — three times a float's unit roundoff, two to the minus twenty-four, times
/// the sum of the terms' sizes — since an engine adds its terms up in its own order and, as libtorch does, in single
/// precision; a value worked out value by value is held to PyTorch's own float tolerance, a relative 1.3e-6 and an absolute
/// 1e-5; and what only moves values moves them exactly.
/// </summary>
/// <param name="engine">The engine held to the light one.</param>
public abstract class AgreementContract(ITensorBackend engine)
{
    /// <summary>A float's unit roundoff: two to the minus twenty-four.</summary>
    public const double Roundoff = 1.0 / 16777216;

    /// <summary>How many roundings of its terms' size a total may lie from the light engine's.</summary>
    public const double Roundings = 3;

    /// <summary>PyTorch's relative tolerance for float values.</summary>
    public const double RelativeTolerance = 1.3e-6;

    /// <summary>PyTorch's absolute tolerance for float values.</summary>
    public const double AbsoluteTolerance = 1e-5;

    /// <summary>The most a total may lie from another engine's: three roundings of the size of the terms it adds up.</summary>
    /// <param name="termsSize">The sum of the terms' sizes, each as large as it is.</param>
    /// <returns>The distance.</returns>
    public static double Bound(double termsSize) => Roundings * Roundoff * termsSize;

    private static readonly Window Walking = new(3, 3) { Padding = 1 };

    // A border TensorFlow splits unevenly on both axes: rows one before and two after, columns none before and one after.
    private static readonly Window Same = new(4, 3) { Stride = 2, PaddingMode = PaddingMode.Same };

    private static readonly Shape Images = new(2, 6, 6, 3);

    private static readonly Shape Uneven = new(2, 7, 8, 3);

    // Each total an engine works out, from values drawn once: worked out on an engine from the values as they are drawn, and
    // — the same total of the terms' sizes — from the values as large as they are, on the light engine, which rounds once.
    private static readonly Dictionary<string, Func<ITensorBackend, Draw, Tensor>> Totals = new()
    {
        ["MatMul"] = (engine, draw) => engine.MatMul(draw(new Shape(32, 64), 1), draw(new Shape(64, 16), 2)),
        ["SumRows"] = (engine, draw) => engine.SumRows(draw(new Shape(256, 8), 3)),
        ["Mean"] = (engine, draw) => engine.Mean(draw(new Shape(1000), 4)),
        ["Fold"] = (engine, draw) => engine.Fold(draw(Walking.PatchesOf(Images), 5), Images, Walking),
        ["Fold padded as 'same'"] = (engine, draw) => engine.Fold(draw(Same.PatchesOf(Uneven), 11), Uneven, Same),
    };

    // Each value-by-value operation, from values drawn once where it is defined.
    private static readonly Dictionary<string, Func<ITensorBackend, Tensor>> ValueByValue = new()
    {
        ["Add"] = engine => engine.Add(Drawn(new Shape(8, 8), 6), Drawn(new Shape(8, 8), 7)),
        ["Subtract"] = engine => engine.Subtract(Drawn(new Shape(8, 8), 6), Drawn(new Shape(8, 8), 7)),
        ["Multiply"] = engine => engine.Multiply(Drawn(new Shape(8, 8), 6), Drawn(new Shape(8, 8), 7)),
        ["Divide"] = engine => engine.Divide(Drawn(new Shape(8, 8), 6), Between(new Shape(8, 8), 7, 0.3f, 3f)),
        ["Scale"] = engine => engine.Scale(Drawn(new Shape(8, 8), 6), Tensor.From(new Shape(), [0.7f])),
        ["AddRow"] = engine => engine.AddRow(Drawn(new Shape(8, 8), 6), Drawn(new Shape(8), 8)),
        ["Relu"] = engine => engine.Relu(Drawn(new Shape(8, 8), 6)),
        ["Positive"] = engine => engine.Positive(Drawn(new Shape(8, 8), 6)),
        ["Tanh"] = engine => engine.Tanh(Between(new Shape(8, 8), 6, -4f, 4f)),
        ["Sigmoid"] = engine => engine.Sigmoid(Between(new Shape(8, 8), 6, -10f, 10f)),
        ["Exp"] = engine => engine.Exp(Between(new Shape(8, 8), 6, -10f, 10f)),
        ["Log"] = engine => engine.Log(Between(new Shape(8, 8), 6, 1e-3f, 1e3f)),
        ["Sqrt"] = engine => engine.Sqrt(Between(new Shape(8, 8), 6, 0f, 1e3f)),
        ["Softplus"] = engine => engine.Softplus(Between(new Shape(8, 8), 6, -40f, 40f)),
        ["LogSoftmax"] = engine => engine.LogSoftmax(Between(new Shape(8, 10), 6, -30f, 30f)),
    };

    // Each operation that only moves values, or writes one.
    private static readonly Dictionary<string, Func<ITensorBackend, Tensor>> Moving = new()
    {
        ["Transpose"] = engine => engine.Transpose(Drawn(new Shape(5, 7), 9)),
        ["Reshape"] = engine => engine.Reshape(Drawn(new Shape(5, 7), 9), new Shape(7, 5)),
        ["Unfold"] = engine => engine.Unfold(Drawn(Images, 10), Walking with { Stride = 2 }),
        ["Unfold padded as 'same'"] = engine => engine.Unfold(Drawn(Uneven, 12), Same),
        ["Fill"] = engine => engine.Fill(new Shape(3, 4), 0.1f),
        ["FirstLargest"] = engine => engine.FirstLargest(Drawn(new Shape(8, 10), 11)),
    };

    private readonly ITensorBackend _backend = engine;

    public static TheoryData<string> TotalsWorkedOut => [.. Totals.Keys];

    public static TheoryData<string> ValuesWorkedOut => [.. ValueByValue.Keys];

    public static TheoryData<string> ValuesMoved => [.. Moving.Keys];

    [Theory]
    [MemberData(nameof(TotalsWorkedOut), MemberType = typeof(AgreementContract))]
    public void EveryTotal_ComesWithinThreeRoundingsOfTheSizeOfItsTerms_OfTheLightEnginesTotal(string operation)
    {
        var light = Totals[operation](new CpuBackend(), Drawn).Values.ToArray();
        var own = Totals[operation](_backend, Drawn).Values.ToArray();
        var sizes = Totals[operation](new CpuBackend(), Sized).Values.ToArray();

        Assert.Equal(light.Length, own.Length);

        for (var at = 0; at < light.Length; at++)
        {
            var apart = Math.Abs((double)own[at] - light[at]);

            Assert.True(
                apart <= Bound(sizes[at]),
                string.Create(CultureInfo.InvariantCulture, $"{operation}[{at}]: {own[at]} against {light[at]}, {apart / (Roundoff * sizes[at]):0.###} roundings of the terms' size {sizes[at]}"));
        }
    }

    [Theory]
    [MemberData(nameof(ValuesWorkedOut), MemberType = typeof(AgreementContract))]
    public void EveryValueWorkedOutValueByValue_ComesWithinPyTorchsFloatTolerance_OfTheLightEngines(string operation)
    {
        var light = ValueByValue[operation](new CpuBackend()).Values.ToArray();
        var own = ValueByValue[operation](_backend).Values.ToArray();

        Assert.Equal(light.Length, own.Length);

        for (var at = 0; at < light.Length; at++)
        {
            Assert.True(
                Math.Abs((double)own[at] - light[at]) <= AbsoluteTolerance + (RelativeTolerance * Math.Abs(light[at])),
                string.Create(CultureInfo.InvariantCulture, $"{operation}[{at}]: {own[at]} against {light[at]}"));
        }
    }

    [Theory]
    [MemberData(nameof(ValuesMoved), MemberType = typeof(AgreementContract))]
    public void WhatOnlyMovesValues_MovesThemExactly(string operation)
    {
        var light = Moving[operation](new CpuBackend());
        var own = Moving[operation](_backend);

        Assert.Equal(light.Shape, own.Shape);
        Assert.Equal(light.Values.ToArray(), own.Values.ToArray());
    }

    // Values between minus one and one, drawn from the run's own generator so every runtime draws the same.
    private static Tensor Drawn(Shape shape, int purpose)
    {
        var draws = new RandomStream(20260930).Draw($"agreement:{purpose}", 0, 0);
        var values = new float[shape.Count];

        for (var at = 0; at < values.Length; at++)
        {
            values[at] = (draws.NextSingle() * 2) - 1;
        }

        return Tensor.From(shape, values);
    }

    // The same values, each as large as it is: what the size of a total's terms is worked out from.
    private static Tensor Sized(Shape shape, int purpose) => Tensor.From(shape, [.. Drawn(shape, purpose).Values.ToArray().Select(Math.Abs)]);

    private static Tensor Between(Shape shape, int purpose, float low, float high)
    {
        var drawn = Drawn(shape, purpose).Values.ToArray();

        return Tensor.From(shape, [.. drawn.Select(value => low + ((high - low) * (value + 1) / 2))]);
    }

    // How the values a total adds up are drawn: as they are, or as large as they are.
    private delegate Tensor Draw(Shape shape, int purpose);
}

/// <summary>The patches a window takes of a batch of images, as the seam lays them out.</summary>
internal static class WindowPatchesExtensions
{
    extension(Window window)
    {
        /// <summary>A row for every image and every place the window stands, each as long as the window holds values.</summary>
        public Shape PatchesOf(Shape images) =>
            new(images[0] * window.RowsOver(images[1]) * window.ColumnsOver(images[2]), window.Height * window.Width * images[3]);
    }
}
