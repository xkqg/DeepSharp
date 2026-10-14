// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text.Json;
using DeepSharp.Tensors;
using DeepSharp.Tests.Backends.Parts;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// The larger of two tensors, the smaller, and a tensor held between minus a bound and the bound: made of the operations
/// every engine has, so nothing is added to the seam and every engine gives the same numbers. Held to PyTorch's
/// <c>maximum</c>, <c>minimum</c> and <c>clamp</c> to the bit — nought of either sign counting as nought, since a step against
/// a gradient of minus nought is the step against one of nought — over the edges of the arithmetic and two thousand ordinary
/// pairs. The fixture is <c>Fixtures/selection.json</c>, which <c>selection-fixtures.py</c> wrote.
/// </summary>
public class TensorSelectionTests
{
    private static readonly JsonElement Fixture = JsonDocument.Parse(File.ReadAllText(Path.Join(AppContext.BaseDirectory, "Tensors", "Fixtures", "selection.json"))).RootElement;

    [Theory]
    [InlineData("cpu")]
    [InlineData("nativememory")]
    public void TheLarger_IsPyTorchsMaximum_ToTheBit(string engine) =>
        Same(Bits("maximum"), Engine(engine).Larger(First(), Second()));

    [Theory]
    [InlineData("cpu")]
    [InlineData("nativememory")]
    public void TheSmaller_IsPyTorchsMinimum_ToTheBit(string engine) =>
        Same(Bits("minimum"), Engine(engine).Smaller(First(), Second()));

    [Theory]
    [InlineData("cpu", "0.5")]
    [InlineData("cpu", "0.001")]
    [InlineData("cpu", "3.0")]
    [InlineData("cpu", "1.0")]
    [InlineData("nativememory", "0.5")]
    [InlineData("nativememory", "0.001")]
    [InlineData("nativememory", "3.0")]
    [InlineData("nativememory", "1.0")]
    public void AValueHeldBetweenMinusABoundAndTheBound_IsPyTorchsClamp_ToTheBit(string engine, string bound) =>
        Same(Fixture.GetProperty("clamped").GetProperty(bound).EnumerateArray().Select(Float).ToArray(), Engine(engine).Clamped(First(), float.Parse(bound, CultureInfo.InvariantCulture)));

    [Fact]
    public void ABoundThatIsNoNumberAboveNothing_IsRefused()
    {
        var engine = new CpuBackend();

        Assert.Throws<ArgumentOutOfRangeException>(() => engine.Clamped(Tensor.From(new Shape(1), [1f]), 0f));
        Assert.Throws<ArgumentOutOfRangeException>(() => engine.Clamped(Tensor.From(new Shape(1), [1f]), -1f));
        Assert.Throws<ArgumentOutOfRangeException>(() => engine.Clamped(Tensor.From(new Shape(1), [1f]), float.NaN));
    }

    [Fact]
    public void TwoTensorsOfOtherShapes_AreRefusedAsTheSeamRefusesThem()
    {
        var engine = new CpuBackend();

        Assert.Throws<ArgumentException>(() => engine.Larger(Tensor.From(new Shape(2), [1f, 2f]), Tensor.From(new Shape(3), [1f, 2f, 3f])));
        Assert.Throws<ArgumentException>(() => engine.Smaller(Tensor.From(new Shape(2), [1f, 2f]), Tensor.From(new Shape(3), [1f, 2f, 3f])));
    }

    private static ITensorBackend Engine(string name) => name == "cpu" ? new CpuBackend() : new NativeMemoryBackend();

    private static Tensor First() => Of("first");

    private static Tensor Second() => Of("second");

    private static Tensor Of(string name) => Tensor.From(new Shape(Fixture.GetProperty("pairs").GetInt32()), Bits(name));

    private static float[] Bits(string name) => [.. Fixture.GetProperty(name).EnumerateArray().Select(Float)];

    private static float Float(JsonElement bits) => BitConverter.Int32BitsToSingle(unchecked((int)bits.GetUInt32()));

    // Nought of either sign is nought; every other number is the same 32 bits.
    private static void Same(float[] expected, Tensor made)
    {
        var values = made.Values;

        Assert.Equal(expected.Length, values.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            if (expected[at] is 0f && values[at] is 0f)
            {
                continue;
            }

            Assert.True(
                BitConverter.SingleToInt32Bits(expected[at]) == BitConverter.SingleToInt32Bits(values[at]),
                string.Create(CultureInfo.InvariantCulture, $"Pair {at}: PyTorch makes {expected[at]:R}, and this makes {values[at]:R}."));
        }
    }
}
