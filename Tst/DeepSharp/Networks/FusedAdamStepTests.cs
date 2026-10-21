// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers.Binary;
using System.Security.Cryptography;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using DeepSharp.Tests.Backends.Parts;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// Adam's step is about twenty operations of the engine for every parameter — the means, their corrections, the root, the
/// quotient, the move — each made into a tensor of its own. The light engine takes the whole step in one pass over the
/// numbers, in the order and with the roundings the twenty make, so it moves a parameter to the very bits they do; an engine
/// that cannot is asked for the twenty, as every engine always was.
/// </summary>
public class FusedAdamStepTests
{
    // What the twenty operations made of the walks below, as 0.13.0 made them: a step that moved one bit of one number moves it.
    private const string ChainDigest = "4BF775AC7434113DE9DF36A3736F9E3261BDCF446A4D8A46830B35EDFE1396DD";

    public static TheoryData<string> Walks { get; } = [.. Variants().Select(variant => variant.Name)];

    [Theory]
    [MemberData(nameof(Walks))]
    public void AStepInOnePass_MovesEveryParameterAndEveryMeanToTheBitsOfTheTwentyOperations(string walk)
    {
        var variant = Variants().Single(each => each.Name == walk);
        var fused = new Walked(variant.Make(), new CpuBackend());
        var chain = new Walked(variant.Make(), new CopyCountingBackend());

        for (var step = 1; step <= 7; step++)
        {
            fused.Step(step);
            chain.Step(step);

            Assert.True(fused.Bits().SequenceEqual(chain.Bits()), $"{walk}: after step {step} a parameter or a mean has other bits than the twenty operations make.");
        }
    }

    [Fact]
    public void TheWalksTheChainMade_AreTheDigestKeptHere()
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);

        foreach (var chain in Variants().Select(variant => new Walked(variant.Make(), new CopyCountingBackend())))
        {
            for (var step = 1; step <= 7; step++)
            {
                chain.Step(step);
            }

            hash.AppendData(chain.Bits());
        }

        var digest = Convert.ToHexString(hash.GetHashAndReset());

        Assert.True(digest == ChainDigest, $"The twenty operations digest to {digest}.");
    }

    [Fact]
    public void AStep_AllocatesTheTensorsItMakesAndNoMore()
    {
        const int Count = 4096;

        var optimizer = new Adam(0.01) { AmsGrad = true };
        var walked = new Walked(optimizer, new CpuBackend(), Count);

        walked.Step(1);
        walked.Step(2);

        var before = GC.GetAllocatedBytesForCurrentThread();

        walked.Step(3);

        var allocated = GC.GetAllocatedBytesForCurrentThread() - before;

        // The gradient and the gradients' dictionary the test builds, and the tensors the step makes: the parameter, the two
        // means and the largest of the squares. Twenty tensors' worth is what the operations made.
        Assert.InRange(allocated, 0, (8 * Count * 4) + 4096);
    }

    [Fact]
    public void AStepThatMakesNoNumber_IsTheTwentyOperationsToo_ForWhichOfTwoNotANumbersAnAdditionKeepsIsTheInstructionsToSay()
    {
        var fused = new Walked(new Adam(0.01), new CpuBackend());
        var chain = new Walked(new Adam(0.01), new CopyCountingBackend());

        fused.Step(1, Specials);
        chain.Step(1, Specials);
        fused.Step(2, Specials);
        chain.Step(2, Specials);

        Assert.True(fused.Bits().SequenceEqual(chain.Bits()));
    }

    private static readonly float[] Specials =
    [
        float.NaN, BitConverter.UInt32BitsToSingle(0x7FC00002), BitConverter.UInt32BitsToSingle(0xFFC00003), float.PositiveInfinity,
        float.NegativeInfinity, -0f, 0f, 1e30f, -1e30f, 1e-30f, float.Epsilon, 3f, -3f, 0.5f, -0.5f, 1f,
    ];

    private static IEnumerable<Variant> Variants() =>
    [
        new("Adam", () => new Adam(0.01)),
        new("Adam with AMSGrad", () => new Adam(0.01) { AmsGrad = true }),
        new("Adam with a weight decay", () => new Adam(0.01) { WeightDecay = 0.01 }),
        new("Adam with betas and an epsilon of its own", () => new Adam(0.003) { Betas = new Betas(0.8, 0.99), Epsilon = 1e-6 }),
        new("AdamW", () => new AdamW(0.01) { WeightDecay = 0.01 }),
        new("AdamW with AMSGrad", () => new AdamW(0.02) { WeightDecay = 0.1, AmsGrad = true }),
        new("AdamW without a decay", () => new AdamW(0.01) { WeightDecay = 0 }),
    ];

    private sealed record Variant(string Name, Func<Optimizer> Make);

    // One parameter of a few shapes, stepped by an optimizer on an engine, with the same gradients whichever engine it is.
    private sealed class Walked
    {
        private readonly Optimizer _optimizer;
        private readonly ITensorBackend _backend;
        private readonly Parameter[] _parameters;

        public Walked(Optimizer optimizer, ITensorBackend backend, int count = 0)
        {
            _optimizer = optimizer;
            _backend = backend;
            _parameters = count > 0
                ? [new Parameter("big", Tensor.From(new Shape(count), Draw(count, 9)))]
                : [new Parameter("w", Tensor.From(new Shape(5, 7), Draw(35, 1))), new Parameter("b", Tensor.From(new Shape(7), Draw(7, 2))), new Parameter("one", Tensor.From(new Shape(1), [0.5f]))];
        }

        public void Step(int step, float[]? specials = null)
        {
            var gradients = new Dictionary<Tensor, Tensor>(ReferenceEqualityComparer.Instance);

            for (var at = 0; at < _parameters.Length; at++)
            {
                var values = Draw(_parameters[at].Value.Shape.Count, (step * 31) + at);

                if (specials is not null)
                {
                    for (var spoil = 0; spoil < values.Length; spoil++)
                    {
                        values[spoil] = specials[(spoil + step) % specials.Length];
                    }
                }
                else if (step == 1)
                {
                    values[0] = -0f;
                }

                gradients[_parameters[at].Value] = Tensor.From(_parameters[at].Value.Shape, values);
            }

            _optimizer.Step(_parameters, new Gradients(gradients), 0.01, _backend);
        }

        // Every number the optimizer has moved or remembers, as the bits it holds: the parameters and then their means.
        public byte[] Bits()
        {
            var bits = new List<byte>();

            foreach (var parameter in _parameters)
            {
                Append(bits, parameter.Value);

                if (_optimizer.KeptOf(parameter) is { } memory)
                {
                    bits.AddRange(BitConverter.GetBytes(memory.Steps));

                    foreach (var name in memory.Tensors.Keys.Order(StringComparer.Ordinal))
                    {
                        Append(bits, memory.Tensors[name]);
                    }
                }
            }

            return [.. bits];
        }

        private static void Append(List<byte> bits, Tensor tensor)
        {
            Span<byte> word = stackalloc byte[4];

            foreach (var value in tensor.Values)
            {
                BinaryPrimitives.WriteUInt32LittleEndian(word, BitConverter.SingleToUInt32Bits(value));
                bits.AddRange(word.ToArray());
            }
        }

        private static float[] Draw(int count, int seed)
        {
            var random = new Random(seed);

            return [.. Enumerable.Range(0, count).Select(_ => (float)((random.NextDouble() * 2) - 1))];
        }
    }
}
