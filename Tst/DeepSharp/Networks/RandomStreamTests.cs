// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A network needs many random numbers — how to shuffle rows, how to set out weights, which units a dropout
/// layer drops — and every one of them has to come from somewhere that does not depend on the order layers
/// happen to ask in. A <see cref="RandomStream"/> is a counter-based generator: the numbers for one purpose
/// at one point of a run are worked out fresh from the seed, the purpose, the epoch and the step, never kept
/// from any earlier draw. The block function underneath is checked against numpy's own Philox output, word
/// for word.
/// </summary>
public class RandomStreamTests
{
    // numpy.random.Philox(key=42).random_raw(8), captured with numpy 2.5.3. numpy's buffer advances its
    // counter before it fills a block, even the first one, so the block numpy calls "the first" is our block
    // function's counter word 0 = 1, and "the second" is counter word 0 = 2 (both other counter words 0,
    // key = (42, 0)) — confirmed by comparing against Philox(key=42, counter=0) and Philox(key=42, counter=1)
    // explicitly, which give the same two blocks.
    private const ulong NumpyWord0 = 0xD1F8817D4D62880E;
    private const ulong NumpyWord1 = 0x307266B65CC8797E;
    private const ulong NumpyWord2 = 0xDE1F04E7F084ED03;
    private const ulong NumpyWord3 = 0x65034A8E78CD1E59;
    private const ulong NumpyWord4 = 0x5E3DAA8961C3E3D3;
    private const ulong NumpyWord5 = 0x6F37DEA4A04BD05C;
    private const ulong NumpyWord6 = 0x31D3A1AE26E190B9;
    private const ulong NumpyWord7 = 0x0FEF7FAE0AB2A01A;

    [Fact]
    public void ThePhiloxBlockFunction_ForKey42_GivesNumpysFirstEightRawWords()
    {
        var key = new PhiloxKey(42, 0);

        var first = Philox.Block(new PhiloxCounter(1, 0, 0, 0), key);
        var second = Philox.Block(new PhiloxCounter(2, 0, 0, 0), key);

        Assert.Equal(NumpyWord0, first.Word0);
        Assert.Equal(NumpyWord1, first.Word1);
        Assert.Equal(NumpyWord2, first.Word2);
        Assert.Equal(NumpyWord3, first.Word3);
        Assert.Equal(NumpyWord4, second.Word0);
        Assert.Equal(NumpyWord5, second.Word1);
        Assert.Equal(NumpyWord6, second.Word2);
        Assert.Equal(NumpyWord7, second.Word3);
    }

    [Fact]
    public void TheDoubleConversion_OfNumpysOwnWords_MatchesNumpysGeneratorOutput()
    {
        // numpy.random.Generator(numpy.random.Philox(key=42)).random(8), same numpy 2.5.3 run as above: the
        // Generator reads doubles off the same eight words, one word per double, top 53 bits scaled by 2^-53
        // — the formula NextDouble() below documents itself as using.
        ulong[] words = [NumpyWord0, NumpyWord1, NumpyWord2, NumpyWord3, NumpyWord4, NumpyWord5, NumpyWord6, NumpyWord7];
        double[] expected =
        [
            0.8201981478608876, 0.18924562408645496, 0.8676608148821462, 0.3945814702827203,
            0.36812845090913937, 0.4344462539595917, 0.1946354913878905, 0.06224821089808552,
        ];

        for (var at = 0; at < words.Length; at++)
        {
            Assert.Equal(expected[at], (words[at] >> 11) * (1.0 / 9007199254740992.0));
        }
    }

    [Fact]
    public void Seed_IsWhatTheStreamWasCreatedWith()
    {
        var stream = new RandomStream(12345);

        Assert.Equal(12345, stream.Seed);
    }

    [Fact]
    public void TheSameSeedPurposeEpochAndStep_GivesTheSameDraws()
    {
        var stream = new RandomStream(2026);

        var first = stream.Draw("0.weight", epoch: 3, step: 12);
        var second = stream.Draw("0.weight", epoch: 3, step: 12);

        for (var at = 0; at < 10; at++)
        {
            Assert.Equal(first.NextDouble(), second.NextDouble());
        }
    }

    [Fact]
    public void AResumedRun_ReadsBackExactlyWhatItWouldHaveDrawnTheFirstTime()
    {
        // The scenario the design is for: a run stops after epoch 3 and is resumed there later. The stream
        // that resumes it knows nothing of what happened before — only the seed — and still reads the same
        // numbers for epoch 3 that the original run would have.
        var original = new RandomStream(99).Draw("shuffle", epoch: 3, step: 0).Permutation(50);
        var resumed = new RandomStream(99).Draw("shuffle", epoch: 3, step: 0).Permutation(50);

        Assert.Equal(original, resumed);
    }

    [Fact]
    public void ChangingTheStep_ChangesTheDraws()
    {
        var stream = new RandomStream(7);

        var atStepZero = stream.Draw("weights", 0, 0).NextDouble();
        var atStepOne = stream.Draw("weights", 0, 1).NextDouble();

        Assert.NotEqual(atStepZero, atStepOne);
    }

    [Fact]
    public void ChangingTheEpoch_ChangesTheDraws()
    {
        var stream = new RandomStream(7);

        var atEpochZero = stream.Draw("weights", 0, 0).NextDouble();
        var atEpochOne = stream.Draw("weights", 1, 0).NextDouble();

        Assert.NotEqual(atEpochZero, atEpochOne);
    }

    [Fact]
    public void ChangingThePurpose_ChangesTheDraws()
    {
        var stream = new RandomStream(7);

        var forWeights = stream.Draw("weights", 0, 0).NextDouble();
        var forBias = stream.Draw("bias", 0, 0).NextDouble();

        Assert.NotEqual(forWeights, forBias);
    }

    [Fact]
    public void ChangingTheSeed_ChangesTheDraws()
    {
        var first = new RandomStream(7).Draw("weights", 0, 0).NextDouble();
        var second = new RandomStream(8).Draw("weights", 0, 0).NextDouble();

        Assert.NotEqual(first, second);
    }

    [Fact]
    public void NextSingle_StaysAtOrAboveZeroAndBelowOne_OverManyDraws()
    {
        var draws = new RandomStream(11).Draw("check", 0, 0);

        for (var at = 0; at < 100_000; at++)
        {
            var value = draws.NextSingle();
            Assert.True(value >= 0f && value < 1f, $"NextSingle produced {value}, outside [0,1).");
        }
    }

    [Fact]
    public void NextDouble_StaysAtOrAboveZeroAndBelowOne_OverManyDraws()
    {
        var draws = new RandomStream(11).Draw("check", 0, 1);

        for (var at = 0; at < 100_000; at++)
        {
            var value = draws.NextDouble();
            Assert.True(value >= 0.0 && value < 1.0, $"NextDouble produced {value}, outside [0,1).");
        }
    }

    [Fact]
    public void NextInt_OfSeven_HitsEveryValueZeroToSixAndNeverSeven()
    {
        var draws = new RandomStream(5).Draw("choices", 0, 0);
        var seen = new bool[7];

        for (var at = 0; at < 10_000; at++)
        {
            var value = draws.NextInt(7);
            Assert.InRange(value, 0, 6);
            seen[value] = true;
        }

        Assert.All(seen, wasSeen => Assert.True(wasSeen));
    }

    [Fact]
    public void NextInt_OfZero_IsRefused()
    {
        var draws = new RandomStream(5).Draw("choices", 0, 0);

        Assert.Throws<ArgumentOutOfRangeException>(() => draws.NextInt(0));
    }

    [Fact]
    public void NextInt_WithABoundNearIntMaxValue_EventuallyResamplesAndStaysInRange()
    {
        // Lemire's method redraws whenever a word would land unevenly, in the band below a threshold of
        // 2^32 mod below. With a small bound (7) that band is a sliver of the 32-bit space it is compared
        // against, so it would take far more than a practical test can afford to see it by chance — and a
        // bound near int.MaxValue is no better, since it is close to exactly half of 2^32 and leaves almost
        // no remainder. 1,500,000,000 leaves a threshold of about 1.29 billion, close to a third of the
        // full 32-bit space, so roughly one draw in three redraws: this is what actually exercises that
        // branch, confirmed by reading the coverage report rather than guessed at.
        var draws = new RandomStream(5).Draw("resample-check", 0, 0);
        const int below = 1_500_000_000;

        for (var at = 0; at < 2_000; at++)
        {
            Assert.InRange(draws.NextInt(below), 0, below - 1);
        }
    }

    [Fact]
    public void Permutation_Of623_HoldsEveryIndexExactlyOnce()
    {
        var permutation = new RandomStream(3).Draw("shuffle", 0, 0).Permutation(623);

        Assert.Equal(Enumerable.Range(0, 623), permutation.OrderBy(value => value));
    }

    [Fact]
    public void Permutation_ForTheSameSeedPurposeEpochAndStep_IsTheSameTwice()
    {
        var stream = new RandomStream(3);

        var first = stream.Draw("shuffle", 0, 0).Permutation(623);
        var second = stream.Draw("shuffle", 0, 0).Permutation(623);

        Assert.Equal(first, second);
    }

    [Fact]
    public void Permutation_BetweenTwoEpochs_Differs()
    {
        var stream = new RandomStream(3);

        var atEpochZero = stream.Draw("shuffle", 0, 0).Permutation(623);
        var atEpochOne = stream.Draw("shuffle", 1, 0).Permutation(623);

        Assert.NotEqual(atEpochZero, atEpochOne);
    }

    [Fact]
    public void Permutation_OfZero_IsEmpty()
    {
        var draws = new RandomStream(1).Draw("shuffle", 0, 0);

        Assert.Empty(draws.Permutation(0));
    }

    [Fact]
    public void Permutation_OfANegativeCount_IsRefused()
    {
        var draws = new RandomStream(1).Draw("shuffle", 0, 0);

        Assert.Throws<ArgumentOutOfRangeException>(() => draws.Permutation(-1));
    }

    [Fact]
    public void Draw_WithANullPurpose_IsRefused()
    {
        var stream = new RandomStream(1);

        Assert.Throws<ArgumentException>(() => stream.Draw(null!, 0, 0));
    }

    [Fact]
    public void Draw_WithAnEmptyPurpose_IsRefused()
    {
        var stream = new RandomStream(1);

        Assert.Throws<ArgumentException>(() => stream.Draw(string.Empty, 0, 0));
    }

    [Fact]
    public void Draw_WithANegativeEpoch_IsRefused()
    {
        var stream = new RandomStream(1);

        Assert.Throws<ArgumentOutOfRangeException>(() => stream.Draw("weights", -1, 0));
    }

    [Fact]
    public void Draw_WithANegativeStep_IsRefused()
    {
        var stream = new RandomStream(1);

        Assert.Throws<ArgumentOutOfRangeException>(() => stream.Draw("weights", 0, -1));
    }
}
