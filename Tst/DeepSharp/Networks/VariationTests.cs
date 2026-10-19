// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Buffers;
using System.Text;
using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A variation of the rows a network learns from, and of those alone: an image flipped, shifted or noised while the network trains,
/// never while it answers. Each has a seed of its own, which keys a stream of its own at the epoch and the step, so the same run
/// varies the same images the same way, a resumed run varies them as it would have, and the run's own seed moves none of it.
/// </summary>
public class VariationTests
{
    private const int Images = 64;

    private static readonly Shape Each = new(Images, 4, 5, 2);

    private static Tensor Batch()
    {
        var values = new float[Each.Count];

        for (var at = 0; at < values.Length; at++)
        {
            values[at] = at % 97 / 97f;
        }

        return Tensor.From(Each, values);
    }

    private static float[] Pixels(Tensor tensor) => tensor.Values.ToArray();

    private static Pass Training(long runSeed = 1, int epoch = 0, int step = 0) => Pass.Training(new CpuBackend(), new RandomStream(runSeed), epoch, step);

    private static Pass Evaluating() => Pass.Evaluation(new CpuBackend());

    // The pixels of one image of the batch: rows, columns and channels, as a flat list.
    private static float[] Image(float[] batch, int at) => batch[(at * 4 * 5 * 2)..((at + 1) * 4 * 5 * 2)];

    private static float[] Mirrored(float[] image, bool horizontal, bool vertical)
    {
        var made = new float[image.Length];

        for (var row = 0; row < 4; row++)
        {
            for (var column = 0; column < 5; column++)
            {
                for (var channel = 0; channel < 2; channel++)
                {
                    var fromRow = vertical ? 3 - row : row;
                    var fromColumn = horizontal ? 4 - column : column;

                    made[(((row * 5) + column) * 2) + channel] = image[(((fromRow * 5) + fromColumn) * 2) + channel];
                }
            }
        }

        return made;
    }

    // ---- what every variation shares ----

    [Fact]
    public void WhileTheNetworkAnswers_NothingIsVaried_TheBatchComesBackAsItWas()
    {
        var batch = Batch();

        Assert.Same(batch, new RandomFlip(horizontal: true, vertical: true, seed: 3).Forward(batch, Evaluating()));
        Assert.Same(batch, new RandomCrop(pad: 1, seed: 3).Forward(batch, Evaluating()));
        Assert.Same(batch, new GaussianNoise(spread: 0.5, seed: 3).Forward(batch, Evaluating()));
    }

    [Fact]
    public void AVariation_IsDrawnFromItsOwnSeed_NotTheRunsSeed_AndFromTheEpochAndTheStep()
    {
        var flip = new RandomFlip(horizontal: true, vertical: false, seed: 11);
        var first = Pixels(flip.Forward(Batch(), Training(runSeed: 1)));
        var again = Pixels(flip.Forward(Batch(), Training(runSeed: 1)));
        var otherRun = Pixels(flip.Forward(Batch(), Training(runSeed: 999)));
        var otherEpoch = Pixels(flip.Forward(Batch(), Training(runSeed: 1, epoch: 1)));
        var otherStep = Pixels(flip.Forward(Batch(), Training(runSeed: 1, step: 1)));
        var otherSeed = Pixels(new RandomFlip(horizontal: true, vertical: false, seed: 12).Forward(Batch(), Training(runSeed: 1)));

        // The run's seed moves nothing; the layer's own seed, the epoch and the step each do — and a resumed run draws what it would have.
        Assert.Equal(first, again);
        Assert.Equal(first, otherRun);
        Assert.NotEqual(first, otherEpoch);
        Assert.NotEqual(first, otherStep);
        Assert.NotEqual(first, otherSeed);
    }

    [Fact]
    public void TwoVariationsOfOneSeed_AtDifferentPlaces_DrawDifferently()
    {
        var network = new LayerStack(new RandomFlip(horizontal: true, vertical: false, seed: 5), new RandomFlip(horizontal: true, vertical: false, seed: 5));
        var held = network.HeldLayers().ToArray();
        var one = Pixels(held[0].Layer.Forward(Batch(), Training()));
        var two = Pixels(held[1].Layer.Forward(Batch(), Training()));

        Assert.NotEqual(one, two);
    }

    // ---- a flip ----

    [Fact]
    public void AFlip_MirrorsEachImageOrLeavesItAlone_AboutHalfOfThemEach()
    {
        var batch = Pixels(Batch());
        var flipped = Pixels(new RandomFlip(horizontal: true, vertical: false, seed: 9).Forward(Batch(), Training()));
        var mirrors = 0;

        for (var at = 0; at < Images; at++)
        {
            var image = Image(flipped, at);

            if (image.SequenceEqual(Mirrored(Image(batch, at), horizontal: true, vertical: false)))
            {
                mirrors++;
            }
            else
            {
                Assert.Equal(Image(batch, at), image);
            }
        }

        Assert.InRange(mirrors, 16, 48);
    }

    [Fact]
    public void AFlipOnBothAxes_IsOneOfFourWaysAnImageCanBeTurned()
    {
        var batch = Pixels(Batch());
        var flipped = Pixels(new RandomFlip(horizontal: true, vertical: true, seed: 9).Forward(Batch(), Training()));
        var seen = new HashSet<(bool, bool)>();

        for (var at = 0; at < Images; at++)
        {
            var image = Image(flipped, at);
            var ways = new[] { (false, false), (true, false), (false, true), (true, true) }
                .Where(way => Mirrored(Image(batch, at), way.Item1, way.Item2).SequenceEqual(image))
                .ToArray();

            Assert.NotEmpty(ways);
            seen.Add(ways[0]);
        }

        Assert.Equal(4, seen.Count);
    }

    [Fact]
    public void AFlipThatFlipsNothing_LeavesTheBatchAlone_AndNeedsImages()
    {
        var batch = Batch();

        Assert.Equal(Pixels(batch), Pixels(new RandomFlip(horizontal: false, vertical: false, seed: 1).Forward(batch, Training())));
        Assert.Throws<ArgumentException>(() => new RandomFlip(horizontal: true, vertical: false, seed: 1).Forward(Tensor.From(new Shape(2, 3), new float[6]), Training()));
    }

    // ---- a crop ----

    [Fact]
    public void ACrop_KeepsTheShape_EachImageShiftedByUpToThePadAndPaddedWithNought()
    {
        var batch = Pixels(Batch());
        var cropped = new RandomCrop(pad: 1, seed: 4).Forward(Batch(), Training());
        var shifts = new HashSet<(int, int)>();

        Assert.Equal(Each, cropped.Shape);

        for (var at = 0; at < Images; at++)
        {
            var image = Image(Pixels(cropped), at);
            var found = false;

            for (var down = -1; down <= 1 && !found; down++)
            {
                for (var right = -1; right <= 1 && !found; right++)
                {
                    if (image.SequenceEqual(Shifted(Image(batch, at), down, right)))
                    {
                        found = true;
                        shifts.Add((down, right));
                    }
                }
            }

            Assert.True(found, $"image {at} is no shift of itself by a pad of one");
        }

        Assert.True(shifts.Count > 4, $"{shifts.Count} different shifts");
    }

    private static float[] Shifted(float[] image, int down, int right)
    {
        var made = new float[image.Length];

        for (var row = 0; row < 4; row++)
        {
            for (var column = 0; column < 5; column++)
            {
                var fromRow = row + down;
                var fromColumn = column + right;

                for (var channel = 0; channel < 2; channel++)
                {
                    made[(((row * 5) + column) * 2) + channel] = fromRow is >= 0 and < 4 && fromColumn is >= 0 and < 5 ? image[(((fromRow * 5) + fromColumn) * 2) + channel] : 0f;
                }
            }
        }

        return made;
    }

    [Fact]
    public void ACropWithNoPad_IsTheIdentity()
    {
        Assert.Equal(Pixels(Batch()), Pixels(new RandomCrop(pad: 0, seed: 1).Forward(Batch(), Training())));
    }

    // ---- noise ----

    [Fact]
    public void Noise_IsAddedWithTheSpreadAsked_AndTheShapeKept()
    {
        var batch = Pixels(Batch());
        var noised = new GaussianNoise(spread: 0.5, seed: 2).Forward(Batch(), Training());
        var added = Pixels(noised).Zip(batch, (after, before) => (double)(after - before)).ToArray();
        var mean = added.Average();
        var spread = Math.Sqrt(added.Sum(each => (each - mean) * (each - mean)) / (added.Length - 1));

        Assert.Equal(Each, noised.Shape);
        Assert.InRange(mean, -0.05, 0.05);
        Assert.InRange(spread, 0.45, 0.55);
    }

    [Fact]
    public void NoiseOfNoSpread_IsTheIdentity()
    {
        Assert.Equal(Pixels(Batch()), Pixels(new GaussianNoise(spread: 0, seed: 1).Forward(Batch(), Training())));
    }

    // ---- what each is, and is saved as ----

    [Fact]
    public void EachVariationIsRegistered_AndRebuiltFromWhatItWrote()
    {
        var catalog = NetworkCatalog.BuiltIn();
        var network = new LayerStack(new RandomFlip(horizontal: true, vertical: false, seed: 31), new RandomCrop(pad: 3, seed: 32), new GaussianNoise(spread: 0.25, seed: 33));
        var buffer = new ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, network, new MeanSquaredError());
            writer.WriteEndObject();
        }

        var read = NetworkDocument.ReadNetwork(Encoding.UTF8.GetString(buffer.WrittenSpan), "network", catalog);
        var layers = read.Network.HeldLayers().Select(held => held.Layer).ToArray();

        Assert.Contains("randomflip", catalog.Names);
        Assert.Contains("randomcrop", catalog.Names);
        Assert.Contains("gaussiannoise", catalog.Names);

        var flip = Assert.IsType<RandomFlip>(layers[0]);
        var crop = Assert.IsType<RandomCrop>(layers[1]);
        var noise = Assert.IsType<GaussianNoise>(layers[2]);

        Assert.Equal((true, false, 31L), (flip.Horizontal, flip.Vertical, flip.Seed));
        Assert.Equal((3, 32L), (crop.Pad, crop.Seed));
        Assert.Equal((0.25, 33L), (noise.Spread, noise.Seed));
    }

    [Fact]
    public void ADescribedNetworkStartsWithItsVariations_AndRefusesOneAfterALayerThatLearns()
    {
        var described = new Sequential().RandomFlip(horizontal: true, vertical: false, seed: 1).RandomCrop(pad: 1, seed: 2).GaussianNoise(spread: 0.1, seed: 3).Flatten().Dense(1);
        var stack = described.Lower(new Shape(4, 5, 2), new RandomStream(7));
        var layers = stack.HeldLayers().Select(held => held.Layer).ToArray();

        Assert.IsType<RandomFlip>(layers[0]);
        Assert.IsType<RandomCrop>(layers[1]);
        Assert.IsType<GaussianNoise>(layers[2]);
        Assert.Throws<InvalidOperationException>(() => new Sequential().Dense(4).RandomFlip(horizontal: true, vertical: false, seed: 1));
        Assert.Throws<InvalidOperationException>(() => new Sequential().Flatten().GaussianNoise(spread: 0.1, seed: 1));
        Assert.Throws<ArgumentException>(() => new Sequential().RandomCrop(pad: 1, seed: 1).Dense(1).Lower(new Shape(10), new RandomStream(7)));
        Assert.Throws<ArgumentOutOfRangeException>(() => new Sequential().RandomCrop(pad: -1, seed: 1));
        Assert.Throws<ArgumentException>(() => new Sequential().RandomFlip(horizontal: true, vertical: false, seed: 1).Dense(1).Lower(new Shape(10), new RandomStream(7)));
    }

    [Fact]
    public void ASettingThatIsNoVariation_IsRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new RandomCrop(pad: -1, seed: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GaussianNoise(spread: -0.1, seed: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new GaussianNoise(spread: double.NaN, seed: 1));
    }
}
