// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// A network trained behind variations is trained again to the same scores, goes on from a checkpoint as the straight run would have
/// gone, and is judged, and answers, by rows that were never varied.
/// </summary>
public class VariationTrainingTests
{
    private const int Examples = 48;

    // Images of 4 rows, 4 columns and a channel, the answer being the mean of the left half less the right half: an answer a mirror
    // reverses, so a flipped image is a different example.
    private static TrainingData Images()
    {
        var random = new Random(5);
        var pixels = new float[Examples * 16];
        var answers = new float[Examples];

        for (var at = 0; at < pixels.Length; at++)
        {
            pixels[at] = (float)((random.NextDouble() * 2) - 1);
        }

        for (var example = 0; example < Examples; example++)
        {
            var left = 0f;
            var right = 0f;

            for (var row = 0; row < 4; row++)
            {
                left += pixels[(example * 16) + (row * 4)] + pixels[(example * 16) + (row * 4) + 1];
                right += pixels[(example * 16) + (row * 4) + 2] + pixels[(example * 16) + (row * 4) + 3];
            }

            answers[example] = (left - right) / 8;
        }

        return new TrainingData(Tensor.From(new Shape(Examples, 4, 4, 1), pixels), Tensor.From(new Shape(Examples, 1), answers));
    }

    private static CompiledNetwork Described(bool varied, long seed = 3) =>
        (varied ? new Sequential().RandomFlip(horizontal: true, vertical: false, seed).GaussianNoise(spread: 0.05, seed + 1) : new Sequential())
            .Flatten()
            .Dense(4)
            .Relu()
            .Dense(1)
            .Compile(new Adam(0.01), new MeanSquaredError());

    private static double[] Losses(History history) => [.. history.Epochs.Select(epoch => epoch.Loss)];

    [Fact]
    public void ANetworkTrainedAgain_BehindTheSameVariations_GivesTheSameScores()
    {
        var first = Described(varied: true).Fit(Images(), null, new FitOptions(seed: 7) { Epochs = 5 });
        var again = Described(varied: true).Fit(Images(), null, new FitOptions(seed: 7) { Epochs = 5 });

        Assert.Equal(Losses(first), Losses(again));
    }

    [Fact]
    public void VariedRows_TrainToOtherScores_ThanRowsLeftAlone()
    {
        var varied = Described(varied: true).Fit(Images(), null, new FitOptions(seed: 7) { Epochs = 5 });
        var plain = Described(varied: false).Fit(Images(), null, new FitOptions(seed: 7) { Epochs = 5 });

        Assert.NotEqual(Losses(varied), Losses(plain));
    }

    [Fact]
    public void ARunGoneOnFromACheckpoint_DrawsTheVariationsTheStraightRunDrew()
    {
        var straight = Described(varied: true).Fit(Images(), null, new FitOptions(seed: 7) { Epochs = 6 });
        Checkpoint? kept = null;

        _ = Described(varied: true).Fit(Images(), null, new FitOptions(seed: 7)
        {
            Epochs = 3,
            Checkpoints = new Checkpoints(checkpoint => kept = checkpoint),
        });

        var goneOn = Described(varied: true).Fit(Images(), null, new FitOptions(seed: 7) { Epochs = 6, ResumeFrom = kept });

        Assert.Equal(Losses(straight)[3..], Losses(goneOn)[3..]);
    }

    [Fact]
    public void WhatIsAnsweredAndWhatJudges_IsNeverVaried()
    {
        var network = Described(varied: true);
        var train = Images();
        var history = network.Fit(train, train, new FitOptions(seed: 7) { Epochs = 3 });
        var backend = new CpuBackend();

        // Answering the same rows twice gives the same answers: an evaluation pass varies nothing.
        Assert.Equal(network.Predict(train.Features, backend).Values.ToArray(), network.Predict(train.Features, backend).Values.ToArray());
        Assert.All(history.Epochs, epoch => Assert.True(epoch.ValidationLoss.HasValue));
    }
}
