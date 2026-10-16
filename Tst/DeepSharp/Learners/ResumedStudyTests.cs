// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A study that took hours can be handed the trials it already tried and go on from there: they keep their numbers, the
/// sampler is handed them as it was, and the study that goes on is the study that never stopped. The trials are the caller's
/// to keep; what the study holds them to is that they are the trials it would have tried — numbered from nought without a
/// gap, of its space, declaring what it declares — and, where the best of them is one it did not train, that training it
/// again gives what it was judged by, since that trial's networks are the ones it hands back.
/// </summary>
public class ResumedStudyTests
{
    private static readonly SearchSpace Space = new([new LogRange("rate", 1e-3, 1e-1), new WholeRange("units", 2, 8)]);

    private static Study Studied(int trials) =>
        new(Space, (candidate, network) => network
            .Dense(candidate.Whole("units")).Relu().Dense(1)
            .Adam(candidate.Number("rate"))
            .BinaryCrossEntropy()
            .Run(seed: 20260929, epochs: 2))
        {
            Sampler = new RandomSampler(11),
            Trials = trials,
        };

    [Fact]
    public void AStudyHandedItsEarlierTrials_IsTheStudyThatNeverStopped()
    {
        var whole = Studied(5).Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);
        var first = Studied(3).Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);
        var resumed = Studied(5).Run(deal => StudyTests.Passengers(deal), first.Trials, TestContext.Current.CancellationToken);

        Assert.Equal(whole.Trials.Select(trial => trial.Number), resumed.Trials.Select(trial => trial.Number));
        Assert.Equal(whole.Trials.Select(trial => trial.Candidate), resumed.Trials.Select(trial => trial.Candidate));
        Assert.Equal(whole.Trials.Select(trial => trial.Scores), resumed.Trials.Select(trial => trial.Scores));
        Assert.Equal(whole.Best.Number, resumed.Best.Number);
        Assert.Equal(
            whole.Winner.Select(network => network.Network.Parameters().Select(parameter => parameter.Value.Values.ToArray())),
            resumed.Winner.Select(network => network.Network.Parameters().Select(parameter => parameter.Value.Values.ToArray())));
    }

    [Fact]
    public void WhenTheBestTrialIsAnEarlierOne_ItIsTrainedAgain_AndItsNetworksAreTheOnesHandedBack()
    {
        var first = Studied(3).Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);
        var resumed = Studied(3).Run(deal => StudyTests.Passengers(deal), first.Trials, TestContext.Current.CancellationToken);

        Assert.Equal(first.Best.Number, resumed.Best.Number);
        Assert.Equal(
            first.Winner.Select(network => network.Network.Parameters().Select(parameter => parameter.Value.Values.ToArray())),
            resumed.Winner.Select(network => network.Network.Parameters().Select(parameter => parameter.Value.Values.ToArray())));
    }

    [Fact]
    public void AnEarlierBestThatTrainsAgainToOtherScores_IsRefused_NamingTheTrial()
    {
        var first = Studied(3).Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);
        Trial[] altered = [.. first.Trials.Select(trial => trial.Number == first.Best.Number ? trial with { Scores = [trial.Scores[0] / 2] } : trial)];

        var refused = Assert.Throws<InvalidOperationException>(() => Studied(3).Run(deal => StudyTests.Passengers(deal), altered, TestContext.Current.CancellationToken));

        Assert.Contains($"trial {first.Best.Number}", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void EarlierTrialsWithAGap_OrOfAnotherSpace_OrDeclaringAnotherNetwork_AreRefused()
    {
        var first = Studied(3).Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);
        var other = first.Trials[1] with { Candidate = new Candidate([new Setting("momentum", 0.9, null)]) };
        var redeclared = first.Trials[1] with { Declared = first.Trials[0].Declared };

        Assert.Throws<InvalidOperationException>(() => Studied(4).Run(deal => StudyTests.Passengers(deal), [first.Trials[0], first.Trials[2]], TestContext.Current.CancellationToken));
        Assert.Throws<InvalidOperationException>(() => Studied(4).Run(deal => StudyTests.Passengers(deal), [first.Trials[0], other], TestContext.Current.CancellationToken));
        Assert.Throws<InvalidOperationException>(() => Studied(4).Run(deal => StudyTests.Passengers(deal), [first.Trials[0], redeclared], TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentNullException>(() => Studied(4).Run(deal => StudyTests.Passengers(deal), null!, TestContext.Current.CancellationToken));
    }
}
