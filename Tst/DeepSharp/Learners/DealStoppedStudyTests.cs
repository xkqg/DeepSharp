// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A trial of several networks is judged by a <see cref="Study.Judge"/> once every deal has finished, and until now
/// there was no way to stop it between deals: a <see cref="Study.Pruner"/> hears a single network's own epochs, and is
/// refused beside a <see cref="Study.Judge"/>, so a trial of several networks ran every deal whatever its first deal
/// said. A <see cref="Study.DealStopper"/> hears the trial's own <see cref="Study.Judge"/> number after each deal that
/// finishes — the same number the trial is judged by — and may end the trial before its remaining deals train.
/// </summary>
public class DealStoppedStudyTests
{
    private static readonly SearchSpace Space = new([new LogRange("rate", 1e-3, 1e-1)]);

    private static Pipeline Passengers(Candidate candidate, int deal) =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("sex").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitAtRandom(0.70, 0.15, 100 + deal, 7)
            .FillMissing("age", With.Median)
            .EncodeCategories()
            .Normalise("age", Scale.MidRange)
            .Normalise("fare", Scale.MidRange)
            .Normalise("sibsp", Scale.MidRange)
            .Normalise("parch", Scale.MidRange)
            .Target("survived")
            .Build();

    // Trials > 1 throughout: a dealStopper that stops every trial it hears would leave Run() with no winner at all,
    // which is a different, already-tested refusal (no trial finished) — not what a deal stopper is about. Stopping
    // only the first trial (no trials before it yet) leaves the second to finish normally and win, so the stopped
    // trial's own fate can be read off result.Trials[0] beside a real winner.
    private static Study StudiedTogether(int trials, int deals, Func<DealsJudged, bool>? dealStopper) =>
        new(Space, candidate => [
            new NetworkDeclaration(Vocabularies.Torch).Dense(4).Relu().Dense(1).Adam(candidate.Number("rate")).BinaryCrossEntropy().Run(seed: 1, epochs: 2).Step(),
            new NetworkDeclaration(Vocabularies.Torch).Dense(2).Relu().Dense(1).Adam(candidate.Number("rate")).BinaryCrossEntropy().Run(seed: 2, epochs: 2).Step()])
        {
            Sampler = new RandomSampler(11),
            Trials = trials,
            Deals = deals,
            Judge = networks => networks.Sum(network => network.ValidationLoss()),
            DealStopper = dealStopper,
        };

    private static readonly Func<DealsJudged, bool> StopOnlyTheFirstTrial = judged => judged.Earlier.Count == 0;

    private static IReadOnlyList<Pipeline> TwoNetworksOf(Candidate candidate, int deal) => [Passengers(candidate, deal), Passengers(candidate, deal)];

    [Fact]
    public void ADealStopperThatStopsTheFirstTrial_StopsItAfterItsFirstDeal_AndTheSecondTrialStillRunsEveryDeal()
    {
        var result = StudiedTogether(trials: 2, deals: 3, dealStopper: StopOnlyTheFirstTrial).Run(
            (candidate, deal) => TwoNetworksOf(candidate, deal), TestContext.Current.CancellationToken);

        Assert.True(result.Trials[0].Pruned);
        Assert.False(result.Trials[1].Pruned);
        Assert.Single(result.Trials[0].Scores);
        Assert.Equal(3, result.Trials[1].Scores.Count);
    }

    [Fact]
    public void ADealStopperThatNeverStops_HearsEveryDealButTheLast_InOrder()
    {
        var heard = new List<int>();

        StudiedTogether(trials: 1, deals: 3, dealStopper: judged =>
        {
            heard.Add(judged.Deal);

            return false;
        }).Run((candidate, deal) => TwoNetworksOf(candidate, deal), TestContext.Current.CancellationToken);

        // Three deals run; the stopper is asked after deal 0 and after deal 1, never after the last — nothing is
        // left to save by stopping once every deal has already trained.
        Assert.Equal([0, 1], heard);
    }

    [Fact]
    public void WhenADealStopperStopsATrial_ItsFinishedDealsScore_IsTheRealOneJudgeGave()
    {
        double? firstScore = null;

        var result = StudiedTogether(trials: 2, deals: 3, dealStopper: judged =>
        {
            if (judged.Earlier.Count == 0)
            {
                firstScore = judged.Scores[0];
            }

            return StopOnlyTheFirstTrial(judged);
        }).Run((candidate, deal) => TwoNetworksOf(candidate, deal), TestContext.Current.CancellationToken);

        Assert.NotNull(firstScore);
        Assert.True(double.IsFinite(firstScore!.Value));
        Assert.Equal([firstScore!.Value], result.Trials[0].Scores);
    }

    [Fact]
    public void ATrialADealStopperStopped_IsNotFinished_AndNeverWinsAStudy()
    {
        var result = StudiedTogether(trials: 2, deals: 3, dealStopper: StopOnlyTheFirstTrial).Run(
            (candidate, deal) => TwoNetworksOf(candidate, deal), TestContext.Current.CancellationToken);

        Assert.False(result.Trials[0].Finished);
        Assert.NotEqual(0, result.Best.Number);
    }

    [Fact]
    public void ADealStopper_MayStandBesideAJudge_WithoutBeingRefused()
    {
        // The standing rule stays: a Judge is refused beside a Pruner, because a Pruner would stop a trial by its own
        // network's raw validation loss, a number the trial is not judged by. A DealStopper hears Judge's own number,
        // so it is never refused beside one.
        var exception = Record.Exception(() => StudiedTogether(trials: 1, deals: 2, dealStopper: _ => false)
            .Run((candidate, deal) => TwoNetworksOf(candidate, deal), TestContext.Current.CancellationToken));

        Assert.Null(exception);
    }
}
