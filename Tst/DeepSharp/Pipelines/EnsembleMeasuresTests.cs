// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Several networks behind one fit answer as one, and the ensemble is measured beside each of its members on the very same rows. The
/// way back from the units a network was trained in is not a straight line — a log is undone by an exponential, and the mean of
/// exponentials is not the exponential of the mean — so every member comes back into the answer's own units first and the ensemble is
/// combined there: numbers are averaged; classes are averaged as chances or counted as votes, a tie going to the higher mean chance.
/// </summary>
public class EnsembleMeasuresTests
{
    private const double Close = 1e-9;

    private static readonly Part[] ThreeParts = [Part.Train, Part.Validation, Part.Test];

    // The answer is read on a log scale, so what a network says is a log and what a person reads is its exponential.
    private static PreparedData Priced() =>
        Pdd.Create()
            .Read(CsvRowSource.FromText("t,x,y\n" + string.Join('\n', Enumerable.Range(1, 20).Select(t => $"{t},{t % 5},{10 + (t * 7 % 13)}")) + "\n"), "twenty rows")
            .Declare(schema => schema.Integer("t", "x", "y"))
            .Reshape("y", Maths.Log)
            .SplitByTime("t", 0.50, 0.25)
            .Target("y")
            .Report(report => report.Measure(Metric.Rmse, Metric.Mae).On(ThreeParts).As(Shown.Numbers))
            .Build()
            .Run();

    private static PreparedData Passengers(Action<ReportBuilder> report) =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "pclass").Category("sex"))
            .SplitStratified("survived", 0.70, 0.15)
            .EncodeCategories()
            .Target("survived")
            .Report(report)
            .Build()
            .Run();

    private static PreparedData Labelled() =>
        Pdd.Create()
            .Read(
                new InMemoryRowSource(
                    ["t", "a", "b", "c"],
                    [.. Enumerable.Range(1, 20).Select(t => new string?[] { $"{t}", t % 3 == 0 ? "1" : "0", t % 3 == 1 ? "1" : "0", t % 3 == 2 ? "1" : "0" })]),
                "twenty rows")
            .Declare(schema => schema.Integer("t", "a", "b", "c"))
            .SplitByTime("t", 0.50, 0.25)
            .Labels(["a", "b", "c"], ones: 1)
            .Report(report => report.Measure(Metric.Accuracy, Metric.ConfusionMatrix).On(ThreeParts).As(Shown.Numbers))
            .Build()
            .Run();

    // What a member says of each row of every part the report names, in the units it was handed over in.
    private static IReadOnlyList<PartPredictions> Said(PreparedData prepared, Func<Batch, int, double[]> member) =>
    [
        .. prepared.Declaration.Report!.Parts.Select(part =>
        {
            var batch = prepared.Batch(part);

            return new PartPredictions(batch, [.. Enumerable.Range(0, batch.RowCount).Select(row => member(batch, row))]);
        }),
    ];

    private static Func<Batch, int, double[]> Off(double by) => (batch, row) => [batch.Answers![row][0] + by];

    private static Func<Batch, int, double[]> Always(params double[] chances) => (_, _) => chances;

    [Fact]
    public void TheMeanOfMembers_IsTakenInTheAnswersOwnUnits_NotInTheUnitsTheyWereTrainedIn()
    {
        var prepared = Priced();
        var measured = prepared.MeasureEnsemble([Said(prepared, Off(+0.3)), Said(prepared, Off(-0.3))]);

        foreach (var part in measured.Ensemble.Parts)
        {
            for (var row = 0; row < part.Rows; row++)
            {
                // exp(y + d) and exp(y - d) average to y * cosh(d), which is not exp of their mean, y.
                Assert.Equal(part.Actual[row][0] * Math.Cosh(0.3), part.Predicted[row][0], Close);
                Assert.NotEqual(part.Actual[row][0], part.Predicted[row][0], Close);
            }
        }

        Assert.Equal(Blend.Average, measured.Blend);
    }

    [Fact]
    public void TheEnsemble_IsMeasuredBesideEachMember_OnTheSameRowsBySameReport()
    {
        var prepared = Priced();
        var measured = prepared.MeasureEnsemble([Said(prepared, Off(+0.3)), Said(prepared, Off(-0.1)), Said(prepared, Off(0))]);

        Assert.Equal(3, measured.Members.Count);

        for (var at = 0; at < measured.Ensemble.Parts.Count; at++)
        {
            var part = measured.Ensemble.Parts[at];

            Assert.All(measured.Members, member => Assert.Equal(part.Part, member.Parts[at].Part));
            Assert.All(measured.Members, member => Assert.Equal(part.Actual, member.Parts[at].Actual));
            Assert.All(measured.Members, member => Assert.Equal(part.Rows, member.Parts[at].Rows));

            // The ensemble's rmse is worked out from its own combined predictions, and the baseline stands beside it unchanged.
            var rmse = Math.Sqrt(Enumerable.Range(0, part.Rows).Average(row => Math.Pow(part.Actual[row][0] - part.Predicted[row][0], 2)));

            Assert.Equal(rmse, part.Values.Single(value => value.Metric == Metric.Rmse).Value, Close);
            Assert.Equal(measured.Members[0].Parts[at].Values.Single(value => value.Metric == Metric.Rmse).Baseline, part.Values.Single(value => value.Metric == Metric.Rmse).Baseline, Close);
        }

        // A member measured here is the member measured alone.
        Assert.Equal(
            prepared.Measure(Said(prepared, Off(+0.3))).Parts[2].Values.Single(value => value.Metric == Metric.Mae).Value,
            measured.Members[0].Parts[2].Values.Single(value => value.Metric == Metric.Mae).Value,
            Close);
    }

    [Fact]
    public void Chances_AreAveraged_ByDefault()
    {
        var prepared = Passengers(report => report.Measure(Metric.Accuracy, Metric.Brier).On(ThreeParts).As(Shown.Numbers));
        var measured = prepared.MeasureEnsemble([Said(prepared, Always(0.2)), Said(prepared, Always(0.6))]);

        Assert.All(measured.Ensemble.Parts, part => Assert.All(part.Predicted, row => Assert.Equal(0.4, row[0], Close)));
        Assert.Equal(Blend.Average, measured.Blend);
    }

    [Fact]
    public void Votes_AreCounted_AMajorityDecides_AndATieGoesToTheHigherMeanChance()
    {
        var prepared = Passengers(report => report.Measure(Metric.Accuracy).On(ThreeParts).As(Shown.Numbers));

        // Two of three say yes, whatever the third says.
        var majority = prepared.MeasureEnsemble([Said(prepared, Always(0.9)), Said(prepared, Always(0.8)), Said(prepared, Always(0.1))], Blend.Vote);

        // One yes against one no is a tie: the members' mean chance, 0.6 here, is a yes, and 0.35 below is a no.
        var tieUp = prepared.MeasureEnsemble([Said(prepared, Always(0.9)), Said(prepared, Always(0.3))], Blend.Vote);
        var tieDown = prepared.MeasureEnsemble([Said(prepared, Always(0.6)), Said(prepared, Always(0.1))], Blend.Vote);

        Assert.All(majority.Ensemble.Parts, part => Assert.All(part.Predicted, row => Assert.Equal(1.0, row[0])));
        Assert.All(tieUp.Ensemble.Parts, part => Assert.All(part.Predicted, row => Assert.Equal(1.0, row[0])));
        Assert.All(tieDown.Ensemble.Parts, part => Assert.All(part.Predicted, row => Assert.Equal(0.0, row[0])));
        Assert.Equal(Blend.Vote, majority.Blend);
    }

    [Fact]
    public void Votes_OfLabelsARowHoldsOneOf_GoToTheLabelMostMembersNamed_ATieToTheHigherMeanChance()
    {
        var prepared = Labelled();

        // Two members name different labels: a (0.6) and b (0.5); the mean chances are a 0.35 and b 0.4, so b wins the tie.
        var tie = prepared.MeasureEnsemble([Said(prepared, Always(0.6, 0.3, 0.1)), Said(prepared, Always(0.1, 0.5, 0.4))], Blend.Vote);
        var majority = prepared.MeasureEnsemble([Said(prepared, Always(0.6, 0.3, 0.1)), Said(prepared, Always(0.5, 0.4, 0.1)), Said(prepared, Always(0.1, 0.5, 0.4))], Blend.Vote);

        Assert.All(tie.Ensemble.Parts, part => Assert.All(part.Predicted, row => Assert.Equal([0.0, 1.0, 0.0], row)));
        Assert.All(majority.Ensemble.Parts, part => Assert.All(part.Predicted, row => Assert.Equal([1.0, 0.0, 0.0], row)));
    }

    [Fact]
    public void VotesOfAnAmount_AreRefused_ForAnAmountIsNoClass()
    {
        var prepared = Priced();
        var refused = Assert.Throws<InvalidOperationException>(() => prepared.MeasureEnsemble([Said(prepared, Off(0.1)), Said(prepared, Off(-0.1))], Blend.Vote));

        Assert.Contains("class", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void FewerThanTwoMembers_AreRefused_AndMembersMeasuredOnOtherPartsAreRefused()
    {
        var prepared = Priced();
        var fewer = Said(prepared, Off(0.1)).Take(2).ToArray();

        Assert.Throws<ArgumentException>(() => prepared.MeasureEnsemble([Said(prepared, Off(0.1))]));
        Assert.Throws<ArgumentException>(() => prepared.MeasureEnsemble([]));
        Assert.Throws<ArgumentException>(() => prepared.MeasureEnsemble([Said(prepared, Off(0.1)), fewer]));
        Assert.Throws<ArgumentNullException>(() => prepared.MeasureEnsemble(null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => prepared.MeasureEnsemble([Said(prepared, Off(0.1)), Said(prepared, Off(-0.1))], (Blend)7));
    }

    [Fact]
    public void WhatTheEnsembleSaid_CannotCrossAsText_ButWhatEachMemberSaidCan()
    {
        var prepared = Priced();
        var measured = prepared.MeasureEnsemble([Said(prepared, Off(0.1)), Said(prepared, Off(-0.1))]);
        var refused = Assert.Throws<InvalidOperationException>(() => measured.Ensemble.PredictionsToJson());

        Assert.Contains("member", refused.Message, StringComparison.Ordinal);
        Assert.All(measured.Members, member => Assert.StartsWith("{", member.PredictionsToJson(), StringComparison.Ordinal));
    }

    [Fact]
    public void MembersCombined_ForServedRows_ComeBackThroughOneWalkAndAreCombinedInTheAnswersUnits()
    {
        var prepared = Priced();
        var handedIn = new InMemoryRowSource(["t", "x"], [.. Enumerable.Range(30, 3).Select(t => new string?[] { $"{t}", "1" })]);
        var served = prepared.Served(handedIn);
        IReadOnlyList<double[]>[] members = [[[2.0], [2.5], [3.0]], [[2.2], [2.3], [2.6]]];

        var combined = prepared.BackToOriginal(members, served, handedIn, Blend.Average);

        for (var row = 0; row < 3; row++)
        {
            Assert.Equal((prepared.BackToOriginal(members[0][row][0]) + prepared.BackToOriginal(members[1][row][0])) / 2, combined[row][0], Close);
        }

        Assert.Equal(prepared.BackToOriginal(members[0], served, handedIn).Select(row => row[0]), prepared.BackToOriginal([members[0], members[0]], served, handedIn, Blend.Average).Select(row => row[0]));
    }
}
