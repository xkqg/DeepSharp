// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// One split cannot say whether a change helped: the same network scores a few percent differently on two splits of the
/// same rows, as much as most changes move it. A paired comparison trains the one network behind two pipelines on each of
/// many deals and reports, deal by deal, how far the second sits from the first — with how much that moves, so a
/// difference is read against its own spread.
/// </summary>
public class ComparisonTests
{
    private static Comparison Compared(int deals = 3) =>
        new(network => network.Dense(3).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy().Run(seed: 20260929, epochs: 2))
        {
            Deals = deals,
        };

    [Fact]
    public void TheSamePipelineOnBothSides_DiffersFromItselfByNothing_OnEveryDeal()
    {
        var result = Compared().Run(deal => StudyTests.Passengers(deal), deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);

        Assert.Equal(3, result.Deals);
        Assert.All(result.Differences, difference => Assert.Equal(0, difference));
        Assert.Equal(0, result.Mean);
        Assert.Equal(0, result.Spread);
        Assert.Equal(result.Firsts, result.Seconds);
    }

    [Fact]
    public void TheDifference_IsHowFarTheSecondSitsFromTheFirst_DealByDeal()
    {
        var scores = new Queue<double>([1.0, 3.0, 2.0, 6.0, 4.0, 7.0]);
        var comparison = new Comparison(network => network.Dense(3).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy().Run(1, 1))
        {
            Deals = 3,
            Score = _ => scores.Dequeue(),
        };

        var result = comparison.Run(deal => StudyTests.Passengers(deal), deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);

        Assert.Equal([1.0, 2.0, 4.0], result.Firsts);
        Assert.Equal([3.0, 6.0, 7.0], result.Seconds);
        Assert.Equal([2.0, 4.0, 3.0], result.Differences);
        Assert.Equal(3.0, result.Mean);
        Assert.Equal(1.0, result.Spread);
        Assert.Equal(1.0 / Math.Sqrt(3), result.StandardError, 12);
    }

    [Fact]
    public void EachSideIsRunOncePerDeal_AndTheDealIsTheSameOnBothSides()
    {
        var first = new List<int>();
        var second = new List<int>();

        _ = Compared().Run(
            deal =>
            {
                first.Add(deal);

                return StudyTests.Passengers(deal);
            },
            deal =>
            {
                second.Add(deal);

                return StudyTests.Passengers(deal);
            },
            TestContext.Current.CancellationToken);

        Assert.Equal([0, 1, 2], first);
        Assert.Equal([0, 1, 2], second);
    }

    [Fact]
    public void TwoSidesThatDiffer_GiveDifferencesThatAreSomethingAndTheSameWhenRunAgain()
    {
        var one = Compared().Run(deal => StudyTests.Passengers(deal), deal => StudyTests.Passengers(deal + 50), TestContext.Current.CancellationToken);
        var again = Compared().Run(deal => StudyTests.Passengers(deal), deal => StudyTests.Passengers(deal + 50), TestContext.Current.CancellationToken);

        Assert.Contains(one.Differences, difference => difference != 0);
        Assert.Equal(one.Differences, again.Differences);
    }

    [Fact]
    public void AComparisonThatIsCancelled_StopsBeforeTheNextDeal_AndReturnsNothing()
    {
        using var source = new CancellationTokenSource();
        var runs = new List<int>();

        var stopped = Assert.Throws<OperationCanceledException>(() => Compared(deals: 4).Run(
            deal =>
            {
                runs.Add(deal);

                if (deal == 1)
                {
                    source.Cancel();
                }

                return StudyTests.Passengers(deal);
            },
            deal => StudyTests.Passengers(deal),
            source.Token));

        Assert.Equal(source.Token, stopped.CancellationToken);
        Assert.Equal([0, 1], runs);
    }

    [Fact]
    public void AScoreThatIsNothing_IsRefusedWhereItIsGiven() =>
        Assert.Throws<ArgumentNullException>(() => new Comparison(network => network) { Score = null! });

    [Fact]
    public void APipelineThatIsNothing_IsRefused_NamingTheSideAndTheDeal()
    {
        var first = Assert.Throws<InvalidOperationException>(() => Compared(deals: 2).Run(deal => null!, deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken));
        var second = Assert.Throws<InvalidOperationException>(() => Compared(deals: 2).Run(deal => StudyTests.Passengers(deal), deal => null!, TestContext.Current.CancellationToken));

        Assert.Contains("first", first.Message, StringComparison.Ordinal);
        Assert.Contains("second", second.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ASpreadNeedsTwoDeals_SoOneIsRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new Comparison(network => network) { Deals = 1 });
    }

    [Fact]
    public void ABadComparison_IsRefusedWhereItIsMade()
    {
        Assert.Throws<ArgumentNullException>(() => new Comparison((Func<NetworkDeclaration, NetworkDeclaration>)null!));
        Assert.Throws<ArgumentNullException>(() => new Comparison((LearnNetworkStep)null!));
        Assert.Throws<ArgumentNullException>(() => Compared().Run(null!, deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken));
        Assert.Throws<ArgumentNullException>(() => Compared().Run(deal => StudyTests.Passengers(deal), null!, TestContext.Current.CancellationToken));
    }

    [Fact]
    public void APipelineWithNoValidationRows_IsRefused_ForTheDefaultScoreIsTheValidationLoss()
    {
        var refused = Assert.Throws<InvalidOperationException>(() => Compared().Run(deal => StudyTests.Passengers(deal, validation: 0), deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken));

        Assert.Contains("validation", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ANetworkJudgedByNothing_IsRefused_SayingOnWhichSideAndDeal()
    {
        var comparison = new Comparison(network => network.Dense(3).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy().Run(1, 1))
        {
            Deals = 2,
            Score = _ => double.NaN,
        };

        var refused = Assert.Throws<InvalidOperationException>(() => comparison.Run(deal => StudyTests.Passengers(deal), deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken));

        Assert.Contains("first", refused.Message, StringComparison.Ordinal);
        Assert.Contains("deal 0", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ANetworkWrittenAsAStep_IsTrainedBehindBothSidesAsItIs()
    {
        var declared = Assert.IsType<LearnNetworkStep>(Pdd.Create().ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp", "parch").Category("sex").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitAtRandom(0.70, 0.15)
            .FillMissing("age", With.Median)
            .EncodeCategories()
            .Normalise("age", Scale.MidRange)
            .Normalise("fare", Scale.MidRange)
            .Normalise("sibsp", Scale.MidRange)
            .Normalise("parch", Scale.MidRange)
            .Target("survived")
            .WithTorch(network => network.Dense(3).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy().Run(5, 2))
            .Build().Declaration.Learner);

        var result = new Comparison(declared) { Deals = 2 }.Run(deal => StudyTests.Passengers(deal), deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken);

        Assert.Equal(2, result.Deals);
        Assert.All(result.Differences, difference => Assert.Equal(0, difference));
    }

    [Fact]
    public void TheValuesOfOneComparison_AreFixedOnceItIsRun()
    {
        var result = Compared().Run(deal => StudyTests.Passengers(deal), deal => StudyTests.Passengers(deal + 50), TestContext.Current.CancellationToken);

        Assert.Equal(result.Differences.Average(), result.Mean, 12);
        Assert.Equal(result.Seconds.Zip(result.Firsts, (second, first) => second - first), result.Differences);
    }
}
