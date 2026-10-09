// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A network trained by how far its shares lie from the answer's along the order of the bands: a word a declared network
/// is written in, a door of the chain, and a loss that only an output whose answers are in an order can mean — refused,
/// before anything is trained, for any other. With what is left of a flock's planned whole, the bands are trained as a shape
/// and what is left by how far it is off, and the shares come back as birds that add up to the planned flock.
/// </summary>
public class OrderedLossTests
{
    private static readonly string[] Bands = ["w500", "w550", "w600"];

    // Twelve flocks, numbered in time: their age, how many were planned, and how many arrived in each band.
    private static InMemoryRowSource Flocks() =>
        new(
            ["t", "age", "planned", .. Bands],
            [
                .. Enumerable.Range(1, 12).Select(t => (IReadOnlyList<string?>)
                [
                    t.ToString(CultureInfo.InvariantCulture),
                    (28 + (t % 6)).ToString(CultureInfo.InvariantCulture),
                    (20 + t).ToString(CultureInfo.InvariantCulture),
                    (3 + (t % 4)).ToString(CultureInfo.InvariantCulture),
                    (8 + (t % 3)).ToString(CultureInfo.InvariantCulture),
                    (2 + (t % 5)).ToString(CultureInfo.InvariantCulture),
                ]),
            ]);

    private static FittingBuilder Split() =>
        Pdd.Create()
            .Read(Flocks(), "twelve flocks")
            .Declare(schema => schema.Integer("t").Number("age", "planned").Number(Bands))
            .SplitByTime("t", 0.50, 0.25);

    private static FittingBuilder Scaled(FittingBuilder output) =>
        output.Normalise("age", Scale.MidRange).Normalise("planned", Scale.MidRange).Normalise("t", Scale.MidRange);

    // A network answering as many numbers as the output names: the bands, and what is left when there is such.
    private static Func<NetworkDeclaration, NetworkDeclaration> Network(int answers) =>
        network => network.Dense(4).Relu().Dense(answers).Adam(0.01).EarthMoversDistance().Run(seed: 20260929, epochs: 3, batch: 4);

    [Fact]
    public void TheDistanceAlongAnOrder_IsAWordADeclaredNetworkIsWrittenIn_AndTheChainWritesIt()
    {
        var step = Assert.IsType<LearnNetworkStep>(
            Scaled(Split().Distribution(Bands, "planned", ordered: true, remainder: "other")).WithTorch(Network(4)).Build().Declaration.Learner);

        Assert.Equal("earthMoversDistance", step.Loss.Kind);
        Assert.Contains(NetworkWords.Losses, kind => kind.Name == "earthMoversDistance");
        Assert.False(Assert.IsType<EarthMoversDistance>(new PartDeclaration("earthMoversDistance", []).Judged()).Remainder);
        Assert.Contains("earthMoversDistance", NetworkCatalog.BuiltIn().Names);
    }

    [Fact]
    public void AFlockShortOfPlanned_IsTrainedAlongTheOrderOfItsBands_AndItsSharesComeBackAsBirdsThatAddUpToThePlannedFlock()
    {
        var trained = Scaled(Split().Distribution(Bands, "planned", ordered: true, remainder: "other")).WithTorch(Network(4)).Build().Train();
        var predicted = trained.Predict(new InMemoryRowSource(["t", "age", "planned"], [["13", "30", "1000"], ["14", "31", "640"]]));

        Assert.True(Assert.IsType<EarthMoversDistance>(trained.Loss).Remainder);
        Assert.Equal([.. Bands, "other"], predicted.AnswerNames);
        // Shares worked out in single precision, as a network works them out, sum to one within a ten-millionth or so.
        Assert.Equal(1000, predicted.Answers[0].Sum(), 1e-3);
        Assert.Equal(640, predicted.Answers[1].Sum(), 1e-3);
        Assert.All(predicted.Answers, birds => Assert.All(birds, count => Assert.True(count > 0)));
        Assert.All(trained.History!.Epochs, epoch => Assert.True(double.IsFinite(epoch.Loss)));
    }

    [Fact]
    public void ADistributionOfTheWholeInAnOrder_IsTrainedAlongIt_WithNothingLeftOver()
    {
        var trained = Scaled(Split().NormaliseRow(Norm.L1, Bands).Distribution(Bands, scaleBy: null, ordered: true)).WithTorch(Network(3)).Build().Train();

        Assert.False(Assert.IsType<EarthMoversDistance>(trained.Loss).Remainder);
        Assert.Equal(1, trained.Predict(new InMemoryRowSource(["t", "age", "planned"], [["13", "30", "25"]])).Answers[0].Sum(), 1e-6);
    }

    [Fact]
    public void TheDistanceAlongAnOrder_IsRefusedBeforeAnythingIsTrained_ForAnOutputWithNoOrder()
    {
        var unordered = Assert.Throws<InvalidOperationException>(
            () => Scaled(Split().NormaliseRow(Norm.L1, Bands).Distribution(Bands)).WithTorch(Network(3)).Build().Train());
        var target = Assert.Throws<InvalidOperationException>(
            () => Scaled(Split().Target("w500")).WithTorch(network => network.Dense(1).Adam(0.01).EarthMoversDistance().Run(seed: 1, epochs: 1)).Build().Train());

        Assert.Contains("'earthMoversDistance'", unordered.Message, StringComparison.Ordinal);
        Assert.Contains("'ordered'", unordered.Message, StringComparison.Ordinal);
        Assert.Contains("'target'", target.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ADistanceWrittenAsCode_IsRefusedWhenItsRemainderIsNotTheOutputs_SayingWhichToCompile()
    {
        var withLeft = Scaled(Split().Distribution(Bands, "planned", ordered: true, remainder: "other")).Build().RunFor(Needs.OneScale);
        var whole = Scaled(Split().NormaliseRow(Norm.L1, Bands).Distribution(Bands, scaleBy: null, ordered: true)).Build().RunFor(Needs.OneScale);

        var noLeft = Assert.Throws<InvalidOperationException>(() => new Sequential().Dense(4)
            .Compile(new Adam(0.01), new EarthMoversDistance())
            .Fit(withLeft, new FitOptions(1) { Epochs = 1 }));
        var left = Assert.Throws<InvalidOperationException>(() => new Sequential().Dense(3)
            .Compile(new Adam(0.01), new EarthMoversDistance(remainder: true))
            .Fit(whole, new FitOptions(1) { Epochs = 1 }));

        Assert.Contains("new EarthMoversDistance(remainder: true)", noLeft.Message, StringComparison.Ordinal);
        Assert.Contains("'other'", noLeft.Message, StringComparison.Ordinal);
        Assert.Contains("new EarthMoversDistance()", left.Message, StringComparison.Ordinal);
        Assert.NotNull(new Sequential().Dense(4).Compile(new Adam(0.01), new EarthMoversDistance(remainder: true)).Fit(withLeft, new FitOptions(1) { Epochs = 1 }).History);
    }
}
