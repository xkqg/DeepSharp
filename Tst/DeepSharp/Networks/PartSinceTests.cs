// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// What a part of a network file says is written from one version on, said once: the table the writer takes the version of a
/// part from and the reader takes its refusal of an earlier one from, so the two cannot say different things. The files they
/// make and refuse are held by the network-file tests; this holds the table, and the one sentence a refusal is made of.
/// </summary>
public class PartSinceTests
{
    [Theory]
    [InlineData(1, "first")]
    [InlineData(2, "second")]
    [InlineData(3, "third")]
    [InlineData(4, "fourth")]
    [InlineData(5, "fifth")]
    [InlineData(6, "sixth")]
    [InlineData(7, "seventh")]
    [InlineData(8, "eighth")]
    [InlineData(9, "ninth")]
    public void EveryVersionAFileCanBe_IsNamedInTheRefusalOfAnEarlierOne(int version, string name)
    {
        var since = new PartSince("a training part", "what a run did", version);

        Assert.Equal(
            $"What a run did, under 'did', is written from the {name} version of a training part on, and this one is of the first.",
            since.Refusal("did", 1));
    }

    [Fact]
    public void ARefusal_SaysWhatItIsAboutWhereItStandsAndWhichVersionHasItAndWhichThePartIs()
    {
        var since = new PartSince("a network's part", "what the learning does to a parameter's numbers", NetworkDocument.FifthVersion);

        Assert.Equal(
            "What the learning does to a parameter's numbers, under 'l1', is written from the fifth version of a network's part on, and this one is of the fourth.",
            since.Refusal("l1", NetworkDocument.FourthVersion));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(10)]
    [InlineData(-1)]
    public void AVersionNoFileHas_IsRefused_ForItHasNoName(int version)
    {
        var since = new PartSince("a training part", "what a run did", NetworkDocument.ThirdVersion);

        Assert.Throws<ArgumentOutOfRangeException>(() => since.Refusal("did", version));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PartSince("a training part", "what a run did", version).Refusal("did", 1));
    }

    [Theory]
    [InlineData(2, true)]
    [InlineData(4, true)]
    [InlineData(5, false)]
    [InlineData(9, false)]
    public void APartOfAnEarlierVersion_CannotSayIt_AndOneOfTheVersionOrLaterCan(int version, bool cannot) =>
        Assert.Equal(cannot, new PartSince("a training part", "what a run did", NetworkDocument.FifthVersion).Cannot(version));

    [Fact]
    public void TheTable_SaysEachThingIsWrittenFromTheVersionItWasFirstWrittenIn()
    {
        Assert.Equal(NetworkDocument.FifthVersion, PartSince.Rules.From);
        Assert.Equal(NetworkDocument.SeventhVersion, PartSince.RateShare.From);
        Assert.Equal(NetworkDocument.ThirdVersion, PartSince.ClipNorm.From);
        Assert.Equal(NetworkDocument.FifthVersion, PartSince.ClipValue.From);
        Assert.Equal(NetworkDocument.FifthVersion, PartSince.StepEvery.From);
        Assert.Equal(NetworkDocument.FifthVersion, PartSince.Average.From);
        Assert.Equal(NetworkDocument.SixthVersion, PartSince.JudgedByAverage.From);
        Assert.Equal(NetworkDocument.NinthVersion, PartSince.LossScale.From);
        Assert.Equal(NetworkDocument.NinthVersion, PartSince.EnginePrecision.From);
        Assert.Equal(NetworkDocument.NinthVersion, PartSince.Skipped.From);
        Assert.Equal(NetworkDocument.FifthVersion, PartSince.OptimizerOption.From);
    }

    [Fact]
    public void NoRowOfTheTable_IsOfAVersionThisLibraryDoesNotWrite() =>
        Assert.All(
            new[]
            {
                PartSince.Rules, PartSince.RateShare, PartSince.ClipNorm, PartSince.ClipValue, PartSince.StepEvery, PartSince.Average,
                PartSince.JudgedByAverage, PartSince.LossScale, PartSince.EnginePrecision, PartSince.Skipped, PartSince.OptimizerOption,
            },
            since => Assert.InRange(since.From, NetworkDocument.SecondVersion, NetworkDocument.Version));
}
