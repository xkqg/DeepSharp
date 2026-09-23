// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The chain describes; it does not act. Everything else this library promises rests on that one property:
/// a chain that did the work while you typed it could never be saved as a file, replayed a year later, or
/// handed to somebody who does not have the data. So these tests check that writing a step records it and
/// changes nothing else.
/// </summary>
public class PipelineBuilderTests
{
    [Fact]
    public void ANewPipeline_HasSaidNothingYet()
    {
        var builder = Pdd.Create();

        Assert.Empty(builder.Declaration.Steps);
    }

    [Fact]
    public void ReadingACsv_RecordsTheStepAndDoesNotOpenTheFile()
    {
        // The path is deliberately nonsense. Declaring where the data will come from is not the same act as
        // going to get it, and if this ever throws, the chain has started doing instead of describing.
        var builder = Pdd.Create().ReadCsv("no-such-file-anywhere.csv");

        var step = Assert.IsType<ReadCsvStep>(Assert.Single(builder.Declaration.Steps));
        Assert.Equal("no-such-file-anywhere.csv", step.Path);
        Assert.Equal("read.csv", step.Verb);
    }

    [Fact]
    public void EachStep_IsKeptInTheOrderItWasWritten()
    {
        var builder = Pdd.Create()
            .ReadCsv("first.csv")
            .SplitByTime("timestamp", train: 0.70, validation: 0.15, test: 0.15)
            .FillMissing("trades", With.Mean);

        Assert.Collection(
            builder.Declaration.Steps,
            step => Assert.Equal("read.csv", step.Verb),
            step => Assert.Equal("split.byTime", step.Verb),
            step => Assert.Equal("fill.missing", step.Verb));
    }

    [Fact]
    public void ASplit_RemembersItsColumnAndItsThreeShares()
    {
        var builder = Pdd.Create()
            .ReadCsv("btceur-1d.csv")
            .SplitByTime("timestamp", train: 0.70, validation: 0.15, test: 0.15);

        var split = Assert.IsType<SplitByTimeStep>(builder.Declaration.Steps[1]);

        Assert.Equal("timestamp", split.Column);
        Assert.Equal(0.70, split.Train);
        Assert.Equal(0.15, split.Validation);
        Assert.Equal(0.15, split.Test);
    }

    [Fact]
    public void ThreeSharesThatDoNotMakeAWhole_AreRefusedWhereTheyAreWritten()
    {
        // 0.95 is the mistake that costs an afternoon: the run works, a twentieth of the data is silently
        // in no split at all, and every number afterwards is computed over less data than you think.
        var refused = Assert.Throws<ArgumentException>(
            () => Pdd.Create().ReadCsv("x.csv").SplitByTime("timestamp", 0.70, 0.15, 0.10));

        Assert.Contains("0.95", refused.Message);
    }

    [Theory]
    [InlineData(0.0, 0.5, 0.5)]
    [InlineData(0.5, 0.0, 0.5)]
    [InlineData(0.5, 0.5, 0.0)]
    [InlineData(1.2, -0.1, -0.1)]
    public void AShareThatIsNotAShare_IsRefused(double train, double validation, double test)
    {
        // An empty split is not a split: a model measured on nothing scores perfectly on nothing.
        Assert.Throws<ArgumentOutOfRangeException>(
            () => Pdd.Create().ReadCsv("x.csv").SplitByTime("timestamp", train, validation, test));
    }

    [Fact]
    public void AColumnWithoutAName_IsRefused()
    {
        Assert.Throws<ArgumentException>(
            () => Pdd.Create().ReadCsv("x.csv").SplitByTime("  ", 0.70, 0.15, 0.15));
    }

    [Fact]
    public void APathWithoutAName_IsRefused()
    {
        Assert.Throws<ArgumentException>(() => Pdd.Create().ReadCsv(""));
    }

    [Fact]
    public void FillingAGap_RemembersWhichColumnAndWhichStrategy()
    {
        var builder = Pdd.Create()
            .ReadCsv("btceur-1d.csv")
            .SplitByTime("timestamp", 0.70, 0.15, 0.15)
            .FillMissing("trades", With.Median);

        var fill = Assert.IsType<FillMissingStep>(builder.Declaration.Steps[2]);

        Assert.Equal("trades", fill.Column);
        Assert.Equal(With.Median, fill.Strategy);
        Assert.Equal("median", fill.Strategy.Name);
    }

    [Fact]
    public void TheStrategiesAreNamedValues_SoACallSiteReadsWithoutTheSignature()
    {
        Assert.Equal("mean", With.Mean.Name);
        Assert.Equal("median", With.Median.Name);
        Assert.Equal("zero", With.Zero.Name);
        Assert.Equal("previous", With.Previous.Name);
        Assert.NotEqual(With.Mean, With.Median);
    }

    [Fact]
    public void ADeclaration_ReadsBackAsTheSentenceItWas()
    {
        var builder = Pdd.Create()
            .ReadCsv("btceur-1d.csv")
            .SplitByTime("timestamp", 0.70, 0.15, 0.15)
            .FillMissing("trades", With.Mean);

        Assert.Equal(
            "read.csv -> split.byTime -> fill.missing",
            builder.Declaration.ToString());
    }
}
