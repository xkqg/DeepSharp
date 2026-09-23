// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Pipelines;
using Microsoft.Extensions.DependencyInjection;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A declaration exists so that a mistake is caught while it is being written rather than discovered in a
/// number three steps later. These are the refusals that make that true: the guards on every public door,
/// and the two ways a file can be wrong — a parameter that is not there, and one that is there as the wrong
/// kind of thing.
/// </summary>
public class RefusalTests
{
    [Fact]
    public void FillingGapsInAColumnWithoutAName_IsRefused()
    {
        var after = Pdd.Create().ReadCsv("x.csv").SplitByTime("t", 0.70, 0.15);

        Assert.Throws<ArgumentException>(() => after.FillMissing("   ", With.Mean));
    }

    [Fact]
    public void AStepThatIsNotThere_IsRefusedByBothBuilders()
    {
        var before = Pdd.Create();
        var after = Pdd.Create().ReadCsv("x.csv").SplitByTime("t", 0.70, 0.15);

        Assert.Throws<ArgumentNullException>(() => before.Add(null!));
        Assert.Throws<ArgumentNullException>(() => after.Add(null!));
    }

    [Fact]
    public void ReadingACsvOntoNothing_IsRefused()
    {
        Assert.Throws<ArgumentNullException>(() => CsvSourceExtensions.ReadCsv(null!, "x.csv"));
    }

    [Fact]
    public void ACatalogRefusesAVerbWithoutAName_OrAReaderThatIsNotThere()
    {
        var catalog = StepCatalog.BuiltIn();

        Assert.Throws<ArgumentException>(() => catalog.Register("  ", ReadCsvStep.ReadFrom));
        Assert.Throws<ArgumentNullException>(() => catalog.Register("read.avro", null!));
    }

    [Fact]
    public void ADeclarationOfNothingAtAll_IsRefused()
    {
        Assert.Throws<ArgumentNullException>(() => new PipelineDeclaration(null!));
        Assert.Throws<ArgumentNullException>(() => PipelineDeclaration.FromJson("{}", null!));
    }

    [Fact]
    public void AStepAskedToWriteItselfNowhere_IsRefused()
    {
        Assert.Throws<ArgumentNullException>(() => new ReadCsvStep("x.csv").WriteTo(null!));
        Assert.Throws<ArgumentNullException>(() => new SplitByTimeStep("t", new SplitShares(0.70, 0.15, 0.15)).WriteTo(null!));
        Assert.Throws<ArgumentNullException>(() => new FillMissingStep("trades", With.Mean).WriteTo(null!));
    }

    [Fact]
    public void RegisteringPipelinesWithNoServices_IsRefused()
    {
        Assert.Throws<ArgumentNullException>(
            () => ServiceCollectionExtensions.AddDeepSharpPipelines(null!));
    }

    [Fact]
    public void ADeclarationThatIsNotAList_IsRefused()
    {
        Assert.Throws<FormatException>(
            () => PipelineDeclaration.FromJson("""{"declaration":"read.csv"}"""));
    }

    [Theory]
    [InlineData("""{"declaration":[{"step":"read.csv"}]}""")]
    [InlineData("""{"declaration":[{"step":"read.csv","path":5}]}""")]
    public void ATextParameterMissingOrOfTheWrongKind_IsTheSameFaultToTheReader(string json)
    {
        // Absent and wrongly typed are two different exceptions in the BCL and neither names the parameter.
        // Somebody editing a file by hand has to be told where to look, so both become one refusal.
        var refused = Assert.Throws<FormatException>(() => PipelineDeclaration.FromJson(json));

        Assert.Contains("path", refused.Message);
    }

    [Theory]
    [InlineData("""{"declaration":[{"step":"split.byTime","column":"t","train":0.7,"validation":0.15}]}""")]
    [InlineData("""{"declaration":[{"step":"split.byTime","column":"t","train":0.7,"validation":0.15,"test":"x"}]}""")]
    public void ANumberMissingOrOfTheWrongKind_IsTheSameFaultToo(string json)
    {
        var refused = Assert.Throws<FormatException>(() => PipelineDeclaration.FromJson(json));

        Assert.Contains("test", refused.Message);
    }

    [Fact]
    public void AStepWrittenAsSomethingOtherThanAnObject_IsRefused()
    {
        Assert.Throws<FormatException>(() => PipelineDeclaration.FromJson("""{"declaration":["read.csv"]}"""));
    }

    [Fact]
    public void TheVerbsAreReadByTheCatalogThatWasHandedOver()
    {
        // Not a shared one, and not one the library reached for: the catalog comes in through the door.
        using var document = JsonDocument.Parse("""{"step":"read.csv","path":"x.csv"}""");

        var step = StepCatalog.BuiltIn().Read(document.RootElement);

        Assert.Equal("read.csv", step.Verb);
    }

    [Fact]
    public void AHostThatRegistersTwice_StillHasOneCatalog()
    {
        var services = new ServiceCollection();
        services.AddDeepSharpPipelines();
        services.AddDeepSharpPipelines();

        using var provider = services.BuildServiceProvider();

        Assert.Same(provider.GetRequiredService<StepCatalog>(), provider.GetRequiredService<StepCatalog>());
    }
}
