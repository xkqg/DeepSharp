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
        var after = Pdd.Create().ReadCsv("x.csv").Declare(schema => schema.Timestamp("t")).SplitByTime("t", 0.70, 0.15);

        Assert.Throws<ArgumentException>(() => after.FillMissing("   ", With.Mean));
    }

    [Fact]
    public void AStepThatIsNotThere_IsRefusedByBothBuilders()
    {
        var before = Pdd.Create();
        var after = Pdd.Create().ReadCsv("x.csv").Declare(schema => schema.Timestamp("t")).SplitByTime("t", 0.70, 0.15);

        Assert.Throws<ArgumentNullException>(() => before.Add(null!));
        Assert.Throws<ArgumentNullException>(() => after.Add(null!));
    }

    [Fact]
    public void ReadingACsvOntoNothing_IsRefused()
    {
        Assert.Throws<ArgumentNullException>(() => CsvSourceExtensions.ReadCsv(null!, "x.csv"));
    }

    [Fact]
    public void ACatalogRefusesAStepTypeThatNeverSaysWhatItIsCalledOrWhatItDoes()
    {
        // A verb nobody can write, and a verb nobody can look up, are both holes in the reference page and
        // in every form a notebook shows; the type is refused when it is registered, not when a file meets it.
        var catalog = StepCatalog.BuiltIn();

        var nameless = Assert.Throws<InvalidOperationException>(() => catalog.Register<NamelessStep>());
        var purposeless = Assert.Throws<InvalidOperationException>(() => catalog.Register<PurposelessStep>());

        Assert.Contains(nameof(NamelessStep), nameless.Message, StringComparison.Ordinal);
        Assert.Contains("read.nothing", purposeless.Message, StringComparison.Ordinal);
        Assert.False(catalog.Knows("read.nothing"));
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
        // A step writes itself through the parameters it describes, so the writing is the interface's own
        // and is reached through it.
        IPipelineStep[] steps =
        [
            new ReadCsvStep("x.csv"),
            new SplitByTimeStep("t", new SplitShares(0.70, 0.15, 0.15)),
            FillMissingStep.Of("trades", With.Mean),
        ];

        Assert.All(steps, step => Assert.Throws<ArgumentNullException>(() => step.WriteTo(null!)));
        Assert.Throws<ArgumentNullException>(
            () => ReadCsvStep.Parameters.Write(new Utf8JsonWriter(new MemoryStream()), null!));
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
        Assert.Throws<PipelineFileException>(
            () => PipelineDeclaration.FromJson("""{"declaration":"read.csv"}""", StepCatalog.BuiltIn()));
    }

    [Theory]
    [InlineData("""{"declaration":[{"step":"read.csv"}]}""")]
    [InlineData("""{"declaration":[{"step":"read.csv","path":5}]}""")]
    public void ATextParameterMissingOrOfTheWrongKind_IsTheSameFaultToTheReader(string json)
    {
        // Absent and wrongly typed are two different exceptions in the BCL and neither names the parameter.
        // Somebody editing a file by hand has to be told where to look, so both become one refusal.
        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));

        Assert.Contains("path", refused.Message);
    }

    [Theory]
    [InlineData("""{"version":2,"declaration":[{"step":"split.byTime","column":"t","train":0.7,"validation":0.15}]}""")]
    [InlineData("""{"version":2,"declaration":[{"step":"split.byTime","column":"t","train":0.7,"validation":0.15,"test":"x"}]}""")]
    public void ANumberMissingOrOfTheWrongKind_IsTheSameFaultToo(string json)
    {
        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));

        Assert.Contains("test", refused.Message);
    }

    [Theory]
    // A seed of 3.5 was read as 3 and written back as 3, and a limit of 2.9 rows as 2: the file said one
    // thing and the pipeline did another, and nothing said so.
    [InlineData("""{"version":2,"declaration":[{"step":"split.atRandom","train":0.7,"validation":0.15,"test":0.15,"seed":3.5}]}""", "seed")]
    [InlineData("""{"version":2,"declaration":[{"step":"drop.warmup","atMost":2.9}]}""", "atMost")]
    [InlineData("""{"version":2,"declaration":[{"step":"split.stratified","column":"g","train":0.7,"validation":0.15,"test":0.15,"seed":0.5}]}""", "seed")]
    public void AWholeNumberWrittenWithAFraction_IsRefused(string json, string key)
    {
        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));

        Assert.Contains(key, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AWholeNumberBeyondWhatItCanHold_IsRefusedRatherThanSaturated()
    {
        // 1e10 used to become 2147483647, the largest value the reader could hold, which is a different
        // seed and so a different split.
        const string json = """{"version":2,"declaration":[{"step":"split.atRandom","train":0.7,"validation":0.15,"test":0.15,"seed":1e10}]}""";

        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));

        Assert.Contains("seed", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AShareHeldBackThatIsWrittenInWords_IsRefused()
    {
        // The one share a file may leave out is still a number when it is there.
        const string json = """{"version":2,"declaration":[{"step":"split.byTime","column":"t","train":0.7,"validation":0.15,"test":0.05,"predict":"a tenth"}]}""";

        var refused = Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn()));

        Assert.Contains("'predict'", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnIndicatorsPeriodIsAWholeNumberToo()
    {
        const string json = """
            {"declaration":[{"step":"feature.indicator","column":"x","indicator":"sma","period":14.5,"columns":["a"]}]}
            """;

        Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn().WithIndicators()));
    }

    [Fact]
    public void AStepWrittenAsSomethingOtherThanAnObject_IsRefused()
    {
        Assert.Throws<PipelineFileException>(() => PipelineDeclaration.FromJson("""{"declaration":["read.csv"]}""", StepCatalog.BuiltIn()));
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
