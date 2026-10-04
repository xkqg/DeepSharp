// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.ML;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// What a model file says, read back by a library that carries no ML.NET at all.
/// </summary>
public class MLModelFileTests
{
    [Fact]
    public void AFileThatSaysEverything_IsReadBackWithEveryPartOfIt()
    {
        var carried = PipelineText.Of(Passengers().WithML(trainer => trainer.FastForest()).Build().Run());
        var written = (MLModelFile.Of(carried, MLWords.Default, [7]) with
        {
            Seed = 11,
            Library = "5.0.0",
            Processor = "Arm64",
            Answer = "survived",
            Classes = false,
            Rows = 3,
            Features = ["fare", "age"],
        }).ToJson();

        var read = MLModelFile.FromJson(written);

        Assert.Equal(11, read.Seed);
        Assert.Equal("5.0.0", read.Library);
        Assert.Equal("Arm64", read.Processor);
        Assert.Equal("survived", read.Answer);
        Assert.False(read.Classes);
        Assert.Equal(["fare", "age"], read.Features);
        Assert.Equal(3, read.Rows);
        Assert.Equal(1, MLModelFile.Version);
    }

    [Fact]
    public void AFileWhoseFeaturesAreNotAList_ReadsAsNone_AndItsClassesAsSaid()
    {
        var read = MLModelFile.FromJson(
            """
            {"version": 1, "mlnet": "5.0.0", "processor": "X64", "trainer": {"kind": "fastTree", "trees": 20},
             "seed": 1, "answer": "a", "classes": true, "rows": 1, "features": 7, "model": "",
             "pipeline": {"version": 7, "declaration": []}}
            """);

        Assert.Empty(read.Features);
        Assert.True(read.Classes);
        Assert.Equal("fastTree", read.Trainer.Kind);
        Assert.Equal(20, Assert.Single(read.Trainer.Settings).Value.Number);
    }

    [Fact]
    public void AFileWhoseProcessorIsNoWord_IsRefusedAtThatKey()
    {
        var refused = Assert.Throws<PipelineFileException>(() => MLModelFile.FromJson(
            """
            {"version": 1, "mlnet": "5.0.0", "processor": 64, "trainer": {"kind": "fastTree"}, "seed": 1,
             "answer": "a", "classes": false, "rows": 1, "model": "", "pipeline": {"version": 7, "declaration": []}}
            """));

        Assert.Contains("processor", Assert.Single(refused.Faults).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFeatureListHoldingNothing_ReadsThatOneAsNoName()
    {
        // A file written by hand can say null where a name belongs; it reads as the empty name rather than throwing,
        // because the names are what a reader shows and one missing name is not a reason to refuse a trained model.
        var read = MLModelFile.FromJson(
            """
            {"version": 1, "mlnet": "5.0.0", "processor": "X64", "trainer": {"kind": "fastTree"}, "seed": 1,
             "answer": "a", "classes": false, "rows": 1, "features": ["fare", null], "model": "",
             "pipeline": {"version": 7, "declaration": []}}
            """);

        Assert.Equal(["fare", ""], read.Features);
    }

    private static FittingBuilder Passengers() =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Number("fare"))
            .SplitStratified("survived", 0.70, 0.15)
            .Target("survived");
}
