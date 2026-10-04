// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.ML;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The verb that names a trainer from ML.NET, and the file its model is kept as.
/// </summary>
/// <remarks>
/// Both live in the package that carries no ML.NET at all, which is why they are read here, in the suite that carries
/// none either: declaring a trainer, writing its file and reading one back are what an application that never trains
/// still does.
/// </remarks>
public class LearnMLTests
{
    [Fact]
    public void ItNamesWhatItNeeds_AndChangesNoColumn()
    {
        var step = new LearnMLStep(MLWords.Default);

        Assert.Equal("learn.ml", step.Verb);
        Assert.Equal("learn.ml", LearnMLStep.Name);
        Assert.Equal(7, LearnMLStep.Since);
        Assert.Equal(Needs.NoScale, step.Needs);
        Assert.Equal(Needs.NoScale, LearnMLStep.FeatureNeeds);
        Assert.Equal(20260929, step.Seed);
        Assert.Contains("ML.NET", LearnMLStep.Purpose, StringComparison.Ordinal);

        var before = ColumnState.Of(new Table([new Column<double>("a", ColumnKind.Number, [1, 2])]));

        Assert.Same(before, step.After(before));
    }

    [Fact]
    public void ATrainerTheWordsDoNotKnow_IsRefusedWhereItIsWritten()
    {
        // Only the trainers whose model repeats from the seed a declaration carries have a word, and a trainer that has
        // none is refused at the line rather than at the fit.
        Assert.Throws<ArgumentException>(() => new LearnMLStep(new PartDeclaration("sdca", [])));
        Assert.Throws<ArgumentException>(() => new LearnMLStep(new PartDeclaration("lightGbm", [])));
    }

    [Fact]
    public void ASettingTheTrainerDoesNotTake_IsRefusedToo()
    {
        var refused = Assert.Throws<ArgumentException>(() => new LearnMLStep(
            new PartDeclaration("fastForest", [new PartSetting("rate", PartValue.Of(0.1))])));

        Assert.Contains("rate", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ItIsReadBackFromItsFile_AsItWroteItself()
    {
        var written = new LearnMLStep(new PartDeclaration("fastTree", [
            new PartSetting("leaves", PartValue.Of(8)),
            new PartSetting("trees", PartValue.Of(40)),
            new PartSetting("leastRows", PartValue.Of(5)),
            new PartSetting("rate", PartValue.Of(0.1)),
        ]))
        { Seed = 7 };

        var read = (LearnMLStep)Shipped.Catalog().ReadStep(
            """
            {"step": "learn.ml", "seed": 7,
             "trainer": {"kind": "fastTree", "leaves": 8, "trees": 40, "leastRows": 5, "rate": 0.1}}
            """);

        Assert.Equal(written, read);
        Assert.Equal(written.GetHashCode(), read.GetHashCode());
        Assert.NotEqual(written, new LearnMLStep(MLWords.Default));
        Assert.False(written.Equals(null));
    }

    [Fact]
    public void TheChainNamesIt_AndOnlyOneLearnerIsAllowed()
    {
        var declaration = Passengers().WithML(trainer => trainer.FastForest(trees: 7), seed: 3).Declaration;
        var step = Assert.IsType<LearnMLStep>(declaration.Learner);

        Assert.Equal("fastForest", step.Trainer.Kind);
        Assert.Equal(3, step.Seed);
        Assert.Throws<DeclarationException>(() => Passengers().WithML(trainer => trainer.FastTree()).WithML(trainer => trainer.FastTree()));
    }

    [Fact]
    public void AModelFileWithoutItsPipeline_IsRefusedByName()
    {
        var refused = Assert.Throws<PipelineFileException>(() => MLModelFile.FromJson(
            """{"version": 1, "mlnet": "5.0.0", "processor": "X64", "trainer": {"kind": "fastTree"}, "seed": 1, "answer": "a", "classes": true, "rows": 1, "model": ""}"""));

        Assert.Contains("pipeline", Assert.Single(refused.Faults).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AModelFileThatSaysNothingOfItsTrainer_OrGivesAWordWhereANumberBelongs_SaysWhich()
    {
        var noTrainer = Assert.Throws<PipelineFileException>(() => MLModelFile.FromJson(
            """{"version": 1, "model": "", "trainer": 7, "pipeline": {"version": 7, "declaration": []}}"""));
        var noNumber = Assert.Throws<PipelineFileException>(() => MLModelFile.FromJson(
            """{"version": 1, "model": "", "trainer": {"kind": "fastTree"}, "seed": "soon", "pipeline": {"version": 7, "declaration": []}}"""));

        Assert.Contains("trainer", Assert.Single(noTrainer.Faults).Message, StringComparison.Ordinal);
        Assert.Contains("seed", Assert.Single(noNumber.Faults).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFileMadeByHand_CarriesWhatItWasGiven()
    {
        var carried = PipelineText.Of(Passengers().WithML(trainer => trainer.FastTree()).Build().Run());
        var file = MLModelFile.Of(carried, MLWords.Default, [1, 2, 3]) with { Library = "5.0.0", Processor = "X64", Rows = 2, Features = ["a"], Answer = "survived" };

        Assert.Equal(carried.Digest, MLModelFile.FromJson(file.ToJson()).Carried.Digest);
        Assert.Equal([1, 2, 3], MLModelFile.FromJson(file.ToJson()).Model);
        Assert.Throws<ArgumentNullException>(() => MLModelFile.Of(carried, MLWords.Default, null!));
        Assert.Throws<ArgumentException>(() => MLModelFile.FromJson(" "));
    }

    private static FittingBuilder Passengers() =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived").Number("fare"))
            .SplitStratified("survived", 0.70, 0.15)
            .Target("survived");
}
