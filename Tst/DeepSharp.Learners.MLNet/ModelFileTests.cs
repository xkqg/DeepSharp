// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.ML;
using DeepSharp.Learners.MLNet;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners.MLNet;

/// <summary>
/// A trained trainer and the pipeline it was trained behind, as one file.
/// </summary>
/// <remarks>
/// The model inside it is ML.NET's own archive, which nothing but ML.NET opens — so the file says which version wrote
/// it and on which processor, and a reader that cannot open it has something to tell its caller. The pipeline is carried
/// exactly as its own file wrote it, so the model stays tied to the steps it learned behind.
/// </remarks>
public class ModelFileTests
{
    [Fact]
    public void AModelFile_CarriesItsPipelineExactlyAsItWasWritten_AndIsReadBackAsItWasWritten()
    {
        var trained = Passengers().TrainWithML();
        var text = trained.ToJson();
        var read = MLModelFile.FromJson(text);

        Assert.Equal(PipelineText.Of(trained.Prepared).Text, read.Carried.Text);
        Assert.Equal(PipelineText.Of(trained.Prepared).Digest, read.Carried.Digest);
        Assert.Equal("fastTree", read.Trainer.Kind);
        Assert.Equal(trained.Seed, read.Seed);
        Assert.Equal(trained.TrainedOn.Features, read.Features);
        Assert.Equal(trained.TrainedOn.Answer, read.Answer);
        Assert.Equal(trained.TrainedOn.Rows, read.Rows);
        Assert.True(read.Classes);
        Assert.NotEmpty(read.Model);
        Assert.DoesNotContain("\r", text, StringComparison.Ordinal);
    }

    [Fact]
    public void AModelAskedForItsFileTwice_WritesTheSameBytes_EvenWhenTheArchivesOwnClockHasTicked()
    {
        // ML.NET stamps the entries of its archive with the time it saves them, in steps of two seconds, so a model saved again
        // after a tick is a different file of the same model, and the file a digest binds is not the one written next. The
        // sleep is not a wait for something to happen: it only makes sure the clock has ticked, which is what the repro needs.
        var trained = Passengers().TrainWithML();
        var first = trained.ToJson();

        Thread.Sleep(TimeSpan.FromMilliseconds(2200));

        Assert.Equal(first, trained.ToJson());
    }

    [Fact]
    public void RowsThePipelineHandsOverAsOtherFeaturesThanTheModelLearned_AreRefusedNamingBoth()
    {
        // The model is the same and what it is said to have learned from is not, which no model trained here can be: the private parts of a
        // trained one are handed to the constructor the training uses, with other names for its features.
        var trained = Passengers().TrainWithML();
        const System.Reflection.BindingFlags Private = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
        var model = (Microsoft.ML.ITransformer)typeof(TrainedMLModel).GetField("_model", Private)!.GetValue(trained)!;
        var context = (Microsoft.ML.MLContext)typeof(TrainedMLModel).GetField("_context", Private)!.GetValue(trained)!;
        var other = new TrainedMLModel(model, context, trained.Prepared, trained.TrainedOn with { Features = ["elsewhere"] });
        var served = CsvRowSource.FromText(string.Join('\n', File.ReadLines(Repository.Data("titanic.csv")).Take(13)) + "\n");

        var refused = Assert.Throws<InvalidOperationException>(() => other.Predict(served));

        Assert.Contains("'elsewhere'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("a model answers only what it learned from", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ItNamesTheLibraryThatWroteTheModel_AndTheProcessorItWasWrittenOn()
    {
        // Without these a reader that cannot open the archive can say nothing useful: the model is ML.NET's own format,
        // and which version and which processor wrote it is the whole of what a refusal needs to be actionable.
        var read = MLModelFile.FromJson(Passengers().TrainWithML().ToJson());

        Assert.StartsWith("5.", read.Library, StringComparison.Ordinal);
        Assert.Contains(read.Processor, new[] { "X64", "Arm64", "X86", "Arm" }, StringComparer.Ordinal);
    }

    [Fact]
    public void AFileFromANewerVersion_IsRefusedWhole_AtTheVersionItNames()
    {
        var refused = Assert.Throws<PipelineFileException>(() => MLModelFile.FromJson("""{"version": 99, "model": ""}"""));

        Assert.Contains("version 99", Assert.Single(refused.Faults).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AFileWithoutTheModelOrWithoutItsPipeline_SaysWhichIsMissing()
    {
        var noModel = Assert.Throws<PipelineFileException>(() => MLModelFile.FromJson("""{"version": 1}"""));

        Assert.Contains("model", Assert.Single(noModel.Faults).Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheModelsArchiveIsNeverWhatTiesItToItsPipeline()
    {
        // ML.NET stamps its archive with the clock it was saved at, so two saves of one model differ byte for byte while
        // the model does not. What ties the file to its steps is the pipeline's own text, which does not move.
        var once = Passengers().TrainWithML();
        var first = MLModelFile.FromJson(once.ToJson());
        var second = MLModelFile.FromJson(once.ToJson());

        Assert.Equal(first.Carried.Digest, second.Carried.Digest);
        Assert.Equal(first.Trainer, second.Trainer);
    }

    private static Pipeline Passengers() =>
        Pdd.Create()
            .ReadCsv(Repository.Data("titanic.csv"))
            .Declare(schema => schema.Integer("survived", "sibsp").Category("sex").Optional("age", ColumnKind.Number).Number("fare"))
            .SplitStratified("survived", train: 0.70, validation: 0.15)
            .FillMissing(fill => fill.Median("age"))
            .EncodeCategories()
            .Normalise("age", "fare")
            .Target("survived")
            .WithML(trainer => trainer.FastTree(trees: 20))
            .Build();
}
