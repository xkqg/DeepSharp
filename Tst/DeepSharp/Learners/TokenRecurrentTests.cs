// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A network declared to read a text a word at a time: the ids a pipeline cut it into, each its own vector, the padding the pipeline
/// declared turned into the mask a recurrent layer reads, so a short text is read as it would be read alone and nothing is learned of the
/// padding. The same words a GRU reads, a layer of its own in the file.
/// </summary>
public class TokenRecurrentTests
{
    private static readonly string[] Praise = ["great", "lovely", "fine", "superb"];
    private static readonly string[] Blame = ["awful", "dull", "bad", "boring"];
    private static readonly string[] Filler = ["film", "plot", "actor", "scene", "music", "ending"];

    // A review is three to seven words: one or two of praise or of blame and the rest filler; it is good when it praises.
    private static InMemoryRowSource Reviews(int count)
    {
        var random = new Random(97);

        return new InMemoryRowSource(
            ["review", "good"],
            [.. Enumerable.Range(0, count).Select(_ =>
            {
                var good = random.NextDouble() < 0.5;
                var words = new List<string>();

                for (var at = random.Next(1, 3); at > 0; at--)
                {
                    words.Add((good ? Praise : Blame)[random.Next(4)]);
                }

                for (var at = random.Next(2, 6); at > 0; at--)
                {
                    words.Add(Filler[random.Next(6)]);
                }

                return new string?[] { string.Join(' ', words.OrderBy(_ => random.Next())), good ? "1" : "0" };
            })]);
    }

    private static PreparedData Prepared(Func<NetworkDeclaration, NetworkDeclaration> declared) =>
        Pdd.Create()
            .Read(Reviews(400), "four hundred reviews")
            .Declare(schema => schema.Text("review").Integer("good"))
            .SplitAtRandom(0.60, 0.20, 3)
            .Tokenize("review", length: 8, vocabulary: 24)
            .Target("good")
            .Report(report => report.Measure(Metric.Accuracy).On(Part.Validation, Part.Test).As(Shown.Numbers))
            .WithTorch(network => declared(network).Adam(0.05).BinaryCrossEntropy().Run(seed: 1, epochs: 25))
            .Build()
            .Run();

    [Theory]
    [InlineData("lstm")]
    [InlineData("gru")]
    public void ANetworkThatReadsAReviewAWordAtATime_LearnsWhichWordsPraise_TheMaskCutFromTheSameIds(string kind)
    {
        var trained = Prepared(net => Recurrent(net.TokenEmbedding(vocabulary: 24, width: 6, keepSteps: true).Named("steps").Unpadded().Named("mask").Takes("input"), kind)).Train();
        var accuracy = trained.Measures!.Parts.Single(part => part.Part == Part.Test).Values.Single(value => value.Metric == Metric.Accuracy);
        var stack = Assert.IsType<LayerStack>(trained.Network);

        Assert.True(accuracy.Value > 0.85, $"accuracy {accuracy.Value}");
        Assert.DoesNotContain(stack.Slots(), slot => slot.Path.StartsWith("1.", StringComparison.Ordinal) || slot.Path.StartsWith("mask.", StringComparison.Ordinal));
        Assert.Equal(["steps", "mask", "2", "3"], stack.Steps.Select(step => step.Name));
        Assert.Equal(["steps", "mask"], stack.Steps[2].Takes);
    }

    [Fact]
    public void ARowServedLater_IsReadAWordAtATime_ThroughTheSameVocabularyAndTheSameMask()
    {
        var trained = Prepared(net => Recurrent(net.TokenEmbedding(vocabulary: 24, width: 6, keepSteps: true).Named("steps").Unpadded().Named("mask").Takes("input"), "gru")).Train();
        var served = new InMemoryRowSource(["review", "good"], [["a great film with zxqv", "1"], ["an awful plot with zxqv", "0"]]);
        var answers = trained.Predict(served).Answers;

        Assert.True(answers[0][0] >= 0.5, $"{answers[0][0]}");
        Assert.True(answers[1][0] < 0.5, $"{answers[1][0]}");
    }

    [Fact]
    public void TheDeclaredLayers_AreInTheFileWithTheirSettings_AndOnesLeftOutMeanTheirDefaults()
    {
        var prepared = Prepared(net => net.TokenEmbedding(vocabulary: 24, width: 6, keepSteps: true).Named("steps").Unpadded().Named("mask").Takes("input")
            .Lstm(8, bothWays: true, layers: 2, sequences: false).Takes("steps", "mask").Dense(1));
        var declared = Assert.IsType<LearnNetworkStep>(prepared.Declaration.Learner);
        var json = prepared.Declaration.ToJson();

        Assert.Equal(["tokenEmbedding", "unpadded", "lstm", "dense"], declared.Layers.Select(layer => layer.Kind));
        Assert.True(declared.Layers[0].YesOrNo("keepSteps"));
        Assert.Equal(8, declared.Layers[2].Whole("units"));
        Assert.Equal(2, declared.Layers[2].Whole("layers"));
        Assert.True(declared.Layers[2].YesOrNo("bothWays"));
        Assert.False(declared.Layers[2].YesOrNo("sequences"));
        Assert.Contains("\"bothWays\": true", json, StringComparison.Ordinal);
        Assert.DoesNotContain("\"sequences\"", json, StringComparison.Ordinal);
        Assert.Equal(prepared.Declaration, PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn().WithNetworks()));
    }

    private static NetworkDeclaration Recurrent(NetworkDeclaration steps, string kind) =>
        (kind == "lstm" ? steps.Lstm(8) : steps.Gru(8)).Takes("steps", "mask").Dense(1);
}
