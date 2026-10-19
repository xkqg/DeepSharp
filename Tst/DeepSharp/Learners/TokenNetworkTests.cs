// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A network trained on text takes the ids a pipeline cut it into: the words the training rows held, as vectors it learns and averages,
/// the padding left out and a word it never saw sharing one vector of its own. The vocabulary is the pipeline's, learned from the
/// training rows alone, so a word only the other rows held is no word the network ever learned a thing about.
/// </summary>
public class TokenNetworkTests
{
    private static readonly string[] Praise = ["great", "lovely", "fine", "superb"];
    private static readonly string[] Blame = ["awful", "dull", "bad", "boring"];
    private static readonly string[] Filler = ["film", "plot", "actor", "scene", "music", "ending"];

    // A review is three to seven words: one or two of praise or of blame and the rest filler; it is good when it praises.
    private static InMemoryRowSource Reviews(int count, int offset = 0)
    {
        var random = new Random(97 + offset);

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

                return (IReadOnlyList<string?>)[string.Join(' ', words.OrderBy(_ => random.Next())), good ? "1" : "0"];
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
            .WithTorch(network => declared(network).Adam(0.05).BinaryCrossEntropy().Run(seed: 1, epochs: 30))
            .Build()
            .Run();

    [Fact]
    public void ANetworkDeclaredWithATokenEmbedding_LearnsWhichWordsPraise_FromTheIdsAlone()
    {
        var trained = Prepared(net => net.TokenEmbedding(vocabulary: 24, width: 6).Dense(1)).Train();
        var accuracy = trained.Measures!.Parts.Single(part => part.Part == Part.Test).Values.Single(value => value.Metric == Metric.Accuracy);

        Assert.True(accuracy.Value > 0.9, $"accuracy {accuracy.Value}");
        Assert.Equal(8, trained.TrainedOn.Features.Count);
        Assert.Equal("review.0", trained.TrainedOn.Features[0]);
    }

    [Fact]
    public void ARowServedLater_IsAnsweredThroughTheSameVocabulary_AndAWordNeverSeenIsHeldByItsOwnVector()
    {
        var trained = Prepared(net => net.TokenEmbedding(vocabulary: 24, width: 6).Dense(1)).Train();
        var served = new InMemoryRowSource(["review", "good"], [["a great film with zxqv", "1"], ["an awful plot with zxqv", "0"]]);
        var answers = trained.Predict(served).Answers;

        Assert.True(answers[0][0] >= 0.5, $"{answers[0][0]}");
        Assert.True(answers[1][0] < 0.5, $"{answers[1][0]}");
    }

    [Fact]
    public void TheDeclaredLayer_IsInTheFileWithItsVocabularyAndWidth_AndStandsBesideTheOthersAsWordsDo()
    {
        var prepared = Prepared(net => net.TokenEmbedding(vocabulary: 24, width: 6).Relu().Dense(1));
        var declared = Assert.IsType<LearnNetworkStep>(prepared.Declaration.Learner);

        Assert.Equal(["tokenEmbedding", "relu", "dense"], declared.Layers.Select(layer => layer.Kind));
        Assert.Equal(24, declared.Layers[0].Whole("vocabulary"));
        Assert.Equal(6, declared.Layers[0].Whole("width"));
        Assert.Throws<ArgumentOutOfRangeException>(() => Prepared(net => net.TokenEmbedding(vocabulary: 2, width: 6).Dense(1)).Train());
    }
}
