// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Learners.Networks;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// A sampler is the caller's, and a candidate it makes is trained as it stands: a learning rate of 99 for a range that stops at a
/// tenth, a layer of two and a half units, a word the space never offered. The study holds every value to its dimension before
/// anything is trained on it, above the sampler, so a sampler written elsewhere can be wrong without a network learning from it.
/// </summary>
public class SamplerValuesTests
{
    private static readonly SearchSpace Space = new(
    [
        new Choices("optimizer", ["adam", "sgd"]),
        new LogRange("rate", 1e-3, 1e-1),
        new NumberRange("dropout", 0, 0.5),
        new WholeRange("units", 2, 8),
    ]);

    public static TheoryData<string, double, string?> Refused { get; } = new()
    {
        { "rate", 99, null },
        { "rate", 1e-4, null },
        { "rate", double.NaN, null },
        { "rate", double.PositiveInfinity, null },
        { "dropout", -0.1, null },
        { "dropout", 0.6, null },
        { "units", 2.5, null },
        { "units", 1, null },
        { "units", 9, null },
        { "rate", 0.01, "adam" },
        { "optimizer", 0, "rmsprop" },
        { "optimizer", 1, "adam" },
        { "optimizer", 0, null },
        { "optimizer", 2, "sgd" },
    };

    public static TheoryData<string, double, string?> Held { get; } = new()
    {
        { "rate", 1e-3, null },
        { "rate", 0.1, null },
        { "rate", 0.01, null },
        { "dropout", 0, null },
        { "dropout", 0.5, null },
        { "units", 2, null },
        { "units", 8, null },
        { "units", 5, null },
        { "optimizer", 0, "adam" },
        { "optimizer", 1, "sgd" },
    };

    [Theory]
    [MemberData(nameof(Refused))]
    public void AValueTheDimensionCannotTake_IsRefused_NamingTheDimensionAndWhatItHolds(string dimension, double number, string? word)
    {
        var study = Studied(dimension, number, word);

        var refused = Assert.Throws<InvalidOperationException>(() => study.Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken));

        Assert.Contains($"'{dimension}'", refused.Message, StringComparison.Ordinal);
    }

    [Theory]
    [MemberData(nameof(Held))]
    public void AValueTheDimensionTakes_IsTrainedOn_EvenAtItsBounds(string dimension, double number, string? word)
    {
        var study = Studied(dimension, number, word);

        Assert.Single(study.Run(deal => StudyTests.Passengers(deal), TestContext.Current.CancellationToken).Trials);
    }

    private static Study Studied(string dimension, double number, string? word)
    {
        Setting[] drawn =
        [
            new("optimizer", 0, "adam"),
            new("rate", 0.01, null),
            new("dropout", 0.1, null),
            new("units", 4, null),
        ];

        drawn = [.. drawn.Select(setting => setting.Name == dimension ? new Setting(dimension, number, word) : setting)];

        return new Study(Space, (candidate, network) => network.Dense(2).Relu().Dense(1).Adam(0.01).BinaryCrossEntropy().Run(seed: 1, epochs: 1))
        {
            Sampler = new Fixed(drawn),
            Trials = 1,
        };
    }

    private sealed class Fixed(Setting[] settings) : ISampler
    {
        public Candidate Suggest(SearchSpace space, int trial, IReadOnlyList<Trial> finished) => new(settings);
    }
}
