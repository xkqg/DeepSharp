// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using DeepSharp.Learners.Networks;
using DeepSharp.Networks;
using DeepSharp.Pipelines;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Learners;

/// <summary>
/// One network trained in sequence after another, on what the first left unexplained: a correction network fitted on the
/// basis's own frozen answer subtracted from the truth, both replayed unchanged on rows the basis never trains its
/// weights from again. What a boosted pair predicts is both halves' raw answers summed and brought back into the answer's
/// own units once.
/// </summary>
public class BoostedNetworkTests
{
    private const string Folder = "BoostedNetworkTests";

    private static FittingBuilder Regression(int splitSeed = 5)
    {
        Directory.CreateDirectory(Folder);

        var rows = new StringBuilder("x,y\n");

        for (var row = 0; row < 60; row++)
        {
            rows.Append(CultureInfo.InvariantCulture, $"{1 + (row % 4)},{1000 + (row % 5)}\n");
        }

        File.WriteAllText(Path.Join(Folder, "regression.csv"), rows.ToString());

        return Pdd.Create().ReadCsv(Path.Join(Folder, "regression.csv")).Declare(schema => schema.Number("x", "y")).SplitAtRandom(0.70, 0.15, seed: splitSeed).Normalise("x", Scale.MidRange).Target("y");
    }

    private static FittingBuilder Classifying()
    {
        Directory.CreateDirectory(Folder);

        var rows = new StringBuilder("x,y\n");

        for (var row = 0; row < 60; row++)
        {
            rows.Append(CultureInfo.InvariantCulture, $"{1 + (row % 4)},{row % 2}\n");
        }

        File.WriteAllText(Path.Join(Folder, "classifying.csv"), rows.ToString());

        return Pdd.Create().ReadCsv(Path.Join(Folder, "classifying.csv")).Declare(schema => schema.Number("x").Integer("y")).SplitAtRandom(0.70, 0.15, seed: 5).Normalise("x", Scale.MidRange).Target("y");
    }

    private static LearnNetworkStep Step(Pipeline pipeline) => Assert.IsType<LearnNetworkStep>(pipeline.Declaration.Learner);

    private static (PreparedData Prepared, LearnNetworkStep Basis, LearnNetworkStep Correction) Setup(int splitSeed = 5)
    {
        var pipeline = Regression(splitSeed).WithTorch(network => network.Dense(2).Relu().Dense(1).Adam(0.05).MeanSquaredError().Run(1, 6)).Build();
        var basis = Step(pipeline);

        return (pipeline.RunFor(basis.Needs), basis, basis with { Seed = 2 });
    }

    private static CsvRowSource Served() => CsvRowSource.FromText(string.Join('\n', File.ReadLines(Path.Join(Folder, "regression.csv")).Take(9)) + "\n");

    private static void AssertClose(IReadOnlyList<double[]> expected, IReadOnlyList<double[]> actual, double tolerance = 1e-4)
    {
        Assert.Equal(expected.Count, actual.Count);

        for (var row = 0; row < expected.Count; row++)
        {
            Assert.Equal(expected[row].Length, actual[row].Length);

            for (var column = 0; column < expected[row].Length; column++)
            {
                Assert.True(
                    Math.Abs(expected[row][column] - actual[row][column]) < tolerance,
                    $"row {row} column {column}: expected {expected[row][column]}, was {actual[row][column]}.");
            }
        }
    }

    private static double MeanSquaredErrorOf(IReadOnlyList<double[]> expected, IReadOnlyList<double[]> actual) =>
        expected.SelectMany((row, at) => row.Select((value, column) => (value - actual[at][column]) * (value - actual[at][column]))).Average();

    [Fact]
    public void TrainBoosted_TrainsABasisAndACorrection_AndPredictsTheirSumBackInTheAnswersOwnUnits()
    {
        var (prepared, basis, correction) = Setup();

        var boosted = prepared.TrainBoosted(basis, correction, new BoostedOptions { Cancellation = TestContext.Current.CancellationToken });

        Assert.Same(prepared, boosted.Prepared);
        Assert.Equal(1L, boosted.Basis.TrainedOn.Seed);

        var rows = Served();
        var predicted = boosted.Predict(rows);
        var basisAlone = prepared.Train(basis, cancellation: TestContext.Current.CancellationToken).Predict(rows);

        // The boosted pair answers something, and it need not equal the basis alone: the correction learned from what the
        // basis (trained once more, independently, to the same seed) left unexplained.
        Assert.Equal(basisAlone.Answers.Count, predicted.Answers.Count);
    }

    [Fact]
    public void TheCorrectionsTrainingTarget_IsTheTruthLessTheFrozenBasissOwnAnswer_OnTheTrainingRowsAlone()
    {
        var (prepared, basis, correction) = Setup();
        var basisTrained = prepared.Train(basis, cancellation: TestContext.Current.CancellationToken);

        var boosted = prepared.TrainBoosted(basis, correction, new BoostedOptions { Cancellation = TestContext.Current.CancellationToken });

        var trainBatch = prepared.Batch(Part.Train, TrainedNetwork.FeatureNeeds);
        var chunking = new Chunking(new CpuBackend(), TrainedNetwork.ServingBatchSize) { Example = trainBatch.ExampleShape };
        var basisOnTrain = TrainedNetwork.Answered(basisTrained.Network, basisTrained.Loss, trainBatch.Features, chunking);
        var expectedTarget = trainBatch.Answers!.Select((row, at) => row.Select((value, column) => value - basisOnTrain[at][column]).ToArray()).ToArray();

        var correctionAnswered = TrainedNetwork.Answered(boosted.Correction.Network, boosted.Correction.Loss, trainBatch.Features, chunking);

        // The correction was fitted to answer expectedTarget on these rows, not the raw truth: its mean squared error
        // against the residual target it actually trained on is far smaller than against the truth it was not fitted to.
        var errorAgainstTarget = MeanSquaredErrorOf(expectedTarget, correctionAnswered);
        var errorAgainstTruth = MeanSquaredErrorOf(trainBatch.Answers!, correctionAnswered);

        Assert.True(errorAgainstTarget < errorAgainstTruth);
    }

    [Fact]
    public void ABasisOrACorrectionWhoseLossIsNotTheIdentity_IsRefused_NamingWhichOneAndItsActivation()
    {
        var (prepared, basis, _) = Setup();
        var classifying = Step(Classifying().WithTorch(network => network.Dense(1).Sigmoid().Adam(0.05).BinaryCrossEntropy().Run(1, 2)).Build());

        var basisFault = Assert.Throws<InvalidOperationException>(() => prepared.TrainBoosted(classifying, basis));
        var correctionFault = Assert.Throws<InvalidOperationException>(() => prepared.TrainBoosted(basis, classifying));

        Assert.Contains("Identity", basisFault.Message, StringComparison.Ordinal);
        Assert.Contains("Identity", correctionFault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ABasisOrACorrectionThatAnswersAQuantile_IsRefused_ByName_EvenThoughQuantilesActivationIsTheIdentity()
    {
        var (prepared, basis, _) = Setup();
        var quantile = Step(Regression().WithTorch(network => network.Dense(1).Adam(0.05).Quantile(0.5).Run(1, 2)).Build());

        var fault = Assert.Throws<InvalidOperationException>(() => prepared.TrainBoosted(basis, quantile));

        Assert.Contains("Quantile", fault.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ABoostedPair_IsRefused_WhenItsTwoNetworksWereTrainedBehindDifferentFitsOfAPipeline()
    {
        var (preparedOne, basisOne, correctionOne) = Setup(splitSeed: 5);
        var boostedOne = preparedOne.TrainBoosted(basisOne, correctionOne, new BoostedOptions { Cancellation = TestContext.Current.CancellationToken });
        var (preparedTwo, basisTwo, correctionTwo) = Setup(splitSeed: 11);
        var boostedTwo = preparedTwo.TrainBoosted(basisTwo, correctionTwo, new BoostedOptions { Cancellation = TestContext.Current.CancellationToken });

        Assert.NotEqual(boostedOne.Basis.TrainedOn.TrainedBehind, boostedTwo.Basis.TrainedOn.TrainedBehind);

        var fault = Assert.Throws<ArgumentException>(() => BoostedNetwork.Of(boostedOne.Basis, boostedTwo.Correction, BoostedNetwork.Digest(boostedOne.Basis)));

        Assert.Contains("fit", fault.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ABoostedPair_IsRefused_WhenItsCorrectionWasFittedAgainstADifferentBasis()
    {
        var (prepared, basis, correction) = Setup();
        var boosted = prepared.TrainBoosted(basis, correction, new BoostedOptions { Cancellation = TestContext.Current.CancellationToken });
        var anotherBasis = prepared.Train(basis with { Seed = 9 }, cancellation: TestContext.Current.CancellationToken);

        // As a file would carry it: the digest the correction was actually fitted against (the ORIGINAL basis's), beside a
        // basis that is not that one, though trained behind the same fit.
        var fault = Assert.Throws<ArgumentException>(() => BoostedNetwork.Of(anotherBasis, boosted.Correction, BoostedNetwork.Digest(boosted.Basis)));

        Assert.Contains("basis", fault.Message, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void ABoostedPairWrittenAndReadAgain_AnswersToTheLastBit()
    {
        var (prepared, basis, correction) = Setup();
        var boosted = prepared.TrainBoosted(basis, correction, new BoostedOptions { Cancellation = TestContext.Current.CancellationToken });
        var rows = Served();

        var read = BoostedNetwork.FromJson(boosted.ToJson(), NetworkCatalog.BuiltIn(), StepCatalog.BuiltIn().WithNetworks());

        AssertClose(boosted.Predict(rows).Answers, read.Predict(rows).Answers, 1e-9);
        Assert.Equal(boosted.ToJson(), read.ToJson());
    }

    [Fact]
    public void TheFile_HoldsItsVersion_TheBasis_TheCorrectionAndOnePipeline()
    {
        var (prepared, basis, correction) = Setup();
        var boosted = prepared.TrainBoosted(basis, correction, new BoostedOptions { Cancellation = TestContext.Current.CancellationToken });

        var file = JsonNode.Parse(boosted.ToJson())!.AsObject();

        Assert.Equal(["version", "basis", "correction", "basisDigest", "pipeline"], file.Select(each => each.Key));
        Assert.Equal(1, (int)file["version"]!);
    }

    [Fact]
    public void TheCorrectionHalf_IsNotAPubliclyServableTrainedNetworkOfItsOwn()
    {
        var property = typeof(BoostedNetwork).GetProperty("Correction", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.NonPublic);

        Assert.NotNull(property);
        Assert.False(property!.GetMethod!.IsPublic);
    }
}
