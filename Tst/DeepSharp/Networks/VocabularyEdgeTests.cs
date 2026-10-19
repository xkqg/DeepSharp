// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// The edges of the words 0.11.0 adds: what a parametric rectifier, a softmax and an embedding refuse to take, the forms a
/// normalisation is begun in and written as, and what a Poisson loss refuses to be told or to be handed.
/// </summary>
public class VocabularyEdgeTests
{
    private static readonly CpuBackend Backend = new();

    private static string Written(Network network)
    {
        var buffer = new System.Buffers.ArrayBufferWriter<byte>();

        using (var writer = new Utf8JsonWriter(buffer))
        {
            writer.WriteStartObject();
            writer.WritePropertyName("network");
            NetworkDocument.WriteNetwork(writer, network, new MeanSquaredError());
            writer.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.WrittenSpan);
    }

    [Fact]
    public void AParametricRectifierOfASharePerChannel_RefusesARowAlone_AndALastAxisOfAnotherCount()
    {
        var each = new PRelu(3);

        Assert.Contains("of 3 shares takes a batch", Assert.Throws<ArgumentException>(() => each.Forward(Tensor.Zeros(new Shape(3)), Pass.Evaluation(Backend))).Message, StringComparison.Ordinal);
        Assert.Contains("was handed a 2x4 one", Assert.Throws<ArgumentException>(() => each.Forward(Tensor.Zeros(new Shape(2, 4)), Pass.Evaluation(Backend))).Message, StringComparison.Ordinal);
        Assert.Equal(new Shape(2, 3), each.Forward(Tensor.Zeros(new Shape(2, 3)), Pass.Evaluation(Backend)).Shape);
        Assert.Equal(new Shape(5), new PRelu().Forward(Tensor.Zeros(new Shape(5)), Pass.Evaluation(Backend)).Shape);
    }

    [Fact]
    public void ASoftmax_RefusesATensorThatIsNoBatch()
    {
        var wrong = Assert.Throws<ArgumentException>(() => new Softmax().Forward(Tensor.Zeros(new Shape(4)), Pass.Evaluation(Backend)));

        Assert.Contains("a 4 tensor is not a batch of any", wrong.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AnInstanceNormalisation_BegunByItsChannelsAlone_KeepsItsScaleAndItsShift()
    {
        var plain = new InstanceNorm(4);

        Assert.Equal(4, plain.Features);
        Assert.Equal(default(Affine), plain.Affine);
        Assert.NotNull(plain.Weight);
        Assert.NotNull(plain.Bias);
    }

    [Fact]
    public void ARootMeanSquareNormalisationWithoutItsScale_IsWrittenSayingSo_AndReadBackWithoutIt()
    {
        var without = Written(new LayerStack(new RmsNorm(3, scale: false)));
        var with = Written(new LayerStack(new RmsNorm(3)));
        var read = Assert.IsType<RmsNorm>(Assert.IsType<LayerStack>(NetworkDocument.ReadNetwork(without, "network", NetworkCatalog.BuiltIn()).Network).Layers[0]);

        Assert.Contains("\"scale\":false", without, StringComparison.Ordinal);
        Assert.DoesNotContain("\"scale\"", with, StringComparison.Ordinal);
        Assert.Null(read.Weight);
        Assert.True(read.Affine.WithoutScale);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1e-8)]
    [InlineData(double.NaN)]
    [InlineData(double.PositiveInfinity)]
    public void APoissonLoss_RefusesAnEpsilonThatIsNoFiniteNumberAboveNothing(double epsilon) =>
        Assert.Throws<ArgumentOutOfRangeException>(() => new Poisson(logInput: false) { Epsilon = epsilon });

    [Fact]
    public void APoissonLoss_TakesCountsOfNothingAndMore_AndRefusesOneBelowNothing()
    {
        var loss = new Poisson(logInput: false) { Epsilon = 1e-7 };

        Assert.Equal(1e-7, loss.Epsilon);
        Assert.Null(loss.Refusal([0, 2, 5]));
        Assert.Contains("an answer of -1 is below nothing", loss.Refusal([3, -1]), StringComparison.Ordinal);
    }

    [Fact]
    public void AnEmbeddingWord_RefusesAnExampleThatIsNoRow_AndABlockBeyondTheRow()
    {
        var notARow = Assert.Throws<ArgumentException>(() => new Sequential().Embedding([0, 1], 2).Lower(new Shape(2, 3), new RandomStream(1)));
        var beyond = Assert.Throws<ArgumentException>(() => new Sequential().Embedding([2, 3], 2).Lower(new Shape(3), new RandomStream(1)));

        Assert.Contains("embedding, takes each example as a row of numbers", notARow.Message, StringComparison.Ordinal);
        Assert.Contains("reads a category from column 4 of the row, and each row reaching it holds 3", beyond.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ATokenEmbeddingWord_RefusesAnExampleThatIsNoRowOfIds_AndLowersOneThatIs()
    {
        var notARow = Assert.Throws<ArgumentException>(() => new Sequential().TokenEmbedding(10, 3).Lower(new Shape(2, 3), new RandomStream(1)));
        var lowered = new Sequential().TokenEmbedding(10, 3).Lower(new Shape(4), new RandomStream(1));

        Assert.Contains("tokenEmbedding, takes each example as a row of ids", notARow.Message, StringComparison.Ordinal);
        Assert.NotNull(lowered);
    }

}
