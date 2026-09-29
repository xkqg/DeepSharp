// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Tensors;

namespace DeepSharp.Tests.Tensors;

/// <summary>
/// A loss, a convolution and a normalisation are not operations of their own: each is written in the seam's
/// operations, and the recording works their gradients out through them. Every number here is PyTorch's for the same
/// rows and the same weights — the rows are the ones the pipeline hands over from the published samples.
/// </summary>
public class CompositionTests
{
    private readonly ITensorBackend _backend = new CpuBackend();

    [Fact]
    public void BinaryCrossEntropyOnLogits_IsTheMeanOfTheSoftplusLessTheLogitWhereTheAnswerIsOne()
    {
        // The logits Titanic's first four training rows reach through a small network, and whether each survived.
        var logits = Tensor.From(new Shape(4, 1), [0.2f, 0.2f, 0.22550064f, 0.2f]);
        var survived = Tensor.From(new Shape(4, 1), [0f, 1f, 1f, 1f]);
        var pass = new RecordingBackend(_backend);

        var loss = pass.Mean(pass.Subtract(pass.Softplus(logits), pass.Multiply(logits, survived)));

        Assert.Equal(0.6452890634536743, loss.Values[0], 1e-7);
        AssertClose([0.13745848834514618, -0.11254151165485382, -0.11096563935279846, -0.11254151165485382], pass.GradientsOf(loss, [logits])[logits]);
    }

    [Fact]
    public void BinaryCrossEntropy_AtALogitOfNothing_SendsBackAHalfEitherWay()
    {
        // Where the logit is nothing the gradient is a half less the answer, as PyTorch's fused loss sends it back; a
        // loss built from a rectifier or an absolute value sends back nought or minus one there.
        var logits = Tensor.From(new Shape(2, 1), [0f, 0f]);
        var answers = Tensor.From(new Shape(2, 1), [0f, 1f]);
        var pass = new RecordingBackend(_backend);

        var total = pass.Scale(
            pass.Mean(pass.Subtract(pass.Softplus(logits), pass.Multiply(logits, answers))), pass.Fill(new Shape(), 2f));

        Assert.Equal<float[]>([0.5f, -0.5f], pass.GradientsOf(total, [logits])[logits].Values.ToArray());
    }

    [Fact]
    public void CrossEntropyAgainstOneClass_IsMinusTheLogSoftmaxOfIt()
    {
        var logits = Tensor.From(new Shape(2, 3), [2f, 1f, 0.1f, 0.5f, 2.5f, 0.2f]);
        var classes = Tensor.From(new Shape(2, 3), [1f, 0f, 0f, 0f, 0f, 1f]);

        Assert.Equal(1.4642908573150635, CrossEntropy(_backend, logits, classes).Values[0], 1e-6);
    }

    [Fact]
    public void CrossEntropyAgainstTheDaysShareOfEachHour_IsPyTorchs_AndSoIsItsGradient()
    {
        // The first two training days of the bike-sharing sample, twenty-one features each, through one layer of fixed
        // weights to twenty-four logits; the answer is how each day's rentals were shared over its hours.
        var days = Tensor.From(new Shape(2, 21),
        [
            0f, 0f, -0.2784217894077301f, -0.2529652416706085f, 0.657240092754364f, -0.430787593126297f, -0.7818315029144287f, 0.6234897971153259f, 0f, 1f, 1f, 0f, 0f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f,
            0f, 0f, -0.22953544557094574f, -0.27891865372657776f, 0.431541383266449f, -0.06757088750600815f, 0f, 1f, 0f, 1f, 1f, 0f, 0f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f,
        ]);
        var shares = Tensor.From(new Shape(2, 24),
        [
            0.01624365523457527f, 0.04060913622379303f, 0.03248731046915054f, 0.013197969645261765f, 0.0010152284521609545f, 0.0010152284521609545f, 0.002030456904321909f, 0.0030456853564828634f, 0.008121827617287636f, 0.014213197864592075f, 0.036548223346471786f, 0.0568527914583683f, 0.0852791890501976f, 0.09543146938085556f, 0.10761421173810959f, 0.11167512834072113f, 0.09441624581813812f, 0.0680203065276146f, 0.03553299605846405f, 0.03756345063447952f, 0.036548223346471786f, 0.034517765045166016f, 0.02842639572918415f, 0.039593908935785294f,
            0.02122347056865692f, 0.02122347056865692f, 0.01123595517128706f, 0.00749063678085804f, 0.00374531839042902f, 0f, 0.0024968788493424654f, 0.0012484394246712327f, 0.009987515397369862f, 0.024968789890408516f, 0.06616728752851486f, 0.08739075809717178f, 0.11610487103462219f, 0.09363295882940292f, 0.07365792989730835f, 0.09238451719284058f, 0.09488140046596527f, 0.08114856481552124f, 0.06616728752851486f, 0.03745318204164505f, 0.027465667575597763f, 0.0387016236782074f, 0.01123595517128706f, 0.009987515397369862f,
        ]);
        var weights = Tensor.From(new Shape(21, 24), [.. Enumerable.Range(0, 21 * 24).Select(at => (((5 * (at / 24)) + (3 * (at % 24))) % 13 - 6) / 40f)]);
        var bias = Tensor.Zeros(new Shape(24));
        var pass = new RecordingBackend(_backend);

        var loss = CrossEntropy(pass, pass.AddRow(pass.MatMul(days, weights), bias), shares);
        var gradients = pass.GradientsOf(loss, [weights, bias]);

        Assert.Equal(3.2375428676605225, loss.Values[0], 1e-6);
        AssertClose(
            [0.029888564720749855, 0.021646419540047646, 0.00921641569584608, 0.0220717191696167, 0.04249461740255356, 0.044172242283821106],
            Tensor.From(new Shape(6), gradients[bias].Values[..6]));
        AssertClose(
            [-0.007620698772370815, -0.005249547306448221, -0.0020554496441036463, -0.005570740904659033],
            Tensor.From(new Shape(4), gradients[weights].Values[48..52]));
    }

    [Fact]
    public void AConvolution_IsTheImagesPatchesTimesTheKernel_PlusItsBias()
    {
        // One image of four by four holding a sixteenth to one, one channel, a kernel of three by three with a border of one.
        var image = Tensor.From(new Shape(1, 4, 4, 1), [.. Enumerable.Range(1, 16).Select(at => at / 16f)]);
        var kernel = Tensor.From(new Shape(9, 1), [0.1f, 0f, -0.1f, 0.2f, 0.5f, -0.2f, 0.1f, 0f, -0.1f]);
        var bias = Tensor.From(new Shape(1), [0.05f]);
        var pass = new RecordingBackend(_backend);

        var output = Convolved(pass, image, kernel, bias, new Window(3, 3) { Padding = 1 });
        var loss = pass.Mean(pass.Multiply(output, output));
        var gradients = pass.GradientsOf(loss, [image, kernel, bias]);

        Assert.Equal(new Shape(1, 4, 4, 1), output.Shape);
        AssertClose(
            [0.01874999701976776, 0.07500000298023224, 0.10625000298023224, 0.2562499940395355, 0.056249991059303284, 0.1875000298023224, 0.2187500149011612, 0.4749999940395355,
             0.08125000447034836, 0.3125000596046448, 0.3437500298023224, 0.699999988079071, 0.21875, 0.44999998807907104, 0.4812500476837158, 0.8062500357627869],
            output);
        AssertClose(
            [0.005390625447034836, 0.008906250819563866, 0.014765625819563866, 0.010624999180436134, 0.013046875596046448, 0.02015625312924385, 0.027968749403953552, 0.018593749031424522,
             0.020859377458691597, 0.031406253576278687, 0.039218753576278687, 0.026406247168779373, 0.028828125447034836, 0.03796875476837158, 0.043828126043081284, 0.034062497317790985],
            gradients[image]);
        AssertClose(
            [0.22114259004592896, 0.27119141817092896, 0.15537109971046448, 0.3536132872104645, 0.4183593988418579, 0.23417970538139343, 0.244384765625, 0.2782226800918579, 0.1411132961511612],
            gradients[kernel]);
        AssertClose([0.5984375476837158], gradients[bias]);
    }

    [Fact]
    public void AConvolutionOfTwoChannelsIntoTwo_WithAStrideOfTwo_IsPyTorchs()
    {
        // PyTorch's image of two channels of five by five and its two kernels, laid out with the channels last: each
        // kernel row is a place in the window and a channel in, each column a channel out.
        var image = Tensor.From(new Shape(1, 5, 5, 2), [.. Enumerable.Range(0, 50).Select(at => (((at % 2) * 25) + (at / 2)) / 50f - 0.5f)]);
        var kernel = Tensor.From(new Shape(18, 2), [.. Enumerable.Range(0, 36).Select(at => (((at % 2) * 18) + ((at / 2 % 2) * 9) + (at / 4)) / 36f - 0.25f)]);
        var bias = Tensor.From(new Shape(2), [0f, 0.1f]);
        var pass = new RecordingBackend(_backend);

        var output = Convolved(pass, image, kernel, bias, new Window(3, 3) { Stride = 2 });
        var loss = pass.Mean(pass.Multiply(output, output));
        var gradients = pass.GradientsOf(loss, [image, kernel, bias]);

        Assert.Equal(new Shape(1, 2, 2, 2), output.Shape);
        AssertClose([0.7016666531562805, -0.3683333396911621, 0.6916666030883789, -0.018333330750465393, 0.6516666412353516, 1.3816665410995483, 0.6416666507720947, 1.7316666841506958], output);
        AssertClose([0.6716666221618652, 0.6816666126251221], gradients[bias]);
        AssertClose(
            [-0.2578333020210266, -0.16803331673145294, 0.07799999415874481, 0.1728000044822693, -0.24439997971057892, -0.15440000593662262, 0.0914333313703537, 0.1864333301782608,
             -0.23096665740013123, -0.14076665043830872, 0.10486665368080139, 0.20006665587425232, -0.19066666066646576, -0.09986665844917297, 0.14516666531562805, 0.24096664786338806,
             -0.17723333835601807, -0.08623332530260086, 0.15860000252723694, 0.25459998846054077, -0.16380000114440918, -0.07259999215602875, 0.17203330993652344, 0.2682332992553711,
             -0.12349998950958252, -0.03169999644160271, 0.2123333215713501, 0.3091333210468292, -0.11006666719913483, -0.018066665157675743, 0.22576665878295898, 0.32276666164398193,
             -0.09663332998752594, -0.004433338064700365, 0.23919998109340668, 0.33640000224113464],
            gradients[kernel]);
        AssertClose(
            [-0.06687499582767487, -0.046041667461395264, -0.0645601898431778, -0.0437268503010273, -0.10662037134170532, -0.043703705072402954, -0.039699070155620575, 0.0023842614609748125,
             -0.035023145377635956, 0.0070601836778223515],
            Tensor.From(new Shape(10), gradients[image].Values[..10]));
    }

    [Fact]
    public void AConvolutionOfABatchOfTwo_ConvolvesEachImageOnItsOwn()
    {
        // Two images of three by three and two kernels with a border of one: the answer PyTorch gives for the batch, which is
        // each image's own answer side by side. The loss weighs every output by its own number.
        var images = Tensor.From(new Shape(2, 3, 3, 1), [.. Enumerable.Range(0, 18).Select(at => (at / 9f) - 1)]);
        var kernel = Tensor.From(new Shape(9, 2), [.. Enumerable.Range(0, 18).Select(at => (((at % 2) * 9) + (at / 2)) / 18f - 0.5f)]);
        var none = Tensor.Zeros(new Shape(2));
        var weighing = Tensor.From(new Shape(2, 3, 3, 2),
        [
            -1f, -0.48571425676345825f, -0.9428571462631226f, -0.4285714030265808f, -0.8857142925262451f, -0.37142854928970337f, -0.8285714387893677f, -0.3142856955528259f, -0.7714285850524902f,
            -0.2571428418159485f, -0.7142857313156128f, -0.19999998807907104f, -0.6571428775787354f, -0.1428571343421936f, -0.6000000238418579f, -0.08571428060531616f, -0.5428571701049805f,
            -0.02857142686843872f, 0.02857142686843872f, 0.5428571701049805f, 0.08571428060531616f, 0.6000000238418579f, 0.1428571343421936f, 0.6571428775787354f, 0.19999998807907104f,
            0.7142857313156128f, 0.2571428418159485f, 0.7714285850524902f, 0.3142856955528259f, 0.8285714387893677f, 0.37142854928970337f, 0.8857142925262451f, 0.4285714030265808f,
            0.9428571462631226f, 0.48571425676345825f, 1f,
        ]);
        var pass = new RecordingBackend(_backend);

        var output = Convolved(pass, images, kernel, none, new Window(3, 3) { Padding = 1 });
        var total = pass.Scale(pass.Mean(pass.Multiply(output, weighing)), pass.Fill(new Shape(), 36f));
        var gradients = pass.GradientsOf(total, [images, kernel]);

        AssertClose(
            [0.5802469253540039, -0.9753085970878601, 0.9506173133850098, -1.2160494327545166, 0.6543209552764893, -0.6790123581886292, 1.1481481790542603, -0.6851851940155029,
             1.7592592239379883, -0.7407407164573669, 1.1481481790542603, -0.3518518805503845, 0.6543209552764893, -0.23456789553165436, 0.9506173133850098, -0.21604938805103302,
             0.5802469253540039, -0.08641976118087769, -0.08641976118087769, 0.3580247163772583, -0.21604938805103302, 0.6172839403152466, -0.23456789553165436, 0.43209877610206604,
             -0.3518518805503845, 0.8148148059844971, -0.7407407164573669, 1.2592592239379883, -0.6851851940155029, 0.8148148059844971, -0.6790123581886292, 0.43209877610206604,
             -1.2160494327545166, 0.6172839403152466, -0.9753085970878601, 0.3580247163772583],
            output);
        AssertClose(
            [1.149206280708313, 1.4603173732757568, 0.8317460417747498, 0.8952380418777466, 1.033333420753479, 0.5333333611488342, 0.3492063581943512, 0.3746032118797302, 0.18412700295448303,
             0.00634924927726388, 0.0888889729976654, 0.14603178203105927, 0.20952388644218445, 0.519047737121582, 0.533333420753479, 0.5777778029441833, 1.0603175163269043, 0.8698412775993347],
            gradients[images]);
        AssertClose(
            [2.501587152481079, 1.3587301969528198, 3.7650792598724365, 2.393650770187378, 2.450793743133545, 1.765079379081726, 3.866666555404663, 3.180952548980713, 5.647619247436523,
             5.133333206176758, 3.5619046688079834, 3.5619046688079834, 2.0444445610046387, 2.2730157375335693, 2.8507936000823975, 3.536508083343506, 1.6888889074325562, 2.374603271484375],
            gradients[kernel]);
    }

    [Fact]
    public void LayerNormalisation_WrittenInTheSeamsOperations_IsPyTorchs_AndSoAreTheGradientsOfItsScaleAndShift()
    {
        // Each row brought to a mean of nothing and a spread of one — the row's own statistics, as a column added into
        // every column by turning the matrix round — then scaled and shifted per column.
        var rows = Tensor.From(new Shape(4, 3), [1f, 2f, -1f, 0.5f, -2f, 3f, 2f, 0f, 1f, -1.5f, 4f, 0f]);
        var scale = Tensor.From(new Shape(3), [1f, 0.5f, 2f]);
        var shift = Tensor.From(new Shape(3), [0f, 0.1f, -0.2f]);
        var pass = new RecordingBackend(_backend);

        var third = pass.Fill(new Shape(), 1f / 3);
        var centred = pass.Subtract(rows, AcrossEachRow(pass, pass.Scale(pass.SumRows(pass.Transpose(rows)), third), 3));
        var variance = pass.Scale(pass.SumRows(pass.Transpose(pass.Multiply(centred, centred))), third);
        var spread = pass.Sqrt(pass.Add(variance, pass.Fill(new Shape(4), 1e-5f)));
        var normalised = pass.Divide(centred, AcrossEachRow(pass, spread, 3));
        var output = pass.AddRow(pass.Multiply(normalised, pass.AddRow(pass.Fill(new Shape(4, 3), 0f), scale)), shift);
        var loss = pass.Mean(pass.Multiply(output, output));
        var gradients = pass.GradientsOf(loss, [scale, shift]);

        AssertClose([0.26726034283638, 0.6345207095146179, -2.87260365486145], Tensor.From(new Shape(3), output.Values[..3]));
        AssertClose([0.4302855134010315, 0.5000317692756653, 1.1538729667663574], gradients[scale]);
        AssertClose([0.08114255964756012, 0.06530679762363434, -0.290179044008255], gradients[shift]);
    }

    // Minus the mean over rows of each row's answers times its log-softmax: the answers may be one class or shares.
    private static Tensor CrossEntropy(ITensorBackend backend, Tensor logits, Tensor answers) =>
        backend.Scale(
            backend.Mean(backend.Multiply(answers, backend.LogSoftmax(logits))),
            backend.Fill(new Shape(), -logits.Shape[1]));

    private static Tensor Convolved(ITensorBackend backend, Tensor images, Tensor kernel, Tensor bias, Window window)
    {
        var patches = backend.Unfold(images, window);
        var height = ((images.Shape[1] + (2 * window.Padding) - window.Height) / window.Stride) + 1;
        var width = ((images.Shape[2] + (2 * window.Padding) - window.Width) / window.Stride) + 1;

        return backend.Reshape(
            backend.AddRow(backend.MatMul(patches, kernel), bias), new Shape(images.Shape[0], height, width, kernel.Shape[1]));
    }

    // A value per row, as a column added into every column of a matrix that many columns wide.
    private static Tensor AcrossEachRow(ITensorBackend backend, Tensor perRow, int columns) =>
        backend.Transpose(backend.AddRow(backend.Fill(new Shape(columns, perRow.Shape[0]), 0f), perRow));

    // Within a hundred-thousandth of the value, or of one where it is smaller: float arithmetic taken in another order
    // than PyTorch's differs in the last digits, and a value in the wrong place differs in the first.
    private static void AssertClose(double[] expected, Tensor actual)
    {
        Assert.Equal(expected.Length, actual.Values.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], actual.Values[at], Math.Max(1e-5, Math.Abs(expected[at]) * 1e-5));
        }
    }
}
