// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using DeepSharp.Tensors;

namespace DeepSharp.Tests.Networks;

/// <summary>
/// The two normalisations, over the last axis — a row's features, or an image's channels when they come last. A batch
/// normalisation measures each feature over the batch while it trains and keeps a running mean and variance, measured on
/// the training batches alone, to use afterwards; a layer normalisation measures each row on its own and keeps nothing.
/// Every number is PyTorch's for the same batch, the same scale and the same shift — put into the slots by the door the
/// library keeps to itself, which is why these run on the light engine alone; what a normalisation does on any engine
/// otherwise runs from the contract.
/// </summary>
public class NormalisationTests
{
    private readonly ITensorBackend _backend = new CpuBackend();

    private static readonly float[] Scale = [1f, 0.5f, 2f];

    private static readonly float[] Shift = [0f, 0.1f, -0.2f];

    // What each output is weighed by in the loss whose gradients are checked: a loss that is the square of a normalised
    // output says nothing, since a normalisation's output does not move when its input is scaled.
    private static Tensor Weighing() => Tensor.From(new Shape(4, 3), [0.3f, -0.2f, 0.5f, 0.1f, 0.4f, -0.3f, -0.6f, 0.2f, 0.1f, 0.2f, -0.1f, 0.7f]);

    [Fact]
    public void ABatchNormalisation_InATrainingPass_IsPyTorchs_AndMovesItsRunningStatisticsAsPyTorchDoes()
    {
        var norm = Started(new BatchNorm(3));

        var output = norm.Forward(WalkedRows.Batch(), Training());

        AssertClose(
            [0.3922310769557953, 0.32360658049583435, -2.5664265155792236, 0.0, -0.5708197355270386, 2.842548370361328,
             1.1766932010650635, -0.12360657006502151, 0.1380608081817627, -1.5689243078231812, 0.7708197236061096, -1.2141828536987305],
            output);
        AssertClose([0.05000000074505806, 0.10000000149011612, 0.07500000298023224], norm.RunningMean.Value);
        AssertClose([1.1166666746139526, 1.566666603088379, 1.191666603088379], norm.RunningVariance.Value);
    }

    [Fact]
    public void ABatchNormalisation_InAnEvaluationPass_UsesWhatItMeasuredInTraining_AndMovesNothing()
    {
        var norm = Started(new BatchNorm(3));
        norm.Forward(WalkedRows.Batch(), Training());
        var mean = norm.RunningMean.Value;
        var variance = norm.RunningVariance.Value;

        var output = norm.Forward(WalkedRows.Batch(), Pass.Evaluation(_backend));

        AssertClose([0.8990004062652588, 0.8589862585067749, -2.1695148944854736], Tensor.From(new Shape(3), output.Values[..3]));
        Assert.Same(mean, norm.RunningMean.Value);
        Assert.Same(variance, norm.RunningVariance.Value);
    }

    [Fact]
    public void ABatchNormalisations_Gradients_ArePyTorchs()
    {
        var norm = Started(new BatchNorm(3));
        var batch = WalkedRows.Batch();
        var pass = new RecordingBackend(_backend);

        var output = norm.Forward(batch, Pass.Training(pass, new RandomStream(1), 0, 0));
        var loss = pass.Scale(pass.Mean(pass.Multiply(output, Weighing())), pass.Fill(new Shape(), 12f));
        var gradients = pass.GradientsOf(loss, [batch, norm.Weight.Value, norm.Bias.Value]);

        AssertClose(
            [0.3047329783439636, -0.04024922475218773, -0.21635639667510986, 0.07844621688127518, 0.008944387547671795, -0.030911793932318687,
             -0.26249435544013977, 0.006708239205181599, -0.12363407760858536, -0.12068483233451843, 0.024596592411398888, 0.3709021806716919],
            gradients[batch]);
        AssertClose([-0.9021315574645996, -0.8497050404548645, -1.386049747467041], gradients[norm.Weight.Value]);
        AssertClose([0, 0.30000001192092896, 1.0], gradients[norm.Bias.Value]);
    }

    [Fact]
    public void ABatchNormalisation_OfImagesWithTheirChannelsLast_NormalisesEachChannelOverEveryPlaceOfEveryImage()
    {
        // Two images of two by two with three channels: PyTorch's BatchNorm2d over the same values, laid out channels first.
        var images = Tensor.From(new Shape(2, 2, 2, 3),
        [
            -0.9974949955940247f, -0.9772627949714661f, -0.9371203780174255f, -0.8778854608535767f, -0.8007650375366211f, -0.7073302865028381f,
            -0.5994846820831299f, -0.4794255495071411f, -0.34959876537323f, -0.21264955401420593f, -0.07136781513690948f, 0.07136781513690948f,
            0.21264955401420593f, 0.34959876537323f, 0.4794255495071411f, 0.5994846224784851f, 0.7073303461074829f, 0.8007650971412659f,
            0.8778854608535767f, 0.9371203184127808f, 0.9772627949714661f, 0.9974949955940247f, 0.9974047541618347f, 0.9769938588142395f,
        ]);
        var norm = Started(new BatchNorm(3));

        var output = norm.Forward(images, Training());

        Assert.Equal(new Shape(2, 2, 2, 3), output.Shape);
        AssertClose(
            [-1.354187250137329, -0.6253463625907898, -3.286909818649292, -1.1918067932128906, -0.5045813322067261, -2.642693042755127,
             -0.8138532638549805, -0.2847113013267517, -1.6397924423217773, -0.2886905074119568, -0.0055061206221580505, -0.45961230993270874,
             0.2886905074119568, 0.2825317084789276, 0.684377908706665, 0.8138531446456909, 0.5273022651672363, 1.5852537155151367,
             1.1918067932128906, 0.6845313906669617, 2.0800650119781494, 1.354187250137329, 0.7257797718048096, 2.0793111324310303],
            output);
        AssertClose([0, 0.008282912895083427, 0.01639707200229168], norm.RunningMean.Value);
        AssertClose([0.962007999420166, 0.9610267281532288, 0.9581623673439026], norm.RunningVariance.Value);
    }

    [Fact]
    public void ALayerNormalisation_IsPyTorchs_InEitherPass_AndSoAreItsGradients()
    {
        var norm = Started(new LayerNorm(3));
        var batch = WalkedRows.Batch();
        var pass = new RecordingBackend(_backend);

        var evaluated = norm.Forward(batch, Pass.Evaluation(_backend));
        var output = norm.Forward(batch, Pass.Training(pass, new RandomStream(1), 0, 0));
        var loss = pass.Scale(pass.Mean(pass.Multiply(output, Weighing())), pass.Fill(new Shape(), 12f));
        var gradients = pass.GradientsOf(loss, [batch, norm.Weight.Value, norm.Bias.Value]);

        var expected = new[]
        {
            0.26726034283638, 0.6345207095146179, -2.87260365486145, 0.0, -0.5123717188835144, 2.2494869232177734,
            1.2247356176376343, -0.5123677849769592, -0.20000000298023224, -1.0051405429840088, 0.7820596098899841, -0.9179575443267822,
        };
        AssertClose(expected, evaluated);
        AssertClose(expected, output);
        AssertClose(
            [0.017180323600769043, -0.011456608772277832, -0.005723834037780762, 0.09797948598861694, -0.048989299684762955, -0.04899020120501518,
             -0.18371686339378357, -0.18370383977890015, 0.3674207031726837, -0.24780619144439697, -0.09292763471603394, 0.3407338261604309],
            gradients[batch]);
        AssertClose([-0.8556914329528809, -1.0850647687911987, -1.2868590354919434], gradients[norm.Weight.Value]);
        AssertClose([0, 0.30000001192092896, 1.0], gradients[norm.Bias.Value]);
    }

    private static TNorm Started<TNorm>(TNorm norm)
        where TNorm : Layer
    {
        var slots = norm.Slots().ToDictionary(slot => slot.Path, slot => slot.Slot);
        slots["weight"].Replace(Tensor.From(new Shape(3), Scale));
        slots["bias"].Replace(Tensor.From(new Shape(3), Shift));

        return norm;
    }

    private Pass Training() => Pass.Training(_backend, new RandomStream(1), 0, 0);

    private static void AssertClose(double[] expected, Tensor actual)
    {
        Assert.Equal(expected.Length, actual.Values.Length);

        for (var at = 0; at < expected.Length; at++)
        {
            Assert.Equal(expected[at], actual.Values[at], Math.Max(1e-5, Math.Abs(expected[at]) * 1e-5));
        }
    }
}
