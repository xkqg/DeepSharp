// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Charts;
using DeepSharp.Networks;
using DeepSharp.Tensors;
using MatPlotLibNet;
using MatPlotLibNet.Models.Series;

namespace DeepSharp.Tests.Charts;

/// <summary>
/// A run's charts are drawn from what the loop kept, epoch by epoch — nobody assembles an array to draw a loss curve: the
/// training loss and the validation loss as two lines, and the rate every epoch trained at, which is the schedule.
/// </summary>
public class HistoryChartsTests
{
    [Fact]
    public void TheLossCurve_HasALineForTheTrainingLoss_AndOneForTheValidationLoss_EpochByEpoch()
    {
        var history = Trained(new Adam(0.01), new ConstantRate(), watched: true);
        var axes = Assert.Single(HistoryCharts.LossFigure(history).SubPlots);
        var lines = axes.Series.OfType<LineSeries>().ToArray();

        Assert.Equal(["training", "validation"], lines.Select(line => line.Label));
        Assert.All(lines, line => Assert.Equal([1.0, 2, 3, 4, 5, 6], line.XData));
        Assert.Equal(history.Epochs.Select(epoch => epoch.Loss), lines[0].YData);
        Assert.Equal(history.Epochs.Select(epoch => epoch.ValidationLoss!.Value), lines[1].YData);
        Assert.Equal("epoch", axes.XAxis.Label);
        Assert.Equal("loss", axes.YAxis.Label);
        Assert.Equal(HistoryCharts.LossFigure(history).ToSvg(), history.LossCurve());
        Assert.StartsWith("<svg", history.LossCurve(), StringComparison.Ordinal);
    }

    [Fact]
    public void ARunWithoutValidationRows_HasTheTrainingLossAlone()
    {
        var history = Trained(new Sgd(0.05), new ConstantRate(), watched: false);

        var line = Assert.Single(Assert.Single(HistoryCharts.LossFigure(history).SubPlots).Series.OfType<LineSeries>());

        Assert.Equal("training", line.Label);
        Assert.Equal(history.Epochs.Select(epoch => epoch.Loss), line.YData);
    }

    [Fact]
    public void TheLearningRateChart_IsTheSchedule_EpochByEpoch()
    {
        var schedule = new CosineDecay(6, 0.001);
        var history = Trained(new Adam(0.01), schedule, watched: false);

        var axes = Assert.Single(HistoryCharts.RatesFigure(history).SubPlots);
        var line = Assert.Single(axes.Series.OfType<LineSeries>());

        Assert.Equal(Enumerable.Range(0, 6).Select(epoch => schedule.RateAt(epoch, 0.01)), line.YData);
        Assert.Equal("learning rate", axes.YAxis.Label);
        Assert.Equal(HistoryCharts.RatesFigure(history).ToSvg(), history.LearningRates());
    }

    [Fact]
    public void AHistoryOfNoEpochs_HasNothingToDraw()
    {
        var empty = new History(1, [], null, Stopping.AllEpochsRan);

        Assert.Throws<ArgumentException>(() => empty.LossCurve());
        Assert.Throws<ArgumentException>(() => empty.LearningRates());
        Assert.Throws<ArgumentNullException>(() => ((History)null!).LossCurve());
        Assert.Throws<ArgumentNullException>(() => ((History)null!).LearningRates());
    }

    // Six epochs of a small network on rows whose answer is twice their first value.
    private static History Trained(Optimizer optimizer, LearningRateSchedule schedule, bool watched)
    {
        var network = new LayerStack(new Dense(2, 4, new RandomStream(5).Draw("initialise:0", 0, 0)), new Tanh(), new Dense(4, 1, new RandomStream(5).Draw("initialise:2", 0, 0)));

        return network.Compile(optimizer, new MeanSquaredError(), schedule)
            .Fit(Rows(32, 0), watched ? Rows(8, 32) : null, new FitOptions(seed: 3) { Epochs = 6, BatchSize = 8 });
    }

    private static TrainingData Rows(int count, int from)
    {
        var features = Enumerable.Range(2 * from, 2 * count).Select(at => MathF.Sin(at) / 2).ToArray();
        var answers = Enumerable.Range(0, count).Select(row => 2 * features[2 * row]).ToArray();

        return new TrainingData(Tensor.From(new Shape(count, 2), features), Tensor.From(new Shape(count, 1), answers));
    }
}
