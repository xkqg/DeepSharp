// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Networks;
using MatPlotLibNet;
using MatPlotLibNet.Models;

namespace DeepSharp.Charts;

/// <summary>
/// The charts of a run, drawn from its history — what the training loop already kept, epoch by epoch, so nobody assembles
/// an array to draw them.
/// </summary>
public static class HistoryCharts
{
    extension(History history)
    {
        /// <summary>The loss curve: the training loss of every epoch, and beside it the validation loss when the run had validation rows.</summary>
        /// <returns>The chart, as the text of an SVG.</returns>
        /// <exception cref="ArgumentException">The history holds no epoch.</exception>
        /// <remarks>A training loss that goes on falling while the validation loss turns back up is a model learning its training rows by heart.</remarks>
        public string LossCurve() => LossFigure(history).ToSvg();

        /// <summary>The learning rate every epoch trained at: the schedule, as the run took it.</summary>
        /// <returns>The chart, as the text of an SVG.</returns>
        /// <exception cref="ArgumentException">The history holds no epoch.</exception>
        public string LearningRates() => RatesFigure(history).ToSvg();
    }

    /// <summary>The figure <see cref="LossCurve"/> draws.</summary>
    internal static Figure LossFigure(History history)
    {
        var epochs = Epochs(history);
        var numbers = Numbers(epochs);

        return new FigureBuilder()
            .WithSize(640, 400)
            .AddSubPlot(1, 1, 1, axes =>
            {
                axes.WithTitle("loss").SetXLabel("epoch").SetYLabel("loss")
                    .Plot(numbers, [.. epochs.Select(epoch => epoch.Loss)], line => line.Label = "training");

                if (epochs.All(epoch => epoch.ValidationLoss is not null))
                {
                    axes.Plot(numbers, [.. epochs.Select(epoch => epoch.ValidationLoss!.Value)], line => line.Label = "validation");
                }

                axes.WithLegend();
            })
            .Build();
    }

    /// <summary>The figure <see cref="LearningRates"/> draws.</summary>
    internal static Figure RatesFigure(History history)
    {
        var epochs = Epochs(history);

        return new FigureBuilder()
            .WithSize(640, 400)
            .AddSubPlot(1, 1, 1, axes => axes
                .WithTitle("learning rate").SetXLabel("epoch").SetYLabel("learning rate")
                .Plot(Numbers(epochs), [.. epochs.Select(epoch => epoch.LearningRate)], line => line.Label = "learning rate"))
            .Build();
    }

    // The epochs of a history that holds some.
    private static IReadOnlyList<Epoch> Epochs(History history)
    {
        ArgumentNullException.ThrowIfNull(history);

        return history.Epochs.Count > 0
            ? history.Epochs
            : throw new ArgumentException("A history of no epochs has nothing to draw.", nameof(history));
    }

    // Each epoch as a person counts it, from one.
    private static double[] Numbers(IReadOnlyList<Epoch> epochs) => [.. epochs.Select(epoch => (double)(epoch.Number + 1))];
}
