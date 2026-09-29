// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Sample.Networks;

namespace DeepSharp.Tests.Samples;

/// <summary>
/// The networks sample is run as it stands, so the page a person reads cannot drift from the code: it trains the three
/// walked networks — a passenger's survival, a price five days on, a day's bikes hour by hour — through both doors, has each
/// pipeline's report measure them, saves each as its one file, reads it back, and predicts a row served afresh.
/// </summary>
public class NetworkSampleTests
{
    [Fact]
    public void TheSample_TrainsMeasuresSavesReadsBackAndServesEachOfTheThreeWalkedNetworks()
    {
        var charts = Directory.CreateTempSubdirectory("deepsharp-sample-").FullName;
        using var output = new StringWriter();

        NetworkSample.Run(Path.GetDirectoryName(Repository.Data("titanic.csv"))!, output, charts);

        var said = output.ToString();

        foreach (var name in new[] { "Titanic", "Apple", "Bike Sharing" })
        {
            Assert.Contains($"== {name}", said, StringComparison.Ordinal);
        }

        Assert.Equal(3, said.Split("read back from its file, it predicts the same", StringSplitOptions.None).Length - 1);
        Assert.Contains("chance of surviving", said, StringComparison.Ordinal);
        Assert.Contains("close five days on", said, StringComparison.Ordinal);
        Assert.Contains("bikes that day", said, StringComparison.Ordinal);
        Assert.Equal(
            ["apple-loss.svg", "apple-measures.svg", "bikes-loss.svg", "bikes-measures.svg", "titanic-confusion.svg", "titanic-loss.svg", "titanic-measures.svg"],
            Directory.GetFiles(charts).Select(Path.GetFileName).Order(StringComparer.Ordinal));
        Assert.All(Directory.GetFiles(charts), chart => Assert.StartsWith("<svg", File.ReadAllText(chart), StringComparison.Ordinal));
    }
}
