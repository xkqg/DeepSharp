// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Notebooks;
using Verso.Serializers;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// The notebook that ships with the samples is run as a person runs it, beside the passenger list in its folder: its blocks
/// show the passengers where each lands, its C# cell trains a network on the pipeline the blocks hand over and hands its
/// predictions back, and its report block then draws what the network was measured by — so the file a first command opens,
/// <c>deepsharp-serve Samples/titanic.verso</c>, cannot drift from the notebook.
/// </summary>
public sealed class SampleNotebookTests : IDisposable
{
    private static readonly string Sample = Path.Join(Repository.Root, "Samples", "titanic.verso");

    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-sample-").FullName;

    // The sample's folder as it stands under Samples: the notebook, and the passenger list in data beside it.
    public SampleNotebookTests()
    {
        Directory.CreateDirectory(Path.Join(_folder, "data"));
        File.Copy(Sample, Path.Join(_folder, "titanic.verso"));
        File.Copy(Repository.Data("titanic.csv"), Path.Join(_folder, "data", "titanic.csv"));
    }

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    [Fact]
    public async Task TheSample_IsTheFileVersoWritesForIt_ABlockReadingThePassengerListFirst_AndAReportBlockLast()
    {
        var text = await File.ReadAllTextAsync(Sample, TestContext.Current.CancellationToken);
        var serializer = new VersoSerializer();
        var notebook = await serializer.DeserializeAsync(text);
        var blocks = notebook.Cells.Where(cell => cell.Type == StepCellType.StepType).ToArray();

        Assert.Equal(text.ReplaceLineEndings("\n"), (await serializer.SerializeAsync(notebook)).ReplaceLineEndings("\n"));
        Assert.Contains("\"read.csv\"", blocks[0].Source, StringComparison.Ordinal);
        Assert.Contains("\"data/titanic.csv\"", blocks[0].Source, StringComparison.Ordinal);
        Assert.Contains("\"evidence.report\"", blocks[^1].Source, StringComparison.Ordinal);

        // The cell names its packages as a person's cell does, by their NuGet ids.
        Assert.StartsWith("#r \"nuget: DeepSharp.Learners.Networks\"", Assert.Single(notebook.Cells, cell => cell.Type == "code").Source, StringComparison.Ordinal);

        // It names no extension it needs: Verso's browser editor, asked to open a notebook that names one, waits on the
        // page's consent while it opens it whenever the package was not installed from its panel at the newest version,
        // and never finishes opening it. A person installing the package from the panel with the notebook open has Verso
        // record it there.
        Assert.Empty(notebook.RequiredExtensions);
    }

    [Fact]
    public async Task TheSample_ShowsThePassengers_TrainsANetworkInItsCell_AndItsReportBlockThenDrawsTheMeasures()
    {
        await using var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));
        var report = notebook.Scaffold.Cells.Last(cell => cell.Type == StepCellType.StepType);
        var trains = notebook.Scaffold.Cells.Single(cell => cell.Type == "code");

        // The data at the report block — every passenger, each marked with the part it lands in — and, with nothing trained
        // yet, where its measures come from.
        await notebook.GestureAsync(report, StepRenderer.Show);

        Assert.Equal(3, report.Outputs.Count);
        Assert.Contains("891 rows", report.Outputs[1].Content, StringComparison.Ordinal);
        Assert.Contains("trains none", report.Outputs[2].Content, StringComparison.Ordinal);

        // The cell trains on the pipeline the blocks handed over, hands the predictions back, and shows its loss curve.
        trains.Source = trains.Source.WithPackagesAsBuilt();

        Assert.Contains(await notebook.RunAsync(trains), output => output.Content.Contains("<svg", StringComparison.Ordinal));

        // Shown again, the report block measures them on the notebook's own run of its blocks and draws them under the grid.
        await notebook.GestureAsync(report, StepRenderer.Show);

        var drawn = report.Outputs[2].Content;

        Assert.False(report.Outputs[2].IsError, drawn);
        Assert.Contains("<th>accuracy</th><th>baseline</th>", drawn, StringComparison.Ordinal);
        Assert.Contains("<td>validation</td><td class=\"deepsharp-number\">133</td>", drawn, StringComparison.Ordinal);
        Assert.Contains("<td>test</td><td class=\"deepsharp-number\">135</td>", drawn, StringComparison.Ordinal);
        Assert.Contains("<svg", drawn, StringComparison.Ordinal);
    }
}
