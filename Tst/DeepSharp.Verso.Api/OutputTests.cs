// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Api;
using Verso.Abstractions;
using Verso.Serializers;

namespace DeepSharp.Tests.Api;

/// <summary>
/// What a cell shows is carried as Verso's engine holds it: a failure with its name and where it happened, and text with
/// the stream it came on — standard error is told apart from ordinary output, and is no failure by itself.
/// </summary>
public sealed class OutputTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-api-outputs-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    [Fact]
    public async Task AFailureShown_CarriesItsNameAndWhereItHappened_AndTextOnStandardErrorSaysWhereItCameFrom()
    {
        var notebook = new NotebookModel { DefaultKernelId = "csharp" };

        notebook.Cells.Add(new CellModel { Type = "code", Language = "csharp", Source = """System.Console.WriteLine("fine"); System.Console.Error.WriteLine("careful");""" });
        notebook.Cells.Add(new CellModel { Type = "code", Language = "csharp", Source = """throw new System.InvalidOperationException("It went wrong.");""" });

        var path = Path.Join(_folder, "outputs.verso");

        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        await using var notebooks = new OpenNotebooks();
        var host = await notebooks.OpenAsync(path, TestContext.Current.CancellationToken);

        var printed = await host.RunAsync(host.Cells[0].Id);

        // What was printed carries no stream of its own, as the kernel writes it; what went to standard error says so.
        Assert.Null(Assert.Single(printed.Outputs, output => output.Content.Contains("fine", StringComparison.Ordinal)).Stream);

        var careful = Assert.Single(printed.Outputs, output => output.Content.Contains("careful", StringComparison.Ordinal));

        Assert.Equal(OutputStream.StandardError, careful.Stream);
        Assert.False(careful.IsError);

        var failed = Assert.Single((await host.RunAsync(host.Cells[1].Id)).Outputs);

        // As the C# kernel writes a failure: the exception's full name before its message.
        Assert.True(failed.IsError);
        Assert.Equal("System.InvalidOperationException: It went wrong.", failed.Content);
        Assert.Equal(nameof(InvalidOperationException), failed.ErrorName);
        Assert.False(string.IsNullOrWhiteSpace(failed.ErrorStack));
    }

    // What a cell names to use DeepSharp's packages: the assemblies beside this suite, which are the ones it was built with.
    private static string References(params string[] assemblies) =>
        string.Concat(assemblies.Select(assembly => $"#r \"{Path.Join(AppContext.BaseDirectory, assembly + ".dll")}\"\n"));

    // A notebook of one C# cell, opened as an application of your own opens one, and the cell run.
    private async Task<HostedOutput> ShownAsync(string source)
    {
        var notebook = new NotebookModel { DefaultKernelId = "csharp" };

        notebook.Cells.Add(new CellModel { Type = "code", Language = "csharp", Source = source });

        var path = Path.Join(_folder, "shown.verso");

        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        await using var notebooks = new OpenNotebooks();
        var host = await notebooks.OpenAsync(path, TestContext.Current.CancellationToken);

        return Assert.Single((await host.RunAsync(host.Cells[0].Id)).Outputs);
    }

    [Fact]
    public async Task ACellEndingWithTheReportOfAModelsMeasures_ShowsItAsHtml()
    {
        // The report as its pipeline declared it shown — the numbers and the charts — is one value with a public ToHtml(),
        // which Verso's engine shows as HTML whichever assembly the cell loaded its type from.
        var shown = await ShownAsync(References("DeepSharp.Pipelines", "DeepSharp.Charts") + """
            using DeepSharp.Pipelines;
            using DeepSharp.Charts;

            var prepared = Pdd.Create()
                .Read(CsvRowSource.FromText("t,y\n1,0\n2,1\n3,1\n4,0\n5,1\n6,0\n7,1\n8,0\n9,1\n10,1\n11,0\n12,1\n"), "twelve rows")
                .Declare(schema => schema.Integer("t", "y"))
                .SplitByTime("t", 0.50, 0.25)
                .Target("y")
                .Report(report => report.Measure(Metric.Accuracy).On(Part.Validation, Part.Test).As(Shown.Numbers, Shown.Drawn))
                .Build()
                .Run();

            prepared.Measure(new[] { Part.Validation, Part.Test }.Select(part =>
            {
                var batch = prepared.Batch(part);

                return new PartPredictions(batch, batch.Features.Select(row => new[] { row[0] % 3 == 0 ? 0.8 : 0.3 }).ToArray());
            }).ToArray()).Report()
            """);

        Assert.False(shown.IsError, shown.Content);
        Assert.Equal("text/html", shown.ContentType);
        Assert.Contains("<th>accuracy</th><th>baseline</th>", shown.Content, StringComparison.Ordinal);
        Assert.Contains("<svg", shown.Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task ACellEndingWithALossCurve_ShowsThePicture()
    {
        // A chart is handed out as the text of an SVG, and Verso's engine shows such text as the picture it is.
        var shown = await ShownAsync(References("DeepSharp", "DeepSharp.Charts") + """
            using DeepSharp.Networks;
            using DeepSharp.Tensors;
            using DeepSharp.Charts;

            var features = Tensor.From(new Shape(4, 2), new[] { 0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f });
            var answers = Tensor.From(new Shape(4, 1), new[] { 0f, 1f, 1f, 0f });
            var history = new Sequential().Dense(4).Relu().Dense(1)
                .Compile(new Adam(0.01), new MeanSquaredError())
                .Fit(new TrainingData(features, answers), null, new FitOptions(seed: 7) { Epochs = 3 });

            history.LossCurve()
            """);

        Assert.False(shown.IsError, shown.Content);
        Assert.Equal("text/html", shown.ContentType);
        Assert.Contains("verso-svg-output", shown.Content, StringComparison.Ordinal);
        Assert.Contains("<svg", shown.Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task TextAKernelWroteToStandardOutput_SaysSo_AsTheFileKeepsIt()
    {
        var notebook = new NotebookModel { DefaultKernelId = "csharp" };
        var cell = new CellModel { Type = "code", Language = "python", Source = "print('fine')" };

        // As Verso's Python kernel writes what a cell printed, and as a part may: on standard output, which the file keeps.
        cell.Outputs.Add(CellOutput.Stdout("fine"));
        notebook.Cells.Add(cell);

        var path = Path.Join(_folder, "printed.verso");

        await File.WriteAllTextAsync(path, await new VersoSerializer().SerializeAsync(notebook), TestContext.Current.CancellationToken);

        await using var notebooks = new OpenNotebooks();
        var host = await notebooks.OpenAsync(path, TestContext.Current.CancellationToken);
        var printed = Assert.Single(host.Cells[0].Outputs);

        Assert.Equal("fine", printed.Content);
        Assert.Equal(OutputStream.StandardOutput, printed.Stream);
    }
}
