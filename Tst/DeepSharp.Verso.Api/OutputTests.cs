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
