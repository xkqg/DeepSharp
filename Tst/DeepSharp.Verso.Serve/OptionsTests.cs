// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Verso.Serve;

namespace DeepSharp.Tests.Serve;

/// <summary>
/// What <c>deepsharp-serve</c> is told on its command line: the notebook to serve, or the folder of notebooks — the
/// folder it runs in when it is told neither — the port, and whether to open a browser. Anything else is refused with
/// what was wrong and how to use it.
/// </summary>
public sealed class OptionsTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-serve-options-").FullName;

    public OptionsTests() => File.WriteAllText(Path.Join(_folder, "titanic.verso"), "{}");

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    [Fact]
    public void TheFolderItRunsIn_IsServed_WhenNothingElseIsNamed_OnAPortTheSystemPicks_InABrowserItOpens()
    {
        var options = ServeOptions.Parse([], _folder);

        Assert.Equal(_folder, options.Path);
        Assert.Equal(0, options.Port);
        Assert.True(options.OpenBrowser);
        Assert.False(options.Help);
    }

    [Fact]
    public void ANotebookOrAFolder_IsServedAsItsFullPath_OnThePortAsked_WithOrWithoutABrowser()
    {
        var notebook = ServeOptions.Parse(["titanic.verso", "--port", "5171", "--no-browser"], _folder);
        var folder = ServeOptions.Parse(["--no-browser", "."], _folder);

        Assert.Equal(Path.Join(_folder, "titanic.verso"), notebook.Path);
        Assert.Equal(5171, notebook.Port);
        Assert.False(notebook.OpenBrowser);
        Assert.Equal(_folder, folder.Path);
        Assert.True(ServeOptions.Parse(["--help"], _folder).Help);
    }

    [Theory]
    [InlineData("--nope", "--nope")]
    [InlineData("--port", "--port")]
    [InlineData("--port x", "x")]
    [InlineData("--port 70000", "70000")]
    [InlineData("--port -1", "-1")]
    [InlineData("missing.verso", "missing.verso")]
    [InlineData("titanic.verso .", ".")]
    public void AnythingElse_IsRefused_SayingWhatWasWrong(string arguments, string named)
    {
        var refused = Assert.Throws<ArgumentException>(() => ServeOptions.Parse(arguments.Split(' '), _folder));

        Assert.Contains(named, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheUsage_SaysEveryOption()
    {
        Assert.Contains("deepsharp-serve", ServeOptions.Usage, StringComparison.Ordinal);
        Assert.Contains("--port", ServeOptions.Usage, StringComparison.Ordinal);
        Assert.Contains("--no-browser", ServeOptions.Usage, StringComparison.Ordinal);
        Assert.Contains("--help", ServeOptions.Usage, StringComparison.Ordinal);
    }
}
