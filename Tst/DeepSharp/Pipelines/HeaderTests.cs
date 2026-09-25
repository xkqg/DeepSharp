// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A comma-separated file's header on its own: read with the same rules the whole file is read by — a quoted comma or
/// line break belongs to its cell — and without reading any further, so a large file costs its first line.
/// </summary>
public sealed class HeaderTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-header-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private string Written(string text)
    {
        var path = Path.Join(_folder, $"{Guid.NewGuid():N}.csv");

        File.WriteAllText(path, text);

        return path;
    }

    [Fact]
    public void TheHeader_IsReadAsTheWholeFileReadsIt_QuotesAndAll()
    {
        var path = Written("id,\"a, quoted\",\"a line\nbreak\",\"say \"\"hi\"\"\"\r\n1,2,3,4\n");

        Assert.Equal(["id", "a, quoted", "a line\nbreak", "say \"hi\""], CsvRowSource.HeaderOf(path));
        Assert.Equal(new CsvRowSource(path).ColumnNames, CsvRowSource.HeaderOf(path));
    }

    [Fact]
    public void NothingBelowTheHeader_IsRead()
    {
        // A row with the wrong number of cells refuses the whole file, and not its header.
        var path = Written("a,b\n1,2\n3\n");

        Assert.Throws<FormatException>(() => new CsvRowSource(path));
        Assert.Equal(["a", "b"], CsvRowSource.HeaderOf(path));
    }

    [Fact]
    public void AHeaderWithNoLineBreakAfterIt_IsTheWholeFile()
    {
        Assert.Equal(["a", "b"], CsvRowSource.HeaderOf(Written("a,b")));
        Assert.Equal(["a", "b"], CsvRowSource.FromText("a,b").ColumnNames);
    }

    [Fact]
    public void AFileWithNoHeader_IsRefused_InTheWordsTheWholeFileIsRefusedIn()
    {
        var path = Written(string.Empty);

        var header = Assert.Throws<FormatException>(() => CsvRowSource.HeaderOf(path));
        var whole = Assert.Throws<FormatException>(() => new CsvRowSource(path));

        Assert.Equal(whole.Message, header.Message);
    }

    [Fact]
    public void AFileThatIsNotThere_IsRefusedAsAnyReadIs()
    {
        Assert.Throws<FileNotFoundException>(() => CsvRowSource.HeaderOf(Path.Join(_folder, "nothing.csv")));
    }
}
