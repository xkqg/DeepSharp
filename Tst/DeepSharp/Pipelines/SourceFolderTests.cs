// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// One rule for where a relative path points, whoever opens it. A pipeline file and a notebook read their
/// data from the folder they sit in, and a pipeline written in code from the working directory; the path is
/// kept as it was written either way, so the same file means the same thing wherever it is moved with its data.
/// </summary>
public sealed class SourceFolderTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    [Fact]
    public void ARelativePathResolvesAgainstTheFolderItIsGiven()
    {
        Assert.Equal(Path.Join(_folder, "data", "rows.csv"), SourceFolder.Of(_folder).Resolve(Path.Join("data", "rows.csv")));
    }

    [Fact]
    public void AnAbsolutePathIsUsedAsWritten()
    {
        var elsewhere = Path.Join(Path.GetTempPath(), "elsewhere.csv");

        Assert.Equal(elsewhere, SourceFolder.Of(_folder).Resolve(elsewhere));
    }

    [Fact]
    public void APipelineWrittenInCode_ReadsFromTheWorkingDirectory()
    {
        Assert.Equal(Path.GetFullPath("rows.csv"), SourceFolder.WorkingDirectory.Resolve("rows.csv"));
    }

    [Fact]
    public void ADocumentsFolder_IsTheOneItSitsIn()
    {
        Assert.Equal(
            Path.Join(_folder, "rows.csv"),
            SourceFolder.OfDocument(Path.Join(_folder, "pipeline.json")).Resolve("rows.csv"));
        Assert.Throws<ArgumentException>(() => SourceFolder.Of(" "));
        Assert.Throws<ArgumentException>(() => SourceFolder.OfDocument(" "));
        Assert.Throws<ArgumentException>(() => SourceFolder.Of(_folder).Resolve(" "));
    }

    [Fact]
    public void APipelineOpensItsSourceFromItsFolder_AndKeepsThePathAsWritten()
    {
        File.WriteAllText(Path.Join(_folder, "rows.csv"), "a\n1\n2\n3\n");

        var declaration = Pdd.Create().ReadCsv("rows.csv").Declare(schema => schema.Number("a")).Declaration;
        var pipeline = new Pipeline(declaration, rows: null, SourceFolder.Of(_folder));

        Assert.Equal(3, pipeline.Run().Table.RowCount);
        Assert.Equal(3, pipeline.ViewAt(1).Table.RowCount);
        Assert.Contains("\"rows.csv\"", declaration.ToJson(), StringComparison.Ordinal);
        Assert.Throws<FileNotFoundException>(() => new Pipeline(declaration).Run());
    }
}
