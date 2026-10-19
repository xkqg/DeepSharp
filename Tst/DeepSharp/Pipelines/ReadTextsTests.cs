// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text;
using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// A folder of files read as rows, a folder to a class: the files in the order of their paths, each row what its file holds and the
/// name of the folder it stands in, and a column of nought and one for each class, named for it. A row is known by its content, not by
/// where the file lies.
/// </summary>
public class ReadTextsTests
{
    private static ScratchFolder Reviews()
    {
        var folder = new ScratchFolder("deepsharp-texts-");

        Directory.CreateDirectory(folder.At("neg"));
        Directory.CreateDirectory(folder.At("pos"));
        File.WriteAllText(folder.At(Path.Join("pos", "b.txt")), "a fine film, truly fine");
        File.WriteAllText(folder.At(Path.Join("pos", "a.txt")), "loved it");
        File.WriteAllText(folder.At(Path.Join("neg", "c.txt")), "dull and slow");
        File.WriteAllText(folder.At(Path.Join("neg", "notes.md")), "not a text file of the reader's");

        return folder;
    }

    [Fact]
    public void EachTextFile_IsARow_InTheOrderOfItsPath_WithTheNameOfItsFolderAndAColumnForEachClass()
    {
        using var folder = Reviews();
        var source = new ReadTextsStep(folder.Path).Open(SourceFolder.WorkingDirectory);

        Assert.Equal(["text", "label", "neg", "pos"], source.ColumnNames);
        Assert.Equal(
            [["dull and slow", "neg", "1", "0"], ["loved it", "pos", "0", "1"], ["a fine film, truly fine", "pos", "0", "1"]],
            source.Rows.Select(row => row.ToArray()));
    }

    [Fact]
    public void ARelativePath_IsReadFromThePipelinesFolder_AndTheStepIsWrittenAndReadBackAsItIs()
    {
        using var parent = new ScratchFolder("deepsharp-texts-");

        Directory.CreateDirectory(parent.At(Path.Join("reviews", "pos")));
        File.WriteAllText(parent.At(Path.Join("reviews", "pos", "a.txt")), "loved it");

        var step = new ReadTextsStep("reviews");
        var from = SourceFolder.Of(parent.Path);

        Assert.Equal("read.texts", step.Verb);
        Assert.Equal(["text", "label", "pos"], step.Open(from).ColumnNames);
        Assert.Single(step.Open(from).Rows);
        Assert.Equal(step, ReadTextsStep.ReadFrom(JsonDocument.Parse(Encoding.UTF8.GetString(step.Canonical())).RootElement));
    }

    [Fact]
    public void ARowIsKnownByWhatItsFileHolds_NotByWhereItLies()
    {
        using var folder = Reviews();
        using var other = new ScratchFolder("deepsharp-texts-");

        Directory.CreateDirectory(other.At("pos"));
        Directory.CreateDirectory(other.At("neg"));
        File.WriteAllText(other.At(Path.Join("pos", "renamed.txt")), "loved it");
        File.WriteAllText(other.At(Path.Join("neg", "x.txt")), "dull and slow");

        var here = Pdd.Create().ReadTexts(folder.Path).Declare(schema => schema.Text("text").Category("label").Integer("neg", "pos")).Build().Run();
        var there = Pdd.Create().ReadTexts(other.Path).Declare(schema => schema.Text("text").Category("label").Integer("neg", "pos")).Build().Run();
        var loved = here.Table.Identities.Select(identity => identity.Key).Intersect(there.Table.Identities.Select(identity => identity.Key)).ToArray();

        Assert.Equal(2, loved.Length);
    }

    [Fact]
    public void AFileBesideTheFolders_ANonsenseFolder_AndAnEmptyOne_AreRefused()
    {
        using var beside = Reviews();
        File.WriteAllText(beside.At("loose.txt"), "belongs to no class");

        using var empty = new ScratchFolder("deepsharp-texts-");
        using var clash = new ScratchFolder("deepsharp-texts-");
        Directory.CreateDirectory(clash.At("label"));
        File.WriteAllText(clash.At(Path.Join("label", "a.txt")), "x");

        Assert.Contains("beside the folders", Assert.Throws<FormatException>(() => new ReadTextsStep(beside.Path).Open(SourceFolder.WorkingDirectory)).Message, StringComparison.Ordinal);
        Assert.Contains("no file", Assert.Throws<FormatException>(() => new ReadTextsStep(empty.Path).Open(SourceFolder.WorkingDirectory)).Message, StringComparison.Ordinal);
        Assert.Contains("rename the folder", Assert.Throws<FormatException>(() => new ReadTextsStep(clash.Path).Open(SourceFolder.WorkingDirectory)).Message, StringComparison.Ordinal);
        Assert.Throws<DirectoryNotFoundException>(() => new ReadTextsStep(Path.Join(empty.Path, "nowhere")).Open(SourceFolder.WorkingDirectory));
    }

    [Fact]
    public void AFileTheReaderCannotMakeACellOf_IsRefused_NamingTheFile()
    {
        using var folder = Reviews();

        var refused = Assert.Throws<FormatException>(() => new FilesAsRows(folder.Path, "text", [".txt"], _ => throw new FormatException("it is no cell")));

        Assert.Contains("neg", refused.Message, StringComparison.Ordinal);
        Assert.Contains("c.txt", refused.Message, StringComparison.Ordinal);
        Assert.Contains("it is no cell", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheClassColumns_AreTheAnswerAsAnyLabelsAre()
    {
        using var folder = Reviews();
        var prepared = Pdd.Create()
            .ReadTexts(folder.Path)
            .Declare(schema => schema.Text("text").Category("label").Integer("neg", "pos"))
            .SplitAtRandom(0.70, 0.15, 1)
            .Labels(["neg", "pos"], ones: 1)
            .Build()
            .Run();

        Assert.Equal("read.texts", prepared.Declaration.Steps[0].Verb);
        Assert.Equal(["neg", "pos"], prepared.Table.Columns.Where(column => column.Kind == ColumnKind.Integer).Select(column => column.Name));
    }
}
