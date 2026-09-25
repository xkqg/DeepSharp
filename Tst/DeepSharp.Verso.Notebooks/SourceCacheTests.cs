// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using DeepSharp.Verso.Notebooks;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Notebooks;

/// <summary>
/// Every gesture runs its prefix from the source up, and parsing the file was measured at 54 to 75 per cent of
/// every run on a file of 5 MB or more. So a notebook's session keeps the rows its source opened last — only the
/// rows as read, never a table or anything a step made of them, since those are replayed every time. The key is
/// the read step, the path the core's one rule resolves, and a SHA-256 of the file's bytes, worked out on every
/// use: a file changed on disk is opened again, and the rows kept are always the ones the fingerprint names.
/// </summary>
public sealed class SourceCacheTests : IDisposable
{
    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-cache-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private static PipelineDeclaration Reading(string path) => new([new ReadCsvStep(path)]);

    private string Written(string name, string text)
    {
        var path = Path.Join(_folder, name);
        File.WriteAllText(path, text);

        return path;
    }

    [Fact]
    public void TheSameFileTwice_IsParsedOnce_AndHandsOutTheSameRows()
    {
        var path = Written("rows.csv", "a,b\n1,2\n3,4\n");
        var cache = new SourceCache();

        var first = cache.RowsFor(Reading(path), SourceFolder.WorkingDirectory);
        var second = cache.RowsFor(Reading(path), SourceFolder.WorkingDirectory);

        Assert.Equal(1, cache.Parsed);
        Assert.Same(first.Rows, second.Rows);
        Assert.Equal(["a", "b"], first.Rows!.ColumnNames);
        Assert.Equal(2, first.Rows.Rows.Count());
        Assert.Equal(64, first.Fingerprint!.Length);
        Assert.Equal(first.Fingerprint, second.Fingerprint);
    }

    [Fact]
    public void AFileChangedOnDisk_IsOpenedAgain()
    {
        var path = Written("rows.csv", "a\n1\n");
        var cache = new SourceCache();

        cache.RowsFor(Reading(path), SourceFolder.WorkingDirectory);
        File.WriteAllText(path, "a\n1\n2\n");

        var again = cache.RowsFor(Reading(path), SourceFolder.WorkingDirectory);

        Assert.Equal(2, cache.Parsed);
        Assert.Equal(2, again.Rows!.Rows.Count());
    }

    [Fact]
    public void AReadStepPointingElsewhere_IsAnotherEntry_EvenOverTheSameBytes()
    {
        var one = Written("one.csv", "a\n1\n");
        var other = Written("other.csv", "a\n1\n");
        var cache = new SourceCache();

        cache.RowsFor(Reading(one), SourceFolder.WorkingDirectory);
        cache.RowsFor(Reading(other), SourceFolder.WorkingDirectory);

        Assert.Equal(2, cache.Parsed);
    }

    [Fact]
    public void ASessionKeepsOneSource_TheLastOneOpened()
    {
        var one = Written("one.csv", "a\n1\n");
        var other = Written("other.csv", "a\n2\n");
        var cache = new SourceCache();

        cache.RowsFor(Reading(one), SourceFolder.WorkingDirectory);
        cache.RowsFor(Reading(other), SourceFolder.WorkingDirectory);
        cache.RowsFor(Reading(one), SourceFolder.WorkingDirectory);

        Assert.Equal(3, cache.Parsed);
    }

    [Fact]
    public void TwoSessions_ShareNothing()
    {
        var path = Written("rows.csv", "a\n1\n");
        var one = new SourceCache();
        var other = new SourceCache();

        Assert.NotSame(one.RowsFor(Reading(path), SourceFolder.WorkingDirectory).Rows, other.RowsFor(Reading(path), SourceFolder.WorkingDirectory).Rows);
        Assert.Equal(1, one.Parsed);
        Assert.Equal(1, other.Parsed);
    }

    [Fact]
    public void ARelativePath_IsResolvedByTheCoresOneRule()
    {
        Written("rows.csv", "a\n1\n");
        var cache = new SourceCache();

        var relative = cache.RowsFor(Reading("rows.csv"), SourceFolder.Of(_folder));
        var absolute = cache.RowsFor(Reading(Path.Join(_folder, "rows.csv")), SourceFolder.Of(_folder));

        Assert.Equal(["a"], relative.Rows!.ColumnNames);

        // The same file, reached by another spelling of its path, is another read step and so another entry.
        Assert.Equal(2, cache.Parsed);
        Assert.Equal(relative.Rows.Rows, absolute.Rows!.Rows);
        Assert.Equal(relative.Fingerprint, absolute.Fingerprint);
    }

    [Fact]
    public void APipelineWhoseRowsAreNotInAFile_IsRefused_ForANotebookHandsNoRowsIn()
    {
        // Rows handed in, or a declaration with nothing to read: a notebook hands no rows in, so there is nothing to
        // open, and it says what would be — not the advice meant for code, which can hand rows in.
        var cache = new SourceCache();

        var handed = Assert.Throws<InvalidOperationException>(
            () => cache.RowsFor(new PipelineDeclaration([new ReadRowsStep("handed in")]), SourceFolder.WorkingDirectory));
        var none = Assert.Throws<InvalidOperationException>(() => cache.RowsFor(new PipelineDeclaration([]), SourceFolder.WorkingDirectory));

        Assert.Equal(handed.Message, none.Message);
        Assert.Contains($"'{ReadCsvStep.Name}'", handed.Message, StringComparison.Ordinal);
        Assert.Equal(0, cache.Parsed);
    }

    [Fact]
    public void WhatIsKnownOfTheBytes_IsTheirFingerprint_OrThatTheyAreGone_OrNothingForAPipelineThatReadsNoFile()
    {
        // Read and hashed, never parsed: what an export compares a fit's bytes with.
        var path = Written("rows.csv", "a,b\n1,2\n");
        var read = new SourceCache().RowsFor(Reading(path), SourceFolder.WorkingDirectory);

        Assert.Equal(SourceBytes.Of(read.Fingerprint), SourceCache.BytesOf(Reading(path), SourceFolder.WorkingDirectory));
        Assert.Equal(SourceBytes.Unreadable, SourceCache.BytesOf(Reading(Path.Join(_folder, "gone.csv")), SourceFolder.WorkingDirectory));
        Assert.Equal(SourceBytes.Unknown, SourceCache.BytesOf(new PipelineDeclaration([new ReadRowsStep("handed in")]), SourceFolder.WorkingDirectory));
        Assert.Equal(SourceBytes.Unknown, SourceCache.BytesOf(new PipelineDeclaration([]), SourceFolder.WorkingDirectory));
    }

    [Fact]
    public void TheCacheHandsOutRowsAsRead_NeverATable()
    {
        // What a step made of the rows is replayed every time; the only thing kept is the input to the first step,
        // handed out with the name of the bytes it was read from.
        var handedOut = typeof(SourceCache).GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
            .Select(method => method.ReturnType)
            .Concat(typeof(SourceCache).GetProperties().Select(property => property.PropertyType))
            .SelectMany(type => type == typeof(SourceRows) ? typeof(SourceRows).GetProperties().Select(property => property.PropertyType) : [type]);

        Assert.All(handedOut, type => Assert.True(
            type == typeof(IRowSource) || type == typeof(string) || type == typeof(IReadOnlyList<string>) || type == typeof(int), type.Name));
    }

    [Fact]
    public async Task AViewOverRowsTheSessionAlreadyOpened_IsTheViewAFreshReadGives_AtEveryBlock()
    {
        var titanic = Path.Join(_folder, "titanic.csv");
        File.Copy(Repository.Data("titanic.csv"), titanic);

        string[] blocks =
        [
            """{"step": "read.csv", "path": "titanic.csv"}""",
            """{"step": "declare", "remainder": "drop", "columns": [{"name": "survived", "kind": "integer", "optional": false}, {"name": "age", "kind": "number", "optional": true}, {"name": "fare", "kind": "number", "optional": false}]}""",
            """{"step": "split.stratified", "column": "survived", "train": 0.7, "validation": 0.15, "test": 0.15, "seed": 20260923}""",
            """{"step": "fill.missing", "column": "age", "with": "median"}""",
            """{"step": "normalise", "column": "fare", "scale": "robust", "outOfRange": "pass"}""",
        ];

        await using var notebook = await Notebook.OpenAsync(Path.Join(_folder, "titanic.verso"));

        foreach (var block in blocks)
        {
            notebook.AddBlock(block);
        }

        var declaration = new PipelineDeclaration([.. blocks.Select(block => StepCatalog.BuiltIn().ReadStep(block))]);

        for (var position = 0; position < blocks.Length; position++)
        {
            var cell = notebook.Scaffold.Cells[position];
            var fresh = DataGrid.Of(new Pipeline(declaration, rows: null, SourceFolder.Of(_folder)).ViewAt(position + 1), 0, declaration).Output.Content;

            await notebook.GestureAsync(cell, StepRenderer.Show);

            Assert.Equal(fresh, cell.Outputs[1].Content);
        }

        var session = notebook.Host.GetCellTypes().OfType<StepCellType>().Single().Session;

        Assert.Equal(1, session.Sources.Parsed);
    }

    [Fact]
    public async Task AKeptView_IsTheViewAFreshSessionGives_AfterAStepBelowReadsAnAbsentColumn()
    {
        // A step added below the split changes neither the key of a view above it nor the bytes, so the view kept is
        // shown again — and it is the view a fresh session works out, because a view checks only the steps it walks.
        // The step below reads a column the rows lack, and is refused at its own block.
        File.WriteAllText(Path.Join(_folder, "rows.csv"), "t,a\n1,1\n2,2\n3,3\n4,4\n5,5\n6,6\n7,7\n8,8\n9,9\n10,10\n");

        string[] blocks =
        [
            """{"step": "read.csv", "path": "rows.csv"}""",
            """{"step": "declare", "remainder": "drop", "columns": [{"name": "t", "kind": "integer", "optional": false}, {"name": "a", "kind": "number", "optional": false}, {"name": "b", "kind": "number", "optional": true}]}""",
            """{"step": "split.byTime", "column": "t", "train": 0.7, "validation": 0.15, "test": 0.15}""",
        ];
        const string readingB = """{"step": "fill.missing", "column": "b", "with": "median"}""";

        await using var kept = await Notebook.OpenAsync(Path.Join(_folder, "kept.verso"));
        await using var fresh = await Notebook.OpenAsync(Path.Join(_folder, "fresh.verso"));

        foreach (var block in blocks)
        {
            kept.AddBlock(block);
            fresh.AddBlock(block);
        }

        var declare = kept.Scaffold.Cells[1];

        await kept.GestureAsync(declare, StepRenderer.Show);

        var fill = kept.AddBlock(readingB);
        fresh.AddBlock(readingB);

        await kept.GestureAsync(declare, StepRenderer.Show);
        await fresh.GestureAsync(fresh.Scaffold.Cells[1], StepRenderer.Show);

        Assert.Equal(1, kept.Host.GetCellTypes().OfType<StepCellType>().Single().Session.ViewsRun);
        Assert.Equal(fresh.Scaffold.Cells[1].Outputs[1].Content, declare.Outputs[1].Content);

        await kept.GestureAsync(fill, StepRenderer.Show);

        Assert.Contains(fill.Outputs, output => output.IsError && output.Content.Contains("fill.missing", StringComparison.Ordinal));
    }
}
