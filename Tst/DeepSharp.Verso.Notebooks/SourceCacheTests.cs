// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Reflection;
using System.Security.Cryptography;
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

    // The reader a first block names for a file, by what the file is.
    private static IPipelineStep ReaderOf(string path) => Path.GetExtension(path) switch
    {
        ".parquet" => new ReadParquetStep(path),
        ".xlsx" => new ReadExcelStep(path),
        ".json" => new ReadJsonStep(path),
        _ => new ReadCsvStep(path),
    };

    public static TheoryData<string> ThePassengerListInEveryOtherFormat() => new() { "titanic.parquet", "titanic.xlsx", "titanic.json" };

    // A copy of one of the files the readers' tests read, in this test's own folder.
    private string Copied(string file)
    {
        var path = Path.Join(_folder, file);
        File.Copy(file == "titanic.csv" ? Repository.Data(file) : Repository.Fixture(file), path);

        return path;
    }

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
    public void WhatTheCellsSayEachColumnHolds_IsProposedOncePerBytes_AndKeptBesideTheirRows()
    {
        var path = Written("rows.csv", "a,b\n1,x\n2,y\n");
        var cache = new SourceCache();

        var first = cache.RowsFor(Reading(path), SourceFolder.WorkingDirectory);
        var second = cache.RowsFor(Reading(path), SourceFolder.WorkingDirectory);

        Assert.Same(first.Proposal, second.Proposal);
        Assert.Same(first.Proposal, cache.KeptFor(Reading(path))!.Value.Proposal);
        Assert.Equal(ColumnKind.Integer, first.Proposal["a"].Kind);

        File.WriteAllText(path, "a,b\nx,1\n");

        Assert.Equal(ColumnKind.Text, cache.RowsFor(Reading(path), SourceFolder.WorkingDirectory).Proposal["a"].Kind);
        Assert.Equal(2, cache.Parsed);
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
    public void APipelineWhoseRowsAreNotInAFile_IsRefused_NamingEveryFileANotebookReads()
    {
        // Rows handed in, or a declaration with nothing to read: a notebook hands no rows in, so there is nothing to
        // open, and it says what would be — every verb of its own that reads a file, as its catalog knows them — not the
        // advice meant for code, which can hand rows in.
        var cache = new SourceCache();
        var catalog = NotebookVerbs.Catalog();
        string[] files = [.. catalog.Descriptions.Where(each => catalog.ReadStep(each.Template) is IReadsAFile or IReadsFiles).Select(each => each.Verb)];

        var handed = Assert.Throws<InvalidOperationException>(
            () => cache.RowsFor(new PipelineDeclaration([new ReadRowsStep("handed in")]), SourceFolder.WorkingDirectory));
        var none = Assert.Throws<InvalidOperationException>(() => cache.RowsFor(new PipelineDeclaration([]), SourceFolder.WorkingDirectory));

        Assert.Equal(["read.csv", "read.excel", "read.join", "read.json", "read.parquet"], files);
        Assert.Equal(handed.Message, none.Message);
        Assert.All(files, verb => Assert.Contains($"'{verb}'", handed.Message, StringComparison.Ordinal));
        Assert.Equal(
            "A notebook reads its rows from a file, with 'read.csv', 'read.excel', 'read.join', 'read.json' or 'read.parquet' as its first block, "
            + "and hands none in; these blocks read no file, so there are no rows to show.",
            handed.Message);
        Assert.Equal(0, cache.Parsed);
    }

    [Theory]
    [MemberData(nameof(ThePassengerListInEveryOtherFormat))]
    public void ThePassengerListInEachFormat_IsParsedOnceFromTheBytesItsFingerprintNames(string file)
    {
        // A fresh step each time, as every gesture reads the blocks afresh: a step is known by what it says, never by which
        // object says it.
        var path = Copied(file);
        var cache = new SourceCache();

        var first = cache.RowsFor(new PipelineDeclaration([ReaderOf(path)]), SourceFolder.WorkingDirectory);
        var second = cache.RowsFor(new PipelineDeclaration([ReaderOf(path)]), SourceFolder.WorkingDirectory);

        Assert.Equal(1, cache.Parsed);
        Assert.Same(first.Rows, second.Rows);
        Assert.Equal(15, first.Rows.ColumnNames.Count);
        Assert.Equal(891, first.Rows.Rows.Count());
        Assert.Equal(Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(path))).ToLowerInvariant(), first.Fingerprint);
        Assert.Equal(first.Fingerprint, second.Fingerprint);
    }

    [Fact]
    public void AWorkbookReadAtANamedSheet_IsAnotherEntry_OverTheSameBytes()
    {
        // The first sheet and the sheet named as the first are one sheet, read by two steps that say different things.
        var path = Copied("titanic.xlsx");
        var cache = new SourceCache();

        var first = cache.RowsFor(new PipelineDeclaration([new ReadExcelStep(path)]), SourceFolder.WorkingDirectory);
        var named = cache.RowsFor(new PipelineDeclaration([new ReadExcelStep(path, "titanic")]), SourceFolder.WorkingDirectory);

        Assert.Equal(2, cache.Parsed);
        Assert.Equal(first.Fingerprint, named.Fingerprint);
        Assert.Equal(first.Rows.Rows, named.Rows.Rows);
    }

    [Fact]
    public void AParquetFilesRows_ProposeWhatTheFileStates_WhereItsCommaSeparatedFileProposesWhatItsCellsLookLike()
    {
        // The Parquet file holds the ages as the words the comma-separated file spells them in, and says so; what a file
        // states outranks cells that read as numbers.
        var parquet = new SourceCache().RowsFor(new PipelineDeclaration([ReaderOf(Copied("titanic.parquet"))]), SourceFolder.WorkingDirectory);
        var csv = new SourceCache().RowsFor(Reading(Copied("titanic.csv")), SourceFolder.WorkingDirectory);

        Assert.Equal(ColumnKind.Text, parquet.Proposal["age"].Stated);
        Assert.Equal(ColumnKind.Text, parquet.Proposal["age"].Kind);
        Assert.Equal(ColumnKind.Integer, parquet.Proposal["survived"].Stated);
        Assert.Null(csv.Proposal["age"].Stated);
        Assert.Equal(ColumnKind.Number, csv.Proposal["age"].Kind);
    }

    [Theory]
    [MemberData(nameof(ThePassengerListInEveryOtherFormat))]
    public void WhatIsKnownOfTheBytes_IsTheirFingerprint_WhateverFileTheFirstBlockReads(string file)
    {
        // What an export compares a fit's bytes with, read and hashed as the rows were, whatever the file is.
        var path = Copied(file);
        var declaration = new PipelineDeclaration([ReaderOf(path)]);
        var read = new SourceCache().RowsFor(declaration, SourceFolder.WorkingDirectory);

        Assert.Equal(SourceBytes.Of(read.Fingerprint), SourceCache.BytesOf(declaration, SourceFolder.WorkingDirectory));
        Assert.Equal(
            SourceBytes.Unreadable,
            SourceCache.BytesOf(new PipelineDeclaration([ReaderOf(Path.Join(_folder, "gone" + Path.GetExtension(file)))]), SourceFolder.WorkingDirectory));
    }

    [Fact]
    public void TheRowsKeptForTheBlocks_AreFoundByTheBlockThatReadsTheFile()
    {
        // What a gesture knows of the source: the rows this session read last, found by the block that reads the file,
        // whatever stands below it — and nothing for another reader of the same path, for rows handed in, or for none.
        var path = Copied("titanic.parquet");
        var cache = new SourceCache();

        Assert.Null(cache.KeptFor(new PipelineDeclaration([ReaderOf(path)])));

        var read = cache.RowsFor(new PipelineDeclaration([ReaderOf(path)]), SourceFolder.WorkingDirectory);
        var kept = cache.KeptFor(new PipelineDeclaration(
            [ReaderOf(path), new DeclareStep([new ColumnDeclaration("survived", ColumnKind.Integer, Optional: false)])]));

        Assert.NotNull(kept);
        Assert.Same(read.Rows, kept.Value.Rows);
        Assert.Same(read.Proposal, kept.Value.Proposal);
        Assert.Equal(read.Fingerprint, kept.Value.Fingerprint);
        Assert.Null(cache.KeptFor(new PipelineDeclaration([new ReadCsvStep(path)])));
        Assert.Null(cache.KeptFor(new PipelineDeclaration([new ReadRowsStep("handed in")])));
        Assert.Null(cache.KeptFor(new PipelineDeclaration([])));
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
        // handed out with the name of the bytes it was read from and what their cells say each column holds.
        var handedOut = typeof(SourceCache).GetMethods(BindingFlags.Public | BindingFlags.Instance | BindingFlags.DeclaredOnly)
            .Select(method => method.ReturnType)
            .Concat(typeof(SourceCache).GetProperties().Select(property => property.PropertyType))
            .Select(type => Nullable.GetUnderlyingType(type) ?? type)
            .SelectMany(type => type == typeof(SourceRows) ? typeof(SourceRows).GetProperties().Select(property => property.PropertyType) : [type]);

        Assert.All(handedOut, type => Assert.True(
            type == typeof(IRowSource) || type == typeof(string) || type == typeof(IReadOnlyList<string>) || type == typeof(int)
            || type == typeof(KindProposal), type.Name));
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
            var fresh = DataGrid.Of(
                new Pipeline(declaration, rows: null, SourceFolder.Of(_folder)).ViewAt(position + 1), 0, declaration,
                column => TakenIn.Of(column, stored: null, KindProposal.Of(new CsvRowSource(titanic)))).Output.Content;

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

    // ---- a source of two files: one entry, keyed on both files' bytes in the order the step names them

    private static PipelineDeclaration Joining(string left, string right) =>
        new([new ReadJoinStep(left, right, ["Flock"], Unmatched.Refuse)]);

    // The key of several files as the cache is to make it: a SHA-256 over each file's own SHA-256, in order.
    private static string Combined(params string[] paths) =>
        Convert.ToHexString(SHA256.HashData([.. paths.SelectMany(path => SHA256.HashData(File.ReadAllBytes(path)))])).ToLowerInvariant();

    [Fact]
    public void AJoinOfTwoFiles_IsParsedOnce_UnderOneKeyOfBothFilesInTheirOrder()
    {
        var left = Written("planned.csv", "Flock,Age\nF1,30\nF2,31\n");
        var right = Written("arrived.csv", "Flock,Weight\nF2,20\nF1,10\n");
        var cache = new SourceCache();

        var first = cache.RowsFor(Joining(left, right), SourceFolder.WorkingDirectory);
        var second = cache.RowsFor(Joining(left, right), SourceFolder.WorkingDirectory);

        Assert.Equal(1, cache.Parsed);
        Assert.Same(first.Rows, second.Rows);
        Assert.Equal(["Flock", "Age", "Weight"], first.Rows.ColumnNames);
        Assert.Equal(Combined(left, right), first.Fingerprint);
        Assert.NotEqual(Combined(right, left), first.Fingerprint);
        Assert.Same(first.Rows, cache.KeptFor(Joining(left, right))!.Value.Rows);
        Assert.Null(cache.KeptFor(Joining(right, left)));
    }

    [Fact]
    public void ChangingOnlyTheSecondFileOfAJoin_MakesAnotherKey_AndItsRowsAreReadAgain()
    {
        // The stale view a key of one file would leave: the right file changes, the left does not.
        var left = Written("planned.csv", "Flock,Age\nF1,30\nF2,31\n");
        var right = Written("arrived.csv", "Flock,Weight\nF1,10\nF2,20\n");
        var cache = new SourceCache();

        var before = cache.RowsFor(Joining(left, right), SourceFolder.WorkingDirectory);
        File.WriteAllText(right, "Flock,Weight\nF1,11\nF2,20\n");
        var after = cache.RowsFor(Joining(left, right), SourceFolder.WorkingDirectory);

        Assert.NotEqual(before.Fingerprint, after.Fingerprint);
        Assert.Equal(2, cache.Parsed);
        Assert.Equal("11", after.Rows.Rows.First()[2]);
        Assert.Equal(SourceBytes.Of(after.Fingerprint), SourceCache.BytesOf(Joining(left, right), SourceFolder.WorkingDirectory));
    }

    [Fact]
    public void TwoFilesSplitAnotherWay_AreAnotherKey_ThoughTheirBytesRunOnTheSame()
    {
        // A digest over the files laid end to end could not tell "ab" and "c" from "a" and "bc"; one over each file's own
        // digest can.
        var one = SourceCache.BytesOf(Joining(Written("l1.csv", "Flock\nF1\nF"), Written("r1.csv", "2\n")), SourceFolder.WorkingDirectory);
        var other = SourceCache.BytesOf(Joining(Written("l2.csv", "Flock\nF1\n"), Written("r2.csv", "F2\n")), SourceFolder.WorkingDirectory);

        Assert.NotEqual(one, other);
        Assert.NotEqual(SourceBytes.Of("Flock\nF1\nF2\n"u8.ToArray().Fingerprint()), one);
    }

    [Fact]
    public void WhatIsKnownOfTheBytesOfAJoin_IsUnreadable_WhenEitherFileIsGone()
    {
        var left = Written("planned.csv", "Flock,Age\nF1,30\n");

        Assert.Equal(SourceBytes.Unreadable, SourceCache.BytesOf(Joining(left, Path.Join(_folder, "gone.csv")), SourceFolder.WorkingDirectory));
        Assert.Equal(SourceBytes.Unreadable, SourceCache.BytesOf(Joining(Path.Join(_folder, "gone.csv"), left), SourceFolder.WorkingDirectory));
    }

    [Fact]
    public async Task AViewOverAJoin_IsWorkedOutAgain_WhenOnlyTheSecondFileChanges_AndKeptWhileNeitherDoes()
    {
        File.WriteAllText(Path.Join(_folder, "planned.csv"), "Flock,Age\nF1,30\nF2,31\n");
        File.WriteAllText(Path.Join(_folder, "arrived.csv"), "Flock,Weight\nF1,10\nF2,20\n");

        await using var notebook = await Notebook.OpenAsync(Path.Join(_folder, "flocks.verso"));
        notebook.AddBlock("""{"step": "read.join", "left": {"kind": "csv", "path": "planned.csv"}, "right": {"kind": "csv", "path": "arrived.csv"}, "on": ["Flock"], "unmatched": "refuse"}""");
        notebook.AddBlock("""{"step": "declare", "remainder": "drop", "columns": [{"name": "Flock", "kind": "text", "optional": false, "id": true}, {"name": "Age", "kind": "number", "optional": false}, {"name": "Weight", "kind": "number", "optional": false}]}""");

        var declare = notebook.Scaffold.Cells[1];
        var session = notebook.Host.GetCellTypes().OfType<StepCellType>().Single().Session;

        await notebook.GestureAsync(declare, StepRenderer.Show);
        var first = declare.Outputs[1].Content;
        await notebook.GestureAsync(declare, StepRenderer.Show);

        Assert.Equal(1, session.ViewsRun);
        Assert.Equal(1, session.Sources.Parsed);

        File.WriteAllText(Path.Join(_folder, "arrived.csv"), "Flock,Weight\nF1,17\nF2,20\n");
        await notebook.GestureAsync(declare, StepRenderer.Show);

        Assert.Equal(2, session.ViewsRun);
        Assert.Equal(2, session.Sources.Parsed);
        Assert.NotEqual(first, declare.Outputs[1].Content);
        Assert.Contains(">17<", declare.Outputs[1].Content, StringComparison.Ordinal);
    }
}
