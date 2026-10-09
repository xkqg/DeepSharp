// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Text.Json;
using DeepSharp.Pipelines;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// Two files read as one source: the rows of the left file, each with the one row of the right file its key names. A team
/// that joined its planned flocks to the flocks that arrived by hand, before the pipeline started, had a pipeline file that
/// said nothing of the join; this puts the join in the file, with how the keys are compared and what becomes of a left row
/// without a partner said, never defaulted.
/// </summary>
public sealed class JoinTests : IDisposable
{
    // What was planned for each flock, and what arrived for it: the right file holds the flocks in another order.
    private const string Planned = "Flock,Farm,Age,Planned\nF1,north,30,100\nF2,south,31,200\nF3,north,32,300\nF4,east,33,400\n";
    private const string Arrived = "Flock,Band1,Band2\nF3,30,270\nF1,10,90\nF2,20,180\nF4,40,360\n";

    private readonly string _folder = Directory.CreateTempSubdirectory("deepsharp-join-").FullName;

    public void Dispose() => Directory.Delete(_folder, recursive: true);

    private string Written(string name, string text)
    {
        var path = Path.Join(_folder, name);
        File.WriteAllText(path, text);

        return path;
    }

    private SourceFolder Folder => SourceFolder.Of(_folder);

    // The two files written beside each other, and the join of them, by their relative paths.
    private ReadJoinStep Join(Unmatched unmatched, string planned = Planned, string arrived = Arrived, params string[] on)
    {
        Written("planned.csv", planned);
        Written("arrived.csv", arrived);

        return new ReadJoinStep("planned.csv", "arrived.csv", on.Length == 0 ? ["Flock"] : on, unmatched);
    }

    private static string[][] Cells(IRowSource rows) => [.. rows.Rows.Select(row => row.Select(cell => cell ?? "<gap>").ToArray())];

    [Fact]
    public void AJoin_IsOneSource_WhoseRowsSayBothFiles_InTheLeftFilesOrder()
    {
        var rows = Join(Unmatched.Refuse).Open(Folder);

        // The left file's columns, then the right file's but its key.
        Assert.Equal(["Flock", "Farm", "Age", "Planned", "Band1", "Band2"], rows.ColumnNames);
        Assert.Equal(
            [
                ["F1", "north", "30", "100", "10", "90"],
                ["F2", "south", "31", "200", "20", "180"],
                ["F3", "north", "32", "300", "30", "270"],
                ["F4", "east", "33", "400", "40", "360"],
            ],
            Cells(rows));
    }

    [Fact]
    public void AKeyTheRightFileHoldsTwice_IsRefused_NamingTheKeyAndItsRows_BeforeAnyLeftRowIsMatched()
    {
        // Many to one: every left row has at most one partner, so a right key that stands twice is two answers to one
        // question. Refused before the left rows are looked at, so an unmatched left row does not hide it.
        var join = Join(
            Unmatched.Refuse,
            planned: Planned + "F9,west,34,500\n",
            arrived: "Flock,Band1,Band2\nF1,10,90\nF2,20,180\nF3,30,270\nF2,21,179\nF4,40,360\n");

        var refused = Assert.Throws<FormatException>(() => join.Open(Folder));

        Assert.Contains("'F2'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("rows 2 and 4", refused.Message, StringComparison.Ordinal);
        Assert.Contains("arrived.csv", refused.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("F9", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AKeyTheLeftFileHoldsTwice_GivesEachOfItsRowsTheSamePartner()
    {
        var rows = Join(Unmatched.Refuse, planned: "Flock,Age\nF1,30\nF2,31\nF1,32\n").Open(Folder);

        Assert.Equal([["F1", "30", "10", "90"], ["F2", "31", "20", "180"], ["F1", "32", "10", "90"]], Cells(rows));
    }

    [Fact]
    public void ALeftRowWithoutAPartner_StopsTheRun_NamingTheRowAndItsKey_WhenTheJoinRefusesThem()
    {
        var join = Join(Unmatched.Refuse, planned: Planned + "F5,west,34,500\n");

        var refused = Assert.Throws<FormatException>(() => join.Open(Folder));

        Assert.Contains("Row 5 of", refused.Message, StringComparison.Ordinal);
        Assert.Contains("planned.csv", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'F5'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("arrived.csv", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ALeftRowWithoutAPartner_IsLeftOut_AndCountedInTheFit_WhenTheJoinDropsThem()
    {
        var join = Join(Unmatched.Drop, planned: "Flock,Farm,Age,Planned\nF1,north,30,100\nF5,west,34,500\nF2,south,31,200\nF6,west,35,600\nF3,north,32,300\nF4,east,33,400\n");

        var prepared = new Pipeline(
                Pdd.Create().Add(join).Declare(schema => schema.Text("Flock").Number("Age", "Band1")).SplitAtRandom(0.5, 0.25).Declaration,
                rows: null,
                Folder)
            .Run();

        var seen = prepared.Fitted[2];

        Assert.Equal(["F1", "F2", "F3", "F4"], Enumerable.Range(0, prepared.Table.RowCount).Select(row => prepared.Table["Flock"].TextAt(row)).Order(StringComparer.Ordinal));
        Assert.Equal(2, seen.Number("rows.unmatched"));
        Assert.Equal(4, seen.Number("rows.train") + seen.Number("rows.validation") + seen.Number("rows.test"));
        Assert.Contains("\"rows.unmatched\": 2", prepared.ToJson(), StringComparison.Ordinal);
    }

    [Fact]
    public void AJoinThatDropsAndFindsEveryPartner_SaysItLeftNoneOut_AndOneThatRefusesSaysNothing()
    {
        // The count stands where the join drops, nought included, so a reader of the fit never has to guess whether the
        // join left rows out; a join that refuses them never leaves one out, and the split writes what it always wrote.
        PreparedData Run(Unmatched unmatched) => new Pipeline(
                Pdd.Create().Add(Join(unmatched)).Declare(schema => schema.Number("Age")).SplitAtRandom(0.5, 0.25).Declaration,
                rows: null,
                Folder)
            .Run();

        Assert.Equal(0, Run(Unmatched.Drop).Fitted[2].Number("rows.unmatched"));
        Assert.DoesNotContain("rows.unmatched", Run(Unmatched.Refuse).Fitted[2].Numbers.Keys);
    }

    [Fact]
    public void TheKey_IsTheExactTextOfTheCell_WithTheSpacesAroundItTakenOff_AndNothingElse()
    {
        // As a row key reads a cell: trimmed, compared character by character. No case is folded and no number is read,
        // so 'f2' is not 'F2' and '03' is not '3'.
        var rows = Join(
                Unmatched.Drop,
                planned: "Flock,Age\nF1,30\nf2,31\n03,32\n F4 ,33\n",
                arrived: "Flock,Band1\n  F1,10\nF2,20\n3,30\nF4,40\n")
            .Open(Folder);

        Assert.Equal([["F1", "30", "10"], [" F4 ", "33", "40"]], Cells(rows));
    }

    [Fact]
    public void AKeyThatIsAGap_MatchesNothing_OnEitherSide()
    {
        // A gap is no value, here as everywhere: a left row with no key has no partner, and right rows with no key are
        // neither a partner nor a key held twice.
        var arrived = "Flock,Band1\nF1,10\n,11\n ,12\nF2,20\n";

        var dropped = Join(Unmatched.Drop, planned: "Flock,Age\nF1,30\n,31\nF2,32\n", arrived: arrived).Open(Folder);
        var refused = Assert.Throws<FormatException>(
            () => Join(Unmatched.Refuse, planned: "Flock,Age\nF1,30\n  ,31\n", arrived: arrived).Open(Folder));

        Assert.Equal([["F1", "30", "10"], ["F2", "32", "20"]], Cells(dropped));
        Assert.Contains("Row 2 of", refused.Message, StringComparison.Ordinal);
        Assert.Contains("'Flock' is a gap", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void SeveralKeyColumns_MatchARowOnAllOfThem()
    {
        var rows = Join(
                Unmatched.Drop,
                planned: "Farm,Flock,Age\nnorth,1,30\nsouth,1,31\nnorth,2,32\n",
                arrived: "Flock,Farm,Band1\n1,south,11\n1,north,10\n",
                on: ["Farm", "Flock"])
            .Open(Folder);

        Assert.Equal(["Farm", "Flock", "Age", "Band1"], rows.ColumnNames);
        Assert.Equal([["north", "1", "30", "10"], ["south", "1", "31", "11"]], Cells(rows));
    }

    [Fact]
    public void AKeyColumnEitherFileLacks_IsRefusedByName()
    {
        var left = Assert.Throws<FormatException>(() => Join(Unmatched.Refuse, planned: "Id,Age\nF1,30\n").Open(Folder));
        var right = Assert.Throws<FormatException>(() => Join(Unmatched.Refuse, arrived: "Id,Band1\nF1,10\n").Open(Folder));

        Assert.Contains("'Flock'", left.Message, StringComparison.Ordinal);
        Assert.Contains("planned.csv", left.Message, StringComparison.Ordinal);
        Assert.Contains("'Flock'", right.Message, StringComparison.Ordinal);
        Assert.Contains("arrived.csv", right.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AColumnBothFilesHold_BesideTheKey_IsRefusedByName_RatherThanRenamed()
    {
        var refused = Assert.Throws<FormatException>(
            () => Join(Unmatched.Refuse, arrived: "Flock,Farm,Band1\nF1,north,10\nF2,south,20\nF3,north,30\nF4,east,40\n").Open(Folder));

        Assert.Contains("'Farm'", refused.Message, StringComparison.Ordinal);
        Assert.Contains("rename", refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void AJoinedRowsKey_IsADigestOfEveryCell_TheRightFilesToo()
    {
        var join = Join(Unmatched.Refuse);

        PreparedData Run() => new Pipeline(
                Pdd.Create().Add(join).Declare(schema => schema.Text("Flock").Number("Age")).Declaration,
                rows: null,
                Folder)
            .Run();

        var first = Run().Table.Identities;
        var again = Run().Table.Identities;

        // Nothing the schema takes in comes from the right file, and a cell of it changing still makes another record.
        Written("arrived.csv", Arrived.Replace("F1,10,90", "F1,11,90", StringComparison.Ordinal));
        var changed = Run().Table.Identities;

        Assert.Equal(first, again);
        Assert.NotEqual(first[0].Key, changed[0].Key);
        Assert.Equal(first.Skip(1), changed.Skip(1));
        Assert.Equal(RowKey.Of(["Flock", "Farm", "Age", "Planned", "Band1", "Band2"], ["F1", "north", "30", "100", "10", "90"]), first[0].Key);
    }

    [Fact]
    public void AJoinedRow_IsReadAtItsPlaceInTheLeftFile()
    {
        // The right file names the flocks in another order; a row is where the left file has it, which is the row a
        // refusal names.
        var prepared = new Pipeline(
                Pdd.Create().Add(Join(Unmatched.Refuse)).Declare(schema => schema.Text("Flock").Number("Band1")).Declaration,
                rows: null,
                Folder)
            .Run();

        Assert.Equal([0, 1, 2, 3], prepared.Table.Identities.Select(identity => identity.ReadAt));
        Assert.Equal(["F1", "F2", "F3", "F4"], Enumerable.Range(0, 4).Select(row => prepared.Table["Flock"].TextAt(row)));
    }

    [Fact]
    public void AJoinedRow_BelowARowLeftOut_IsReadAtItsPlaceAmongTheRowsTheJoinKept()
    {
        // A row's place is where it stands among the rows read — the place every way back reads the row as it was read by —
        // so below a left row with no partner, the rows read are one fewer than the left file's.
        var prepared = new Pipeline(
                Pdd.Create().Add(Join(Unmatched.Drop, planned: "Flock,Farm,Age,Planned\nF9,west,40,900\nF1,north,30,100\nF2,south,31,200\n"))
                    .Declare(schema => schema.Text("Flock").Number("Band1")).Declaration,
                rows: null,
                Folder)
            .Run();

        Assert.Equal([0, 1], prepared.Table.Identities.Select(identity => identity.ReadAt));
        Assert.Equal(["F1", "F2"], Enumerable.Range(0, 2).Select(row => prepared.Table["Flock"].TextAt(row)));
    }

    [Fact]
    public void ServedRows_ArriveAlreadyJoined_AndTheJoinIsOpenedAtTheFitAlone()
    {
        // The join reads training rows. A row served later is handed in as the rows it was trained on read, both files'
        // columns in it; the files are not opened again, so serving goes on without them.
        var join = Join(Unmatched.Refuse);
        var prepared = new Pipeline(
                Pdd.Create().Add(join).Declare(schema => schema.Number("Age", "Band1", "Planned")).SplitAtRandom(0.5, 0.25).Target("Planned").Declaration,
                rows: null,
                Folder)
            .Run();

        File.Delete(Path.Join(_folder, "planned.csv"));
        File.Delete(Path.Join(_folder, "arrived.csv"));

        var served = prepared.Served(new InMemoryRowSource(["Age", "Band1"], [["40", "55"]]));
        var notJoined = Assert.Throws<InvalidOperationException>(() => prepared.Served(new InMemoryRowSource(["Age"], [["40"]])));

        Assert.Equal(["Age", "Band1"], served.FeatureNames);
        Assert.Equal([40d, 55d], served.Features[0]);
        Assert.Contains("'Band1'", notJoined.Message, StringComparison.Ordinal);
        Assert.Contains("already joined", notJoined.Message, StringComparison.Ordinal);
        Assert.Throws<FileNotFoundException>(() => join.Open(Folder));
    }

    [Fact]
    public void AJoin_IsWrittenAndReadBackAsTheVerbItIs_NewInTheEighthVersion()
    {
        var join = Join(Unmatched.Drop, on: ["Farm", "Flock"]);
        var declaration = Pdd.Create().Add(join).Declare(schema => schema.Number("Age")).Declaration;
        var json = declaration.ToJson();

        var back = PipelineDeclaration.FromJson(json, StepCatalog.BuiltIn());

        Assert.Equal(declaration, back);
        Assert.Contains(
            """{"step":"read.join","left":{"kind":"csv","path":"planned.csv"},"right":{"kind":"csv","path":"arrived.csv"},"on":["Farm","Flock"],"unmatched":"drop"}""",
            JsonSerializer.Serialize(JsonDocument.Parse(json).RootElement.GetProperty("declaration")[0]),
            StringComparison.Ordinal);
        Assert.Equal(8, StepCatalog.BuiltIn().Describe("read.join").Since);
        Assert.Contains("read.join", StepCatalog.BuiltIn().JsonSchema(), StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("""{"step":"read.join","left":{"kind":"csv","path":"a.csv"},"right":{"kind":"csv","path":"b.csv"},"on":["k"]}""", "unmatched")]
    [InlineData("""{"step":"read.join","left":{"kind":"csv","path":"a.csv"},"right":{"kind":"csv","path":"b.csv"},"on":["k"],"unmatched":"keep"}""", "keep")]
    [InlineData("""{"step":"read.join","left":{"kind":"csv","path":"a.csv"},"right":{"kind":"csv","path":"b.csv"},"on":[],"unmatched":"drop"}""", "on")]
    [InlineData("""{"step":"read.join","left":{"kind":"csv","path":"a.csv"},"on":["k"],"unmatched":"drop"}""", "right")]
    [InlineData("""{"step":"read.join","left":{"kind":"csv","path":" "},"right":{"kind":"csv","path":"b.csv"},"on":["k"],"unmatched":"drop"}""", "path")]
    public void AJoinThatLeavesSomethingUnsaid_IsRefusedWhereItIsRead_AndWhatBecomesOfAnUnmatchedRowHasNoDefault(string step, string named)
    {
        var refused = Assert.Throws<PipelineFileException>(
            () => PipelineDeclaration.FromJson($$"""{"version":8,"declaration":[{{step}}]}""", StepCatalog.BuiltIn()));

        Assert.Contains(named, refused.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void TheDoor_TakesTheTwoFilesTheKeysAndWhatBecomesOfAnUnmatchedRow_WithNoDefaultForIt()
    {
        Written("planned.csv", Planned);
        Written("arrived.csv", Arrived);

        var door = typeof(PipelineBuilder).GetMethod(nameof(PipelineBuilder.ReadJoin))!;
        var declaration = Pdd.Create()
            .ReadJoin(Path.Join(_folder, "planned.csv"), Path.Join(_folder, "arrived.csv"), ["Flock"], Unmatched.Drop)
            .Declare(schema => schema.Number("Band2"))
            .Declaration;

        Assert.Equal(4, door.GetParameters().Length);
        Assert.False(door.GetParameters()[^1].HasDefaultValue);
        Assert.Equal(
            new ReadJoinStep(Path.Join(_folder, "planned.csv"), Path.Join(_folder, "arrived.csv"), ["Flock"], Unmatched.Drop),
            declaration.Steps[0]);
        Assert.Equal(4, new Pipeline(declaration).Run().Table.RowCount);
        Assert.Throws<ArgumentOutOfRangeException>(() => new ReadJoinStep("a.csv", "b.csv", ["k"], (Unmatched)7));
        Assert.Throws<ArgumentException>(() => new ReadJoinStep("a.csv", "b.csv", [], Unmatched.Drop));
        Assert.Throws<ArgumentException>(() => new ReadJoinStep("a.csv", " ", ["k"], Unmatched.Drop));
    }

    [Fact]
    public void AJoin_ReadsTwoFiles_InTheOrderItNamesThem_FromTheBytesHandedIn()
    {
        // What a caller that read the files' bytes — to fingerprint them — hands back to have the rows parsed from those
        // very bytes, never from a second read.
        var join = Join(Unmatched.Refuse);
        IReadsFiles files = join;

        var fromBytes = files.Open([File.ReadAllBytes(Path.Join(_folder, "planned.csv")), File.ReadAllBytes(Path.Join(_folder, "arrived.csv"))], ["planned.csv", "arrived.csv"]);

        Assert.Equal(["planned.csv", "arrived.csv"], files.Paths);
        Assert.Equal(Cells(join.Open(Folder)), Cells(fromBytes));
        Assert.Throws<ArgumentException>(() => files.Open([File.ReadAllBytes(Path.Join(_folder, "planned.csv"))], ["planned.csv"]));
        Assert.Throws<ArgumentException>(() => files.Open([[], []], ["planned.csv"]));
    }

    [Fact]
    public void AJoin_NamesItsColumnsFromTheTwoHeadersAlone_SoSavedDecisionsAreTakenOverOntoIt()
    {
        // A file's first line is what a take-over compares saved decisions with; a join says the columns its rows will
        // have from the two first lines, and nothing else is read.
        var planned = Written("planned.csv", Planned + "F1,\"unclosed\n");
        var arrived = Written("arrived.csv", Arrived);
        var join = new ReadJoinStep(planned, arrived, ["Flock"], Unmatched.Refuse);
        var saved = PipelinePreset.Of(Pdd.Create().ReadCsv(planned).Declare(schema => schema.Text("Flock").Number("Age")).Declaration, ["Flock", "Farm", "Age", "Planned"]);

        Pdd.Create().Add(join).Declare(saved, out var declared);

        Assert.Equal(["Flock", "Farm", "Age", "Planned", "Band1", "Band2"], join.ColumnNamesIn(SourceFolder.WorkingDirectory));
        Assert.Equal(["Flock", "Farm", "Age", "Planned", "Band1", "Band2"], PipelineBuilder.HeaderOf([join], rows: null));
        Assert.Equal(["Band1", "Band2"], declared.NewColumns);
        Assert.Null(PipelineBuilder.HeaderOf([new ReadJoinStep(planned, Path.Join(_folder, "gone.csv"), ["Flock"], Unmatched.Refuse)], rows: null));
        Assert.Null(PipelineBuilder.HeaderOf([new ReadJoinStep(planned, arrived, ["Barn"], Unmatched.Refuse)], rows: null));
    }

    [Fact]
    public void TwoJoins_AreOneStep_OnlyWhenTheySayTheSameInEveryPart()
    {
        var join = new ReadJoinStep("a.csv", "b.csv", ["Farm", "Flock"], Unmatched.Drop);

        Assert.Equal(join, new ReadJoinStep("a.csv", "b.csv", ["Farm", "Flock"], Unmatched.Drop));
        Assert.Equal(join.GetHashCode(), new ReadJoinStep("a.csv", "b.csv", ["Farm", "Flock"], Unmatched.Drop).GetHashCode());
        Assert.NotEqual(join, new ReadJoinStep("c.csv", "b.csv", ["Farm", "Flock"], Unmatched.Drop));
        Assert.NotEqual(join, new ReadJoinStep("a.csv", "c.csv", ["Farm", "Flock"], Unmatched.Drop));
        Assert.NotEqual(join, new ReadJoinStep("a.csv", "b.csv", ["Farm", "Flock"], Unmatched.Refuse));
        Assert.NotEqual(join, new ReadJoinStep("a.csv", "b.csv", ["Flock", "Farm"], Unmatched.Drop));
        Assert.False(join.Equals(null));
    }

    [Fact]
    public void RowsServedWithoutTheColumnsOfBothFiles_AreRefused_SayingTheyArriveJoined_AndOtherSourcesAreRefusedAsEver()
    {
        // Only a source of several files says so; rows served to a pipeline of one file are refused in the words they always
        // were, and a column the schema allows to be absent is not asked of served rows.
        var join = new Pipeline(
                Pdd.Create().Add(Join(Unmatched.Refuse)).Declare(schema => schema.Number("Age", "Band1", "Planned").Optional("Band2", ColumnKind.Number))
                    .SplitAtRandom(0.5, 0.25).Target("Planned").Declaration,
                rows: null,
                Folder)
            .Run();
        var single = Pdd.Create().Read(new InMemoryRowSource(["Age", "Band1"], [["1", "2"], ["3", "4"], ["5", "6"], ["7", "8"]]), "rows")
            .Declare(schema => schema.Number("Age", "Band1")).Build().Run();

        var one = Assert.Throws<InvalidOperationException>(() => single.Served(new InMemoryRowSource(["Age"], [["40"]])));

        Assert.Equal(1, join.Served(new InMemoryRowSource(["Age", "Band1"], [["40", "55"]])).RowCount);
        Assert.Equal(1, join.Served(new InMemoryRowSource(["Age", "Band1", "Planned"], [["40", "55", "300"]])).RowCount);
        Assert.DoesNotContain("joined", one.Message, StringComparison.Ordinal);
        Assert.Contains("'Band1'", one.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ACourseHasNoPlaceForAJoin_EveryCourseStartsByReadingOneFile()
    {
        Assert.All(PipelineCourse.Named, course => Assert.Equal("read.csv", course.Steps[0].Verb));
    }

    [Fact]
    public void AJoin_ReadsNoColumnOfTheTable_ItNamesColumnsOfItsFiles()
    {
        // The keys are columns of the two files, not of the rows the schema reads: no step above the schema reads one,
        // and nothing that follows the table's columns may move a drop above the source for them.
        IPipelineStep join = new ReadJoinStep("a.csv", "b.csv", ["Flock"], Unmatched.Drop);

        Assert.Empty(join.ColumnsRead);
    }
}
