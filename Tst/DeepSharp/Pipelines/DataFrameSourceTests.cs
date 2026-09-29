// Copyright (c) 2026 H.P. Gansevoort. All rights reserved.
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

using System.Data;
using DeepSharp.Pipelines;
using Microsoft.Data.Analysis;
using Microsoft.Data.Sqlite;

namespace DeepSharp.Tests.Pipelines;

/// <summary>
/// The second reader, and the one that proves the seam is real: a data frame is nothing like a file, and
/// the pipeline cannot tell the difference. Everything a frame can be filled from — a query, a file, rows
/// somebody already had — comes in through this one door, so the long tail of formats stays somebody
/// else's solved problem.
/// </summary>
public class DataFrameSourceTests
{
    private static DataFrame Frame()
    {
        var age = new PrimitiveDataFrameColumn<double>("age", [22.0, null, 35.0]);
        var sex = new StringDataFrameColumn("sex", ["male", "female", "female"]);
        var fare = new PrimitiveDataFrameColumn<double>("fare", [7.25, 71.28, 8.05]);

        return new DataFrame(age, sex, fare);
    }

    [Fact]
    public void AFrameIsReadLikeAnyOtherRows()
    {
        var source = new DataFrameRowSource(Frame());

        Assert.Equal(["age", "sex", "fare"], source.ColumnNames);
        Assert.Equal(3, source.Rows.Count());
        Assert.Equal("male", source.Rows.First()[1]);
    }

    [Fact]
    public void AnAbsentValueStaysAbsent()
    {
        // The obvious shortcut is to hand back a not-a-number, which is what a plotting reader does because
        // a not-a-number means "do not draw". Here it would mean "arithmetic went wrong", and the pipeline
        // would fill a column with a number nobody measured.
        var source = new DataFrameRowSource(Frame());

        Assert.Null(source.Rows.ElementAt(1)[0]);
    }

    [Fact]
    public void AWholePipelineRunsOffAFrame()
    {
        var prepared = Pdd.Create()
            .ReadDataFrame(Frame())
            .Declare(schema => schema.Optional("age", ColumnKind.Number).Text("sex").Number("fare"))
            .SplitAtRandom(0.60, 0.20, seed: 3)
            .FillMissing("age", With.Median)
            .Encode("sex")
            .Normalise("fare")
            .Build()
            .Run();

        Assert.Equal(3, prepared.Table.RowCount);
        Assert.Equal(1, ((Column<double>)prepared.Table["age_was_missing"])[1]);
        Assert.True(prepared.Table.Has("sex_male") || prepared.Table.Has("sex_female"));
    }

    [Fact]
    public void TheFileSaysTheRowsWereHandedIn_AndSaysSoAgainWhenItIsRun()
    {
        var declaration = Pdd.Create()
            .ReadDataFrame(Frame())
            .Declare(schema => schema.Number("fare"))
            .Declaration;

        var returned = PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn());

        Assert.Equal(declaration, returned);
        Assert.Equal("read.rows", returned.Steps[0].Verb);

        // Loaded from a file, the rows are not in it — which is the truth, said out loud rather than by a
        // null reference three steps later.
        var refused = Assert.Throws<InvalidOperationException>(() => new Pipeline(returned).Run());

        Assert.Contains("handed in", refused.Message);
        Assert.Contains("data frame", refused.Message);
    }

    [Fact]
    public void AndTheSameDeclarationRunsOnRowsHandedInLater()
    {
        var declaration = Pdd.Create()
            .ReadDataFrame(Frame())
            .Declare(schema => schema.Number("fare"))
            .Declaration;

        var prepared = new Pipeline(PipelineDeclaration.FromJson(declaration.ToJson(), StepCatalog.BuiltIn()))
            .Run(new DataFrameRowSource(Frame()));

        Assert.Equal(3, prepared.Table.RowCount);
    }

    [Fact]
    public void AFrameReadFromAFile_IsTheSameDataAsTheReaderOfOurOwn()
    {
        var path = Repository.Data("titanic.csv");

        var ours = Pdd.Create()
            .ReadCsv(path)
            .Declare(schema => schema.Integer("survived").Number("fare"))
            .Build()
            .Prepare();

        var theirs = Pdd.Create()
            .ReadCsvFrame(path)
            .Declare(schema => schema.Integer("survived").Number("fare"))
            .Build()
            .Prepare();

        Assert.Equal(ours.RowCount, theirs.RowCount);
        Assert.Equal(((Column<double>)ours["fare"])[0], ((Column<double>)theirs["fare"])[0]);
        Assert.Equal(((Column<long>)ours["survived"])[890], ((Column<long>)theirs["survived"])[890]);
    }

    [Theory]
    [InlineData("titanic.csv")]
    [InlineData("apple.csv")]
    public void AFrameReadFromAFile_HandsOverTheFilesOwnText_SoItsRowsAreTheRowsOurOwnReaderReads(string file)
    {
        // The frame used to guess each column's kind from its first ten rows and hand back its own spelling of what it
        // read — 133.1285 where the file says 133.1284878 — so the same file split differently through the two doors.
        var path = Repository.Data(file);
        var first = CsvRowSource.HeaderOf(path)[0];

        Table Everything(PipelineBuilder source) => source.Declare(schema => schema.Text(first), Remainder.Keep).Build().Prepare();

        var ours = Everything(Pdd.Create().ReadCsv(path));
        var theirs = Everything(Pdd.Create().ReadCsvFrame(path));

        Assert.Equal(ours.Identities, theirs.Identities);
        Assert.All(ours.Columns, column => Assert.Equal(
            Enumerable.Range(0, ours.RowCount).Select(row => ((TextColumn)column)[row]),
            Enumerable.Range(0, theirs.RowCount).Select(row => ((TextColumn)theirs[column.Name])[row])));
    }

    [Fact]
    public async Task ADatabaseStatesWhatItsColumnsHold_AndWhatItStatesOutranksWhatTheCellsLookLike()
    {
        await using var connection = new SqliteConnection("Data Source=:memory:");
        await connection.OpenAsync(TestContext.Current.CancellationToken);

        await using (var create = connection.CreateCommand())
        {
            create.CommandText =
                "create table part (code text, stock integer, price real);"
                + "insert into part values ('007', 3, 1.5), ('012', 4, 2.0), ('345', 5, 3.0);";

            await create.ExecuteNonQueryAsync(TestContext.Current.CancellationToken);
        }

        await using var query = connection.CreateCommand();
        query.CommandText = "select code, stock, price from part";

        await using var reader = await query.ExecuteReaderAsync(TestContext.Current.CancellationToken);

        var proposal = (await Pdd.Create().ReadDbAsync(reader)).ProposedKinds();

        // Codes written in digits are words to the database, and stay words: read as whole numbers, 007 would be 7.
        Assert.Equal(ColumnKind.Text, proposal["code"].Kind);
        Assert.Equal(ColumnKind.Text, proposal["code"].Stated);
        Assert.Equal(ColumnKind.Integer, proposal["stock"].Kind);
        Assert.Equal(ColumnKind.Integer, proposal["stock"].Stated);

        // A price of 2.0 is handed over as 2, and is still the number with a fraction the database says it is.
        Assert.Equal(ColumnKind.Number, proposal["price"].Kind);
        Assert.Equal(ColumnKind.Number, proposal["price"].Stated);
    }

    [Fact]
    public async Task EveryKindAReaderHandsOver_IsStated_AndBindsAsWhatItIs()
    {
        using var table = new DataTable();
        table.Columns.Add("flag", typeof(bool));
        table.Columns.Add("when", typeof(DateTime));
        table.Columns.Add("count", typeof(int));
        table.Columns.Add("share", typeof(decimal));
        table.Columns.Add("name", typeof(string));
        table.Columns.Add("grade", typeof(char));
        table.Rows.Add(true, new DateTime(2015, 2, 18, 9, 30, 0, DateTimeKind.Utc), 3, 0.25m, "a", 'x');
        table.Rows.Add(false, new DateTime(2015, 2, 19, 9, 30, 0, DateTimeKind.Utc), 4, 0.5m, "b", 'y');

        await using var reader = table.CreateDataReader();

        var pipeline = await Pdd.Create().ReadDbAsync(reader);
        var proposal = pipeline.ProposedKinds();

        Assert.Equal(
            [ColumnKind.Boolean, ColumnKind.Timestamp, ColumnKind.Integer, ColumnKind.Number, ColumnKind.Text, ColumnKind.Text],
            proposal.Columns.Select(column => column.Stated!.Value));
        Assert.Equal(proposal.Columns.Select(column => column.Stated!.Value), proposal.Columns.Select(column => column.Kind));

        var bound = pipeline
            .Declare(schema => schema.Boolean("flag").Timestamp("when").Integer("count").Number("share").Text("name", "grade"))
            .Build()
            .Prepare();

        Assert.Equal(new DateTime(2015, 2, 19, 9, 30, 0, DateTimeKind.Utc), ((Column<DateTime>)bound["when"])[1]);
        Assert.False(((Column<bool>)bound["flag"])[1]);
    }

    [Fact]
    public void AFrameHandedIn_StatesNothing_ForItsKindsMayBeAGuess()
    {
        var proposal = Pdd.Create().ReadDataFrame(Frame()).ProposedKinds();

        Assert.All(proposal.Columns, column => Assert.Null(column.Stated));
        Assert.Equal(ColumnKind.Number, proposal["fare"].Kind);
    }

    [Fact]
    public void ReadingNothingAtAll_IsRefused()
    {
        Assert.Throws<ArgumentNullException>(() => new DataFrameRowSource(null!));
        Assert.Throws<ArgumentNullException>(() => Pdd.Create().ReadDataFrame(null!));
        Assert.Throws<ArgumentNullException>(() => DataFrameSourceExtensions.ReadDataFrame(null!, Frame()));
        Assert.Throws<ArgumentNullException>(() => Pdd.Create().Read(null!, "nothing"));
        Assert.Throws<ArgumentException>(() => Pdd.Create().Read(new DataFrameRowSource(Frame()), " "));
    }

    [Fact]
    public async Task AQueryIsWaitedForOnceAndThenReadLikeAnythingElse()
    {
        // A real database rather than a stand-in for one: the question is whether a provider's reader
        // arrives as rows, and only a provider can answer that.
        await using var connection = new SqliteConnection("Data Source=:memory:");
        await connection.OpenAsync(TestContext.Current.CancellationToken);

        await using (var create = connection.CreateCommand())
        {
            create.CommandText =
                "create table passenger (fare real, sex text);"
                + "insert into passenger values (7.25, 'male'), (71.28, 'female'), (8.05, 'female');";

            await create.ExecuteNonQueryAsync(TestContext.Current.CancellationToken);
        }

        await using var query = connection.CreateCommand();
        query.CommandText = "select fare, sex from passenger";

        await using var reader = await query.ExecuteReaderAsync(TestContext.Current.CancellationToken);

        var pipeline = await Pdd.Create().ReadDbAsync(reader);

        var prepared = pipeline
            .Declare(schema => schema.Number("fare").Text("sex"))
            .SplitAtRandom(0.60, 0.20, seed: 1)
            .Encode("sex")
            .Build()
            .Run();

        Assert.Equal(3, prepared.Table.RowCount);
        Assert.Equal("read.rows", prepared.Declaration.Steps[0].Verb);
        Assert.Equal(7.25, ((Column<double>)prepared.Table["fare"])[0]);
    }

    [Fact]
    public async Task ReadingFromNoQueryAtAll_IsRefused()
    {
        await using var connection = new SqliteConnection("Data Source=:memory:");
        await connection.OpenAsync(TestContext.Current.CancellationToken);

        await using var query = connection.CreateCommand();
        query.CommandText = "select 1";

        await using var reader = await query.ExecuteReaderAsync(TestContext.Current.CancellationToken);

        await Assert.ThrowsAsync<ArgumentNullException>(() => Pdd.Create().ReadDbAsync(null!));
        await Assert.ThrowsAsync<ArgumentNullException>(
            () => DataFrameSourceExtensions.ReadDbAsync(null!, reader));
    }

    [Fact]
    public void AMomentInTimeCrossesTheSeamAsAMomentInTime()
    {
        // Everything leaves a frame as text, and a date written the short way is read back as a different
        // day on a machine set to another country. The round-trip form is the one that cannot be.
        var frame = new DataFrame(
            new PrimitiveDataFrameColumn<DateTime>("when", [new DateTime(2016, 12, 8, 14, 30, 0, DateTimeKind.Utc)]),
            new PrimitiveDataFrameColumn<double>("close", [111.5]));

        var source = new DataFrameRowSource(frame);
        var row = source.Rows.Single();

        Assert.StartsWith("2016-12-08T14:30:00", row[0], StringComparison.Ordinal);
        Assert.Equal("111.5", row[1]);
    }

    [Fact]
    public void AMomentAFrameHoldsAsAMoment_IsReadWhateverFormatItsColumnSaysAFileWritesItIn()
    {
        // The format says how a file writes its moments; a frame or a database that holds them as moments hands each
        // one over in the round-trip form, which no format mistakes for another moment.
        var frame = new DataFrame(new PrimitiveDataFrameColumn<DateTime>(
            "when", [new DateTime(2015, 11, 27, 9, 30, 0, DateTimeKind.Utc), new DateTime(2015, 3, 2, 0, 0, 0, DateTimeKind.Unspecified)]));
        var declared = new DeclareStep([new ColumnDeclaration("when", ColumnKind.Timestamp, false) { Format = "dd/MM/yyyy" }]);

        var when = (Column<DateTime>)SchemaBinding.Bind(declared, new DataFrameRowSource(frame))["when"];

        Assert.Equal(new DateTime(2015, 11, 27, 9, 30, 0, DateTimeKind.Utc), when[0]);
        Assert.Equal(new DateTime(2015, 3, 2, 0, 0, 0, DateTimeKind.Utc), when[1]);
    }
}
